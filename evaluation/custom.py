"""Accuracy + speed benchmark on our own labeled corpus: a folder of real files plus a CSV of queries.

    Corpus/                   any nesting; .txt .md .pdf .docx .csv code etc. (images/archives are skipped)
    queries.csv               query,file,category
                              "notes from the lecture on gradient descent",school/ml/lecture3.pdf,student
                              - file: path relative to Corpus/ (a bare filename works if it's unique);
                                several acceptable answers can be joined with "|"
                              - category: optional free-text label (persona, query style, ...) for breakdowns

Every file is extracted, chunked (~200 tokens, overlapping) and embedded exactly as the app will; a file's
score is its best chunk's cosine similarity. A BM25 keyword baseline runs on the same text for reference.

Accuracy: Hit@1/5/10 (a correct file in the top k), MRR@10, nDCG@10, overall and per category / file type.
Speed: extraction, model load, indexing throughput, index size, and per-query search latency (p50/p95),
measured one query at a time like a user typing into the search box.

    python -m evaluation.custom                                     # Corpus/ + queries.csv, MiniLM vs BM25
    python -m evaluation.custom --corpus path/to/Corpus --queries path/to/queries.csv
    python -m evaluation.custom --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
    python -m evaluation.custom --fresh                             # ignore embedding cache (re-time indexing)
    python -m evaluation.custom --distractors 300                   # + 300 unlabeled web-page files as noise
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from evaluation.common import REPO_ROOT, RESULTS_DIR, embed_corpus
from refinder.embedder import DEFAULT_MODEL, Embedder
from refinder.extract import UnsupportedFile, extract_text

KS = (1, 5, 10)
LATENCY_RUNS = 200


# ---------------------------------------------------------------- data

def load_corpus(root: Path) -> tuple[list[str], list[str], dict]:
    """Returns (relative paths, texts, extraction stats) for every supported file under root."""
    paths, texts, skipped, empty, failed = [], [], Counter(), [], {}
    t0 = time.perf_counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        try:
            text = extract_text(p)
        except UnsupportedFile:
            skipped[p.suffix.lower() or p.name] += 1
            continue
        except Exception as e:  # corrupt PDF, password-protected docx, ...
            failed[rel] = f"{type(e).__name__}: {e}"
            continue
        if not text:
            empty.append(rel)
        paths.append(rel)
        texts.append(text)
    return paths, texts, {
        "files": len(paths),
        "by_type": dict(Counter(Path(p).suffix.lower() or "(none)" for p in paths)),
        "text_mb": sum(len(t.encode()) for t in texts) / 1e6,
        "extract_s": time.perf_counter() - t0,
        "skipped": dict(skipped),
        "empty": empty,
        "failed": failed,
    }


def load_distractors(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """n unrelated saved-web-page files from the Kaggle corpus: harder retrieval and a bigger index to time."""
    from evaluation.semantic_matching import dataset_dir

    files = sorted(dataset_dir().glob("*.csv"))
    files = random.Random(seed).sample(files, min(n, len(files)))
    texts = [f.read_text(encoding="utf-8", errors="ignore").strip() for f in files]
    return [f"_distractors/{f.stem}.txt" for f, t in zip(files, texts) if t], [t for t in texts if t]


def resolve(name: str, index: dict[str, int], by_basename: dict[str, list[int]]) -> int | None:
    name = name.strip().replace("\\", "/").removeprefix("./").removeprefix("Corpus/")
    if name in index:
        return index[name]
    hits = by_basename.get(Path(name).name, [])
    return hits[0] if len(hits) == 1 else None


def load_queries(csv_path: Path, paths: list[str]) -> tuple[list[dict], list[str]]:
    index = {p: i for i, p in enumerate(paths)}
    by_basename = defaultdict(list)
    for i, p in enumerate(paths):
        by_basename[Path(p).name].append(i)

    queries, problems = [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_cols = {"query", "file"} - set(reader.fieldnames or [])
        if missing_cols:
            sys.exit(f"{csv_path} needs columns query,file[,category]; missing {sorted(missing_cols)}")
        for line_no, row in enumerate(reader, start=2):
            query = (row.get("query") or "").strip()
            names = [n for n in (row.get("file") or "").split("|") if n.strip()]
            if not query or not names:
                problems.append(f"line {line_no}: empty query or file")
                continue
            gold = [resolve(n, index, by_basename) for n in names]
            unresolved = [n for n, g in zip(names, gold) if g is None]
            if unresolved:
                problems.append(f"line {line_no}: file not found or ambiguous: {unresolved}")
                continue
            queries.append({"query": query, "gold": set(gold), "category": (row.get("category") or "").strip() or "-"})
    return queries, problems


# ---------------------------------------------------------------- scoring

def ranks_of_gold(scores: np.ndarray, queries: list[dict]) -> np.ndarray:
    """1-based rank of the best-ranked correct file for each query."""
    out = np.empty(len(queries), dtype=int)
    for i, q in enumerate(queries):
        best = max(scores[i, g] for g in q["gold"])
        out[i] = int((scores[i] > best).sum()) + 1
    return out


def ndcg10(scores: np.ndarray, queries: list[dict]) -> np.ndarray:
    out = np.empty(len(queries))
    discounts = 1.0 / np.log2(np.arange(2, 12))
    for i, q in enumerate(queries):
        top = np.argsort(-scores[i])[:10]
        dcg = sum(discounts[r] for r, d in enumerate(top) if d in q["gold"])
        out[i] = dcg / discounts[: min(len(q["gold"]), 10)].sum()
    return out


def metrics(ranks: np.ndarray, ndcg: np.ndarray) -> dict[str, float]:
    out = {f"Hit@{k}": float((ranks <= k).mean()) for k in KS}
    out["MRR@10"] = float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean())
    out["nDCG@10"] = float(ndcg.mean())
    out["n"] = int(len(ranks))
    return out


def breakdowns(ranks: np.ndarray, ndcg: np.ndarray, queries: list[dict], paths: list[str]) -> dict:
    groups = {"overall": {"all": np.arange(len(queries))}, "category": defaultdict(list), "file_type": defaultdict(list)}
    for i, q in enumerate(queries):
        groups["category"][q["category"]].append(i)
        groups["file_type"][Path(paths[min(q["gold"])]).suffix.lower() or "(none)"].append(i)
    return {
        kind: {name: metrics(ranks[idx], ndcg[idx]) for name, idx in sorted(members.items())}
        for kind, members in groups.items()
    }


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    a = np.array(samples_ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


# ---------------------------------------------------------------- retrievers

class DenseRetriever:
    def __init__(self, model_name: str, texts: list[str], use_cache: bool):
        t0 = time.perf_counter()
        self.embedder = Embedder(model_name)
        self.load_s = time.perf_counter() - t0
        print(f"\n[{model_name}] dim={self.embedder.dim} device={self.embedder.model.device} (loaded in {self.load_s:.1f}s)")
        self.n_files = len(texts)
        self.vecs, owner, self.index_stats = embed_corpus(self.embedder, texts, use_cache=use_cache)
        # Chunks are contiguous per file, so a file's score is the max over its slice of chunk scores.
        self.starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]) if len(owner) else np.array([], int)
        self.doc_ids = owner[self.starts] if len(owner) else np.array([], int)

    def score(self, q_vecs: np.ndarray) -> np.ndarray:
        scores = np.full((len(q_vecs), self.n_files), -np.inf, dtype=np.float32)
        if len(self.starts):
            scores[:, self.doc_ids] = np.maximum.reduceat(q_vecs @ self.vecs.T, self.starts, axis=1)
        return scores

    def search(self, query: str, k: int = 10) -> np.ndarray:
        s = self.score(self.embedder.encode_query(query)[None])[0]
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return top[np.argsort(-s[top])]

    def all_scores(self, queries: list[str]) -> np.ndarray:
        return self.score(self.embedder.encode_queries(queries))


class BM25Retriever:
    def __init__(self, texts: list[str]):
        self.stemmer = Stemmer.Stemmer("english")
        t0 = time.perf_counter()
        self.bm25 = bm25s.BM25()
        self.bm25.index(self._tok(texts), show_progress=False)
        self.index_stats = {"index_s": time.perf_counter() - t0}
        self.n_files = len(texts)

    def _tok(self, texts: list[str]):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)

    def all_scores(self, queries: list[str]) -> np.ndarray:
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)
        return np.stack([self.bm25.get_scores(t) for t in tokens]).astype(np.float32)

    def search(self, query: str, k: int = 10) -> np.ndarray:
        s = self.all_scores([query])[0]
        return np.argsort(-s)[:k]


def time_queries(retriever, queries: list[str]) -> dict[str, float]:
    sample = (queries * (LATENCY_RUNS // max(len(queries), 1) + 1))[:LATENCY_RUNS]
    for q in sample[:5]:  # warm-up (first MPS/CUDA calls compile kernels)
        retriever.search(q)
    samples = []
    for q in sample:
        t0 = time.perf_counter()
        retriever.search(q)
        samples.append(1000 * (time.perf_counter() - t0))
    return latency_stats(samples)


# ---------------------------------------------------------------- report

def print_metrics_table(title: str, rows: dict[str, dict]) -> None:
    cols = [f"Hit@{k}" for k in KS] + ["MRR@10", "nDCG@10"]
    width = max([len(r) for r in rows] + [len(title)]) + 2
    print(f"\n{title:<{width}}" + "".join(f"{c:>9}" for c in cols) + f"{'n':>6}")
    for name, m in rows.items():
        print(f"{name:<{width}}" + "".join(f"{m[c]:>9.3f}" for c in cols) + f"{m['n']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "Corpus")
    ap.add_argument("--queries", type=Path, default=REPO_ROOT / "queries.csv")
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--distractors", type=int, default=0, help="add N unlabeled Kaggle web-text files as noise")
    ap.add_argument("--fresh", action="store_true", help="re-embed even if cached, so indexing speed is measured")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        sys.exit(f"corpus folder not found: {args.corpus}")
    if not args.queries.is_file():
        sys.exit(f"queries CSV not found: {args.queries}")

    paths, texts, corpus_stats = load_corpus(args.corpus)
    if args.distractors:
        d_paths, d_texts = load_distractors(args.distractors)
        paths, texts = paths + d_paths, texts + d_texts
        corpus_stats["distractors"] = len(d_paths)
        corpus_stats["text_mb"] += sum(len(t.encode()) for t in d_texts) / 1e6
    queries, problems = load_queries(args.queries, paths)
    corpus_stats["files"] = len(paths)
    print(f"Corpus: {corpus_stats['files']} files ({corpus_stats['text_mb']:.1f} MB text) "
          f"extracted in {corpus_stats['extract_s']:.1f}s; types {corpus_stats['by_type']}"
          + (f" + {corpus_stats['distractors']} distractor files" if args.distractors else ""))
    if corpus_stats["skipped"]:
        print(f"  skipped (unsupported): {corpus_stats['skipped']}")
    for rel, err in corpus_stats["failed"].items():
        print(f"  FAILED to extract {rel}: {err}")
    for rel in corpus_stats["empty"]:
        print(f"  WARNING no text extracted (scanned PDF?): {rel}")
    print(f"Queries: {len(queries)} usable, covering {len({g for q in queries for g in q['gold']})} files")
    for p in problems:
        print(f"  SKIPPED query, {p}")
    if not queries:
        sys.exit("no usable queries")

    q_texts = [q["query"] for q in queries]
    retrievers = [(m, lambda m=m: DenseRetriever(m, texts, use_cache=not args.fresh)) for m in args.models]
    if not args.no_bm25:
        retrievers.append(("bm25", lambda: BM25Retriever(texts)))

    report, per_query = {}, {}
    for name, build in retrievers:
        r = build()
        scores = r.all_scores(q_texts)
        ranks, nd = ranks_of_gold(scores, queries), ndcg10(scores, queries)
        top3 = np.argsort(-scores, axis=1)[:, :3]
        per_query[name] = (ranks, top3)
        speed = {"index": r.index_stats, "search": time_queries(r, q_texts)}
        if isinstance(r, DenseRetriever):
            speed["model_load_s"] = r.load_s
            speed["index_mb"] = r.vecs.nbytes / 1e6
        report[name] = {"accuracy": breakdowns(ranks, nd, queries, paths), "speed": speed}

    print_metrics_table("ACCURACY", {n: rep["accuracy"]["overall"]["all"] for n, rep in report.items()})
    for kind in ("category", "file_type"):
        groups = next(iter(report.values()))["accuracy"][kind]
        if len(groups) > 1:
            for name, rep in report.items():
                print_metrics_table(f"{name} by {kind}", rep["accuracy"][kind])

    print("\nSPEED")
    for name, rep in report.items():
        s, idx = rep["speed"], rep["speed"]["index"]
        line = f"  {name}: search p50 {s['search']['p50_ms']:.2f} ms, p95 {s['search']['p95_ms']:.2f} ms"
        if "embed_s" in idx:
            total = corpus_stats["extract_s"] + idx["chunk_s"] + idx["embed_s"]
            line += (f" | index {idx['chunks']:,} chunks in {idx['chunk_s'] + idx['embed_s']:.1f}s "
                     f"({idx['chunks_per_s']:.0f} chunks/s, {corpus_stats['files'] / total:.1f} files/s incl. extraction)"
                     f" | {s['index_mb']:.1f} MB vectors | model load {s['model_load_s']:.1f}s")
        elif idx.get("cached"):
            line += f" | index from cache (use --fresh to time it) | {s['index_mb']:.1f} MB vectors"
        else:
            line += f" | index built in {idx['index_s']:.2f}s"
        print(line)

    out_dir = RESULTS_DIR / f"custom_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "corpus": corpus_stats,
         "query_problems": problems, "results": report}, indent=2))
    with (out_dir / "per_query.csv").open("w", newline="") as f:
        w = csv.writer(f)
        names = list(per_query)
        w.writerow(["query", "category", "expected"] + [f"{n} rank" for n in names] + [f"{n} top3" for n in names])
        for i, q in enumerate(queries):
            w.writerow([q["query"], q["category"], " | ".join(paths[g] for g in sorted(q["gold"]))]
                       + [int(per_query[n][0][i]) for n in names]
                       + [" | ".join(paths[d] for d in per_query[n][1][i]) for n in names])
    print(f"\nsaved {out_dir.relative_to(REPO_ROOT)}/ (summary.json, per_query.csv for reviewing misses)")


if __name__ == "__main__":
    main()
