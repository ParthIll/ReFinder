"""Known-item retrieval benchmark on the Kaggle "Corpus for Semantic Matching" dataset.

The corpus is ~4.2k files of scraped web text, each named after a topic keyword (tesla.csv, winds.csv, ...).
It ships without queries or relevance labels, so we build a known-item task, which is what ReFinder does:
"find the file I'm thinking of from something I remember about it".

For each sampled eval file we pick a sentence that mentions the file's keyword, cut it out of the indexed
text (so it can't be matched verbatim), and ask each retriever to find the source file using:

  keyword   the keyword alone ("tesla")                        -> favors exact-match/keyword search
  sentence  the held-out sentence                              -> paraphrase-free recall of a detail
  masked    the held-out sentence with the keyword removed     -> only the meaning is left

Retrievers: embedding models (chunked, max-pooled per file, cosine) and a BM25 keyword baseline over whole files.
Metrics: Recall@1/5/10 and MRR@10 (one relevant file per query).

    python -m evaluation.semantic_matching                         # 1000 files, 300 queries, MiniLM + BM25
    python -m evaluation.semantic_matching --docs 0 --queries 1000 # full corpus
    python -m evaluation.semantic_matching --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
"""

import argparse
import json
import random
import re
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from evaluation.common import REPO_ROOT, RESULTS_DIR, dense_scores
from refinder.embedder import DEFAULT_MODEL

KAGGLE_DATASET = "mantunes/corpus-for-semantic-matching"
TASKS = ("keyword", "sentence", "masked")
KS = (1, 5, 10)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Query:
    doc_id: int
    task: str
    text: str


def dataset_dir() -> Path:
    """Download (cached by kagglehub) and extract the corpus; returns the folder of .csv text files."""
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    out = root / "extracted"
    if not (out / "dataset").is_dir():
        with tarfile.open(root / "dataset.tar.xz") as tar:
            tar.extractall(out, filter="data")
    return out / "dataset"


def load_corpus(n_docs: int, seed: int) -> tuple[list[str], list[str]]:
    paths = sorted(dataset_dir().glob("*.csv"))
    names, texts = [], []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            names.append(p.stem)
            texts.append(text)
    if n_docs and n_docs < len(names):
        keep = sorted(random.Random(seed).sample(range(len(names)), n_docs))
        names, texts = [names[i] for i in keep], [texts[i] for i in keep]
    return names, texts


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def build_queries(names: list[str], texts: list[str], n_queries: int, seed: int) -> tuple[list[Query], list[str]]:
    """Pick one held-out sentence per eval file and return (queries, texts with those sentences removed)."""
    # A held-out sentence must be unique across the corpus, or retrieving another file that contains
    # the same boilerplate would be scored as a miss.
    counts = Counter(s for t in texts for s in sentences(t))
    rng = random.Random(seed)
    order = list(range(len(names)))
    rng.shuffle(order)

    queries, texts = [], list(texts)
    for i in order:
        if len(queries) // len(TASKS) >= n_queries:
            break
        kw = names[i].lower()
        if not kw.isalpha() or len(kw) < 3:
            continue
        kw_re = re.compile(rf"\b{re.escape(kw)}\w*", re.IGNORECASE)
        candidates = [
            s for s in sentences(texts[i])
            if counts[s] == 1 and 10 <= len(s.split()) <= 40 and kw_re.search(s)
        ]
        if not candidates:
            continue
        held_out = rng.choice(candidates)
        masked = re.sub(r"\s+", " ", kw_re.sub("", held_out)).strip()
        texts[i] = texts[i].replace(held_out, " ")
        queries += [Query(i, "keyword", kw), Query(i, "sentence", held_out), Query(i, "masked", masked)]
    return queries, texts


def ranks_from_scores(scores: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """scores: (n_queries, n_docs). Returns the 1-based rank of each query's gold doc."""
    gold_scores = scores[np.arange(len(gold)), gold]
    return (scores > gold_scores[:, None]).sum(axis=1) + 1


def metrics(ranks: np.ndarray) -> dict[str, float]:
    out = {f"R@{k}": float((ranks <= k).mean()) for k in KS}
    out["MRR@10"] = float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean())
    return out


def bm25_tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def run_bm25(texts: list[str], queries: list[Query]) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    bm25 = BM25Okapi([bm25_tokens(t) for t in texts])
    t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = np.stack([bm25.get_scores(bm25_tokens(q.text)) for q in queries])
    t_query = time.perf_counter() - t0
    return scores, {"index_s": t_index, "query_ms": 1000 * t_query / len(queries)}


def print_table(results: dict[str, dict[str, dict[str, float]]]) -> None:
    cols = [f"R@{k}" for k in KS] + ["MRR@10"]
    width = max(len(r) for r in results) + 2
    for task in TASKS:
        print(f"\n{task}")
        print(" " * width + "".join(f"{c:>9}" for c in cols))
        for name, by_task in results.items():
            print(f"{name:<{width}}" + "".join(f"{by_task[task][c]:>9.3f}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--docs", type=int, default=1000, help="files to index (0 = all ~4.2k)")
    ap.add_argument("--queries", type=int, default=300, help="eval files (each yields one query per task)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bm25", action="store_true")
    args = ap.parse_args()

    names, texts = load_corpus(args.docs, args.seed)
    queries, texts = build_queries(names, texts, args.queries, args.seed)
    n_eval = len(queries) // len(TASKS)
    print(f"Indexed {len(texts):,} files, {n_eval} eval files x {len(TASKS)} tasks = {len(queries)} queries")
    for q in queries[:3]:
        print(f"  e.g. [{q.task}] {names[q.doc_id]}.csv <- {q.text!r}")

    gold = np.array([q.doc_id for q in queries])
    task_of = np.array([q.task for q in queries])
    results, stats = {}, {}

    runs = [(m, lambda m=m: dense_scores(m, texts, [q.text for q in queries])) for m in args.models]
    if not args.no_bm25:
        runs.append(("bm25", lambda: run_bm25(texts, queries)))
    for name, run in runs:
        scores, stats[name] = run()
        ranks = ranks_from_scores(scores, gold)
        results[name] = {t: metrics(ranks[task_of == t]) for t in TASKS}

    print_table(results)
    print("\nspeed")
    for name, s in stats.items():
        extra = f", {s['chunks_per_s']:.0f} chunks/s embedding" if "chunks_per_s" in s else ""
        print(f"  {name}: {s['query_ms']:.1f} ms/query{extra}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"semantic_matching_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"args": vars(args), "n_files": len(texts), "n_eval_files": n_eval,
                               "results": results, "stats": stats}, indent=2))
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
