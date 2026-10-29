"""Accuracy + speed benchmark on our own labeled corpus: a folder of real files plus a CSV of queries.

    Corpus/                   any nesting; .pdf .docx .pptx .xlsx .txt .md .csv .tsv (other types are skipped)
    queries.csv               query,file,category
                              "notes from the lecture on gradient descent",school/ml/lecture3.pdf,student
                              - file: path relative to the CSV's folder or Corpus/ (a bare filename works if
                                it's unique); several acceptable answers can be joined with "|"
                              - category: optional free-text label (persona, query style, ...) for breakdowns

The team template is also understood: source_file (instead of file), secondary_relevant_files (";"-separated,
counted as partially relevant in nDCG), document_type (as category), query_type and split (tune/test).
Without --queries, every CSV inside the corpus that has a query column is used, with paths resolved relative
to its own folder; those CSVs (and same-named spreadsheet copies) are left out of the index.

Every file is extracted, chunked (whole paragraphs/slides/rows packed to ~200 tokens, or fixed windows with
--chunking window) and embedded as the app will; the location of each chunk is kept, and a file's
score is its best chunk's cosine similarity. A BM25 keyword baseline runs on the same text for reference.

Accuracy: Hit@1/5/10 (a correct file in the top k), MRR@10, nDCG@10, overall and per category / file type.
Speed: extraction, model load, indexing throughput, index size, and per-query search latency (p50/p95),
measured one query at a time like a user typing into the search box.

    python -m evaluation.custom                                     # Corpus/ + queries.csv, MiniLM vs BM25
    python -m evaluation.custom --corpus path/to/Corpus --queries path/to/queries.csv
    python -m evaluation.custom --corpus "ReFinder File Corpus"       # finds the query CSVs inside the corpus
    python -m evaluation.custom ... --split test                    # only queries whose split column is "test"
    python -m evaluation.custom --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
    python -m evaluation.custom --fresh                             # ignore embedding cache (re-time indexing)
    python -m evaluation.custom --distractors 300                   # + 300 unlabeled web-page files as noise
    python -m evaluation.custom --chunking window paragraph         # compare chunking strategies
    python -m evaluation.custom --tree-beams 1 2 4                  # + coarse-to-fine category tree search
"""

import argparse
import csv
import hashlib
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from evaluation.common import CACHE_DIR, CHUNKING, REPO_ROOT, RESULTS_DIR, embed_corpus
from refinder.embedder import DEFAULT_MODEL, Embedder
from refinder.extract import Piece, UnsupportedFile, describe_location, extract_pieces, pieces_from_text
from refinder.tree import VectorTree

KS = (1, 5, 10)
LATENCY_RUNS = 200


# ---------------------------------------------------------------- data

EXTRACT_CACHE = CACHE_DIR.parent / "extract"


def extract_cached(p: Path, use_cache: bool) -> list[Piece]:
    """extract_pieces with a per-file disk cache keyed by path, size and mtime (some PDFs take minutes to parse)."""
    st = p.stat()
    key = hashlib.sha256(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:24]
    cache = EXTRACT_CACHE / f"{key}.pkl"
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    file_pieces = extract_pieces(p)
    EXTRACT_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(file_pieces))
    return file_pieces


def load_corpus(
    root: Path, exclude: set[Path] = frozenset(), use_cache: bool = True
) -> tuple[list[str], list[str], list[list[Piece]], dict]:
    """Returns (relative paths, texts, pieces, extraction stats) for every supported file under root."""
    paths, texts, pieces, skipped, empty, failed = [], [], [], Counter(), [], {}
    t0 = time.perf_counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve() in exclude:
            continue
        rel = p.relative_to(root).as_posix()
        try:
            file_pieces = extract_cached(p, use_cache)
        except UnsupportedFile:
            skipped[p.suffix.lower() or p.name] += 1
            continue
        except Exception as e:  # corrupt PDF, password-protected docx, ...
            failed[rel] = f"{type(e).__name__}: {e}"
            continue
        text = "\n\n".join(piece.text for piece in file_pieces)
        if not text:
            empty.append(rel)
        paths.append(rel)
        texts.append(text)
        pieces.append(file_pieces)
    return paths, texts, pieces, {
        "files": len(paths),
        "by_type": dict(Counter(Path(p).suffix.lower() or "(none)" for p in paths)),
        "text_mb": sum(len(t.encode()) for t in texts) / 1e6,
        "extract_s": time.perf_counter() - t0,
        "skipped": dict(skipped),
        "empty": empty,
        "failed": failed,
    }


def load_distractors(n: int, seed: int = 0) -> tuple[list[str], list[str], list[list[Piece]]]:
    """n unrelated saved-web-page files from the Kaggle corpus: harder retrieval and a bigger index to time."""
    from evaluation.semantic_matching import dataset_dir

    files = sorted(dataset_dir().glob("*.csv"))
    files = random.Random(seed).sample(files, min(n, len(files)))
    texts = [f.read_text(encoding="utf-8", errors="ignore").strip() for f in files]
    kept = [(f, t) for f, t in zip(files, texts) if t]
    return [f"_distractors/{f.stem}.txt" for f, _ in kept], [t for _, t in kept], [pieces_from_text(t) for _, t in kept]


FILE_COLUMNS = ("file", "source_file", "corpus_path")
CATEGORY_COLUMNS = ("category", "document_type")


def _header(csv_path: Path) -> list[str]:
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            return [c.strip() for c in next(csv.reader(f), [])]
    except (OSError, UnicodeDecodeError):
        return []


def is_query_csv(csv_path: Path) -> bool:
    cols = set(_header(csv_path))
    return "query" in cols and bool(cols & set(FILE_COLUMNS))


def find_query_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if is_query_csv(p))


def query_file_exclusions(csvs: list[Path]) -> set[Path]:
    """The query CSVs plus same-named copies next to them (e.g. a Google Sheet exported as .xlsx)."""
    out = set()
    for c in csvs:
        out |= {p.resolve() for p in c.parent.glob(f"{c.stem}.*")}
    return out


class Resolver:
    """Maps a path written in a query CSV to a corpus file index."""

    def __init__(self, root: Path, paths: list[str]):
        self.root = root.resolve()
        self.paths = paths
        self.index = {p: i for i, p in enumerate(paths)}
        self.by_basename = defaultdict(list)
        for i, p in enumerate(paths):
            self.by_basename[Path(p).name].append(i)

    def __call__(self, name: str, base: str) -> int | None:
        name = name.strip().replace("\\", "/").removeprefix("./").removeprefix("Corpus/")
        for candidate in (f"{base}/{name}" if base else name, name):
            if candidate in self.index:
                return self.index[candidate]
        hits = self.by_basename.get(Path(name).name, [])
        local = [i for i in hits if not base or self.paths[i].startswith(base + "/")]
        if len(local) == 1:
            return local[0]
        return hits[0] if len(hits) == 1 else None


def _split_names(value: str | None) -> list[str]:
    return [n.strip() for n in (value or "").replace("|", ";").split(";") if n.strip() and n.strip() != "..."]


def load_queries(csv_paths: list[Path], root: Path, paths: list[str]) -> tuple[list[dict], list[str]]:
    resolve = Resolver(root, paths)
    queries, problems = [], []
    for csv_path in csv_paths:
        try:
            base = csv_path.resolve().parent.relative_to(root.resolve()).as_posix()
        except ValueError:  # CSV lives outside the corpus: paths are relative to the corpus root
            base = ""
        base = "" if base == "." else base
        label = csv_path.name if not base else f"{base}/{csv_path.name}"
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = [c.strip() for c in reader.fieldnames or []]
            file_col = next((c for c in FILE_COLUMNS if c in cols), None)
            if "query" not in cols or not file_col:
                sys.exit(f"{label} needs a query column and a file (or source_file) column; has {cols}")
            cat_col = next((c for c in CATEGORY_COLUMNS if c in cols), None)
            for line_no, row in enumerate(reader, start=2):
                row = {(k or "").strip(): v for k, v in row.items()}
                query = (row.get("query") or "").strip()
                names = _split_names(row.get(file_col))
                if not query or not names:
                    if query or names:
                        problems.append(f"{label} line {line_no}: empty query or file")
                    continue
                gold = [resolve(n, base) for n in names]
                unresolved = [n for n, g in zip(names, gold) if g is None]
                if unresolved:
                    problems.append(f"{label} line {line_no}: file not found or ambiguous: {unresolved}")
                    continue
                secondary = {g for g in (resolve(n, base) for n in _split_names(row.get("secondary_relevant_files")))
                             if g is not None} - set(gold)
                queries.append({
                    "query": query,
                    "gold": set(gold),
                    "secondary": secondary,
                    "category": ((row.get(cat_col) or "").strip() if cat_col else "") or "-",
                    "query_type": (row.get("query_type") or "").strip() or "-",
                    "split": (row.get("split") or "").strip() or "-",
                    "source": label,
                })
    return queries, problems


# ---------------------------------------------------------------- scoring

def ranks_of_gold(scores: np.ndarray, queries: list[dict]) -> np.ndarray:
    """1-based rank of the best-ranked correct file for each query (past the last file if it was never scored)."""
    out = np.empty(len(queries), dtype=int)
    for i, q in enumerate(queries):
        best = max(scores[i, g] for g in q["gold"])
        out[i] = int((scores[i] > best).sum()) + 1 if np.isfinite(best) else scores.shape[1] + 1
    return out


def ndcg10(scores: np.ndarray, queries: list[dict]) -> np.ndarray:
    """Graded nDCG@10: the target file(s) count 2, secondary relevant files count 1."""
    out = np.empty(len(queries))
    discounts = 1.0 / np.log2(np.arange(2, 12))
    for i, q in enumerate(queries):
        gain = {d: 2.0 for d in q["gold"]} | {d: 1.0 for d in q.get("secondary", ())}
        top = np.argsort(-scores[i])[:10]
        dcg = sum(discounts[r] * gain.get(int(d), 0.0) for r, d in enumerate(top))
        ideal = sorted(gain.values(), reverse=True)[:10]
        out[i] = dcg / sum(discounts[r] * g for r, g in enumerate(ideal))
    return out


def metrics(ranks: np.ndarray, ndcg: np.ndarray) -> dict[str, float]:
    out = {f"Hit@{k}": float((ranks <= k).mean()) for k in KS}
    out["MRR@10"] = float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean())
    out["nDCG@10"] = float(ndcg.mean())
    out["n"] = int(len(ranks))
    return out


BREAKDOWNS = ("folder", "category", "query_type", "split", "file_type")


def breakdowns(ranks: np.ndarray, ndcg: np.ndarray, queries: list[dict], paths: list[str]) -> dict:
    groups = {"overall": {"all": np.arange(len(queries))}} | {kind: defaultdict(list) for kind in BREAKDOWNS}
    for i, q in enumerate(queries):
        target = paths[min(q["gold"])]
        groups["folder"][target.split("/")[0] if "/" in target else "(root)"].append(i)
        groups["file_type"][Path(target).suffix.lower() or "(none)"].append(i)
        for kind in ("category", "query_type", "split"):
            groups[kind][q[kind]].append(i)
    return {
        kind: {name: metrics(ranks[idx], ndcg[idx]) for name, idx in sorted(members.items())}
        for kind, members in groups.items()
    }


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    a = np.array(samples_ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


# ---------------------------------------------------------------- retrievers

class DenseRetriever:
    def __init__(self, model_name: str, texts: list[str], pieces: list[list[Piece]], chunking: str, use_cache: bool):
        t0 = time.perf_counter()
        self.embedder = Embedder(model_name)
        self.load_s = time.perf_counter() - t0
        print(f"\n[{model_name} | {chunking} chunks] dim={self.embedder.dim} device={self.embedder.model.device} "
              f"(loaded in {self.load_s:.1f}s)")
        self.n_files = len(texts)
        self.pieces = pieces
        self.vecs, self.owner, self.spans, self.index_stats = embed_corpus(
            self.embedder, texts, use_cache=use_cache, pieces=[[p.text for p in ps] for ps in pieces], chunking=chunking
        )
        owner = self.owner
        # Chunks are contiguous per file, so a file's score is the max over its slice of chunk scores.
        self.starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]) if len(owner) else np.array([], int)
        self.doc_ids = owner[self.starts] if len(owner) else np.array([], int)
        self.q_vecs = None

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
        self.q_vecs = self.embedder.encode_queries(queries)
        return self.score(self.q_vecs)

    def match(self, query_idx: int, file_id: int) -> str:
        """Where in file_id the query's best chunk is, plus the start of that passage (for result snippets)."""
        chunk_ids = np.flatnonzero(self.owner == file_id)
        if not len(chunk_ids):
            return ""
        best = chunk_ids[np.argmax(self.vecs[chunk_ids] @ self.q_vecs[query_idx])]
        number, total = best - chunk_ids[0] + 1, len(chunk_ids)
        first, last = self.spans[best]
        if first < 0:  # window chunks don't track pieces
            return f"chunk {number}/{total}"
        ps = self.pieces[file_id]
        snippet = ps[first].text.replace("\n", " ")
        return f"{describe_location(number, total, ps[first].location, ps[last].location)}: {snippet[:90]}"


class TreeRetriever:
    """Search through a VectorTree built over a DenseRetriever's vectors (same model, same chunks).

    beam:   prune, scoring only the files reached by keeping `beam` nodes per level.
    weight: boost, flat best-chunk score + weight x the file's category match (see refinder/tree.py).
    """

    def __init__(self, dense: DenseRetriever, tree: VectorTree, build_s: float,
                 beam: int | None = None, weight: float | None = None):
        assert (beam is None) != (weight is None)
        self.dense, self.tree, self.beam, self.weight = dense, tree, beam, weight
        self.n_files = dense.n_files
        self.index_stats = {"index_s": build_s, "nodes": sum(1 for _ in tree.nodes()), "depth": tree.depth,
                            "parts": len(tree.part_vecs)}
        self.candidates: list[np.ndarray] = []
        self.chunks_scored: list[int] = []

    def _prune_one(self, q: np.ndarray) -> np.ndarray:
        s = np.full(self.n_files, -np.inf, dtype=np.float32)
        files = self.tree.candidates(q, self.beam)
        if len(files):
            s[files], n_chunks = self.tree.score_files(q, files)
            self.chunks_scored.append(n_chunks)
        self.candidates.append(files)
        return s

    def score(self, q_vecs: np.ndarray) -> np.ndarray:
        if self.beam is not None:
            return np.stack([self._prune_one(q) for q in q_vecs])
        return self.dense.score(q_vecs) + self.weight * self.tree.category_scores(q_vecs)

    def all_scores(self, queries: list[str]) -> np.ndarray:
        self.candidates, self.chunks_scored = [], []
        d = self.dense
        q_vecs = d.q_vecs if d.q_vecs is not None and len(d.q_vecs) == len(queries) else d.embedder.encode_queries(queries)
        return self.score(q_vecs)

    def search(self, query: str, k: int = 10) -> np.ndarray:
        s = self.score(self.dense.embedder.encode_query(query)[None])[0]
        return np.argsort(-s)[:k]

    def routing(self, queries: list[dict]) -> dict[str, float] | None:
        if self.beam is None:
            return None
        """How often the tree kept a correct file in reach, and how much of the index each query touched."""
        n = len(queries)
        reached = [bool(q["gold"] & set(map(int, c))) for q, c in zip(queries, self.candidates[:n])]
        return {"gold_reached": float(np.mean(reached)),
                "files_searched": float(np.mean([len(c) for c in self.candidates[:n]])),
                "chunks_scored": float(np.mean(self.chunks_scored[:n])),
                "chunks_total": int(len(self.dense.vecs))}


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
    ap.add_argument("--queries", type=Path, nargs="+",
                    help="query CSV(s); default: every query CSV found inside the corpus, else ./queries.csv")
    ap.add_argument("--split", help="only use queries whose split column has this value (e.g. test)")
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--chunking", nargs="+", default=["paragraph"], choices=CHUNKING,
                    help="paragraph: pack whole paragraphs/slides/rows (default); window: fixed token windows")
    ap.add_argument("--tree-beams", type=int, nargs="*", default=[],
                    help="also search each model through a semantic category tree, keeping this many nodes per level")
    ap.add_argument("--tree-boost", type=float, nargs="*", default=[],
                    help="also rank each model by best chunk + this weight x the file's category match")
    ap.add_argument("--tree-parts", type=int, nargs="+", default=[1],
                    help="max vectors per file when placing it in the tree (big files split by topic)")
    ap.add_argument("--tree-part-chunks", type=int, default=40, help="roughly one part vector per this many chunks")
    ap.add_argument("--tree-branching", type=int, default=8, help="max child categories per tree node")
    ap.add_argument("--tree-leaf", type=int, default=10, help="max file parts in a leaf category")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--distractors", type=int, default=0, help="add N unlabeled Kaggle web-text files as noise")
    ap.add_argument("--fresh", action="store_true",
                    help="re-extract and re-embed even if cached, so extraction and indexing speed are measured")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        sys.exit(f"corpus folder not found: {args.corpus}")
    query_csvs = args.queries or find_query_csvs(args.corpus) or [REPO_ROOT / "queries.csv"]
    for q in query_csvs:
        if not q.is_file():
            sys.exit(f"queries CSV not found: {q}")

    paths, texts, pieces, corpus_stats = load_corpus(
        args.corpus, exclude=query_file_exclusions(query_csvs), use_cache=not args.fresh
    )
    if args.distractors:
        d_paths, d_texts, d_pieces = load_distractors(args.distractors)
        paths, texts, pieces = paths + d_paths, texts + d_texts, pieces + d_pieces
        corpus_stats["distractors"] = len(d_paths)
        corpus_stats["text_mb"] += sum(len(t.encode()) for t in d_texts) / 1e6
    queries, problems = load_queries(query_csvs, args.corpus, paths)
    if args.split:
        queries = [q for q in queries if q["split"] == args.split]
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
    print(f"Queries: {len(queries)} usable from {len(query_csvs)} CSV(s), covering "
          f"{len({g for q in queries for g in q['gold']})} files" + (f" (split={args.split})" if args.split else ""))
    for c, n in sorted(Counter(q["source"] for q in queries).items()):
        print(f"  {n:4d}  {c}")
    for p in problems:
        print(f"  SKIPPED query, {p}")
    if not queries:
        sys.exit("no usable queries")

    q_texts = [q["query"] for q in queries]
    out_dir = RESULTS_DIR / f"custom_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dense_cache, tree_cache = {}, {}

    def dense(m: str, c: str) -> DenseRetriever:
        if (m, c) not in dense_cache:
            dense_cache[(m, c)] = DenseRetriever(m, texts, pieces, c, use_cache=not args.fresh)
        return dense_cache[(m, c)]

    def tree(m: str, c: str, parts: int, **mode) -> TreeRetriever:
        d = dense(m, c)
        if (m, c, parts) not in tree_cache:
            t0 = time.perf_counter()
            t = VectorTree(d.vecs, d.owner, len(texts), args.tree_branching, args.tree_leaf,
                           max_parts=parts, part_chunks=args.tree_part_chunks)
            build_s = time.perf_counter() - t0
            t.label(texts)
            tree_cache[(m, c, parts)] = (t, build_s)
            short = m.split("/")[-1]
            print(f"\n[{short} tree, up to {parts} part(s)/file] {len(t.part_vecs)} parts, "
                  f"{sum(1 for _ in t.nodes())} nodes, depth {t.depth}, built in {build_s:.2f}s; top two levels:\n"
                  f"{t.outline(paths)}")
            (out_dir / f"tree_{short}_{c}_parts{parts}.txt").write_text(t.outline(paths, max_depth=99) + "\n")
        t, build_s = tree_cache[(m, c, parts)]
        return TreeRetriever(d, t, build_s, **mode)

    retrievers = []
    for m in args.models:
        for c in args.chunking:
            name = f"{m.split('/')[-1]} [{c}]"
            retrievers.append((name, lambda m=m, c=c: dense(m, c)))
            for p in args.tree_parts:
                suffix = f" p={p}" if len(args.tree_parts) > 1 else ""
                retrievers += [(f"{name} tree b={b}{suffix}", lambda m=m, c=c, p=p, b=b: tree(m, c, p, beam=b))
                               for b in args.tree_beams]
                retrievers += [(f"{name} boost {w:g}{suffix}", lambda m=m, c=c, p=p, w=w: tree(m, c, p, weight=w))
                               for w in args.tree_boost]
    if not args.no_bm25:
        retrievers.append(("bm25", lambda: BM25Retriever(texts)))

    report, per_query = {}, {}
    for name, build in retrievers:
        r = build()
        scores = r.all_scores(q_texts)
        ranks, nd = ranks_of_gold(scores, queries), ndcg10(scores, queries)
        top3 = np.argsort(-scores, axis=1)[:, :3]
        matches = [r.match(i, top3[i, 0]) for i in range(len(queries))] if isinstance(r, DenseRetriever) else None
        per_query[name] = (ranks, top3, matches)
        routing = r.routing(queries) if isinstance(r, TreeRetriever) else None
        speed = {"index": r.index_stats, "search": time_queries(r, q_texts)}
        if isinstance(r, DenseRetriever):
            speed["model_load_s"] = r.load_s
            speed["index_mb"] = r.vecs.nbytes / 1e6
        report[name] = {"accuracy": breakdowns(ranks, nd, queries, paths), "speed": speed}
        if routing:
            report[name]["routing"] = routing

    print_metrics_table("ACCURACY", {n: rep["accuracy"]["overall"]["all"] for n, rep in report.items()})
    for kind in BREAKDOWNS:
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
        if "routing" in rep:
            rt = rep["routing"]
            line += (f" | correct file reachable {rt['gold_reached']:.1%}, searched {rt['files_searched']:.0f} files / "
                     f"{rt['chunks_scored']:,.0f} of {rt['chunks_total']:,} chunks per query")
        print(line)

    (out_dir / "summary.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "corpus": corpus_stats,
         "query_problems": problems, "results": report}, indent=2))
    with (out_dir / "per_query.csv").open("w", newline="") as f:
        w = csv.writer(f)
        names = list(per_query)
        dense = [n for n in names if per_query[n][2] is not None]
        w.writerow(["query", "category", "query_type", "split", "expected"] + [f"{n} rank" for n in names] + [f"{n} top3" for n in names]
                   + [f"{n} top1 match" for n in dense])
        for i, q in enumerate(queries):
            w.writerow([q["query"], q["category"], q["query_type"], q["split"], " | ".join(paths[g] for g in sorted(q["gold"]))]
                       + [int(per_query[n][0][i]) for n in names]
                       + [" | ".join(paths[d] for d in per_query[n][1][i]) for n in names]
                       + [per_query[n][2][i] for n in dense])
    print(f"\nsaved {out_dir.relative_to(REPO_ROOT)}/ (summary.json, per_query.csv for reviewing misses)")


if __name__ == "__main__":
    main()
