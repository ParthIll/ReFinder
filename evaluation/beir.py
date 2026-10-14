"""Retrieval benchmark on public BEIR datasets: natural-language queries with human relevance labels.

Unlike the Kaggle corpus, these have real description/question-style queries and graded qrels, so the
numbers are comparable to published results (MTEB / BEIR paper). Documents go through the same pipeline
ReFinder uses for files: ~200-token overlapping chunks, each document scored by its best chunk.

  scifact   scientific claims -> supporting abstracts         5.2k docs,  300 queries
  nfcorpus  plain-language health questions -> articles       3.6k docs,  323 queries
  fiqa      financial questions -> forum answers              57k docs,   648 queries

    python -m evaluation.beir                              # all three, MiniLM vs BM25
    python -m evaluation.beir --datasets scifact --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
"""

import argparse
import json
import time
from collections import defaultdict

import bm25s
import numpy as np
import Stemmer
from datasets import load_dataset

from evaluation.common import REPO_ROOT, RESULTS_DIR, dense_scores
from refinder.embedder import DEFAULT_MODEL

DATASETS = ("scifact", "nfcorpus", "fiqa")
TOP_K = 100

# Published nDCG@10 for sanity-checking our pipeline (truncation-based, not chunked, so expect small gaps).
# all-MiniLM-L6-v2: MTEB leaderboard. BM25: BEIR paper (Thakur et al., 2021).
PUBLISHED_NDCG10 = {
    "scifact": {DEFAULT_MODEL: 0.645, "bm25": 0.665},
    "nfcorpus": {DEFAULT_MODEL: 0.316, "bm25": 0.325},
    "fiqa": {DEFAULT_MODEL: 0.369, "bm25": 0.236},
}


def load_beir(name: str) -> tuple[list[str], list[str], list[str], list[dict[int, int]]]:
    """Returns (doc_ids, doc_texts, query_texts, qrels) where qrels[i] maps doc index -> relevance grade."""
    corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
    queries = load_dataset(f"mteb/{name}", "queries", split="queries")
    judgments = load_dataset(f"mteb/{name}", "default", split="test")

    doc_ids = corpus["_id"]
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    doc_texts = [f"{t} {x}".strip() for t, x in zip(corpus["title"], corpus["text"])]

    rels = defaultdict(dict)
    for row in judgments:
        if row["score"] > 0 and row["corpus-id"] in doc_index:
            rels[row["query-id"]][doc_index[row["corpus-id"]]] = int(row["score"])
    query_text = dict(zip(queries["_id"], queries["text"]))
    q_ids = [q for q in rels if q in query_text]
    return doc_ids, doc_texts, [query_text[q] for q in q_ids], [rels[q] for q in q_ids]


def top_k(scores: np.ndarray, k: int = TOP_K) -> np.ndarray:
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def bm25_rankings(texts: list[str], queries: list[str]) -> tuple[np.ndarray, dict]:
    """BM25 over whole documents with English stopwords + stemming (like the BEIR paper's Anserini baseline)."""
    stemmer = Stemmer.Stemmer("english")
    t0 = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
    t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    q_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    ranked, _ = retriever.retrieve(q_tokens, k=TOP_K, show_progress=False)
    t_query = time.perf_counter() - t0
    return ranked, {"index_s": t_index, "query_ms": 1000 * t_query / len(queries)}


def evaluate(ranked: np.ndarray, qrels: list[dict[int, int]]) -> dict[str, float]:
    """nDCG@10 (linear gain, as in trec_eval/BEIR), Recall@10/100, MRR@10, averaged over queries."""
    ndcg, r10, r100, mrr = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, 12))
    for docs, rel in zip(ranked, qrels):
        gains = np.array([rel.get(int(d), 0) for d in docs[:10]], dtype=float)
        ideal = np.sort(np.array(list(rel.values()), dtype=float))[::-1][:10]
        ndcg.append((gains * discounts[: len(gains)]).sum() / (ideal * discounts[: len(ideal)]).sum())
        hits = [int(d) in rel for d in docs]
        r10.append(sum(hits[:10]) / len(rel))
        r100.append(sum(hits) / len(rel))
        first = next((i for i, h in enumerate(hits[:10]) if h), None)
        mrr.append(0.0 if first is None else 1.0 / (first + 1))
    return {"nDCG@10": float(np.mean(ndcg)), "R@10": float(np.mean(r10)),
            "R@100": float(np.mean(r100)), "MRR@10": float(np.mean(mrr))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--no-bm25", action="store_true")
    args = ap.parse_args()

    results, stats = {}, {}
    for ds in args.datasets:
        _, texts, queries, qrels = load_beir(ds)
        print(f"\n=== {ds}: {len(texts):,} docs, {len(queries)} test queries ===")
        print(f"  e.g. {queries[0]!r}")
        results[ds], stats[ds] = {}, {}
        for model in args.models:
            scores, stats[ds][model] = dense_scores(model, texts, queries)
            results[ds][model] = evaluate(top_k(scores), qrels)
        if not args.no_bm25:
            ranked, stats[ds]["bm25"] = bm25_rankings(texts, queries)
            results[ds]["bm25"] = evaluate(ranked, qrels)

    cols = ["nDCG@10", "R@10", "R@100", "MRR@10"]
    width = max(len(n) for r in results.values() for n in r) + 2
    for ds, by_model in results.items():
        print(f"\n{ds}")
        print(" " * width + "".join(f"{c:>9}" for c in cols) + "  published nDCG@10")
        for name, m in by_model.items():
            pub = PUBLISHED_NDCG10[ds].get(name)
            print(f"{name:<{width}}" + "".join(f"{m[c]:>9.3f}" for c in cols) + (f"  {pub:.3f}" if pub else ""))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"beir_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results, "stats": stats}, indent=2))
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
