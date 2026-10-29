"""Shared retrieval-eval plumbing: chunk + embed a corpus (with disk cache) and score files by max chunk similarity."""

import hashlib
import time
from pathlib import Path

import numpy as np

from refinder.embedder import CHUNK_OVERLAP, CHUNK_TOKENS, Embedder

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "eval"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"


def chunk_corpus(embedder: Embedder, texts: list[str]) -> tuple[list[str], np.ndarray]:
    chunk_texts, owner = [], []
    for doc_id, text in enumerate(texts):
        for c in embedder.chunk(text, CHUNK_TOKENS, CHUNK_OVERLAP):
            chunk_texts.append(c.text)
            owner.append(doc_id)
    return chunk_texts, np.array(owner)


def embed_corpus(embedder: Embedder, texts: list[str], use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Chunk + embed every file, caching vectors on disk keyed by model, chunking params and corpus content."""
    h = hashlib.sha256(f"{embedder.model_name}|{CHUNK_TOKENS}|{CHUNK_OVERLAP}".encode())
    for t in texts:
        h.update(t.encode())
    cache = CACHE_DIR / f"{h.hexdigest()[:16]}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return z["vecs"], z["owner"], {"cached": True}

    t0 = time.perf_counter()
    chunk_texts, owner = chunk_corpus(embedder, texts)
    t_chunk = time.perf_counter() - t0
    print(f"  {len(chunk_texts):,} chunks from {len(texts):,} files; embedding...")
    t0 = time.perf_counter()
    vecs = embedder.encode(chunk_texts, batch_size=128, show_progress=True)
    t_embed = time.perf_counter() - t0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, vecs=vecs, owner=owner)
    return vecs, owner, {
        "cached": False,
        "chunks": len(chunk_texts),
        "chunk_s": t_chunk,
        "embed_s": t_embed,
        "chunks_per_s": len(chunk_texts) / t_embed,
    }


def dense_scores(model_name: str, texts: list[str], queries: list[str]) -> tuple[np.ndarray, dict]:
    """Returns (n_queries, n_files) cosine scores, each file scored by its best-matching chunk."""
    embedder = Embedder(model_name)
    print(f"\n[{model_name}] dim={embedder.dim} device={embedder.model.device}")
    vecs, owner, stats = embed_corpus(embedder, texts)
    # Chunks are stored contiguously per file, so a file's score is a max over its slice of chunk scores.
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    doc_ids = owner[starts]

    t0 = time.perf_counter()
    q_vecs = embedder.encode_queries(queries)
    scores = np.full((len(queries), len(texts)), -np.inf, dtype=np.float32)
    for b in range(0, len(queries), 64):
        chunk_scores = q_vecs[b : b + 64] @ vecs.T
        scores[b : b + 64, doc_ids] = np.maximum.reduceat(chunk_scores, starts, axis=1)
    stats["query_ms"] = 1000 * (time.perf_counter() - t0) / len(queries)
    stats["dim"] = embedder.dim
    return scores, stats
