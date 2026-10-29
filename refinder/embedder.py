"""Embedding model wrapper: loads a Sentence Transformers model, chunks text, encodes docs and queries."""

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

# Chunking tokenizes whole files, which trips a harmless "longer than max length" warning.
hf_logging.set_verbosity_error()

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 50


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Chunk:
    text: str
    index: int
    start_token: int


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device or pick_device())
        self.tokenizer = self.model.tokenizer
        self.dim = self.model.get_embedding_dimension()
        # all-MiniLM-L6-v2 truncates at 256 word pieces; chunks must stay under that.
        self.max_seq_length = self.model.max_seq_length

    def chunk(self, text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
        """Split text into ~chunk_tokens-token windows that overlap by `overlap` tokens."""
        if overlap >= chunk_tokens:
            raise ValueError("overlap must be smaller than chunk_tokens")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return []
        step = chunk_tokens - overlap
        chunks = []
        for i, start in enumerate(range(0, len(ids), step)):
            window = ids[start : start + chunk_tokens]
            chunks.append(Chunk(self.tokenizer.decode(window), i, start))
            if start + chunk_tokens >= len(ids):
                break
        return chunks

    def _encode(self, fn, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        return fn(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode document-side texts to L2-normalized float32 vectors, shape (n, dim). Dot product == cosine.

        Uses the model's document prompt if it defines one (e.g. "passage: " for e5); a no-op for MiniLM.
        """
        return self._encode(self.model.encode_document, texts, batch_size, show_progress)

    def encode_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode search queries, applying the model's query prompt if it defines one."""
        return self._encode(self.model.encode_query, queries, batch_size, False)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_document(self, text: str) -> tuple[list[Chunk], np.ndarray]:
        chunks = self.chunk(text)
        return chunks, self.encode([c.text for c in chunks]) if chunks else np.empty((0, self.dim), np.float32)


def cosine_top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Brute-force nearest neighbors on normalized vectors. Used by the CLI smoke test; the API uses ChromaDB."""
    scores = doc_vecs @ query_vec
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]
