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

# Query instructions recommended by the model authors but missing from the model's own config. Models that ship
# their prompts (e.g. Qwen3-Embedding) don't need an entry.
QUERY_PROMPTS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


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
    start_token: int = 0
    # For chunks built from extractor pieces (paragraphs, slides, rows...): the inclusive range of pieces covered,
    # so a search hit can point back to e.g. "page 3, paragraphs 4-6".
    first_piece: int | None = None
    last_piece: int | None = None


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device or pick_device())
        self.tokenizer = self.model.tokenizer
        self.dim = self.model.get_embedding_dimension()
        # all-MiniLM-L6-v2 truncates at 256 word pieces; chunks must stay under that.
        self.max_seq_length = self.model.max_seq_length
        if model_name in QUERY_PROMPTS and not self.model.prompts.get("query"):
            self.model.prompts["query"] = QUERY_PROMPTS[model_name]

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

    def chunk_pieces(
        self, pieces: list[str], chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP
    ) -> list[Chunk]:
        """Pack consecutive pieces (paragraphs, slides, rows) into chunks of up to chunk_tokens tokens.

        Pieces are only cut when a single piece is longer than chunk_tokens (then it's split into overlapping
        windows). Each packed chunk starts with up to `overlap` tokens carried over from the end of the previous
        one: whole trailing pieces if they fit, otherwise the tail of the last piece. So a detail that straddles a
        chunk boundary still appears intact in one chunk.
        """
        if overlap >= chunk_tokens:
            raise ValueError("overlap must be smaller than chunk_tokens")
        if not pieces:
            return []
        ids = self.tokenizer(pieces, add_special_tokens=False)["input_ids"]
        chunks: list[Chunk] = []
        # Buffer entries are (piece index, text, token count); text may be a tail fragment of a carried piece.
        buf: list[tuple[int, str, int]] = []
        n_carried = 0  # how many leading buffer entries are overlap from the previous chunk

        def buf_tokens() -> int:
            return sum(n for _, _, n in buf)

        def carry_tail() -> list[tuple[int, str, int]]:
            carried, budget = [], overlap
            for i, text, n in reversed(buf):
                if n <= budget:
                    carried.insert(0, (i, text, n))
                    budget -= n
                    continue
                if not carried:  # last piece alone is too big: carry just its final `overlap` tokens
                    tail = ids[i][-overlap:] if text == pieces[i] else self.tokenizer.encode(text, add_special_tokens=False)[-overlap:]
                    carried = [(i, self.tokenizer.decode(tail), len(tail))]
                break
            return carried

        def flush(carry: bool) -> None:
            nonlocal buf, n_carried
            if len(buf) > n_carried:  # don't emit a chunk that is only carried-over text
                chunks.append(Chunk("\n".join(t for _, t, _ in buf), len(chunks),
                                    first_piece=buf[0][0], last_piece=buf[-1][0]))
            buf = carry_tail() if carry and len(buf) > n_carried else []
            n_carried = len(buf)

        for i, piece_ids in enumerate(ids):
            n = len(piece_ids)
            if n == 0:
                continue
            if n > chunk_tokens:
                flush(carry=False)
                for c in self.chunk(pieces[i], chunk_tokens, overlap):
                    chunks.append(Chunk(c.text, len(chunks), c.start_token, first_piece=i, last_piece=i))
                tail = piece_ids[-overlap:]
                buf, n_carried = [(i, self.tokenizer.decode(tail), len(tail))], 1
                continue
            if buf_tokens() + n > chunk_tokens:
                flush(carry=True)
                if buf_tokens() + n > chunk_tokens:  # carried text + this piece still too big: drop the carry
                    buf, n_carried = [], 0
            buf.append((i, pieces[i], n))
        flush(carry=False)
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
