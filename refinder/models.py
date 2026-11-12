"""The embedding models a user can pick in the app, each with the indexing and search settings measured best for it.

To offer a new model, add a ModelSpec to MODELS (and its prompts to refinder.embedder.MODEL_PROMPTS if the model
card asks for a query/document prefix). Changing a model re-embeds the index; it never mixes vectors from two models.
Numbers in descriptions: team corpus (280 files), M5 Pro; see evaluation/ITERATIONS.md.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str  # what the frontend sends, e.g. "bge-small"
    name: str  # Hugging Face model id
    label: str  # shown in the model picker
    description: str
    max_chunks: int  # paragraph chunks embedded per file; longer files keep a sample of this many
    sampling: str  # which chunks a long file keeps: "head" (first half of the budget + the rest spread) or "even"
    file_vector: bool = False  # also score each file by the mean of its chunk vectors, minus a giant-file penalty
    reranker: str | None = None  # cross-encoder that re-scores the top chunks of each search, fused by rank
    gpu_recommended: bool = False
    fp16: bool = True  # half precision on a GPU (~1.3x faster); off for models whose values overflow in fp16


MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("bge-small", "BAAI/bge-small-en-v1.5", "Fast",
              "bge-small: quickest to index (about 1-2 min for 280 files on a laptop CPU)",
              max_chunks=20, sampling="head"),
    ModelSpec("harrier-270m", "microsoft/harrier-oss-v1-270m", "Accurate",
              "Harrier 270M + reranker: best results without a GPU (about 4 min for 280 files on CPU)",
              max_chunks=100, sampling="even", file_vector=True, reranker="Alibaba-NLP/gte-reranker-modernbert-base",
              fp16=False),  # Harrier (Gemma 3) gives NaN vectors in fp16; it runs in its own bfloat16 on a GPU
    ModelSpec("harrier-0.6b", "microsoft/harrier-oss-v1-0.6b", "Most accurate",
              "Harrier 0.6B + reranker: highest accuracy; slow without a GPU",
              max_chunks=100, sampling="even", file_vector=True, reranker="Alibaba-NLP/gte-reranker-modernbert-base",
              gpu_recommended=True, fp16=False),
    ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", "Lightweight",
              "MiniLM: smallest model, least accurate", max_chunks=20, sampling="head"),
)}
DEFAULT_MODEL = "bge-small"


def get(key: str | None) -> ModelSpec:
    return MODELS.get(key or DEFAULT_MODEL) or MODELS[DEFAULT_MODEL]


def by_name(name: str | None) -> ModelSpec | None:
    """The spec for a Hugging Face id (older indexes stored only the model name)."""
    return next((m for m in MODELS.values() if m.name == name), None)


def sample_chunks(n: int, cap: int, how: str = "head") -> list[int]:
    """Which of a file's n chunks to embed when only `cap` fit (all of them when n <= cap).
    head: the first half of the budget (titles, abstracts, intros say what a file is about), the rest spread over
          the remainder. even: `cap` evenly spaced chunks, first and last included."""
    if not cap or n <= cap:
        return list(range(n))
    if how == "even":
        return np.unique(np.linspace(0, n - 1, cap).round().astype(int)).tolist()
    head = cap // 2
    tail = np.linspace(head, n - 1, cap - head).round().astype(int)
    return sorted(set(range(head)) | set(tail.tolist()))
