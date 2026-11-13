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
    w_kw: float = 1.0  # BM25's weight against the meaning ranking (weight 1) in the app's rank fusion
    gpu_recommended: bool = False
    fp16: bool = True  # half precision on a GPU (~1.3x faster); off for models whose values overflow in fp16


MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("bge-small", "BAAI/bge-small-en-v1.5", "Fast",
              "bge-small: quickest to index (about 1-2 min for 280 files on a laptop CPU)",
              max_chunks=20, sampling="head"),
    ModelSpec("harrier-270m", "microsoft/harrier-oss-v1-270m", "Accurate",
              "Harrier 270M + reranker: best results without a GPU (about 4 min for 280 files on CPU)",
              max_chunks=50,  # 100 -> 50: same Hit@1 (team +0.3 pts), 47% fewer chunks on a 14k-file folder
              sampling="even", file_vector=True, reranker="Alibaba-NLP/gte-reranker-modernbert-base",
              w_kw=0.25,  # team Hit@1 0.598 -> 0.633, OrgSmith and sample within 1 point (ITERATIONS.md)
              fp16=False),  # Harrier (Gemma 3) gives NaN vectors in fp16; it runs in its own bfloat16 on a GPU
    ModelSpec("harrier-0.6b", "microsoft/harrier-oss-v1-0.6b", "Most accurate",
              "Harrier 0.6B + reranker: highest accuracy; slow without a GPU",
              max_chunks=100, sampling="even", file_vector=True, reranker="Alibaba-NLP/gte-reranker-modernbert-base",
              gpu_recommended=True, fp16=False),
    ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", "Lightweight",
              "MiniLM: smallest model, least accurate", max_chunks=20, sampling="head"),
    # 23M model distilled from snowflake-arctic-embed-m-v1.5. With Harrier's search extras it beat Harrier 270M on
    # every test set (team 0.653 / OrgSmith 0.623 / sample 0.870 vs 0.636 / 0.621 / 0.861) and embeds ~8x faster
    # (M5 Pro: 750 vs 90 chunks/s on GPU, 187 vs 22 on CPU). Its Dense layer keeps it on PyTorch, not ONNX.
    ModelSpec("leaf-ir", "MongoDB/mdbr-leaf-ir", "Fast + accurate",
              "mdbr-leaf-ir + reranker: about 8x faster to index than Harrier 270M, as accurate",
              max_chunks=20, sampling="head", file_vector=True, reranker="Alibaba-NLP/gte-reranker-modernbert-base",
              w_kw=0.25),  # team 0.653; 0.1 to 1 all within 1.3 pts of it
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
