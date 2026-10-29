"""Persistent ChromaDB store for chunk embeddings used by the ReFinder API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
import numpy as np
from chromadb.config import Settings

if TYPE_CHECKING:
    from refinder.embedder import Embedder

COLLECTION_NAME = "refinder"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
MANIFEST_NAME = "manifest.json"
ADD_BATCH = 5000


class ChromaIndex:
    """Chunk + embed files, persist vectors in Chroma, and search by max-pooled cosine."""

    
    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.persist_dir / MANIFEST_NAME
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self._get_or_create()

    def _get_or_create(self):
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self, model_name: str, corpus_root: str) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "model_name": model_name,
                "corpus_root": corpus_root,
            },
        )

    def count(self) -> int:
        return int(self.collection.count())

    def ready(self) -> bool:
        return self.manifest_path.is_file()

    def load(self) -> tuple[Path, list[str], list[str]] | None:
        """Restore corpus root, relative paths, and extracted texts from the last index."""
        if not self.ready():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return Path(data["corpus_root"]), list(data["paths"]), list(data["texts"])

    def model_name(self) -> str | None:
        if not self.ready():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return data.get("model_name")

    def replace(self, corpus_root: Path, paths: list[str], texts: list[str], embedder: "Embedder") -> int:
        """Rebuild the collection from extracted files. Returns the number of stored chunks."""
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for doc_id, (path, text) in enumerate(zip(paths, texts)):
            for chunk in embedder.chunk(text):
                ids.append(f"{doc_id}:{chunk.index}")
                documents.append(chunk.text)
                metadatas.append({
                    "path": path,
                    "chunk_index": chunk.index,
                    "doc_id": doc_id,
                })

        vectors: np.ndarray | None = None
        if documents:
            print(f"  {len(documents):,} chunks from {len(texts):,} files; embedding...")
            vectors = embedder.encode(documents, batch_size=128, show_progress=True)

        self._reset_collection(embedder.model_name, str(corpus_root))
        if vectors is not None:
            for start in range(0, len(ids), ADD_BATCH):
                end = start + ADD_BATCH
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=vectors[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )

        self.manifest_path.write_text(
            json.dumps(
                {
                    "corpus_root": str(corpus_root),
                    "model_name": embedder.model_name,
                    "paths": paths,
                    "texts": texts,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return len(ids)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Return up to k files, each scored by its best-matching chunk."""
        n_chunks = self.count()
        if n_chunks == 0:
            return []

        n_results = min(n_chunks, max(k * 20, 50))
        raw = self.collection.query(
            query_embeddings=[np.asarray(query_vec, dtype=np.float32)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        best: dict[str, dict[str, Any]] = {}
        for document, metadata, distance in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            if not metadata:
                continue
            path = str(metadata["path"])
            score = 1.0 - float(distance)
            current = best.get(path)
            if current is None or score > current["score"]:
                best[path] = {
                    "path": path,
                    "score": score,
                    "snippet": (document or "").replace("\n", " ")[:300],
                    "chunk_id": int(metadata["chunk_index"]),
                }

        return sorted(best.values(), key=lambda item: -item["score"])[:k]
