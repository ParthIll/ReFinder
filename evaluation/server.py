"""ReFinder local API (FastAPI). Run: uvicorn evaluation.server:app --host 127.0.0.1 --port 8000

Indexing a folder:  extract text (parallel processes) -> BM25 keyword index (searchable now) -> background thread
embeds a capped sample of paragraph chunks per file with bge-small into Chroma. Search fuses BM25 and vector
rankings (see evaluation/database.py), so results improve while embedding runs.
"""

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evaluation.database import Cancelled, ChromaIndex

# Fastest model at good accuracy in the benchmark; fp16 is ~1.3x faster on GPU/MPS with the same results.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Process start-up costs a few seconds, so small folders extract in-process.
PARALLEL_MIN_FILES = 200


# ---------------------------------------------------------
# Request and response models
# ---------------------------------------------------------

class IndexRequest(BaseModel):
    folder: str


class IndexResponse(BaseModel):
    status: str
    files_queued: int
    failed: int
    extract_s: float


class SearchResult(BaseModel):
    path: str
    # 0-1 relevance from rank fusion: 1.0 = ranked first by both keyword and meaning search.
    score: float
    snippet: str
    chunk_id: int | None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class StatusResponse(BaseModel):
    indexed_files: int
    chunks: int
    # idle | extracting | embedding | error
    state: str
    embedded_chunks: int
    total_chunks: int
    error: str | None = None


class FileResponse(BaseModel):
    path: str
    text: str


# ---------------------------------------------------------
# Application and server memory
# ---------------------------------------------------------

app = FastAPI(title="ReFinder API")

# The packaged Tauri webview accesses the local API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

corpus_root: Path | None = None
paths: list[str] = []
texts: list[str] = []

store = ChromaIndex()
embedder: Any | None = None
model_lock = threading.Lock()  # one model, used by the embedding thread and by query encoding
index_lock = threading.Lock()  # one /index at a time
index_state = "idle"
index_error: str | None = None


def _get_embedder():
    """Load the sentence-transformer on first use (a few seconds)."""
    global embedder
    with model_lock:
        if embedder is None:
            from refinder.embedder import Embedder

            embedder = Embedder(store.model_name() or MODEL_NAME, fp16=True)
    return embedder


def _embed_in_background(generation: int) -> None:
    global index_state, index_error
    try:
        store.embed(_get_embedder(), generation, model_lock)
    except Cancelled:
        return  # a newer /index owns the state now
    except Exception as e:  # keep keyword search working even if embedding fails
        if generation == store.generation:
            index_state, index_error = "error", f"Embedding failed: {type(e).__name__}: {e}"
        return
    if generation == store.generation:
        index_state = "idle"


def _start_embedding() -> None:
    global index_state, index_error
    index_state, index_error = "embedding", None
    threading.Thread(target=_embed_in_background, args=(store.generation,), daemon=True).start()


def _restore_from_disk() -> None:
    """Reload the last index; finish embedding it if the server stopped part way."""
    global corpus_root, paths, texts
    loaded = store.load()
    if loaded is None:
        return
    corpus_root, paths, texts = loaded
    if not store.embedding_complete():
        store.build_keyword(corpus_root, paths, store.manifest["pieces"], store.model_name() or MODEL_NAME)
        _start_embedding()


_restore_from_disk()


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.post("/index", response_model=IndexResponse)
def index_folder(request: IndexRequest) -> IndexResponse:
    """Index a folder, replacing the current index. Returns once keyword search is ready; embedding continues in
    the background (poll /status)."""
    global corpus_root, paths, texts, index_state, index_error, embedder

    from evaluation.custom import load_corpus

    folder = Path(request.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    if not index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Already indexing a folder")
    try:
        index_state, index_error = "extracting", None
        n_files = sum(1 for p in folder.rglob("*") if p.is_file())
        workers = min(8, os.cpu_count() or 1) if n_files >= PARALLEL_MIN_FILES else 1
        new_paths, new_texts, new_pieces, stats = load_corpus(folder, workers=workers)
        if not new_paths:
            index_state = "idle"
            raise HTTPException(status_code=400, detail="Folder contains no supported files")

        if embedder is not None and embedder.model_name != MODEL_NAME:
            embedder = None  # index built with an older model: switch to the current one
        store.build_keyword(folder, new_paths, [[p.text for p in ps] for ps in new_pieces], MODEL_NAME)
        corpus_root, paths, texts = folder, new_paths, new_texts
        _start_embedding()
    except HTTPException:
        raise
    except Exception as e:
        index_state, index_error = "error", f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=index_error)
    finally:
        index_lock.release()

    return IndexResponse(status="searchable", files_queued=len(paths), failed=len(stats["failed"]),
                         extract_s=round(stats["extract_s"], 2))


@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = Query(default=5, ge=1, le=20),
           mode: str = Query(default="smart", pattern="^(smart|keyword)$")) -> SearchResponse:
    """mode=smart: filename + phrase + keyword + meaning. mode=keyword: plain BM25 keyword match only."""
    if not store.ready():
        raise HTTPException(status_code=409, detail="Index a folder before searching")
    if mode == "keyword":
        return SearchResponse(results=[SearchResult(**hit) for hit in store.keyword_search(q, k=k)])
    query_vec = None
    if store.count():  # skip loading the model while only keyword search is available
        enc = _get_embedder()
        with model_lock:
            query_vec = enc.encode_query(q)
    return SearchResponse(results=[SearchResult(**hit) for hit in store.search(q, query_vec, k=k)])


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        indexed_files=len(paths),
        chunks=store.count(),
        state=index_state,
        embedded_chunks=store.embedded_chunks,
        total_chunks=store.total_chunks,
        error=index_error,
    )


@app.get("/file", response_model=FileResponse)
def get_file(path: str) -> FileResponse:
    try:
        index = paths.index(path)
    except ValueError:
        raise HTTPException(status_code=404, detail="Indexed file not found")
    return FileResponse(path=paths[index], text=texts[index])


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ReFinder API is running"}
