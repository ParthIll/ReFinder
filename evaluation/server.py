"""ReFinder local API (FastAPI). Run: uvicorn evaluation.server:app --host 127.0.0.1 --port 8000

Indexing a folder:  extract text (parallel processes) -> BM25 keyword index (searchable now) -> background thread
embeds a capped sample of paragraph chunks per file into Chroma, with the model picked in the app
(refinder/models.py; bge-small by default). Search fuses BM25 and vector rankings (see evaluation/database.py), so
results improve while embedding runs. POST /model switches an index to another model without re-reading files.
"""

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as RawFileResponse
from pydantic import BaseModel

from evaluation.database import Cancelled, ChromaIndex, Filters
from refinder import models

# Process start-up costs a few seconds, so small folders extract in-process.
PARALLEL_MIN_FILES = 200


# ---------------------------------------------------------
# Request and response models
# ---------------------------------------------------------

class IndexRequest(BaseModel):
    # Folder to index; ignored when whole_computer is set.
    folder: str = ""
    # Index the user's home folder, skipping hidden, system, app and dependency folders.
    whole_computer: bool = False
    # "replace": the index becomes this folder only. "add": index this folder's new files alongside the folders
    # already indexed, without re-reading or re-embedding those.
    mode: str = "replace"
    # Embedding model key from GET /models, for "replace" (default: the current index's model, else bge-small).
    # "add" always keeps the index's model, so one index never mixes two models' vectors.
    model: str | None = None


class IndexResponse(BaseModel):
    status: str
    files_queued: int  # new files read by this request
    total_files: int
    failed: int
    extract_s: float


class SearchResult(BaseModel):
    path: str
    # 0-1 relevance from rank fusion: 1.0 = ranked first by both keyword and meaning search.
    score: float
    snippet: str
    # The full matched passage (best chunk or paragraph), for highlighting it in the preview.
    passage: str = ""
    chunk_id: int | None
    # Unix seconds; "added" is macOS Date Added (falls back to creation time).
    modified: float
    added: float
    size: int
    type: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    # Keyword mode only: the query with misspelled words corrected, e.g. "paystbu" -> "paystub".
    suggestion: str | None = None


class StatusResponse(BaseModel):
    indexed_files: int
    chunks: int
    # idle | extracting | embedding | error
    state: str
    embedded_chunks: int
    total_chunks: int
    error: str | None = None
    # The indexed folder (absolute path), if any; `folders` lists every indexed folder.
    folder: str | None = None
    folders: list[str] = []
    whole_computer: bool = False
    # Indexed files per type, e.g. {"pdf": 312, "docx": 80}, for the type filter.
    file_types: dict[str, int] = {}
    # Key of the index's embedding model (see GET /models).
    model: str | None = None


class ModelInfo(BaseModel):
    key: str
    label: str
    description: str
    gpu_recommended: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    # The index's model, or the default when nothing is indexed yet.
    active: str
    default: str


class ModelRequest(BaseModel):
    model: str


class FileResponse(BaseModel):
    path: str
    text: str


class OpenRequest(BaseModel):
    path: str
    # Show the file in Finder / Explorer instead of opening it.
    reveal: bool = False
    # Open with this application (macOS .app path) instead of the default one.
    app: str | None = None


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

store = ChromaIndex()
embedder: Any | None = None
model_lock = threading.Lock()  # one model, used by the embedding thread and by query encoding
index_lock = threading.Lock()  # one /index at a time
index_state = "idle"
index_error: str | None = None


def _get_embedder():
    """The index's sentence-transformer, loaded on first use and reloaded when the index's model changes."""
    global embedder
    with model_lock:
        spec = store.spec()
        if embedder is None or embedder.model_name != spec.name:
            from refinder.embedder import Embedder

            embedder = None  # free the old model before loading the next
            # fp16 only applies on a GPU; on CPU the embedder uses float32 / ONNX instead.
            embedder = Embedder(spec.name, fp16=spec.fp16)
    return embedder


def _model_key(key: str | None) -> str:
    if key is not None and key not in models.MODELS:
        raise HTTPException(status_code=422, detail=f"Unknown model {key!r}; see GET /models")
    return key or store.model_key() or models.DEFAULT_MODEL


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
    """Reload the last index; finish embedding it if the server stopped part way (resumes after the files that
    were already embedded)."""
    if store.load() and not store.embedding_complete():
        _start_embedding()


_restore_from_disk()


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.post("/index", response_model=IndexResponse)
def index_folder(request: IndexRequest) -> IndexResponse:
    """Index a folder, replacing the current index or adding to it. Returns once keyword search is ready; embedding
    continues in the background (poll /status)."""
    global index_state, index_error

    from evaluation.custom import list_files, load_corpus

    model_key = _model_key(request.model)

    folder = Path.home() if request.whole_computer else Path(request.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    if not index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Already indexing a folder")
    adding = request.mode == "add" and store.ready()
    prev_state = index_state
    try:
        index_state, index_error = "extracting", None
        files = list_files(folder, whole_computer=request.whole_computer)
        if adding:  # never re-read files that are already indexed (e.g. a parent of an indexed folder)
            known = set(store.manifest["paths"])
            files = [f for f in files if str(f) not in known]
            if not files:
                index_state = prev_state
                return IndexResponse(status="unchanged", files_queued=0, total_files=len(known), failed=0,
                                     extract_s=0.0)
        workers = min(8, os.cpu_count() or 1) if len(files) >= PARALLEL_MIN_FILES else 1
        new_paths, _, new_pieces, stats = load_corpus(folder, workers=workers, files=files)
        if not new_paths:
            index_state = prev_state if adding else "idle"
            raise HTTPException(status_code=400, detail="Folder contains no supported files")
        abs_paths = [str(folder / p) for p in new_paths]
        piece_texts = [[p.text for p in ps] for ps in new_pieces]
        if adding:
            store.add(folder, abs_paths, piece_texts)
        else:
            store.replace(folder, abs_paths, piece_texts, model_key)
        _start_embedding()
    except HTTPException:
        raise
    except Exception as e:
        index_state, index_error = "error", f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=index_error)
    finally:
        index_lock.release()

    return IndexResponse(status="searchable", files_queued=len(new_paths), total_files=len(store.manifest["paths"]),
                         failed=len(stats["failed"]),
                         extract_s=round(stats["extract_s"], 2))


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = "",
    k: int = Query(default=5, ge=1, le=100),
    mode: str = Query(default="smart", pattern="^(smart|keyword)$"),
    types: str | None = Query(default=None, description="Comma-separated extensions, e.g. pdf,docx"),
    modified_after: float | None = Query(default=None, description="Unix seconds"),
    modified_before: float | None = None,
    added_after: float | None = None,
    added_before: float | None = None,
    min_size: int | None = Query(default=None, ge=0, description="Bytes"),
    max_size: int | None = Query(default=None, ge=0),
    folder: str | None = Query(default=None, description="Substring of the file's folder path"),
) -> SearchResponse:
    """mode=smart: filename + phrase + keyword + meaning. mode=keyword: plain BM25 keyword match only.
    Filters narrow either mode; with filters and no query, lists matching files newest first."""
    if not store.ready():
        raise HTTPException(status_code=409, detail="Index a folder before searching")
    filters = Filters(
        types={t.strip().lower().lstrip(".") for t in types.split(",") if t.strip()} if types else None,
        modified_after=modified_after, modified_before=modified_before,
        added_after=added_after, added_before=added_before,
        min_size=min_size, max_size=max_size, folder=folder or None,
    )
    if not q.strip():
        if not filters.active():
            raise HTTPException(status_code=422, detail="Enter a search or set a filter")
        return SearchResponse(results=[SearchResult(**hit) for hit in store.browse(filters, k=k)])
    if mode == "keyword":
        return SearchResponse(results=[SearchResult(**hit) for hit in store.keyword_search(q, k=k, filters=filters)],
                              suggestion=store.suggest(q))
    query_vec = None
    if store.count():  # skip loading the model while only keyword search is available
        enc = _get_embedder()
        with model_lock:
            query_vec = enc.encode_query(q)
    return SearchResponse(results=[SearchResult(**hit) for hit in store.search(q, query_vec, k=k, filters=filters)])


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    roots = store.roots()
    return StatusResponse(
        indexed_files=len(store.manifest["paths"]) if store.manifest else 0,
        chunks=store.count(),
        state=index_state,
        embedded_chunks=store.embedded_chunks,
        total_chunks=store.total_chunks,
        error=index_error,
        folder=roots[0] if len(roots) == 1 else None,
        folders=roots,
        whole_computer=str(Path.home()) in roots,
        file_types=store.file_types(),
        model=store.model_key(),
    )


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """The embedding models the app offers (refinder/models.py), and the one the index uses."""
    return ModelsResponse(
        models=[ModelInfo(key=m.key, label=m.label, description=m.description, gpu_recommended=m.gpu_recommended)
                for m in models.MODELS.values()],
        active=store.model_key() or models.DEFAULT_MODEL, default=models.DEFAULT_MODEL)


@app.post("/model", response_model=StatusResponse)
def set_model(request: ModelRequest) -> StatusResponse:
    """Switch the index to another embedding model. Files aren't re-read and keyword search keeps working; their
    vectors are redone in the background with the new model (poll /status). No-op if it's already the model."""
    key = _model_key(request.model)
    if not store.ready():
        raise HTTPException(status_code=409, detail="Index a folder first, or pass the model to /index")
    if key == store.model_key():
        return status()
    if not index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Already indexing a folder")
    try:
        store.set_model(key)
        _start_embedding()
    finally:
        index_lock.release()
    return status()


@app.get("/file", response_model=FileResponse)
def get_file(path: str) -> FileResponse:
    text = store.file_text(path) if store.ready() else None
    if text is None:
        raise HTTPException(status_code=404, detail="Indexed file not found")
    return FileResponse(path=path, text=text)


def _indexed_file(path: str) -> tuple[Path, str]:
    """The path and extracted text of an indexed file; 404 for anything else, so these endpoints can't read or
    open arbitrary files."""
    text = store.file_text(path) if store.ready() else None
    if text is None:
        raise HTTPException(status_code=404, detail="Indexed file not found")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=410, detail="File was moved or deleted since it was indexed")
    return p, text


@app.get("/preview")
def preview_file(path: str) -> dict[str, Any]:
    """What the preview panel shows: {"kind": "pdf" | "docx" | "table" | "slides" | "text", ...}."""
    from evaluation.preview import preview

    p, text = _indexed_file(path)
    return preview(p, text)


@app.get("/raw")
def raw_file(path: str) -> RawFileResponse:
    """The file's bytes, for rendering PDFs and Word documents in the app."""
    p, _ = _indexed_file(path)
    return RawFileResponse(p, filename=p.name, content_disposition_type="inline")


@app.get("/thumbnail")
def file_thumbnail(path: str) -> RawFileResponse:
    """A PNG of the first page or slide (macOS Quick Look)."""
    from evaluation.preview import thumbnail

    p, _ = _indexed_file(path)
    png = thumbnail(p)
    if png is None:
        raise HTTPException(status_code=404, detail="No thumbnail available")
    return RawFileResponse(png, media_type="image/png")


@app.get("/quicklook/{key}/{name}")
def quicklook_file(key: str, name: str) -> RawFileResponse:
    """Files of a Quick Look HTML render made by /preview (the page and its images)."""
    from evaluation.preview import QUICKLOOK_FILE, quicklook_dir

    if not re.fullmatch(r"[0-9a-f]{40}", key) or not QUICKLOOK_FILE.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    f = quicklook_dir(key) / name
    if not f.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return RawFileResponse(f)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif"}


@app.get("/asset")
def asset_file(path: str) -> RawFileResponse:
    """An image inside an indexed folder, for pictures that Markdown files link to (e.g. ![](img/chart.png))."""
    p = Path(path).resolve()
    roots = [Path(r).resolve() for r in store.roots()]
    if p.suffix.lower() not in IMAGE_SUFFIXES or not any(p.is_relative_to(r) for r in roots) or not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return RawFileResponse(p)


@app.post("/open")
def open_file(request: OpenRequest) -> dict[str, str]:
    """Open an indexed file in its default app, or reveal it in Finder."""
    from evaluation.preview import is_app, open_file as open_path

    p, _ = _indexed_file(request.path)
    app = Path(request.app) if request.app else None
    if app is not None and not is_app(app):
        raise HTTPException(status_code=400, detail="Not an application")
    open_path(p, reveal=request.reveal, app=app)
    return {"status": "opened"}


@app.get("/open-with")
def open_with(path: str) -> dict[str, Any]:
    """Apps that can open this file, as in Finder's Open With menu (macOS)."""
    from evaluation.preview import open_with_apps

    p, _ = _indexed_file(path)
    return open_with_apps(p)


@app.get("/app-icon")
def application_icon(app: str) -> RawFileResponse:
    from evaluation.preview import app_icon

    png = app_icon(Path(app))
    if png is None:
        raise HTTPException(status_code=404, detail="No icon")
    return RawFileResponse(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ReFinder API is running"}
