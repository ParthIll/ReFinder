"""ReFinder local API (FastAPI). Run: uvicorn evaluation.server:app --host 127.0.0.1 --port 8000

Indexing a folder:  extract text (parallel processes) -> BM25 keyword index (searchable now) -> background thread
embeds a capped sample of paragraph chunks per file with bge-small into Chroma. Search fuses BM25 and vector
rankings (see evaluation/database.py), so results improve while embedding runs.
"""

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as RawFileResponse
from pydantic import BaseModel

from evaluation.database import Cancelled, ChromaIndex, Filters, _stamp
from evaluation.watcher import FolderWatcher

# Fastest model at good accuracy in the benchmark; fp16 is ~1.3x faster on GPU/MPS with the same results.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Process start-up costs a few seconds, so small folders extract in-process.
PARALLEL_MIN_FILES = 200
# The packaged app keeps no extraction cache next to its (read-only) code; the manifest already stores the text.
USE_EXTRACT_CACHE = not getattr(sys, "frozen", False) and os.environ.get("REFINDER_EXTRACT_CACHE", "1") != "0"
log = logging.getLogger("refinder")


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
    visual_match: dict[str, int] | None = None
    # Unix seconds; "added" is macOS Date Added (falls back to creation time).
    modified: float
    added: float
    size: int
    type: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    # Keyword mode only: the query with misspelled words corrected, e.g. "paystbu" -> "paystub".
    suggestion: str | None = None
    # Hints read from the query and used for ranking, e.g. ["PDF", "from 2025"].
    understood: list[str] = []


class StatusResponse(BaseModel):
    indexed_files: int
    chunks: int
    # idle | extracting | embedding | error
    state: str
    embedded_chunks: int
    total_chunks: int
    embedded_images: int = 0
    total_images: int = 0
    error: str | None = None
    # The indexed folder (absolute path), if any; `folders` lists every indexed folder.
    folder: str | None = None
    folders: list[str] = []
    whole_computer: bool = False
    # Indexed files per type, e.g. {"pdf": 312, "docx": 80}, for the type filter.
    file_types: dict[str, int] = {}


class FileResponse(BaseModel):
    path: str
    text: str


class RemoveFolderRequest(BaseModel):
    folder: str


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
visual_embedder: Any | None = None
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


def _get_visual_embedder():
    """Only load MobileCLIP when at least one OCR-fallback image needs indexing/search."""
    global visual_embedder
    with model_lock:
        if visual_embedder is None:
            from refinder.vision import VisualEmbedder

            visual_embedder = VisualEmbedder()
    return visual_embedder


def _embed_in_background(generation: int) -> None:
    global index_state, index_error
    errors = []
    try:
        store.embed(_get_embedder() if store.pending_text() else None, generation, model_lock)
    except Cancelled:
        return  # a newer /index owns the state now
    except Exception as e:  # keep keyword search working even if embedding fails
        errors.append(f"Text embedding failed: {type(e).__name__}: {e}")
    try:
        store.embed_visuals(_get_visual_embedder, generation, model_lock)
    except Cancelled:
        return
    except Exception as e:
        errors.append(f"Visual embedding failed: {type(e).__name__}: {e}")
    if generation == store.generation:
        index_state, index_error = ("error", "; ".join(errors)) if errors else ("idle", None)


def _start_embedding() -> None:
    global index_state, index_error
    index_state, index_error = "embedding", None
    threading.Thread(target=_embed_in_background, args=(store.generation,), daemon=True).start()


def _extract(paths_by_root: dict[int, list[Path]]) -> list[tuple[int, str, list[str], list[dict]]]:
    """(root index, absolute path, piece texts) for each file that extracts; failures are left out."""
    from evaluation.custom import load_corpus
    from refinder.extract import visual_assets

    out = []
    roots = store.roots()
    for r, files in paths_by_root.items():
        root = Path(roots[r])
        workers = min(8, os.cpu_count() or 1) if len(files) >= PARALLEL_MIN_FILES else 1
        rel, _, pieces, _ = load_corpus(root, workers=workers, files=files, use_cache=USE_EXTRACT_CACHE)
        out += [(r, str(root / p), [x.text for x in ps if x.text.strip()], visual_assets(ps)) for p, ps in zip(rel, pieces)]
    return out


def reconcile(files: set[str], folders: set[str]) -> bool:
    """Bring the index in line with the disk for these changed files and folders (a folder means everything under
    it): new files are read and embedded, changed files re-read, deleted or moved-away files dropped. Returns False
    if another indexing job is running, so the caller retries."""
    global index_state, index_error
    from evaluation.custom import list_files, wanted_file

    if not store.ready():
        return True
    if not index_lock.acquire(blocking=False):
        return False
    try:
        roots, whole = store.roots(), store.whole_roots()
        inside = lambda child, parent: child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)

        def root_index(path: str) -> int | None:  # the innermost indexed folder holding this path
            matches = [i for i, r in enumerate(roots) if inside(path, r)]
            return max(matches, key=lambda i: len(roots[i])) if matches else None

        current: dict[str, int] = {}  # files that should be indexed, within the changed scope
        scope: set[int] = set()  # indexed files within the changed scope
        for folder in folders:
            # A changed folder may sit inside an indexed folder or (e.g. deleted parent) contain whole ones.
            targets = [folder] if root_index(folder) is not None else [r for r in roots if inside(r, folder)]
            for target in targets:
                r = root_index(target)
                if Path(target).is_dir():
                    for f in list_files(Path(target), whole_computer=whole[r]):
                        if wanted_file(f, Path(roots[r]), whole[r]):
                            current[str(f)] = r
                scope |= set(store.live_ids(under=target))
                if (i := store.path_id(target)) is not None:
                    scope.add(i)
        for f in files:
            r = root_index(f)
            if r is None:
                continue
            if (i := store.path_id(f)) is not None:
                scope.add(i)
            if wanted_file(Path(f), Path(roots[r]), whole[r]):
                current[f] = r
        paths = store.manifest["paths"]
        removed = [i for i in scope if paths[i] not in current]
        changed = []
        for i in scope:
            if paths[i] in current:
                old, now = store.stamp(i), _stamp(paths[i])
                if abs(old[0] - now[0]) > 1e-3 or old[1] != now[1]:
                    changed.append(i)
        new = [p for p in current if store.path_id(p) is None]
        if not (removed or changed or new):
            return True
        prev_state = index_state
        index_state = "updating"
        by_root: dict[int, list[Path]] = {}
        for p in new + [paths[i] for i in changed]:
            by_root.setdefault(current[p], []).append(Path(p))
        try:
            added = _extract(by_root)
            store.apply_changes([(r, p, pieces) for r, p, pieces, _ in added], removed + changed,
                                visuals=[assets for _, _, _, assets in added])
        except Exception as e:
            index_state, index_error = prev_state, None
            log.warning("Index update failed: %s: %s", type(e).__name__, e)
            return True
        log.info("Index updated: %d new, %d changed, %d removed", len(new), len(changed), len(removed))
        _start_embedding()
        return True
    finally:
        index_lock.release()


watcher = FolderWatcher(reconcile)


def _sync_watching() -> None:
    """Watch exactly the indexed folders."""
    try:
        watcher.watch(store.roots())
    except Exception as e:  # e.g. too many folders for the OS watcher; the startup scan still catches up
        log.warning("Could not watch folders: %s", e)


def _restore_from_disk() -> None:
    """Reload the last index, finish embedding it if the server stopped part way, then catch up on files that
    changed while ReFinder was closed and start watching for new changes."""
    if not store.load():
        return
    if not store.embedding_complete():
        _start_embedding()

    def catch_up() -> None:
        for _ in range(60):  # wait out a running /index
            if reconcile(set(), set(store.roots())):
                break
            time.sleep(5)
        _sync_watching()

    threading.Thread(target=catch_up, daemon=True, name="refinder-catch-up").start()


def _exit_with_parent() -> None:
    """When started by the desktop app (REFINDER_PARENT_PID), quit if the app goes away, even after a crash."""
    pid = int(os.environ.get("REFINDER_PARENT_PID", "0") or 0)
    if not pid:
        return

    def loop() -> None:
        while True:
            time.sleep(2)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass

    threading.Thread(target=loop, daemon=True, name="refinder-parent-watch").start()


_exit_with_parent()
_restore_from_disk()


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.post("/index", response_model=IndexResponse)
def index_folder(request: IndexRequest) -> IndexResponse:
    """Index a folder, replacing the current index or adding to it. Returns once keyword search is ready; embedding
    continues in the background (poll /status)."""
    global index_state, index_error, embedder

    from evaluation.custom import list_files, load_corpus
    from refinder.extract import visual_assets

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
            known = set(store.keyword["path_id"])
            files = [f for f in files if str(f) not in known]
            if not files:
                index_state = prev_state
                return IndexResponse(status="unchanged", files_queued=0, total_files=len(known), failed=0,
                                     extract_s=0.0)
        workers = min(8, os.cpu_count() or 1) if len(files) >= PARALLEL_MIN_FILES else 1
        new_paths, _, new_pieces, stats = load_corpus(folder, workers=workers, files=files,
                                                      use_cache=USE_EXTRACT_CACHE)
        if not new_paths:
            index_state = prev_state if adding else "idle"
            raise HTTPException(status_code=400, detail="Folder contains no supported files")
        abs_paths = [str(folder / p) for p in new_paths]
        piece_texts = [[p.text for p in ps if p.text.strip()] for ps in new_pieces]
        visuals = [visual_assets(ps) for ps in new_pieces]
        if adding:
            store.add(folder, abs_paths, piece_texts, whole=request.whole_computer, visuals=visuals)
        else:
            if embedder is not None and embedder.model_name != MODEL_NAME:
                embedder = None  # index built with an older model: switch to the current one
            store.replace(folder, abs_paths, piece_texts, MODEL_NAME, whole=request.whole_computer, visuals=visuals)
        _start_embedding()
        _sync_watching()
    except HTTPException:
        raise
    except Exception as e:
        index_state, index_error = "error", f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=index_error)
    finally:
        index_lock.release()

    return IndexResponse(status="searchable", files_queued=len(new_paths), total_files=store.live_count(),
                         failed=len(stats["failed"]),
                         extract_s=round(stats["extract_s"], 2))


@app.post("/folders/remove")
def remove_folder(request: RemoveFolderRequest) -> dict[str, Any]:
    """Stop searching one indexed folder. Its files and vectors are dropped; other folders are untouched."""
    global index_state, index_error
    if not index_lock.acquire(timeout=30):
        raise HTTPException(status_code=409, detail="Busy indexing; try again shortly")
    try:
        try:
            generation = store.remove_root(request.folder)
        except KeyError:
            raise HTTPException(status_code=404, detail="That folder isn't indexed")
        if generation is None:
            index_state, index_error = "idle", None
        elif not store.embedding_complete():
            _start_embedding()
    finally:
        index_lock.release()
    _sync_watching()
    return {"folders": store.roots(), "indexed_files": store.live_count()}


def filter_params(
    types: str | None = Query(default=None, description="Comma-separated extensions, e.g. pdf,docx"),
    modified_after: float | None = Query(default=None, description="Unix seconds"),
    modified_before: float | None = None,
    added_after: float | None = None,
    added_before: float | None = None,
    min_size: int | None = Query(default=None, ge=0, description="Bytes"),
    max_size: int | None = Query(default=None, ge=0),
    folder: str | None = Query(default=None, description="Substring of the file's folder path"),
) -> Filters:
    return Filters(
        types={t.strip().lower().lstrip(".") for t in types.split(",") if t.strip()} if types else None,
        modified_after=modified_after, modified_before=modified_before,
        added_after=added_after, added_before=added_before,
        min_size=min_size, max_size=max_size, folder=folder or None,
    )


def _encode(text: str):
    if not store.count():  # skip loading the model while only keyword search is available
        return None
    enc = _get_embedder()
    with model_lock:
        return enc.encode_query(text)


def _encode_visual(text: str):
    if not store.visual_count():
        return None
    try:
        encoder = _get_visual_embedder()
        with model_lock:
            return encoder.encode_query(text)
    except Exception as exc:
        log.warning("Visual search unavailable: %s: %s", type(exc).__name__, exc)
        return None  # the filename, keyword and text searches remain usable


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = "",
    k: int = Query(default=5, ge=1, le=100),
    mode: str = Query(default="smart", pattern="^(smart|keyword)$"),
    filters: Filters = Depends(filter_params),
) -> SearchResponse:
    """mode=smart: filename + phrase + keyword + meaning, plus file-type/date hints read from the query ("tax pdf
    from last year"). mode=keyword: plain BM25 keyword match only. Filters narrow either mode; with filters and no
    query, lists matching files newest first."""
    from evaluation.query_meta import RECENT, parse

    if not store.ready():
        raise HTTPException(status_code=409, detail="Index a folder before searching")
    if not q.strip():
        if not filters.active():
            raise HTTPException(status_code=422, detail="Enter a search or set a filter")
        return SearchResponse(results=[SearchResult(**hit) for hit in store.browse(filters, k=k)])
    if mode == "keyword":
        return SearchResponse(results=[SearchResult(**hit) for hit in store.keyword_search(q, k=k, filters=filters)],
                              suggestion=store.suggest(q))
    meta = parse(q)
    if meta.active() and not meta.content:  # only hints, e.g. "pdfs from last week"
        hits = store.browse_meta(meta, filters, k=k)
    else:
        # Search the full wording: "during May" or "pitch deck" often describe the content too. The hints only add
        # a small boost for files of that type or date.
        # "recent"/"latest" only ask for newer files, so they are left out of the text itself.
        text = " ".join(RECENT.sub(" ", q).split()) if meta.recent else q
        hits = store.search(text or q, _encode(text or q), k=k, filters=filters, meta=meta,
                            visual_vec=_encode_visual(text or q))
    return SearchResponse(results=[SearchResult(**hit) for hit in hits], understood=meta.labels)


@app.get("/similar", response_model=SearchResponse)
def similar(path: str, k: int = Query(default=20, ge=1, le=100),
            filters: Filters = Depends(filter_params)) -> SearchResponse:
    """"More like this": files most similar to an indexed file."""
    _indexed_file(path)
    return SearchResponse(results=[SearchResult(**hit) for hit in store.similar(path, k=k, filters=filters)])


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    roots = store.roots()
    return StatusResponse(
        indexed_files=store.live_count(),
        chunks=store.count(),
        state=index_state,
        embedded_chunks=store.embedded_chunks,
        total_chunks=store.total_chunks,
        embedded_images=store.embedded_images,
        total_images=store.total_images,
        error=index_error,
        folder=roots[0] if len(roots) == 1 else None,
        folders=roots,
        whole_computer=str(Path.home()) in roots,
        file_types=store.file_types(),
    )


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
