"""Turns files into everything the search index stores: chunks, chunk vectors and a whole-file vector.

This is the ML half of indexing, with no storage in it: the backend decides which files need (re)indexing, calls
index_files, and saves each IndexedFile as it arrives. Settings match what the benchmark measured
(evaluation/ITERATIONS.md, "Indexing speed").

    for f in index_files(paths, Embedder("microsoft/harrier-oss-v1-270m"), on_progress=print):
        if f.status == "ready":
            save(f.path, f.chunks, f.vectors, f.file_vector, f.n_chunks)
"""

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from refinder.embedder import CHUNK_OVERLAP, CHUNK_TOKENS, Embedder
from refinder.extract import Piece, UnsupportedFile, describe_location, extract_pieces

# Giant files (textbooks, big spreadsheets) keep this many evenly spaced chunks. On the team corpus this embeds 22% of
# the chunks with no accuracy loss on any test set (Harrier 270M, full pipeline; 50 also passed, 200 saved less).
MAX_CHUNKS_PER_FILE: int | None = 100
BATCH_CHUNKS = 256  # chunks embedded together, across files, per model call


@dataclass
class IndexedChunk:
    text: str
    ordinal: int  # position in the file's full chunk list (kept when a giant file is thinned out)
    location: str  # e.g. "chunk 3/12, page 2"


@dataclass
class IndexedFile:
    path: Path
    status: str  # "ready", "empty" (no text, e.g. a scanned PDF), "skipped" (unsupported type) or "error"
    error: str = ""
    content_hash: str = ""  # of the extracted text; identical files share vectors
    chunks: list[IndexedChunk] = field(default_factory=list)
    vectors: np.ndarray | None = None  # (len(chunks), dim), L2-normalized float32
    file_vector: np.ndarray | None = None  # normalized mean of the chunk vectors, scored alongside the best chunk
    n_chunks: int = 0  # the file's full chunk count before thinning; the search size penalty uses it


def keep_evenly(n: int, cap: int | None) -> np.ndarray:
    """Which of n chunks to embed: all of them, or `cap` evenly spaced ones (first and last included)."""
    if cap is None or n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(int))


def chunk_file(path: Path, embedder: Embedder, overlap: int, cap: int | None) -> IndexedFile:
    """Extract and chunk one file (no embedding yet)."""
    try:
        pieces: list[Piece] = extract_pieces(path)
    except UnsupportedFile as e:
        return IndexedFile(path, "skipped", str(e))
    except Exception as e:  # corrupt or password-protected file: report it, keep indexing the rest
        return IndexedFile(path, "error", f"{type(e).__name__}: {e}")
    texts = [p.text for p in pieces]
    chunks = embedder.chunk_pieces(texts, CHUNK_TOKENS, overlap)
    if not chunks:
        return IndexedFile(path, "empty")
    keep = keep_evenly(len(chunks), cap)
    out = [IndexedChunk(chunks[i].text, int(i), describe_location(int(i) + 1, len(chunks),
                                                                  pieces[chunks[i].first_piece].location,
                                                                  pieces[chunks[i].last_piece].location))
           for i in keep]
    digest = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()
    return IndexedFile(path, "ready", content_hash=digest, chunks=out, n_chunks=len(chunks))


def _file_vector(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(0)
    return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)


def index_files(
    paths: Iterable[Path],
    embedder: Embedder,
    overlap: int = CHUNK_OVERLAP,
    max_chunks_per_file: int | None = MAX_CHUNKS_PER_FILE,
    batch_chunks: int = BATCH_CHUNKS,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> Iterator[IndexedFile]:
    """Yield an IndexedFile per path, smallest files first, so most files are searchable early.

    Chunks from several files are embedded in one model call, and identical texts (duplicate files, repeated
    boilerplate chunks) are embedded once. A file is yielded as soon as all its chunks have vectors.
    on_progress(done, total, path) fires after each file is extracted and chunked.
    """
    paths = sorted({Path(p) for p in paths}, key=lambda p: (p.stat().st_size if p.exists() else 0, str(p)))
    seen: dict[str, np.ndarray] = {}  # chunk text -> vector, for this run
    pending: list[IndexedFile] = []
    todo: dict[str, None] = {}  # texts waiting for a vector (a dict keeps order and drops repeats)

    def flush() -> Iterator[IndexedFile]:
        if todo:
            texts = list(todo)
            seen.update(zip(texts, embedder.encode(texts)))
            todo.clear()
        for f in pending:
            f.vectors = np.stack([seen[c.text] for c in f.chunks])
            f.file_vector = _file_vector(f.vectors)
            yield f
        pending.clear()

    for done, path in enumerate(paths, 1):
        f = chunk_file(path, embedder, overlap, max_chunks_per_file)
        if f.status != "ready":
            yield f
        else:
            pending.append(f)
            todo.update((c.text, None) for c in f.chunks if c.text not in seen)
            if len(todo) >= batch_chunks:
                yield from flush()
        if on_progress:
            on_progress(done, len(paths), path)
    yield from flush()
