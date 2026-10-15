# Path helps us work with and validate folder paths.
from pathlib import Path
from typing import Any

# NumPy is used to sort similarity scores.
import numpy as np

# FastAPI creates the API.
# HTTPException returns clear API errors.
# Query adds validation to URL query parameters.
from fastapi import FastAPI, HTTPException, Query

# BaseModel describes the shape of incoming and outgoing JSON.
from pydantic import BaseModel

# ---------------------------------------------------------
# Request and response models
# ---------------------------------------------------------

# JSON the client sends when asking us to index a folder.
class IndexRequest(BaseModel):
    folder: str


# JSON returned after indexing a folder.
class IndexResponse(BaseModel):
    status: str
    files_queued: int


# One individual search result.
class SearchResult(BaseModel):
    path: str
    score: float
    snippet: str
    chunk_id: int | None


# The complete response returned by /search.
class SearchResponse(BaseModel):
    results: list[SearchResult]


# Information returned by /status.
class StatusResponse(BaseModel):
    indexed_files: int
    chunks: int
    state: str


# File contents returned by /file.
class FileResponse(BaseModel):
    path: str
    text: str


# ---------------------------------------------------------
# Application and server memory
# ---------------------------------------------------------

# Create the FastAPI application.
app = FastAPI(title="ReFinder API")


# Folder currently loaded into the search engine.
# None means no folder has been indexed yet.
corpus_root: Path | None = None

# Relative paths of all indexed files.
paths: list[str] = []

# Extracted text from each indexed file.
# texts[0] belongs to paths[0], and so on.
texts: list[str] = []

# Search system containing the model and document vectors.
retriever: Any | None = None

# Current state of the indexing process.
index_state = "idle"


# ---------------------------------------------------------
# Index endpoint
# ---------------------------------------------------------

# Accept a JSON body such as:
# {"folder": "evaluation/sample/Corpus"}
@app.post("/index", response_model=IndexResponse)
def index_folder(request: IndexRequest) -> IndexResponse:
    # We assign new values to these module-level variables.
    global corpus_root, paths, texts, retriever, index_state

    # Import the ML stack only when indexing is requested. This keeps server
    # startup and the API docs available even while model dependencies load.
    from evaluation.custom import DenseRetriever, load_corpus
    from refinder.embedder import DEFAULT_MODEL

    # Convert the provided folder into a complete absolute path.
    # expanduser() supports paths beginning with "~".
    folder = Path(request.folder).expanduser().resolve()

    # Stop and return HTTP 404 if the folder does not exist.
    if not folder.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Folder not found",
        )

    # Tell /status that indexing is currently happening.
    index_state = "indexing"

    try:
        # Read supported files and extract their text.
        # new_paths contains filenames.
        # new_texts contains the corresponding file contents.
        # stats contains extraction statistics.
        new_paths, new_texts, stats = load_corpus(folder)

        # Do not create an empty search index.
        if not new_paths:
            raise HTTPException(
                status_code=400,
                detail="Folder contains no supported files",
            )

        # Split the documents into chunks and create embeddings.
        new_retriever = DenseRetriever(
            DEFAULT_MODEL,
            new_texts,
            use_cache=True,
        )

        # Save the completed index in server memory.
        corpus_root = folder
        paths = new_paths
        texts = new_texts
        retriever = new_retriever

    finally:
        # Return to idle whether indexing succeeds or fails.
        index_state = "idle"

    # Send a JSON response to the caller.
    return IndexResponse(
        status="completed",
        files_queued=len(paths),
    )


# ---------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------

# Example request:
# GET /search?q=notes about taxes&k=5
@app.get("/search", response_model=SearchResponse)
def search(
    # The user's natural-language search.
    q: str,

    # Number of results to return.
    # FastAPI only allows values from 1 through 20.
    k: int = Query(default=5, ge=1, le=20),
) -> SearchResponse:

    # Searching is impossible before a folder is indexed.
    if retriever is None:
        raise HTTPException(
            status_code=409,
            detail="Index a folder before searching",
        )

    # Convert the user's query into an embedding vector.
    query_vector = retriever.embedder.encode_query(q)

    # score() expects multiple query vectors.
    # [None] wraps our one vector inside a batch.
    # [0] gets the score results for that one query.
    scores = retriever.score(query_vector[None])[0]

    # Ignore files that do not have a valid score.
    valid_indices = np.flatnonzero(np.isfinite(scores))

    # Sort valid file indexes from highest score to lowest.
    # [:k] keeps only the best k results.
    ranked_indices = valid_indices[
        np.argsort(-scores[valid_indices])
    ][:k]

    # This list will hold the formatted search results.
    results = []

    # Turn each matching file into a SearchResult.
    for index in ranked_indices:
        # Use the first 300 characters as a preview.
        # Replacing line breaks makes the preview easier to display.
        snippet = texts[index].replace("\n", " ")[:300]

        results.append(
            SearchResult(
                path=paths[index],
                score=float(scores[index]),
                snippet=snippet,

                # The current retriever ranks files rather than
                # returning the exact matching chunk.
                chunk_id=None,
            )
        )

    return SearchResponse(results=results)


# ---------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------

# Example request:
# GET /status
@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    # Each stored vector represents one document chunk.
    # If there is no retriever, there are zero chunks.
    chunk_count = len(retriever.vecs) if retriever else 0

    return StatusResponse(
        indexed_files=len(paths),
        chunks=chunk_count,
        state=index_state,
    )


# ---------------------------------------------------------
# File endpoint
# ---------------------------------------------------------

# Example request:
# GET /file?path=finance/taxes/2025_tax_notes.txt
@app.get("/file", response_model=FileResponse)
def get_file(path: str) -> FileResponse:
    try:
        # Find where this filename appears in paths.
        index = paths.index(path)

    except ValueError:
        # paths.index() raises ValueError when the path is absent.
        raise HTTPException(
            status_code=404,
            detail="Indexed file not found",
        )

    # Use the same index to retrieve the matching text.
    return FileResponse(
        path=paths[index],
        text=texts[index],
    )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

# A simple way to verify that the server is running.
@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ReFinder API is running"}
