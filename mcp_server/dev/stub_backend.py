"""Tiny fake ReFinder backend so the MCP server can be tested without the real one.
Uses naive word-overlap scoring instead of embeddings.

Run: uvicorn stub_backend:app --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

DOCS = {
    "/docs/budget_q3.txt": "Q3 budget meeting notes: marketing spend cut by 10 percent, hiring freeze until November.",
    "/docs/pasta.txt": "Grandma's pasta recipe: boil water, add salt, cook spaghetti nine minutes, toss with tomato sauce.",
    "/docs/ml_notes.txt": "Lecture notes on embeddings: cosine similarity compares vectors, nearest neighbor search retrieves chunks.",
}
state = {"state": "idle"}


class IndexReq(BaseModel):
    folder: str


@app.get("/search")
def search(q: str, k: int = 5):
    words = set(q.lower().split())
    scored = []
    for path, text in DOCS.items():
        overlap = len(words & set(text.lower().replace(",", "").replace(".", "").split()))
        if overlap:
            scored.append({"path": path, "score": overlap / len(words), "snippet": text, "chunk_id": path + "#0"})
    scored.sort(key=lambda r: -r["score"])
    return {"results": scored[:k]}


@app.post("/index")
def index(req: IndexReq):
    state["state"] = "idle"
    return {"status": "started", "files_queued": len(DOCS)}


@app.get("/status")
def status():
    return {"indexed_files": len(DOCS), "chunks": len(DOCS), "state": state["state"]}


@app.get("/file")
def file(path: str):
    if path not in DOCS:
        raise HTTPException(404, "not indexed")
    return {"path": path, "text": DOCS[path]}
