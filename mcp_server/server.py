"""ReFinder MCP server (MVP).

Exposes the ReFinder FastAPI backend as MCP tools so an LLM client
(Claude Desktop, Claude Code, etc.) can search and manage the local index.

The MCP server holds NO model or index itself. It is a thin adapter that
calls the backend over HTTP, so the backend stays the single source of truth
and everything keeps running locally.

Backend contract (change the route strings in the tools below if your routes differ):
    GET  /search?q=...&k=5          -> {"results": [{"path", "score", "snippet", "chunk_id"}]}
    POST /index   {"folder": "..."} -> {"status": "started", "files_queued": int}
    GET  /status                    -> {"indexed_files": int, "chunks": int, "state": "idle|indexing"}
    GET  /file?path=...             -> {"path", "text"}

Run:
    REFINDER_URL=http://127.0.0.1:8000 python mcp_server/server.py      # stdio transport
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("REFINDER_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.environ.get("REFINDER_TIMEOUT", "30"))

mcp = FastMCP("refinder")


def _call(method: str, route: str, **kwargs) -> dict:
    """Call the backend and turn failures into readable errors for the LLM."""
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            resp = client.request(method, route, **kwargs)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach the ReFinder backend at {BASE_URL}. Is it running?"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Backend error {e.response.status_code}: {e.response.text}")


@mcp.tool()
def search_files(query: str, top_k: int = 5) -> str:
    """Semantically search the user's local files by meaning, not filename.

    Use natural language (e.g. "notes about the Q3 budget meeting").
    Returns the best-matching chunks with file path, similarity score, and snippet.
    """
    top_k = max(1, min(top_k, 20))
    data = _call("GET", "/search", params={"q": query, "k": top_k})
    results = data.get("results", [])
    if not results:
        return "No matching files found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['path']}  (score {r['score']:.3f})\n   ...{r['snippet'].strip()}..."
        )
    return "\n".join(lines)


@mcp.tool()
def index_folder(folder: str) -> str:
    """Start indexing a folder (scan, extract text, embed, store). Returns immediately;
    call index_status to check progress."""
    data = _call("POST", "/index", json={"folder": folder})
    return f"Indexing {data.get('status', 'started')}: {data.get('files_queued', '?')} files queued."


@mcp.tool()
def index_status() -> str:
    """Report how many files/chunks are indexed and whether indexing is running."""
    d = _call("GET", "/status")
    return f"State: {d['state']}. Indexed files: {d['indexed_files']}, chunks: {d['chunks']}."


@mcp.tool()
def read_file(path: str, max_chars: int = 4000) -> str:
    """Read the extracted text of an indexed file (e.g. after search_files).
    Truncated to max_chars to keep context small."""
    d = _call("GET", "/file", params={"path": path})
    text = d["text"]
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[truncated, {len(text) - max_chars} more chars]"
    return text


if __name__ == "__main__":
    mcp.run()  # stdio transport
