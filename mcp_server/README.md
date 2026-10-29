# ReFinder MCP server

Lets an LLM client (Claude Desktop, Claude Code, ...) search and manage the ReFinder
index through [MCP](https://modelcontextprotocol.io). It is a thin adapter: it holds no
model or index itself and just calls the ReFinder FastAPI backend over HTTP.

```
LLM client  <--stdio/MCP-->  server.py  <--HTTP-->  FastAPI backend (Chroma + SQLite)
```

## Tools

| Tool | Purpose | Backend route |
|---|---|---|
| `search_files(query, top_k=5)` | Semantic search; returns path, score, snippet | `GET /search?q=&k=` |
| `index_folder(folder)` | Start indexing a folder | `POST /index` |
| `index_status()` | File/chunk counts and indexing state | `GET /status` |
| `read_file(path, max_chars=4000)` | Extracted text of an indexed file | `GET /file?path=` |

The backend routes above are a proposed contract. Adjust the route strings in `server.py`
to match the real backend.

## Setup

```bash
pip install -r mcp_server/requirements.txt
```

`mcp` is pinned below 2 because mcp 2.x renamed `FastMCP`.

## Try it without the real backend

```bash
cd mcp_server/dev
uvicorn stub_backend:app --port 8000     # fake backend (word-overlap, no embeddings)
python smoke_test.py                     # in another terminal
```

## Use it with a client

```bash
# Claude Code
claude mcp add refinder -- python /absolute/path/to/ReFinder/mcp_server/server.py
```

For Claude Desktop, add the same command and args under `mcpServers` in its config JSON.
Set `REFINDER_URL` (default `http://127.0.0.1:8000`) if the backend is elsewhere.

## TODO

- Restrict `read_file` to indexed folders.
- More tools: `list_indexed_folders`, `remove_from_index`, `get_similar_files`.
