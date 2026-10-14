# ReFinder
A semantic search engine for your local file system that finds files by contents rather than exact filename or keyword matches. Ideally, everything runs off-cloud.

## Features
- Model inference and storage system for file content embedding
- Natural language search queries for input
- Nearest neighbor retrieval for relevant results (similarity metric of team’s choice)

## Project structure
- `main.py` - embedding / similarity experiments (ML)
- `mcp_server/` - MCP server so an LLM can search and manage the index through the backend API (see `mcp_server/README.md`)
