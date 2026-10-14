# ReFinder
A semantic search engine for your local file system that finds files by contents rather than exact filename or keyword matches. Ideally, everything runs off-cloud.

## Features
- Model inference and storage system for file content embedding
- Natural language search queries for input
- Nearest neighbor retrieval for relevant results (similarity metric of team’s choice)

## Project structure
- `main.py` - embedding / similarity experiments (ML)
- `mcp_server/` - MCP server so an LLM can search and manage the index through the backend API (see `mcp_server/README.md`)

## Project overview

ReFinder is a local semantic file search project developed by our
Texas Luminescence team. Users search with natural language to find
files by meaning rather than exact filenames or matching keywords.

For example, searching "notes about conditional probability"
should return relevant class notes, even with a different filename.

## Planned workflow

1. Scan a folder selected by the user.
2. Extract text from supported files.
3. Use a machine learning embedding model to convert text into vectors.
4. Store those vectors in a searchable index.
5. Embed the user's query and retrieve the closest matches.
6. Display results with file paths and previews.

## Team responsibilities

- ML: Select and evaluate embedding models and similarity search.
- Backend: Build file indexing, index updates, and the search API.
- Frontend: Build the search interface and file previews.

## Status

ReFinder is in early development. Our goal is to run file processing
and search locally, without uploading personal files to the cloud.
