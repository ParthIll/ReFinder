# ReFinder

Local semantic file search: a user picks folders, ReFinder indexes PDF/DOCX/TXT/MD (and more) on their machine,
and finds files from a natural-language description of their contents. File contents and queries never leave
the computer. UT Austin student team (ML, Backend, Frontend). Planning lives in Notion "ReFinder HQ"
(System Architecture page + six drilldowns, Feature Backlog with task IDs, Decision Log, Experiments & Quality).

## Target architecture (from Notion System Architecture)

```
React + Tauri UI ──HTTP──> FastAPI (127.0.0.1)  ──enqueue job──> Python worker: scan → extract → chunk → embed
                              │                                         │ upsert             │ file/job status
                              │ search: embed query, cosine top-k       v                    v
                              └──────────────────────────────> Chroma (chunks+vectors)   SQLite (file ledger, jobs)
MCP adapter (mcp_server/) ──HTTP──> same FastAPI
```

- Two paths: indexing (scan → extract → chunk → embed → store) and search (embed query → nearest chunks →
  one result per file → snippet + open). Both must use the same pinned model.
- Chroma holds chunk text, vectors and light metadata; SQLite is the source of truth for file state
  (path, hash, status, error, chunk_count, index_version). They join on `file_id`.
- One FastAPI process with a background worker thread sharing one Chroma `PersistentClient`; never run
  several Uvicorn workers against the same Chroma dir.
- Model/dimension/chunking change ⇒ new collection + rebuild, never mixed embeddings.

Shared contracts (agree before parallel work):
- FileState: file_id, root_id, canonical_path, content_hash, status, error, last_indexed_at, chunk_count, index_version
- ChunkRecord: chunk_id, file_id, ordinal, text, page_or_section, model_id, chunker_version, vector
- SearchResult: file_id, display_name, path, type, score, snippet, page_or_section
- Status: queued → extracting → embedding → ready; plus skipped, error, stale
- API: POST /index, GET /status, GET /search (k default 5, max 20), GET /file (reject paths outside roots)

Ownership: ML owns embedding + retrieval quality and the relevance report; Backend owns scanning, indexing,
persistence, API; Frontend owns setup/search/preview UX.

## What exists in this repo (main / andrew)

- `refinder/embedder.py` — `Embedder` wraps Sentence Transformers: token-window chunking (200 tokens, 50 overlap),
  `encode` (documents) / `encode_queries` (queries), L2-normalized float32 so dot product == cosine.
  `DEFAULT_MODEL` is still `all-MiniLM-L6-v2`; the PRD's chosen MVP model is `BAAI/bge-small-en-v1.5` (384-d) with
  the query instruction "Represent this sentence for searching relevant passages: " on queries only.
- `refinder/extract.py` — simple PDF/DOCX/plain-text extraction used by the API and eval harness.
- `evaluation/server.py` + `evaluation/database.py` — the current FastAPI backend (Sanjay). `ChromaIndex` persists
  chunks in `evaluation/chroma_db/` (cosine HNSW) and a `manifest.json` with every file's extracted text.
  `/index` is synchronous and rebuilds the whole collection for one folder; no SQLite ledger, no incremental
  rescan, no delete/rename handling yet. Search fetches `max(20k, 50)` chunks and keeps the best chunk per file.
- `evaluation/custom.py` — main benchmark: corpus folder + `queries.csv` (query,file[,category]); dense vs BM25,
  Hit@1/5/10, MRR@10, nDCG@10, latency p50/p95. `evaluation/sample/` is a 45-file stand-in corpus, 115 queries.
- `evaluation/semantic_matching.py` (Kaggle known-item), `evaluation/beir.py` (SciFact/NFCorpus/FiQA nDCG@10).
- `refinder-extractors/` — richer per-format extractors (pdf/docx/pptx/xlsx/txt/md/csv/tsv) and a SQLite
  one-row-per-file indexer with size/mtime/sha256 skip. Not importable as-is: hyphenated dir, relative imports, and
  `pyproject.toml`/`test.py` reference a `refinder.` package path that doesn't exist (and would collide with the
  root `refinder/`). The `testing-environment` branch loads it via importlib under the name `refinder_extractors`.
- `mcp_server/` — MCP adapter calling the FastAPI routes; `dev/stub_backend.py` is a fake backend for testing.
- `main.py` — toy embedding smoke test.

## Other branches

- `origin/testing-environment` (Parth, ML) is ahead of main with: extractor bridge + paragraph-aware chunking with
  page/slide/row locations, `MODEL_PROMPTS` (bge query instruction, nomic prefixes), Qwen3/granite/arctic support,
  `refinder/tree.py` semantic category tree (prune/boost), team-corpus query CSV auto-discovery, OrgSmith and
  FinanceBench corpora, `.eml` extraction. Check it before re-implementing any of these; it isn't merged to main.

## Evaluation data

- Team corpus = Google Drive folder "ReFinder File Corpus", one subfolder per member (Andrew, Parth, Sanjay, Dagem,
  Elliot) with per-folder query CSVs (columns like document_type, corpus_path/source_file, query_id, query_type,
  query, split). Mirror it locally as `ReFinder File Corpus/` at the repo root (gitignored) and run
  `python -m evaluation.custom --corpus "ReFinder File Corpus"` on testing-environment.
- Andrew/ holds 50 synthetic PDFs (Agreements and Contracts, Careers and Resumes, Email and Correspondence,
  Financial Statements, Tax and Government Forms) with 150 labeled queries. Baseline (Oct 14): bge-small
  Hit@1 134/150, Hit@5 148/150, MRR 0.935; BM25 133/147; BGE+BM25 RRF Hit@1 137/150.
- On the full team corpus, Hit@1 is much lower (≈0.44 bge-small, ≈0.57–0.60 Qwen3-0.6B per testing-environment
  README), well under the PRD gate (Hit@1 ≥ 0.75, Hit@5 ≥ 0.90). Treat that gap as the main ML problem.
- Never commit corpora, results, or Chroma data (`.cache/`, `evaluation/results/`, `chroma_db/`, `datasets/`).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn evaluation.server:app --reload          # API at http://127.0.0.1:8000/docs
python -m evaluation.custom --corpus evaluation/sample/Corpus --queries evaluation/sample/queries.csv \
    --models BAAI/bge-small-en-v1.5
python -m evaluation.beir --datasets scifact --models BAAI/bge-small-en-v1.5
```

Embedding caches go to `.cache/eval/` keyed by model + chunk params + corpus content; pass `--fresh` to re-time indexing.

## Conventions

- Evaluate at the file level (unique files, file score = best chunk), not chunk-level hits. Report Hit@1/3/5, MRR,
  per-category misses, latency, memory, index size; review misses with the Notion failure-review template.
- Freeze corpus, labels, model revision and chunk settings before comparing models; change one thing at a time.
- Don't embed filenames or folder names into chunk text; apply query instructions only to queries.
- Keep everything offline after first-run model download. Don't log full file text or raw queries.
- Tie work to Feature Backlog IDs (e.g. M04, S07, Q02) in commit messages / PR descriptions.
- Open decisions (Notion Decision Log DEC01–DEC08): first OS/hardware, scanned-PDF behavior, hidden/symlink policy,
  model revision + offline cache, SQLite/Chroma consistency, BGE-only vs hybrid ranking, log privacy, sprint length.
  Chunk overlap is 50 tokens in code vs 40 in the PRD — unresolved.
