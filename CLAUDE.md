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

## What exists in this repo

`andrew` = `main` (Sanjay's FastAPI + Chroma backend) merged with `testing-environment` (Parth's ML/eval work).

- `refinder/embedder.py` — `Embedder` wraps Sentence Transformers; `encode` (documents) / `encode_queries`
  (queries), L2-normalized float32 so dot product == cosine. Two chunkers: `chunk` (fixed 200-token windows,
  50 overlap) and `chunk_pieces` (packs whole extractor pieces to ~200 tokens with a 50-token carry-over and keeps
  piece ranges for locations). `MODEL_PROMPTS` adds missing query/document prefixes (bge query instruction, nomic);
  `REMOTE_CODE_MODELS` enables trust_remote_code. `DEFAULT_MODEL` is still `all-MiniLM-L6-v2`; the PRD's MVP model
  is `BAAI/bge-small-en-v1.5` (384-d). On CPU, models load in float32, and `ONNX_MODELS` (Harrier 270M, bge-small,
  MiniLM) run through `refinder/onnx_encoder.py`, which exports to `.cache/onnx/` on first use, reproduces the
  model's pooling (last-token, CLS or mean) and falls back to PyTorch on failure. `MAX_BATCH = 32` (bigger was slower).
- `refinder/indexer.py` — the ML half of indexing, no storage: `index_files(paths, embedder)` yields per file its
  chunks (text, ordinal, location), vectors, whole-file vector and full chunk count, smallest files first, batching
  across files and embedding repeated texts once. Giant files keep `MAX_CHUNKS_PER_FILE = 100` evenly spaced chunks.
  Meant to be called by the backend's worker; the API doesn't use it yet.
- `refinder/extract.py` — bridge that loads `refinder-extractors/` via importlib as `refinder_extractors`, adds
  `.eml`, and turns labelled pieces ("Page 3 Paragraph 2", "Slide 4", "Row 12") into `Piece(text, location)`.
  `extract_text` joins pieces for callers that want one string.
- `refinder/tree.py` — experimental semantic category tree over files (spherical k-means); `prune` (beam search)
  and `boost` (category-match bonus) search modes.
- `evaluation/server.py` + `evaluation/database.py` — the FastAPI backend (Sanjay). `ChromaIndex` persists chunks
  in `evaluation/chroma_db/` (cosine HNSW) plus a `manifest.json` holding every file's extracted text.
  `/index` is synchronous and rebuilds the whole collection for one folder; no SQLite ledger, no incremental
  rescan, no delete/rename handling. Search fetches `max(20k, 50)` chunks and keeps the best chunk per file.
  Note: the API indexes with window `chunk()` and the default model, so it does not yet match what the benchmark
  measures (paragraph chunking, chosen model, locations).
- `evaluation/index_bench.py` — indexing profiler: times extraction (uncached, per type, slowest files), chunking and
  embedding per model / device / batch size, with `--cap`, `--int8`, `--limit-chunks` (extrapolates).
- `evaluation/custom.py` — main benchmark: corpus folder + query CSV(s); auto-discovers the team's per-folder
  query CSVs (source_file/corpus_path, secondary_relevant_files, query_type, split). Dense vs BM25, Hit@1/5/10,
  Recall@10, MRR@10, graded nDCG@10, breakdowns by folder/category/file type, latency, `--chunking`, `--tree-*`.
  `evaluation/sample/` is a 45-file stand-in corpus with 115 queries.
- `evaluation/orgsmith.py`, `evaluation/financebench.py` — download external test corpora into `datasets/`.
  `evaluation/semantic_matching.py` (Kaggle known-item), `evaluation/beir.py` (SciFact/NFCorpus/FiQA nDCG@10).
- `refinder-extractors/` — the team's per-format extractors (pdf/docx/pptx/xlsx/txt/md/csv/tsv) and a standalone
  SQLite one-row-per-file indexer with size/mtime/sha256 skip. Kept unmodified; not importable on its own
  (hyphenated dir, `pyproject.toml`/`test.py` point at a nonexistent `refinder.` path), so use it via
  `refinder/extract.py`.
- `mcp_server/` — MCP adapter calling the FastAPI routes; `dev/stub_backend.py` is a fake backend for testing.
- `main.py` — toy embedding smoke test.

## Evaluation data

- Team corpus = Google Drive folder "ReFinder File Corpus", one subfolder per member (Andrew, Parth, Sanjay, Dagem,
  Elliot) with per-folder query CSVs (columns like document_type, corpus_path/source_file, query_id, query_type,
  query, split). Mirror it locally as `ReFinder File Corpus/` at the repo root (gitignored) and run
  `python -m evaluation.custom --corpus "ReFinder File Corpus"`.
- Andrew/ holds 50 synthetic PDFs (Agreements and Contracts, Careers and Resumes, Email and Correspondence,
  Financial Statements, Tax and Government Forms) with 150 labeled queries. Baseline (Oct 14): bge-small
  Hit@1 134/150, Hit@5 148/150, MRR 0.935; BM25 133/147; BGE+BM25 RRF Hit@1 137/150.
- On the full team corpus, Hit@1 is much lower (≈0.44 bge-small, ≈0.57–0.60 Qwen3-0.6B per README), well under
  the PRD gate (Hit@1 ≥ 0.75, Hit@5 ≥ 0.90). On OrgSmith (441 questions) BM25 Hit@1 0.782 beat bge-base 0.773,
  bge-small 0.762, Qwen3-0.6B 0.694. Treat the team-corpus gap and exact-term misses as the main ML problems.
- Sample corpus sanity check (merged code, bge-small, paragraph chunks): Hit@1 0.878, Hit@5 1.000; BM25 0.783/0.930.
- Best setup so far (evaluation/ITERATIONS.md): Harrier-0.6b + whole-file vector (`--file-vector 1`) + outlier
  size penalty (`--size-penalty 0.02`) + gte-reranker-modernbert-base fused by rank (`--rerank-fuse 1`):
  team 0.679 / OrgSmith 0.838 / sample 0.957 Hit@1, ~0.4 s per search. Not yet wired into the API.
- Indexing speed (evaluation/ITERATIONS.md, "Indexing speed"): embedding is the cost (extraction ~1 min, chunking
  ~2 s on the team corpus); 5 files hold 72% of its chunks. Capping files at 100 chunks passed the accuracy check;
  overlap 25 or 0 failed; int8 and static models were measured and not adopted.
  `python -m evaluation.lab --model harrier270 --set w_mean=1 w_size=0.02 w_ce=1 ce=gte --speed --key index --grid cap100`
- 54 team-corpus files are exact duplicates (Dagem/Sanjay CS 314 slides). Ties count as hits; any rank fusion must
  give tied scores the same rank. Use `python -m evaluation.lab` (leave-one-member-out + bootstrap) for new changes.
- Never commit corpora, results, or Chroma data (`.cache/`, `evaluation/results/`, `chroma_db/`, `datasets/`).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn evaluation.server:app --reload          # API at http://127.0.0.1:8000/docs
HF_HUB_OFFLINE=1 ...                            # prove a command runs offline once models are cached
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
