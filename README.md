# ReFinder
A semantic search engine for your local file system that finds files by contents rather than exact filename or keyword matches. Ideally, everything runs off-cloud.

## Features
- Model inference and storage system for file content embedding
- Natural language search queries for input
- Nearest neighbor retrieval for relevant results (similarity metric of team’s choice)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py "recipe for dinner"   # first run downloads all-MiniLM-L6-v2 (~90MB), then runs offline
```

## Evaluation
Benchmarks embedding models against a BM25 keyword baseline on the Kaggle
[Corpus for Semantic Matching](https://www.kaggle.com/datasets/mantunes/corpus-for-semantic-matching)
(~4.2k topic-named web-text files; downloaded automatically, no Kaggle login needed).

The corpus has no labels, so it's scored as known-item search: for each eval file, a sentence mentioning
the file's topic is removed from the index and used to find the file again, as the topic word alone (`keyword`),
the full sentence (`sentence`), and the sentence with the topic word stripped out (`masked`, meaning only).
Reports Recall@1/5/10 and MRR@10; results are saved to `evaluation/results/`.

```bash
python -m evaluation.semantic_matching                                   # 1000 files, MiniLM vs BM25
python -m evaluation.semantic_matching --docs 0 --queries 1000           # full corpus (slow: ~400k chunks)
python -m evaluation.semantic_matching --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
```
Chunk embeddings are cached in `.cache/eval/`, so re-runs with the same model and corpus skip embedding.

### BEIR (public benchmarks with natural-language queries)
Scores on SciFact, NFCorpus and FiQA, which have human relevance labels and published baselines to compare against
(nDCG@10). Documents go through the same chunk + max-pool pipeline as real files; BM25 uses `bm25s` with stemming.
```bash
python -m evaluation.beir
python -m evaluation.beir --datasets scifact --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
```

### Custom corpus (our own files + labeled queries)
The benchmark that matters most: a folder of realistic files plus a CSV of description-style queries.
Reports accuracy (Hit@1/5/10, MRR@10, nDCG@10, overall / per category / per file type) and speed (indexing
throughput, index size, model load, per-query search latency p50/p95). A BM25 baseline runs alongside.

```bash
python -m evaluation.custom --corpus Corpus --queries queries.csv
python -m evaluation.custom --corpus evaluation/sample/Corpus --queries evaluation/sample/queries.csv  # demo set
python -m evaluation.custom ... --distractors 300   # mix in 300 unrelated web-page files (harder, bigger index)
python -m evaluation.custom ... --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
```

`queries.csv` format (UTF-8, header row required):

| column | required | meaning |
|---|---|---|
| `query` | yes | what a user would type, e.g. `the newer lease with the higher rent` |
| `file` | yes | path relative to `Corpus/` (`finance/lease_2025.pdf`); a bare filename works if unique; join several acceptable answers with `\|` |
| `category` | no | free label for breakdowns (persona, query style, ...) |

Supported files: `.pdf`, `.docx`, and any UTF-8 text (`.txt`, `.md`, `.csv`, ...); images/archives are skipped and
reported. Rows pointing at missing files are reported and skipped. Each run writes `summary.json` and
`per_query.csv` (rank + top-3 files per query, for reviewing misses) to `evaluation/results/custom_<timestamp>/`.
`evaluation/sample/` is a 45-file stand-in corpus (6 personas, confusable pairs like two leases) with 115 queries.
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
