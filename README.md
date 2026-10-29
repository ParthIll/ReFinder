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
python -m evaluation.custom ... --chunking window paragraph   # compare chunking strategies
```

`queries.csv` format (UTF-8, header row required):

| column | required | meaning |
|---|---|---|
| `query` | yes | what a user would type, e.g. `the newer lease with the higher rent` |
| `file` | yes | path relative to `Corpus/` (`finance/lease_2025.pdf`); a bare filename works if unique; join several acceptable answers with `\|` |
| `category` | no | free label for breakdowns (persona, query style, ...) |

Supported files (via the team extractors in `refinder-extractors/`): `.pdf .docx .pptx .xlsx .txt .md .csv .tsv`; anything else is skipped and
reported. Rows pointing at missing files are reported and skipped. Each run writes `summary.json` and
`per_query.csv` (rank + top-3 files per query, and where in the top file the match was, e.g.
`chunk 7/9, page 3: Recommendation: shift parcel volume...`) to `evaluation/results/custom_<timestamp>/`.
`evaluation/sample/` is a 45-file stand-in corpus (6 personas, confusable pairs like two leases) with 115 queries.

#### Chunking and match locations
The team extractors label pieces (`Page 3 Paragraph 2`, `Slide 4`, `Row 12`, `Table 1`, sheet ranges). `refinder/extract.py`
strips those labels from the text the model sees and keeps them as a `location` on each piece. `paragraph` chunking
(default) packs whole consecutive pieces into ~200-token chunks, splitting only pieces that are too long on their own.
Each chunk starts with up to 50 tokens carried over from the end of the previous one, so details on a boundary
stay intact. Matches are reported by chunk number plus page/slide/rows where the format has them (`chunk 3/12, page 2`). `window` chunking is the older fixed 200-token
windows with 50-token overlap; accuracy was the same on the sample set, but windows can't say where a match is.

#### Category tree (experimental)
`refinder/tree.py` groups files into a hierarchy of semantic categories (spherical k-means on file vectors, recursively,
with keyword labels) and supports two search modes, compared against flat search with:

    python -m evaluation.custom --corpus "ReFinder File Corpus" --models Qwen/Qwen3-Embedding-0.6B \
        --tree-beams 2 4 --tree-boost 0.25 0.5 1 --tree-parts 1 8

- `--tree-beams`: prune. Descend the tree keeping the best *b* categories per level and score only the files reached.
  Fewer chunks scored, but a file in a pruned branch can't be found.
- `--tree-boost`: boost. Flat best-chunk score + weight x how well the query matches the file's categories.
- `--tree-parts`: give big files up to *n* vectors (one per topic cluster of their chunks) when placing them in the tree.

On the team corpus (Oct 2026): pruning at beam 4 matched flat accuracy while scoring ~1/5 of the chunks, with no
latency gain at this size (query encoding dominates). Boost 0.5 raised Qwen3-0.6B Hit@1 from 0.571 to 0.601 and helped
in every folder; for bge-small the gain was small (0.444 to 0.454). Multiple parts per file didn't help.
Each run writes the full tree outline to `tree_<model>_<chunking>_parts<n>.txt` in its results folder.
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
