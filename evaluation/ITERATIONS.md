# Retrieval improvement iterations (Nov 2, 2026)

Goal: raise "right file ranked first" (Hit@1) on the team corpus without overfitting, using semantic methods only.
Harness: `python -m evaluation.lab` (one change at a time; every run is logged to `.cache/lab/lab_log.jsonl`).

## Test sets
| Set | What | Queries | Role |
|---|---|---|---|
| Team corpus | `ReFinder File Corpus/` from Drive: 280 files, 4 members' query CSVs | 660 | main target; tuned only via leave-one-member-out |
| OrgSmith | 8 synthetic companies (`python -m evaluation.orgsmith --company all`; brackenridge-civil skipped: .doc/.ppt only) | 419 | never tuned on |
| Sample | `evaluation/sample/` | 115 | never tuned on |

## Protocol (fixed before any experiment)
- A tunable value is chosen by **leave-one-member-out**: pick it on 3 members' queries, score the 4th, rotate.
  The team number reported for a change is the pooled held-out result.
- **Keep** only if: team held-out ΔHit@1 > 0 with the 90% cluster-bootstrap interval (clusters = target file)
  above 0, OrgSmith ΔHit@1 ≥ −1 point, and the sample loses at most 1 query. Otherwise revert.
- Each kept change becomes the base for the next one.

## Result
| Model | Setup | Team | OrgSmith | Sample | Search p50 |
|---|---|---|---|---|---|
| bge-small (current default) | baseline | 0.445 | 0.759 | 0.878 | 7 ms |
| bge-small | **final pipeline** | **0.582** | **0.811** | **0.939** | not measured (reranker dominates) |
| Qwen3-Embedding-0.6B | baseline | 0.570 | 0.687 | 0.896 | 18 ms |
| Qwen3-Embedding-0.6B | **final pipeline** | **0.677** | **0.807** | **0.974** | 394 ms |
| Harrier-OSS-v1-0.6B | baseline | 0.565 | 0.761 | 0.939 | 17 ms |
| **Harrier-OSS-v1-0.6B** | **final pipeline (recommended)** | **0.679** | **0.838** | **0.957** | 380 ms (p95 425) |

Final pipeline = best chunk + 1.0 × whole-file vector − outlier size penalty (0.02), then the gte reranker fused
by reciprocal rank (weight 1, top 30 chunks). Speeds measured on an M-series MacBook (MPS), Nov 2026.

```bash
python -m evaluation.custom --corpus "ReFinder File Corpus" --models microsoft/harrier-oss-v1-0.6b \
    --file-vector 1 --size-penalty 0.02 --rerank Alibaba-NLP/gte-reranker-modernbert-base --rerank-fuse 1
```

## Iterations
Base model: Qwen3 for 1–11, Harrier from 12. "Team" = held-out Hit@1 change; brackets = 90% interval.

| # | Change | Team | OrgSmith | Decision |
|---|---|---|---|---|
| 1 | Query instruction written for file finding | −0.5 [−1.7, +0.8] | +0.5 | revert |
| 2 | **Whole-file vector**: + 1.0 × cos(query, normalized mean of the file's chunk vectors) | **+5.2 [+3.0, +7.4]**, every member up | −0.2 | **keep** |
| 2b | Larger file-vector weights (1.5, 2, 3) | −0.3 | 0 | revert (plateau at 1.0) |
| 3 | Mean of top-2/3 chunks instead of best chunk | 0 (all members chose best) | 0 | revert |
| 4 | **Outlier size penalty**: files > 10× median chunk count lose 0.02 × log(excess) | **+1.2 [+0.2, +2.3]** | 0 (no outliers) | **keep** (weak: only the team corpus has outliers) |
| 5 | BM25 rank fusion | (measured with the tie bug below; keyword search then dropped by request) | | revert |
| 6 | **gte-reranker-modernbert-base, rank-fused** (after the tie fix) | **+3.3 [+0.9, +5.8]** | **+12.2** | **keep** |
| 7 | Bonus for matching the file's first chunk (title/intro) | +0.8 [−0.9, +2.3] | 0 | revert |
| 8 | Pseudo-relevance feedback (query moved toward top-3 files) | 0 (all chose off) | 0 | revert |
| 9 | Embed each chunk with its document's opening words | −0.6 [−2.3, +0.9] | top-5 −5.0 | revert |
| 10 | Ensemble with bge-small | −0.9 [−1.8, 0] | (+1.9 if forced) | revert |
| 11 | 400-token chunks | 0 (all chose 200) | 0 | revert |
| 12 | **Harrier-0.6b instead of Qwen3** (same pipeline, not retuned) | −2.3 [−4.8, +0.3], not significant | **+9.1 [+5.7, +12.3]** | **switch** (better all-round) |
| 13 | **gte reranker, rank-fused, on Harrier** | **+5.8 [+3.5, +8.0]**, all members chose w=1 | **+6.2 [+3.5, +9.2]** | **keep** |
| 13b | …only for queries that read like fact questions (semantic router) | worse | | revert |
| 14 | Bigger reranker bge-reranker-v2-m3 | 0 (all chose gte) | | revert (also 2.5× slower) |
| 15 | Rerank 15 or 60 chunks instead of 30 | −1.5 [−2.7, −0.3] | −0.7 | revert (15 ≈ same accuracy at half the latency) |
| 16 | Reranker sees each passage with its document's opening words | +1.4 [−0.2, +2.9], all members chose on | **+4.1 [+1.9, +6.4]** | revert by rule — **top candidate to confirm** |
| 17 | Reranker candidates = pipeline's top 15 files × 2 chunks | −0.8 [−1.8, +0.2] | 0 | revert |

## Findings
- **Rank-fusion tie bug (fixed).** 54 files are exact text duplicates of another file (Dagem's and Sanjay's CS 314
  slides). Duplicates tie, and a tie for first correctly counts as a hit. Rank fusion that broke ties by file order
  marked every query labeled with Sanjay's copy as a miss, which made the reranker look harmful (−5 to −8).
  `evaluation/lab.py` and `RerankRetriever(fuse=...)` in `evaluation/custom.py` now give tied scores the same rank.
- **The reranker must be fused, not trusted outright.** Replacing the ranking with the reranker's helped OrgSmith but
  not the team corpus; rank fusion helped both, at ~0.4 s per search.
- **Year/number queries remain the weakest:** 58 team queries naming a year or number ("Fall 2025 Exam 2",
  "Lecture 14") score 0.19 Hit@1 vs 0.69 for the rest. Embeddings barely separate "2023" from "2025".
- **Companion files are ambiguous.** Sanjay's stats Slides N and Worksheet N cover the same topic; queries that don't
  say "slides" or "worksheet" can't be answered with a single label. Only 51 team queries list secondary files.
- **Harrier vs Qwen3:** equal on the team corpus (difference not significant), Harrier clearly better on fact-style
  questions; same size (0.6B, 1024-d) and speed.

## Next steps
1. Put the final pipeline into the API (`evaluation/server.py`, `database.py`): Harrier, file mean vectors stored per
   file, size penalty, gte reranker fused by rank. Group exact-duplicate files in results.
2. Confirm #16 and the whole pipeline on fresh, blind queries: write queries for Elliot's and Jishnu's files
   (~60 files with none yet) without looking at results.
3. Year/number queries: decide as a team whether parsing course / exam / lecture numbers and years from a
   document's first lines into metadata (used as a filter or tie-breaker) fits the semantic-only goal.
4. Clean labels: list duplicate copies and companion files as secondary relevant files.
5. Speed: rerank 15 chunks instead of 30, or an ONNX/quantized reranker, if 0.4 s is too slow on target hardware.

# Indexing speed (Nov 11, 2026)

Goal: index a folder in 5–10 minutes on a laptop without a GPU, keeping Harrier 270M's accuracy (final pipeline:
team 0.642, OrgSmith 0.823, sample 0.957 Hit@1). Measured on an M5 Pro (CPU = 5 threads); other laptops will differ.
Profiler: `python -m evaluation.index_bench`. Accuracy check: `python -m evaluation.lab --speed` (a shortcut passes if
team and OrgSmith Hit@1 drop at most 1 point and the sample loses at most 1 query).

## Result: team corpus (280 files), CPU, whole index including extraction
| Setup | Chunks embedded | Whole index | 90% of files searchable |
|---|---|---|---|
| Harrier 270M as it loaded before (bfloat16, PyTorch) | 26,298 | ~4.5 h (1.6 chunks/s) | — |
| + float32 | 26,298 | ~24 min (18 chunks/s) | — |
| + ONNX Runtime | 26,298 | ~15 min (31 chunks/s) | — |
| **+ files capped at 100 chunks, smallest first (`refinder/indexer.py`)** | **5,709** | **3.9 min** | **2.7 min** |
| bge-small, same indexer | 5,617 | 2.0 min | 1.1 min |

The last two rows are real end-to-end runs of `index_files` (offline); the others are extrapolated from 2,000 chunks.
On the Mac GPU the full 26,298 chunks take about 3.5 min with Harrier 270M and 9–10 min with Qwen3-0.6B.

## Where the time goes
- Embedding dominates. Uncached extraction takes 1.2 min (PDFs 55 s, one 19 s; one spreadsheet 15 s), chunking 2 s.
  With the cap, extraction is now about a quarter of Harrier's time and half of bge-small's.
- 5 files (textbook PDFs, a spreadsheet) hold 72% of the team corpus's chunks; the median file has 12.

## Experiments
| Change | Chunks | Accuracy (team / OrgSmith / sample Hit@1) | Decision |
|---|---|---|---|
| **Cap files at 200 / 100 / 50 evenly spaced chunks** | 24% / 22% / 19% | +0.5 / +0.6 / +1.4 pts team (within noise); OrgSmith and sample unchanged (no file that big) | **keep 100** |
| Chunk overlap 25 instead of 50 | 92% | 0.0 / −1.2 / +2 queries | revert |
| Chunk overlap 0 | 94% | −1.1 / −1.0 / +3 queries | revert |
| ONNX Runtime on CPU (float32) | — | identical vectors (cosine 1.0000) | **keep**: Harrier 270M 1.7x, bge-small and MiniLM ~10% |
| ONNX int8 | — | vectors 0.99 cosine, OrgSmith −2 / sample −3 queries (Harrier 270M, embeddings only) | revert: no faster on Apple silicon; `index_bench --int8` to retest on Intel/AMD |
| Batch size 16–32 instead of 64–128 | — | identical | **keep** (`MAX_BATCH = 32`): 5–15% faster on CPU and GPU |
| Static model `minishlab/potion-retrieval-32M` | — | final pipeline 0.544 / 0.759 / 0.930 (bge-small 0.582 / 0.811 / 0.939) | not adopted: ~26,000 chunks/s on CPU, so a possible "instant first pass" later |
| Pruning / distilling Harrier | — | not tried | needs retraining; the cap gave more for far less |

### Fewer chunks in the app (Nov 12, `api_eval`, app's real search, Harrier 270M)
Andrew's 14,223-file Downloads index, real chunker: 84,725 chunks at the 100 cap, of which 32% repeat another
chunk's text exactly (copied files, shared boilerplate); 233 files at the cap hold 31%.

| Change | Chunks embedded (14k folder) | Accuracy (team / OrgSmith / sample Hit@1) | Decision |
|---|---|---|---|
| Embed each repeated chunk text once (`ChromaIndex.embed`), all models | 84,725 → 57,714 | identical for Harrier 270M and bge-small; stop + resume gives identical vectors | **keep** |
| Cap 50 instead of 100 (w_kw 0.25) | 57,714 → 44,795 | 0.633 → 0.636 / 0.621 = / 0.861 = | **keep** for Harrier 270M |
| Spreadsheets (csv/tsv/xlsx) capped at 10 | 44,795 → 38,539 | within 1 pt (w_kw 1) | not adopted: test-set spreadsheets are tiny (removed 95 chunks), so untested on big CSVs |

Total: 84,725 → 44,795 chunks (−47%), roughly 11 → 6 min of embedding on the M5 Pro GPU (~130 chunks/s) and
47 → 25 min on its CPU (~30 chunks/s); estimates from chunk counts, since the timed runs shared the GPU with other
jobs. The repeated-text check costs ~4 ms per 85,000 chunks, so folders without copies lose nothing.
`api_eval --max-chunks N` overrides the cap.

## Notes
- The cap's cost shows up in the 24 team queries aimed at a file over 100 chunks: Hit@1 0.917 → 0.833 (2 queries).
  A query for a detail deep inside a textbook can miss; the file-level queries we have mostly don't.
- A capped file keeps its full chunk count (`IndexedFile.n_chunks`) for the size penalty.
- Parth's ~90 min with Qwen on a GPU isn't explained by this Mac (about 10 min). Running
  `python -m evaluation.index_bench --models Qwen/Qwen3-Embedding-0.6B --limit-chunks 2000` on his machine will show
  which stage is slow.

# Models inside the app's search (Nov 12, 2026)

`python -m evaluation.api_eval` runs each model in `refinder/models.py` through `evaluation/database.py` exactly as
the app does: BM25 + filename/folder bonus + exact-phrase bonus + meaning ranking, fused by rank, with each model's
own chunk budget, and the whole-file vector + gte reranker for the Harrier models. Exact-duplicate files count as
hits, as in `evaluation.custom` (the app ranks two identical copies one after the other). Mac GPU (MPS).

| Model (chunks per file) | Team Hit@1 / @5 | OrgSmith | Sample | Team embedding | Search p50 |
|---|---|---|---|---|---|
| bge-small (20, head) | 0.552 / 0.776 | 0.609 / 0.926 | 0.878 / 0.974 | 7 s | 17 ms |
| MiniLM (20, head) | 0.555 / 0.776 | 0.630 / 0.936 | 0.887 / 0.974 | 17 s | 25 ms |
| **Harrier 270M (100, even) + file vector + reranker** | **0.600 / 0.788** | 0.623 / 0.928 | 0.870 / 0.974 | 54 s | 406 ms |
| Harrier 0.6B (100, even) + file vector + reranker | 0.614 / 0.808 | 0.625 / 0.931 | 0.887 / 0.974 | 117 s | 423 ms |

## What the app's fusion does to each model (same indexes, only the weights changed)
| Change to the fusion | bge-small team / OrgSmith / sample | Harrier 270M | Harrier 0.6B |
|---|---|---|---|
| as shipped (name 0.05, phrase 0.03, BM25 1, meaning 1) | 0.552 / 0.609 / 0.878 | 0.600 / 0.623 / 0.870 | 0.614 / 0.625 / 0.887 |
| no filename/folder bonus | 0.509 / 0.783 / 0.852 | 0.577 / 0.816 / 0.843 | 0.586 / 0.833 / 0.861 |
| meaning ranking only | 0.491 / 0.754 / 0.878 | 0.614 / 0.785 / 0.904 | 0.664 / 0.816 / 0.939 |
| BM25 weight 0.25 | 0.567 / 0.613 / 0.887 | 0.635 / 0.621 / 0.861 | 0.645 / 0.616 / 0.896 |
| BM25 only | 0.405 / 0.776 / 0.783 | | |

- The filename/folder bonus costs OrgSmith 16–21 points for every model (its questions share words with folder
  names, so whole folders get the same boost) and gains the team corpus 2–4 points. Parth tuned it on filename
  and exact-phrase queries, which these test sets don't have, so it may still be right for real users: needs a
  team decision, not changed here.
- Harrier's own meaning ranking is the best single signal on the team corpus and sample; mixed with BM25 at equal
  weight it loses 1–5 points there. A per-model fusion weight (in `ModelSpec`) would let Harrier keep it.
- The app's Harrier meaning ranking (0.614 / 0.785 / 0.904) is below the benchmark's (0.648 / 0.823 / 0.957):
  the app scores only files near the top 400 chunks and 100 file vectors, while the benchmark scores every file.
- Search: the Harrier reranker adds ~0.4 s per query on the Mac GPU; CPU not measured yet.

## Per-model BM25 weight (Nov 12, 2026)
`ModelSpec.w_kw` sets BM25's weight against the meaning ranking (weight 1); filename and phrase bonuses unchanged.
`python -m evaluation.api_eval --models harrier-270m --w-kw 1 0.5 0.25 0.1 0` (one index, five fusions, Mac GPU):

| Harrier 270M, w_kw | Team Hit@1 / @5 | OrgSmith | Sample |
|---|---|---|---|
| 1 (old) | 0.598 / 0.789 | 0.623 / 0.928 | 0.870 / 0.974 |
| 0.5 | 0.611 / 0.806 | 0.623 / 0.931 | 0.861 / 0.974 |
| **0.25 (shipped)** | **0.633 / 0.820** | 0.621 / 0.926 | 0.861 / 0.983 |
| 0.1 | 0.618 / 0.842 | 0.618 / 0.924 | 0.861 / 1.000 |
| 0 (meaning only + bonuses) | 0.624 / 0.841 | 0.616 / 0.919 | 0.861 / 1.000 |

Rule: best team Hit@1 losing at most 1 point on OrgSmith and 1 sample query. 0.25 passes (+23 team queries).
Keywords still help a little: 0 is below 0.25 on every set. Only Harrier 270M was swept; Harrier 0.6B, bge-small
and MiniLM keep 1. (w_kw 1 gives 0.598 here vs 0.600 above: run-to-run GPU noise.) Idea from scry
(github.com/xxx12e/scry), which weights meaning 0.7 / keywords 0.3.

# Fast models with the search extras (Nov 12, 2026)

Question: can a small model index much faster than Harrier 270M and stay as accurate? `api_eval --extras` gives any
model Harrier's search extras (whole-file vector + size penalty + gte reranker fused by rank); every model was swept
over BM25 weights 1 / 0.5 / 0.25 / 0.1 / 0, through the app's real search, 20 chunks per file (first half + spread)
unless noted. Rule: team Hit@1 >= 0.606 (at most 3 points under Harrier, agreed with Andrew), OrgSmith and sample at
most 1 point under Harrier, indexing at least 3x faster.

| Model (params) + extras, best weight | Team | OrgSmith | Sample | GPU chunks/s | CPU chunks/s | Result |
|---|---|---|---|---|---|---|
| Harrier 270M, 50/file, 0.25 (before) | 0.636 | 0.621 | 0.861 | 90 | 22 (ONNX) | baseline |
| **MongoDB/mdbr-leaf-ir (23M), 0.25** | **0.653** | **0.623** | **0.870** | **750** | **187** | **pass: better on all three, ~8x faster** |
| Snowflake/snowflake-arctic-embed-s (33M), 0.5 | 0.627 | 0.632 | 0.878 | 434 | 95 | pass |
| ibm-granite/granite-embedding-small-english-r2 (47M), 0.5 | 0.617 | 0.609 | 0.870 | — | — | borderline (OrgSmith −1.2) |
| bge-small (33M), 0.1 | 0.594 | 0.601 | 0.887 | — | — | fail |
| MiniLM (22M), 0.25 | 0.598 | 0.613 | 0.878 | — | — | fail (team) |
| potion-retrieval-32M (static), 1 | 0.555 | 0.625 | 0.896 | — | — | fail (team) |

- mdbr-leaf-ir passes at every BM25 weight (team 0.642–0.655), so the choice of 0.25 isn't what makes it win.
- Speeds: `index_bench --limit-chunks 2000`, M5 Pro, nothing else running. mdbr-leaf-ir has a Dense projection, so
  it runs on PyTorch on CPU (the ONNX path only reproduces pooling); 187 chunks/s is without ONNX.
- On Andrew's 14k-file Downloads folder: ~31,600 chunks to embed (20/file, repeated text once) vs 44,795 for
  Harrier, so about 40 s vs 8 min on the GPU and 3 min vs 35 min on the CPU (estimates from chunk counts).
- Search through `/search` (sample corpus, GPU, reranker and sentence snippets included): 0.36–0.49 s.
- `MongoDB/mdbr-leaf-ir-asym` was not tested: it encodes documents with the 109M teacher, so indexing gets slower.
- Added to the registry as `leaf-ir` ("Fast + accurate"). The app's default model is unchanged.
