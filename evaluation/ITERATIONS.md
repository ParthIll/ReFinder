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
