"""Retrieval experiment lab: try one change at a time with held-out checks, keep it or revert it.

Pipeline = a dict of params. Features (best chunk, top-k chunk means, file mean vector, first chunk, size, BM25,
cross-encoder) are computed once per (dataset, model, query instruction, index) and combined by score().
Data: the team corpus ("ReFinder File Corpus/" at the repo root), every OrgSmith company in datasets/ (run
`python -m evaluation.orgsmith --company all` first), and evaluation/sample/.

Protocol (fixed before running any fix):
  - Tunable values are chosen by leave-one-member-out (LOMO) on the team corpus: pick on 3 members, score the 4th.
  - KEEP if team LOMO dHit@1 > 0 with the 90% cluster-bootstrap CI (clusters = target file) above 0,
    and OrgSmith dHit@1 >= -1.0 pt, and sample loses at most 1 query. OrgSmith and sample are never tuned on.
  - A kept value is saved to the state file and becomes the base for the next experiment.

    python -m evaluation.lab --model harrier                      # evaluate the current state
    python -m evaluation.lab --model harrier --key w_mean --grid 0 0.25 0.5 1 --label "file mean vector"
    python -m evaluation.lab --model harrier --set w_ce=0 --dry   # one-off variation, not saved
    python -m evaluation.lab --model harrier270 --set w_mean=1 w_size=0.02 w_ce=1 ce=gte \
        --speed --key index --grid cap100 ov0                     # indexing shortcuts: how much accuracy they cost

Nov 2026 results and the reasoning behind each kept/reverted change: evaluation/ITERATIONS.md.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import custom
from evaluation.common import CACHE_DIR, REPO_ROOT as REPO, chunk_corpus, embed_corpus
from refinder.embedder import Embedder, pick_device

CACHE = CACHE_DIR.parent / "lab"
CACHE.mkdir(parents=True, exist_ok=True)
STATE = CACHE / "lab_state.json"
LOG = CACHE / "lab_log.jsonl"

MODELS = {"qwen3": "Qwen/Qwen3-Embedding-0.6B", "bge": "BAAI/bge-small-en-v1.5",
          "harrier": "microsoft/harrier-oss-v1-0.6b", "harrier270": "microsoft/harrier-oss-v1-270m"}
INSTR = {
    "default": None,
    "file": "Instruct: Given a description of a document someone wants to find on their computer, "
            "retrieve passages from that document\nQuery:",
}
CE_MODELS = {"minilm": "cross-encoder/ms-marco-MiniLM-L6-v2", "gte": "Alibaba-NLP/gte-reranker-modernbert-base",
             "bgerr": "BAAI/bge-reranker-v2-m3"}
BASE = {"instr": "default", "agg": "max", "w_mean": 0.0, "w_first": 0.0, "w_size": 0.0, "w_bm25": 0.0,
        "w_ce": 0.0, "ce": "minilm", "index": "plain", "prf": 0.0, "ens": 0.0, "ens_model": "bge", "route": "none",
        "ce_depth": 30, "ce_hdr": 0, "ce_cand": "chunks"}
RRF_K = 60
CE_DEPTH = 30


# ---------------------------------------------------------------- data

def load_datasets() -> list[dict]:
    specs = []
    root = REPO / "ReFinder File Corpus"
    csvs = custom.find_query_csvs(root)
    specs.append(("team", root, csvs, set()))
    for q in sorted((REPO / "datasets").glob("orgsmith_*_queries.csv")):
        co = q.name[len("orgsmith_"):-len("_queries.csv")]
        if co in ("dev-mini", "brackenridge-civil"):  # tiny test company; all .doc/.ppt (unsupported)
            continue
        r = REPO / "datasets/OrgSmith/companies" / co
        specs.append((f"org/{co}", r, [q], {r / "TOC.md", r / "PERMISSIONS.md"}))
    specs.append(("sample", REPO / "evaluation/sample/Corpus", [REPO / "evaluation/sample/queries.csv"], set()))
    out = []
    for name, root, csvs, excl in specs:
        paths, texts, pieces, _ = custom.load_corpus(root, exclude=set(excl) | custom.query_file_exclusions(csvs))
        queries, _ = custom.load_queries(csvs, root, paths)
        group = [paths[min(q["gold"])].split("/")[0] if name == "team" else name.split("/")[0] for q in queries]
        cluster = [f"{name}:{paths[min(q['gold'])]}" for q in queries]
        out.append(dict(name=name, root=root, paths=paths, texts=texts, pieces=pieces, queries=queries,
                        group=np.array(group), cluster=np.array(cluster)))
    return out


# ---------------------------------------------------------------- features

_EMB: dict[str, Embedder] = {}
_FEAT: dict[tuple, dict] = {}


def embedder(model: str) -> Embedder:
    if model not in _EMB:
        _EMB[model] = Embedder(MODELS[model])
    return _EMB[model]


def _key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode() + b"\x1f")
    return h.hexdigest()[:16]


HEADER_WORDS = 20


def _header(pieces: list[str]) -> str:
    """The document's own opening words (title line), e.g. 'CS 314 - Exam 2 - Fall 2025', never the filename."""
    first = next((p for p in pieces if p.strip()), "")
    return " ".join(first.split()[:HEADER_WORDS]) + "\n"


def cap_keep(owner: np.ndarray, cap: int) -> np.ndarray:
    """Chunk ids to keep when a file may have at most `cap` chunks: all of a small file's, evenly spaced ones of a
    big file's (first and last included). Chunks are contiguous per file."""
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]) if len(owner) else np.array([], int)
    keep = []
    for s, e in zip(starts, np.r_[starts[1:], len(owner)]):
        keep.append(np.arange(s, e) if e - s <= cap else s + np.unique(np.linspace(0, e - s - 1, cap).round().astype(int)))
    return np.concatenate(keep) if keep else np.array([], int)


def chunk_index(ds: dict, model: str, index: str = "plain"):
    """plain: the app's chunks. hdr: each chunk prefixed with its document's opening words. c400: 400-token chunks.
    capN: plain, but a file keeps at most N evenly spaced chunks. ovN: plain chunking with N tokens of overlap."""
    if index.startswith("cap"):
        vecs, owner = chunk_index(ds, model, "plain")
        keep = cap_keep(owner, int(index[3:]))
        return vecs[keep], owner[keep]
    if index == "plain":
        vecs, owner, _, _ = embed_corpus(embedder(model), ds["texts"], use_cache=True,
                                         pieces=[[p.text for p in ps] for ps in ds["pieces"]], chunking="paragraph")
        return vecs, owner
    f = CACHE / f"idx_{_key(ds['name'], model, index, HEADER_WORDS, *[len(t) for t in ds['texts']])}.npz"
    if f.exists():
        z = np.load(f)
        return z["vecs"], z["owner"]
    e = embedder(model)
    piece_texts = [[p.text for p in ps] for ps in ds["pieces"]]
    if index == "hdr":
        texts, owner, _ = chunk_corpus(e, ds["texts"], piece_texts, "paragraph")
        heads = [_header(ps) for ps in piece_texts]
        texts = [heads[o] + t for t, o in zip(texts, owner)]
    elif index == "c400" or index.startswith("ov"):
        size, overlap = (400, 80) if index == "c400" else (200, int(index[2:]))
        texts, owner = [], []
        for i, ps in enumerate(piece_texts):
            for c in e.chunk_pieces(ps, size, overlap):
                texts.append(c.text)
                owner.append(i)
        owner = np.array(owner, dtype=int)
    else:
        raise ValueError(index)
    print(f"  [{ds['name']}] embedding {len(texts):,} {index} chunks with {model}...", flush=True)
    vecs = e.encode(texts, batch_size=64)
    np.savez(f, vecs=vecs, owner=owner)
    return vecs, owner


_TEXTS: dict[tuple, list[str]] = {}


def index_texts(ds: dict, model: str, index: str) -> list[str]:
    """Chunk texts aligned with chunk_index(ds, model, index) (plain, capN and ovN), for the reranker."""
    key = (ds["name"], model, index)
    if key not in _TEXTS:
        piece_texts = [[p.text for p in ps] for ps in ds["pieces"]]
        if index.startswith("ov"):
            _TEXTS[key] = [c.text for ps in piece_texts for c in embedder(model).chunk_pieces(ps, 200, int(index[2:]))]
        else:
            texts, _, _ = chunk_corpus(embedder(model), ds["texts"], piece_texts, "paragraph")
            if index.startswith("cap"):
                texts = [texts[i] for i in cap_keep(chunk_index(ds, model, "plain")[1], int(index[3:]))]
            _TEXTS[key] = texts
    return _TEXTS[key]


def query_vecs(ds: dict, model: str, instr: str) -> np.ndarray:
    qs = [q["query"] for q in ds["queries"]]
    f = CACHE / f"q_{_key(ds['name'], model, instr, *qs)}.npy"
    if f.exists():
        return np.load(f)
    e = embedder(model)
    if INSTR[instr] is None:
        v = e.encode_queries(qs)
    else:
        v = e.model.encode(qs, prompt=INSTR[instr], normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=64).astype(np.float32)
    np.save(f, v)
    return v


def query_features(q: np.ndarray, I: dict) -> dict:
    """Per-file scores for query vectors q against index I (chunk vectors + file mean vectors)."""
    n_f, n_q = I["n_files"], len(q)
    C = q @ I["vecs"].T
    F = {k: np.full((n_q, n_f), -np.inf, np.float32) for k in ("max", "top2", "top3", "mean", "first")}
    for f, s, e in zip(I["doc_ids"], I["starts"], I["ends"]):
        srt = -np.sort(-C[:, s:e], axis=1)
        F["max"][:, f] = srt[:, 0]
        F["top2"][:, f] = srt[:, :2].mean(1)
        F["top3"][:, f] = srt[:, :3].mean(1)
        F["first"][:, f] = C[:, s]
    F["mean"][:, I["has"]] = q @ I["mean_vecs"][I["has"]].T
    F["C"] = C
    return F


_IDX: dict[tuple, dict] = {}


def index_data(ds: dict, model: str, index: str) -> dict:
    key = (ds["name"], model, index)
    if key not in _IDX:
        vecs, owner = chunk_index(ds, model, index)
        n_f = len(ds["texts"])
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]) if len(owner) else np.array([], int)
        ends = np.r_[starts[1:], len(owner)]
        doc_ids = owner[starts]
        mean_vecs = np.zeros((n_f, vecs.shape[1]), np.float32)
        has = np.zeros(n_f, bool)
        for f, s, e in zip(doc_ids, starts, ends):
            mv = vecs[s:e].mean(0)
            mean_vecs[f] = mv / max(float(np.linalg.norm(mv)), 1e-12)
            has[f] = True
        # A capped file still counts all its chunks for the size penalty, as the app would store the real count.
        counts = chunk_index(ds, model, "plain")[1] if index.startswith("cap") else owner
        n_chunks = np.bincount(counts, minlength=n_f).clip(min=1)
        size = np.maximum(np.log(n_chunks / (10 * np.median(n_chunks))), 0).astype(np.float32)
        _IDX[key] = dict(vecs=vecs, owner=owner, starts=starts, ends=ends, doc_ids=doc_ids, mean_vecs=mean_vecs,
                         has=has, size=size, n_files=n_f)
    return _IDX[key]


def features(ds: dict, model: str, instr: str, index: str = "plain") -> dict:
    key = (ds["name"], model, instr, index)
    if key not in _FEAT:
        I = index_data(ds, model, index)
        q = query_vecs(ds, model, instr)
        F = query_features(q, I)
        F.update(q=q, size=I["size"], owner=I["owner"], I=I)
        _FEAT[key] = F
    return _FEAT[key]


def bm25_scores(ds: dict) -> np.ndarray:
    f = CACHE / f"bm25_{_key(ds['name'], *[q['query'] for q in ds['queries']], len(ds['texts']))}.npy"
    if f.exists():
        return np.load(f)
    s = custom.BM25Retriever(ds["texts"]).all_scores([q["query"] for q in ds["queries"]])
    np.save(f, s)
    return s


_CE = {}


def ce_scores_files(ds: dict, model: str, p: dict, s_first: np.ndarray) -> np.ndarray:
    """Cross-encoder over the pipeline's top CE_DEPTH//2 files, each with its 2 best chunks (same pair budget)."""
    F = features(ds, model, p["instr"], "plain")
    n_files_c = CE_DEPTH // 2
    f = CACHE / f"cef_{_key(ds['name'], model, p['instr'], p['ce'], CE_DEPTH, p['w_mean'], p['w_size'], p['agg'], *[q['query'] for q in ds['queries']])}.npy"
    if f.exists():
        return np.load(f)
    from sentence_transformers import CrossEncoder
    ce = p["ce"]
    if ce not in _CE:
        _CE[ce] = CrossEncoder(CE_MODELS[ce], device=pick_device())
    texts, _, _ = chunk_corpus(embedder(model), ds["texts"], [[x.text for x in ps] for ps in ds["pieces"]], "paragraph")
    I, C = F["I"], F["C"]
    span = {int(fi): (int(a), int(b)) for fi, a, b in zip(I["doc_ids"], I["starts"], I["ends"])}
    out = np.full(s_first.shape, -np.inf, np.float32)
    for i, q in enumerate(ds["queries"]):
        k = min(n_files_c, int(np.isfinite(s_first[i]).sum()))
        files = np.argpartition(-s_first[i], k - 1)[:k]
        chunks = [a + j for fi in files for (a, b) in [span[int(fi)]] for j in np.argsort(-C[i, a:b])[:2]]
        sc = np.asarray(_CE[ce].predict([(q["query"], texts[c]) for c in chunks], show_progress_bar=False), float)
        if sc.min() < 0 or sc.max() > 1:
            sc = 1 / (1 + np.exp(-sc))
        for c, v in zip(chunks, sc):
            o = I["owner"][c]
            out[i, o] = max(out[i, o], v)
    np.save(f, out)
    return out


CE_MAX_DEPTH = 60


def ce_scores_depth(ds: dict, model: str, instr: str, ce: str, depth: int) -> np.ndarray:
    """Like ce_scores, but for any depth <= CE_MAX_DEPTH from one cached scoring of the top CE_MAX_DEPTH chunks."""
    F = features(ds, model, instr, "plain")
    f = CACHE / f"ced_{_key(ds['name'], model, instr, ce, CE_MAX_DEPTH, *[q['query'] for q in ds['queries']])}.npz"
    if f.exists():
        z = np.load(f)
        top, sc = z["top"], z["sc"]
    else:
        from sentence_transformers import CrossEncoder
        if ce not in _CE:
            _CE[ce] = CrossEncoder(CE_MODELS[ce], device=pick_device())
        texts, _, _ = chunk_corpus(embedder(model), ds["texts"], [[p.text for p in ps] for ps in ds["pieces"]],
                                   "paragraph")
        C = F["C"]
        d = min(CE_MAX_DEPTH, C.shape[1])
        top = np.argpartition(-C, d - 1, axis=1)[:, :d]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(C, top, 1), axis=1), 1)  # best chunk first
        sc = np.zeros(top.shape, np.float32)
        for i, q in enumerate(ds["queries"]):
            s = np.asarray(_CE[ce].predict([(q["query"], texts[c]) for c in top[i]], show_progress_bar=False), float)
            sc[i] = 1 / (1 + np.exp(-s)) if (s.min() < 0 or s.max() > 1) else s
        np.savez(f, top=top, sc=sc)
    out = np.full((len(top), len(ds["texts"])), -np.inf, np.float32)
    owner = F["owner"]
    for i in range(len(top)):
        for c, v in zip(top[i, :depth], sc[i, :depth]):
            out[i, owner[c]] = max(out[i, owner[c]], v)
    return out


def ce_scores(ds: dict, model: str, instr: str, ce: str, hdr: bool = False, index: str = "plain") -> np.ndarray:
    """Best cross-encoder probability per file among the CE_DEPTH best chunks (by raw chunk similarity); -inf else.

    hdr: the reranker sees each passage prefixed with its document's own opening words (never the filename).
    index: whose chunks are candidates; capN / ovN indexes rerank their own chunks, everything else plain's."""
    F = features(ds, model, instr, index)
    tag = (ce, "hdr", HEADER_WORDS) if hdr else (ce,)
    if index != "plain":
        tag += (index,)
    f = CACHE / f"ce_{_key(ds['name'], model, instr, *tag, CE_DEPTH, *[q['query'] for q in ds['queries']])}.npy"
    if f.exists():
        return np.load(f)
    from sentence_transformers import CrossEncoder
    if ce not in _CE:
        _CE[ce] = CrossEncoder(CE_MODELS[ce], device=pick_device())
    texts = index_texts(ds, model, index)
    C, owner = F["C"], F["owner"]
    if hdr:
        heads = [_header([p.text for p in ps]) for ps in ds["pieces"]]
        texts = [heads[o] + t for t, o in zip(texts, owner)]
    out = np.full(C.shape[0:1] + (len(ds["texts"]),), -np.inf, np.float32)
    for i, q in enumerate(ds["queries"]):
        depth = min(CE_DEPTH, C.shape[1])
        top = np.argpartition(-C[i], depth - 1)[:depth]
        s = np.asarray(_CE[ce].predict([(q["query"], texts[c]) for c in top], show_progress_bar=False), float)
        if s.min() < 0 or s.max() > 1:
            s = 1 / (1 + np.exp(-s))
        for c, v in zip(top, s):
            out[i, owner[c]] = max(out[i, owner[c]], v)
    np.save(f, out)
    return out


# ---------------------------------------------------------------- scoring

def ranks_matrix(s: np.ndarray) -> np.ndarray:
    """1-based rank of every file per query; tied scores share the best rank (exact duplicate files tie, so fusion
    must not break the tie by file order, or the labeled copy of a duplicate loses to the other copy)."""
    r = np.empty(s.shape, dtype=np.int64)
    for i, row in enumerate(s):
        srt = np.sort(-row)
        r[i] = np.searchsorted(srt, -row, side="left") + 1
    return r


def _combine(F: dict, p: dict) -> np.ndarray:
    s = F[p["agg"]].copy()
    if p["w_mean"]:
        s += p["w_mean"] * F["mean"]
    if p["w_first"]:
        s += p["w_first"] * F["first"]
    if p["w_size"]:
        s -= p["w_size"] * F["size"][None, :]
    return s


PRF_K = 3


def score(ds: dict, model: str, p: dict) -> np.ndarray:
    F = features(ds, model, p["instr"], p["index"])
    s = _combine(F, p)
    if p["prf"]:  # pseudo-relevance feedback: move the query toward its top files' meaning, then search again
        I = F["I"]
        k = min(PRF_K, int(np.isfinite(s).sum(1).min()))
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        q2 = F["q"] + p["prf"] * I["mean_vecs"][top].mean(1)
        q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
        F2 = query_features(q2.astype(np.float32), I)
        F2["size"] = F["size"]
        s = _combine(F2, p)
    if p["ens"]:  # second embedding model, same pipeline, its own default index and query prompt
        s = s + p["ens"] * _combine(features(ds, p["ens_model"], "default", "plain"), p)
    if p["w_bm25"] or p["w_ce"]:  # rank fusion: scale-free, so weights mean the same on every corpus
        fused = 1.0 / (RRF_K + ranks_matrix(s))
        if p["w_bm25"]:
            fused += p["w_bm25"] / (RRF_K + ranks_matrix(bm25_scores(ds)))
        if p["w_ce"]:
            if p["ce_cand"] == "files":
                ce = ce_scores_files(ds, model, p, s)
            elif p["ce_depth"] == CE_DEPTH:
                ce_index = p["index"] if p["index"].startswith(("cap", "ov")) else "plain"
                ce = ce_scores(ds, model, p["instr"], p["ce"], bool(p["ce_hdr"]), ce_index)
            else:
                ce = ce_scores_depth(ds, model, p["instr"], p["ce"], int(p["ce_depth"]))
            w = np.full(len(ce), p["w_ce"])
            if p["route"] == "fact":  # rerank only queries that read like fact questions
                w = w * route_fact(ds, model, p["instr"])
            fused += np.where(np.isfinite(ce), w[:, None] / (RRF_K + ranks_matrix(ce)), 0.0)
        s = np.where(np.isfinite(s), fused, -np.inf)
    return s


ROUTE_PROTOS = ("Which document states a specific fact, such as a date, an amount, a name, or a decision?",
                "Find the file that is about a topic, such as notes, slides, an essay, or an assignment.")
_PROTO: dict[str, np.ndarray] = {}


def route_fact(ds: dict, model: str, instr: str) -> np.ndarray:
    """True where the query is closer (in the model's own query space) to a fact question than to a topic search."""
    if model not in _PROTO:
        _PROTO[model] = embedder(model).encode_queries(list(ROUTE_PROTOS))
    s = query_vecs(ds, model, instr) @ _PROTO[model].T
    return s[:, 0] > s[:, 1]


def ranks(ds: dict, model: str, p: dict) -> np.ndarray:
    return custom.ranks_of_gold(score(ds, model, p), ds["queries"])


def metrics(r: np.ndarray) -> dict:
    return {"Hit@1": float(np.mean(r <= 1)), "Hit@5": float(np.mean(r <= 5)),
            "MRR": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))), "n": int(len(r))}


def boot_ci(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n: int = 4000, seed: int = 0):
    """Paired cluster bootstrap of Hit@1(a) - Hit@1(b); returns (diff, 5th pct, 95th pct)."""
    d = (a <= 1).astype(float) - (b <= 1).astype(float)
    _, inv = np.unique(clusters, return_inverse=True)
    sums, cnt = np.bincount(inv, d), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(sums), (n, len(sums)))
    stat = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(d.mean()), float(np.percentile(stat, 5)), float(np.percentile(stat, 95))


def pooled(dsets, model, p, prefix):
    rs = [ranks(ds, model, p) for ds in dsets if ds["name"].startswith(prefix)]
    cl = [ds["cluster"] for ds in dsets if ds["name"].startswith(prefix)]
    return np.concatenate(rs), np.concatenate(cl)


# ---------------------------------------------------------------- experiment

def parse(v: str):
    try:
        return float(v)
    except ValueError:
        return v


SPEED_MARGIN = 0.01  # an indexing shortcut may cost at most 1 Hit@1 point on team and OrgSmith, 1 sample query


def speed_check(dsets, team, args, cur):
    """Non-inferiority check for changes that make indexing cheaper (no tuning, so no LOMO needed)."""
    base = {name: pooled(dsets, args.model, cur, prefix) for name, prefix in (("team", "team"), ("orgsmith", "org/"),
                                                                              ("sample", "sample"))}
    n_plain = {name: sum(len(index_data(d, args.model, cur["index"])["owner"]) for d in dsets
                         if d["name"].startswith(prefix)) for name, prefix in (("team", "team"), ("orgsmith", "org/"))}
    print(f"\n== speed check: {args.key} in {args.grid}   (base: {cur[args.key]}, "
          f"team {n_plain['team']:,} chunks, orgsmith {n_plain['orgsmith']:,})")
    for name, (r, _) in base.items():
        m = metrics(r)
        print(f"  base {name:<9} Hit@1 {m['Hit@1']:.3f}  Hit@5 {m['Hit@5']:.3f}  MRR {m['MRR']:.3f}")
    for g in args.grid:
        new = {**cur, args.key: g}
        out = {name: pooled(dsets, args.model, new, prefix)[0] for name, prefix in (("team", "team"),
                                                                                  ("orgsmith", "org/"), ("sample", "sample"))}
        n_new = {name: sum(len(index_data(d, args.model, new["index"])["owner"]) for d in dsets
                           if d["name"].startswith(prefix)) for name, prefix in (("team", "team"), ("orgsmith", "org/"))}
        d_team = boot_ci(out["team"], base["team"][0], base["team"][1])
        d_org = boot_ci(out["orgsmith"], base["orgsmith"][0], base["orgsmith"][1])
        d_q = int((out["sample"] <= 1).sum() - (base["sample"][0] <= 1).sum())
        ok = d_team[0] >= -SPEED_MARGIN and d_org[0] >= -SPEED_MARGIN and d_q >= -1
        print(f"\n  {g}: team {n_new['team']:,} chunks ({n_new['team'] / n_plain['team'] * 100:.0f}%), "
              f"orgsmith {n_new['orgsmith']:,} ({n_new['orgsmith'] / n_plain['orgsmith'] * 100:.0f}%)")
        for name in out:
            ma, mb = metrics(base[name][0]), metrics(out[name])
            print(f"    {name:<9} Hit@1 {ma['Hit@1']:.3f} -> {mb['Hit@1']:.3f}   Hit@5 {ma['Hit@5']:.3f} -> {mb['Hit@5']:.3f}"
                  f"   MRR {ma['MRR']:.3f} -> {mb['MRR']:.3f}")
        print(f"    team dHit@1 {d_team[0]*100:+.1f} pts [{d_team[1]*100:+.1f}, {d_team[2]*100:+.1f}]   orgsmith "
              f"{d_org[0]*100:+.1f} [{d_org[1]*100:+.1f}, {d_org[2]*100:+.1f}]   sample {d_q:+d} queries")
        if args.key == "index" and str(g).startswith("cap"):
            counts = np.bincount(chunk_index(team, args.model, "plain")[1], minlength=len(team["texts"]))
            hit = np.array([any(counts[f] > int(g[3:]) for f in q["gold"]) for q in team["queries"]])
            if hit.any():
                print(f"    team queries aimed at a capped file: {hit.sum()}   Hit@1 {np.mean(base['team'][0][hit] <= 1):.3f}"
                      f" -> {np.mean(out['team'][hit] <= 1):.3f}   Hit@5 {np.mean(base['team'][0][hit] <= 5):.3f}"
                      f" -> {np.mean(out['team'][hit] <= 5):.3f}")
        print(f"    {'PASS' if ok else 'FAIL'} (allowed: -{SPEED_MARGIN * 100:.0f} pt team and OrgSmith, -1 sample query)")
        with LOG.open("a") as f:
            f.write(json.dumps({"model": args.model, "label": args.label, "speed": True, "key": args.key, "value": g,
                                "base": cur[args.key], "team": [metrics(base["team"][0]), metrics(out["team"])],
                                "team_ci": d_team, "org_ci": d_org, "sample_dq": d_q, "chunks": n_new, "pass": ok}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", choices=MODELS)
    ap.add_argument("--key")
    ap.add_argument("--grid", nargs="+", type=parse)
    ap.add_argument("--label", default="")
    ap.add_argument("--set", nargs="*", default=[], help="extra fixed params k=v for this run (not saved)")
    ap.add_argument("--dry", action="store_true", help="don't update the saved state")
    ap.add_argument("--speed", action="store_true",
                    help="grid values are indexing shortcuts: PASS if accuracy stays within SPEED_MARGIN, nothing saved")
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    cur = {**BASE, **state.get(args.model, {})}
    for kv in args.set:
        k, v = kv.split("=")
        cur[k] = parse(v)
    dsets = load_datasets()
    team = next(d for d in dsets if d["name"] == "team")
    print(f"\n### {args.model}  current: { {k: v for k, v in cur.items() if v != BASE[k]} or 'baseline' }")

    if not args.key:
        rt = ranks(team, args.model, cur)
        ro, _ = pooled(dsets, args.model, cur, "org/")
        rs, _ = pooled(dsets, args.model, cur, "sample")
        for name, r in (("team", rt), ("orgsmith", ro), ("sample", rs)):
            m = metrics(r)
            print(f"  {name:<9} Hit@1 {m['Hit@1']:.3f}  Hit@5 {m['Hit@5']:.3f}  MRR {m['MRR']:.3f}  n={m['n']}")
        for g in sorted(set(team["group"])):
            print(f"    {g:<8} Hit@1 {metrics(rt[team['group'] == g])['Hit@1']:.3f}")
        return

    if args.speed:
        return speed_check(dsets, team, args, cur)

    # ---- LOMO on the team corpus
    R = {g: ranks(team, args.model, {**cur, args.key: g}) for g in args.grid}
    base_r = ranks(team, args.model, cur)
    groups = team["group"]
    held = np.empty_like(base_r)
    chosen = {}
    for m in sorted(set(groups)):
        tr = groups != m
        best = max(args.grid, key=lambda g: (np.mean(R[g][tr] <= 1), metrics(R[g][tr])["MRR"]))
        held[~tr] = R[best][~tr]
        chosen[m] = best
    deploy = max(args.grid, key=lambda g: (np.mean(R[g] <= 1), metrics(R[g])["MRR"]))
    new = {**cur, args.key: deploy}

    d_team = boot_ci(held, base_r, team["cluster"])
    ob, oc = pooled(dsets, args.model, cur, "org/")
    on, _ = pooled(dsets, args.model, new, "org/")
    sb, _ = pooled(dsets, args.model, cur, "sample")
    sn, _ = pooled(dsets, args.model, new, "sample")
    d_org = boot_ci(on, ob, oc)
    d_sample_q = int((sn <= 1).sum() - (sb <= 1).sum())

    keep = d_team[0] > 0 and d_team[1] > 0 and d_org[0] >= -0.01 and d_sample_q >= -1
    print(f"\n== {args.label or args.key}: {args.key} in {args.grid}")
    print(f"  in-sample team Hit@1 per value: " + ", ".join(f"{g}: {np.mean(R[g] <= 1):.3f}" for g in args.grid))
    print(f"  LOMO choices per held-out member: {chosen}   deployed value: {deploy}")
    rows = [("team (held-out)", base_r, held), ("orgsmith", ob, on), ("sample", sb, sn)]
    for name, a, b in rows:
        ma, mb = metrics(a), metrics(b)
        print(f"  {name:<16} Hit@1 {ma['Hit@1']:.3f} -> {mb['Hit@1']:.3f}   Hit@5 {ma['Hit@5']:.3f} -> "
              f"{mb['Hit@5']:.3f}   MRR {ma['MRR']:.3f} -> {mb['MRR']:.3f}")
    for g in sorted(set(groups)):
        print(f"    {g:<8} Hit@1 {np.mean(base_r[groups == g] <= 1):.3f} -> {np.mean(held[groups == g] <= 1):.3f}"
              f"  (chose {chosen[g]})")
    print(f"  team held-out dHit@1 {d_team[0]*100:+.1f} pts, 90% CI [{d_team[1]*100:+.1f}, {d_team[2]*100:+.1f}]")
    print(f"  orgsmith dHit@1 {d_org[0]*100:+.1f} pts, 90% CI [{d_org[1]*100:+.1f}, {d_org[2]*100:+.1f}]"
          f"   sample d = {d_sample_q:+d} queries")
    print(f"  DECISION: {'KEEP ' + args.key + '=' + str(deploy) if keep else 'REVERT'}")

    rec = {"model": args.model, "label": args.label, "key": args.key, "grid": args.grid, "chosen": chosen,
           "deploy": deploy, "team_before": metrics(base_r), "team_after": metrics(held), "team_ci": d_team,
           "org_before": metrics(ob), "org_after": metrics(on), "org_ci": d_org,
           "sample_before": metrics(sb), "sample_after": metrics(sn), "keep": keep}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if keep and not args.dry:
        state.setdefault(args.model, {})[args.key] = deploy
        STATE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
