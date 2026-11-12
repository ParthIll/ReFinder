"""Accuracy and speed of each selectable model (refinder/models.py) through the app's real search code.

Unlike evaluation.custom / evaluation.lab, which score embeddings alone, this runs evaluation/database.py as the
app does: BM25 + filename + phrase + meaning fusion, each model's own chunk budget, and the file vector / reranker
for the models that have them. Each test set is indexed into a throwaway Chroma folder.

    python -m evaluation.api_eval                                    # every model, team + OrgSmith + sample
    python -m evaluation.api_eval --models bge-small harrier-270m --device cpu
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluation import custom, lab
from evaluation.database import ChromaIndex
from refinder import models
from refinder.embedder import Embedder


def merge_duplicates(scores: np.ndarray, ds: dict) -> np.ndarray:
    """Give files with identical text the best score among them. The app ranks exact duplicates one after the
    other, but either copy answers the query (54 team-corpus files are copies), as evaluation.custom counts it."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(ds["texts"]):
        groups.setdefault(t, []).append(i)
    for idx in groups.values():
        if len(idx) > 1:
            scores[:, idx] = scores[:, idx].max(axis=1, keepdims=True)
    return scores


def evaluate(spec: models.ModelSpec, ds: dict, embedder: Embedder, work: Path) -> dict:
    store = ChromaIndex(work / f"{ds['name'].replace('/', '_')}_{spec.key}")
    paths = [str(ds["root"] / p) for p in ds["paths"]]
    store.replace(ds["root"], paths, [[p.text for p in ps] for ps in ds["pieces"]], spec.key)
    t = time.perf_counter()
    store.embed(embedder, store.generation, _NoLock())
    embed_s = time.perf_counter() - t
    scores, latency = np.zeros((len(ds["queries"]), len(paths))), []
    for i, q in enumerate(ds["queries"]):
        t = time.perf_counter()
        sig = store.signals(q["query"], embedder.encode_query(q["query"]))
        scores[i] = store.fuse(sig, len(paths))
        latency.append(time.perf_counter() - t)
    return {"ranks": custom.ranks_of_gold(merge_duplicates(scores, ds), ds["queries"]), "embed_s": embed_s, "chunks": store.count(),
            "latency": latency}


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(models.MODELS), choices=list(models.MODELS))
    ap.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    args = ap.parse_args()
    dsets = lab.load_datasets()
    work = Path(tempfile.mkdtemp(prefix="refinder_api_eval_"))
    print(f"{len(dsets)} test sets; throwaway indexes in {work}")
    for key in args.models:
        spec = models.MODELS[key]
        embedder = Embedder(spec.name, device=args.device, fp16=spec.fp16)
        out = {ds["name"]: evaluate(spec, ds, embedder, work) for ds in dsets}
        print(f"\n{key} ({spec.name}, {embedder.model.device}, {'ONNX' if embedder.onnx_session else 'PyTorch'}): "
              f"{spec.max_chunks} chunks/file ({spec.sampling})"
              + (", file vector" if spec.file_vector else "") + (", reranker" if spec.reranker else ""))
        for name, prefix in (("team", "team"), ("orgsmith", "org/"), ("sample", "sample")):
            res = [v for k, v in out.items() if k.startswith(prefix)]
            r = np.concatenate([v["ranks"] for v in res])
            lat = np.concatenate([v["latency"] for v in res]) * 1000
            print(f"  {name:9s} Hit@1 {np.mean(r <= 1):.3f}  Hit@5 {np.mean(r <= 5):.3f}  Hit@10 {np.mean(r <= 10):.3f}"
                  f"  MRR {np.mean(np.where(r <= 10, 1 / r, 0)):.3f}  n={len(r)}   embedding {sum(v['embed_s'] for v in res):6.1f} s"
                  f" ({sum(v['chunks'] for v in res):,} chunks)   search p50 {np.median(lat):.0f} ms p95 "
                  f"{np.percentile(lat, 95):.0f} ms", flush=True)
        del embedder


if __name__ == "__main__":
    main()
