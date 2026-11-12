"""Where does indexing time go? Times extraction, chunking and embedding separately, on any machine.

    python -m evaluation.index_bench                                    # team corpus, Harrier 270M, best device
    python -m evaluation.index_bench --device cpu --limit-chunks 2000   # quick CPU estimate (extrapolates)
    python -m evaluation.index_bench --models Qwen/Qwen3-Embedding-0.6B --batch-sizes 16 64 128
    python -m evaluation.index_bench --device cpu --int8                # also time 8-bit ONNX (worth it on Intel/AMD?)

Extraction is always timed fresh (no cache), since that's what a first index costs. Giant files are thinned to
refinder.indexer.MAX_CHUNKS_PER_FILE chunks as the app does (--cap 0 embeds everything). Embedding runs on all those
chunks, or on a fixed random sample with --limit-chunks; the full-corpus time is then extrapolated from that sample.
"""

import argparse
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluation.common import REPO_ROOT, chunk_corpus
from evaluation.custom import find_query_csvs, query_file_exclusions
from refinder.embedder import Embedder, pick_device
from refinder.extract import UnsupportedFile, extract_pieces
from refinder.indexer import MAX_CHUNKS_PER_FILE, keep_evenly
from refinder.onnx_encoder import encode, load_session


def peak_memory_gb() -> float | None:
    """Peak resident memory of this process so far (None on Windows, which lacks the resource module)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # macOS reports bytes, Linux kilobytes


def extract_all(root: Path, exclude: set[Path]):
    """Extract every supported file under root, timing each one."""
    paths, pieces, seconds, failed, skipped = [], [], [], {}, Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve() in exclude:
            continue
        t = time.perf_counter()
        try:
            file_pieces = extract_pieces(p)
        except UnsupportedFile:
            skipped[p.suffix.lower() or p.name] += 1
            continue
        except Exception as e:  # corrupt or password-protected files
            failed[str(p.relative_to(root))] = type(e).__name__
            continue
        paths.append(p.relative_to(root).as_posix())
        pieces.append([x.text for x in file_pieces])
        seconds.append(time.perf_counter() - t)
    return paths, pieces, np.array(seconds), failed, skipped


def time_embedding(model: str, device: str, chunks: list[str], batch_size: int, int8: bool = False):
    """Chunks per second for one model / device / batch size. Model load and a warm-up batch are not counted."""
    e = Embedder(model, device=device)
    backend = "ONNX" if e.onnx_session else "PyTorch"
    run = lambda texts: e.encode(texts, batch_size=batch_size)  # noqa: E731
    if int8:
        if not e.onnx_session:
            return None, "no ONNX"
        session, backend = load_session(model, e.tokenizer, int8=True), "ONNX int8"
        prompt = e.model.prompts.get("document") or ""
        run = lambda texts: encode(session, e.tokenizer, [prompt + t for t in texts], e.dim,  # noqa: E731
                                   e.max_seq_length, batch_size, e.pooling_mode)
    run(chunks[: min(8, len(chunks))])
    t = time.perf_counter()
    run(chunks)
    return len(chunks) / (time.perf_counter() - t), backend


def fmt_min(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "ReFinder File Corpus")
    ap.add_argument("--models", nargs="+", default=["microsoft/harrier-oss-v1-270m"])
    ap.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    ap.add_argument("--limit-chunks", type=int, default=0, help="embed a random sample of this many chunks")
    ap.add_argument("--cap", type=int, default=MAX_CHUNKS_PER_FILE or 0,
                    help=f"max chunks embedded per file (default {MAX_CHUNKS_PER_FILE}, the app's; 0 = all)")
    ap.add_argument("--int8", action="store_true", help="also time 8-bit quantized ONNX on CPU")
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"{platform.platform()} | {platform.processor() or platform.machine()} | device {device}")

    exclude = query_file_exclusions(find_query_csvs(args.corpus))
    t0 = time.perf_counter()
    paths, pieces, ext_s, failed, skipped = extract_all(args.corpus, exclude)
    extract_total = time.perf_counter() - t0

    # Chunk counts depend only on the tokenizer, so the first model's tokenizer stands in for all of them.
    chunker = Embedder(args.models[0], device="cpu")
    t0 = time.perf_counter()
    chunks, owner, _ = chunk_corpus(chunker, ["\n\n".join(p) for p in pieces], pieces, "paragraph")
    chunk_total = time.perf_counter() - t0
    del chunker

    n_chunks = np.bincount(owner, minlength=len(paths))
    by_type, ext_by_type = defaultdict(int), defaultdict(float)
    for path, n, s in zip(paths, n_chunks, ext_s):
        by_type[Path(path).suffix.lower()] += int(n)
        ext_by_type[Path(path).suffix.lower()] += float(s)

    print(f"\n{len(paths)} files, {len(chunks):,} chunks (median {int(np.median(n_chunks))} per file)"
          + (f", {len(failed)} failed" if failed else "") + (f", skipped {dict(skipped)}" if skipped else ""))
    print(f"\nEXTRACTION  {fmt_min(extract_total)}  (uncached)")
    for suffix in sorted(ext_by_type, key=lambda s: -ext_by_type[s]):
        print(f"  {suffix:6s} {fmt_min(ext_by_type[suffix]):>9s}   {by_type[suffix]:>7,} chunks")
    print("  slowest files:")
    for i in np.argsort(-ext_s)[:10]:
        print(f"    {ext_s[i]:7.1f} s  {n_chunks[i]:>6,} chunks  {paths[i][:90]}")
    print(f"\nCHUNKING    {fmt_min(chunk_total)}")

    order = np.argsort(-n_chunks)
    share = np.cumsum(n_chunks[order]) / max(len(chunks), 1)
    print("\nWHERE THE CHUNKS ARE")
    for k in (1, 5, 10, 20):
        if k <= len(paths):
            print(f"  largest {k:>2} files: {share[k - 1] * 100:5.1f}% of chunks")
    for i in order[:6]:
        print(f"    {n_chunks[i]:>6,} chunks  {paths[i][:90]}")

    starts = np.r_[0, np.cumsum(n_chunks)[:-1]]
    embedded = [chunks[s + i] for s, n in zip(starts, n_chunks) for i in keep_evenly(int(n), args.cap or None)]
    print(f"\nEMBEDDED CHUNKS  {len(embedded):,} of {len(chunks):,}"
          + (f" (files keep at most {args.cap})" if args.cap else ""))

    sample = embedded
    if args.limit_chunks and args.limit_chunks < len(embedded):
        rng = np.random.default_rng(0)
        sample = [embedded[i] for i in rng.choice(len(embedded), args.limit_chunks, replace=False)]
    print(f"\nEMBEDDING  ({len(sample):,} chunks timed" + (", rest extrapolated)" if sample is not embedded else ")"))
    runs = [(m, bs, False) for m in args.models for bs in args.batch_sizes]
    runs += [(m, bs, True) for m in args.models for bs in args.batch_sizes] if args.int8 else []
    for model, bs, int8 in runs:
        rate, backend = time_embedding(model, device, sample, bs, int8)
        if rate is None:
            print(f"  {model:42s} int8 needs ONNX (CPU and a model in ONNX_MODELS)")
            continue
        embed_total = len(embedded) / rate
        total = extract_total + chunk_total + embed_total
        mem = peak_memory_gb()
        print(f"  {model:42s} {backend:9s} batch {bs:<4d} {rate:7.1f} chunks/s   embed {fmt_min(embed_total):>9s}"
              f"   whole index {fmt_min(total):>9s}" + (f"   peak memory {mem:.1f} GB" if mem else ""), flush=True)


if __name__ == "__main__":
    main()
