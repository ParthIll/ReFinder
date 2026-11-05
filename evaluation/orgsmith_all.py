"""Run the final pipeline on every OrgSmith company and print pooled Hit@1/5/10.

    python -m evaluation.orgsmith_all
    python -m evaluation.orgsmith_all --model Qwen/Qwen3-Embedding-0.6B

Needs the companies fetched first: python -m evaluation.orgsmith --company all
Each company runs through evaluation.custom (TOC.md and PERMISSIONS.md excluded: they list every filename); the pooled number weights each company by its query count.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPANIES = ["ashcombe-advisory", "calderwood-partners", "hollowell-ip", "meridian-actuarial",
             "northgate-staffing", "quillon-harbor", "saltmarsh-environmental", "verdant-health"]
METRICS = ["Hit@1", "Hit@5", "Hit@10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/harrier-oss-v1-0.6b")
    ap.add_argument("--rerank", default="Alibaba-NLP/gte-reranker-modernbert-base")
    ap.add_argument("--companies", nargs="+", default=COMPANIES)
    args = ap.parse_args()

    totals = defaultdict(lambda: defaultdict(float))  # setup -> metric -> sum of hits
    counts = defaultdict(int)
    for c in args.companies:
        queries = REPO / "datasets" / f"orgsmith_{c}_queries.csv"
        corpus = REPO / "datasets" / "OrgSmith" / "companies" / c
        if not queries.exists() or not corpus.exists():
            sys.exit(f"{c} missing: run python -m evaluation.orgsmith --company {c} first")
        cmd = [sys.executable, "-m", "evaluation.custom", "--corpus", str(corpus), "--queries", str(queries),
               "--models", args.model, "--file-vector", "0", "1", "--size-penalty", "0", "0.02",
               "--rerank", args.rerank, "--rerank-fuse", "1", "--no-bm25",
               "--exclude", "TOC.md", "PERMISSIONS.md"]
        print(f"=== {c}", flush=True)
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        saved = re.search(r"saved (\S+/)", out.stdout)
        if out.returncode or not saved:
            print(out.stdout[-1500:], out.stderr[-1500:])
            sys.exit(f"{c} failed")
        summary = json.loads((REPO / saved.group(1) / "summary.json").read_text())
        for setup, res in summary["results"].items():
            o = res["accuracy"]["overall"]["all"]
            for m in METRICS:
                totals[setup][m] += o[m] * o["n"]
            counts[setup] += o["n"]
            print(f"  {setup[-60:]:60s} n={o['n']:3d} " + " ".join(f"{m} {o[m]:.3f}" for m in METRICS), flush=True)

    print("\nPOOLED (all companies, weighted by queries)")
    for setup, n in counts.items():
        print(f"  {n:4d} queries  " + "  ".join(f"{m} {totals[setup][m] / n:.3f}" for m in METRICS) + f"   {setup}")


if __name__ == "__main__":
    main()
