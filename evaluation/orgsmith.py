"""Use an OrgSmith synthetic company (github.com/peterzat/OrgSmith) as a test corpus.

Each OrgSmith company is a realistic file share (engagement letters, memos, minutes, emails, spreadsheets,
scanned PDFs, plus drafts, duplicates and misfiled noise) with a labeled retrieval suite: "which documents
state X?" questions whose answer is a set of files. This fetches one company (sparse clone, no OrgSmith code is
run) and converts its retrieval.jsonl into the query CSV evaluation.custom reads:

    file                      expected_docs, joined with "|" (any of them counts for Hit@k; Recall@10 needs all)
    secondary_relevant_files  acceptable_docs (partial credit in nDCG)
    category                  question family (fact:money, mention:person, workbook, ...)

    python -m evaluation.orgsmith                          # northgate-staffing
    python -m evaluation.orgsmith --company <name> [<name> ...]
    python -m evaluation.orgsmith --company all            # every company in the repo
then run the command(s) it prints.
"""

import argparse
import csv
import json
import subprocess

from evaluation.common import REPO_ROOT

REPO_URL = "https://github.com/peterzat/OrgSmith.git"
DATA_DIR = REPO_ROOT / "datasets" / "OrgSmith"
# Generator bookkeeping at the share root: TOC.md lists every document, so indexing it would leak the answers.
NOT_DOCUMENTS = ("TOC.md", "PERMISSIONS.md")
FAMILIES = ("fact:", "mention:", "workbook", "firm")


def all_companies() -> list[str]:
    out = subprocess.run(["git", "-C", str(DATA_DIR), "ls-tree", "-d", "--name-only", "HEAD", "companies/"],
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(c.split("/")[-1] for c in out if not c.endswith("-metadata"))


def fetch(company: str) -> None:
    if not (DATA_DIR / ".git").exists():
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(DATA_DIR)],
                       check=True)
    current = subprocess.run(["git", "-C", str(DATA_DIR), "sparse-checkout", "list"],
                             capture_output=True, text=True, check=True).stdout.split()
    wanted = [f"companies/{company}", f"companies/{company}-metadata/evals"]
    subprocess.run(["git", "-C", str(DATA_DIR), "sparse-checkout", "set", *sorted(set(current) | set(wanted))],
                   check=True)


def write_queries(company: str) -> tuple[str, int, int]:
    """Converts the retrieval suite; returns (csv path, questions written, unanswerable questions skipped)."""
    src = DATA_DIR / "companies" / f"{company}-metadata" / "evals" / "retrieval.jsonl"
    out = REPO_ROOT / "datasets" / f"orgsmith_{company}_queries.csv"
    written = skipped = 0
    with src.open() as f_in, out.open("w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["query", "file", "secondary_relevant_files", "category", "query_type"])
        for line in f_in:
            q = json.loads(line)
            if not q["expected_docs"]:  # unanswerable: the right response is to abstain, which ranking can't express
                skipped += 1
                continue
            family = next((t for t in q["tags"] if t.startswith(FAMILIES)), "-")
            n = len(q["expected_docs"])
            w.writerow([q["question"], "|".join(q["expected_docs"]), ";".join(q["acceptable_docs"]), family,
                        "1 doc" if n == 1 else "2-5 docs" if n <= 5 else "6+ docs"])
            written += 1
    return str(out.relative_to(REPO_ROOT)), written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", nargs="+", default=["northgate-staffing"])
    args = ap.parse_args()
    if args.company == ["all"]:
        fetch("northgate-staffing")  # makes sure the clone exists before listing
        args.company = all_companies()
    for company in args.company:
        fetch(company)
        path, n, skipped = write_queries(company)
        corpus = (DATA_DIR / "companies" / company).relative_to(REPO_ROOT)
        print(f"{company}: wrote {n} questions to {path}" + (f" ({skipped} unanswerable skipped)" if skipped else ""))
        print(f'  python -m evaluation.custom --corpus "{corpus}" --queries "{path}" --exclude {" ".join(NOT_DOCUMENTS)}')


if __name__ == "__main__":
    main()
