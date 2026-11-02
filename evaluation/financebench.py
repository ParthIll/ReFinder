"""Use FinanceBench (github.com/patronus-ai/financebench) as a test corpus of real, long PDFs.

The open-source sample is 368 public filings (10-Ks, 10-Qs, 8-Ks, earnings reports) from 32+ US companies,
2015-2023, and 150 analyst questions, each answered from one labeled filing ("What is the FY2018 capital
expenditure for 3M?"). Every filing stays in the index, so the other years' filings of the same company act as
near-duplicate distractors. This fetches the PDFs and questions (sparse clone, ~700 MB, no code is run) and
writes the query CSV evaluation.custom reads:

    file        the filing the question is answered from
    category    question_type (domain-relevant, metrics-generated, novel-generated)
    query_type  question_reasoning (information extraction, numerical reasoning, ...)

    python -m evaluation.financebench
then run the command it prints.
"""

import argparse
import csv
import json
import subprocess

from evaluation.common import REPO_ROOT

REPO_URL = "https://github.com/patronus-ai/financebench.git"
DATA_DIR = REPO_ROOT / "datasets" / "FinanceBench"


def fetch() -> None:
    if not (DATA_DIR / ".git").exists():
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(DATA_DIR)],
                       check=True)
    subprocess.run(["git", "-C", str(DATA_DIR), "sparse-checkout", "set", "pdfs", "data"], check=True)


def write_queries() -> tuple[str, int]:
    out = REPO_ROOT / "datasets" / "financebench_queries.csv"
    n = 0
    with (DATA_DIR / "data" / "financebench_open_source.jsonl").open() as f_in, out.open("w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["query", "file", "category", "query_type"])
        for line in f_in:
            q = json.loads(line)
            docs = {q["doc_name"]} | {e["doc_name"] for e in q["evidence"]}
            reasoning = (q["question_reasoning"] or "-").split(" OR ")[0].split(" (")[0]
            w.writerow([q["question"], "|".join(f"{d}.pdf" for d in sorted(docs)), q["question_type"], reasoning])
            n += 1
    return str(out.relative_to(REPO_ROOT)), n


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fetch()
    path, n = write_queries()
    corpus = (DATA_DIR / "pdfs").relative_to(REPO_ROOT)
    print(f"wrote {n} questions to {path}")
    print(f'  python -m evaluation.custom --corpus "{corpus}" --queries "{path}"')


if __name__ == "__main__":
    main()
