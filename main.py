"""Smoke test: embed a few toy documents and run a semantic query against them."""

import sys

from refinder.embedder import Embedder, cosine_top_k

DOCS = {
    "taxes_2025.txt": "Receipts and W-2 forms for filing my federal income tax return this April.",
    "pasta.md": "Boil the spaghetti for nine minutes, then toss with garlic, olive oil and parmesan.",
    "trip_notes.txt": "Flight to Tokyo departs at 10am; hotel booked in Shinjuku for five nights.",
    "resume.docx": "Software engineer with experience in Python, distributed systems and machine learning.",
    "gym_log.csv": "Squat 3x5 at 225 lbs, bench press 3x5 at 165 lbs, deadlift 1x5 at 275 lbs.",
}


def main() -> None:
    query = " ".join(sys.argv[1:]) or "recipe for dinner"
    embedder = Embedder()
    print(f"model={embedder.model_name} dim={embedder.dim} device={embedder.model.device}")

    names = list(DOCS)
    vecs = embedder.encode([DOCS[n] for n in names])
    print(f"\nquery: {query!r}")
    for i, score in cosine_top_k(embedder.encode_query(query), vecs, k=3):
        print(f"  {score:.3f}  {names[i]}")


if __name__ == "__main__":
    main()
