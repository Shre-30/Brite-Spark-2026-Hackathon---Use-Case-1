"""
Systematic retrieval health check -- not a substitute for the 10-question
test set (that tests the FULL pipeline: retrieval + conflict + refusal +
citation). This only checks retrieval in isolation, but does it for every
clause in the manual at once, in seconds, with no LLM calls needed.

Method: for each clause, use a shortened version of its own text as a
query and check whether the clause retrieves itself in the top-k. If a
clause can't find itself using its own words, no real question about
that clause is likely to find it either -- this is the earliest, cheapest
signal that a clause needs a retrieval fix (more stemming coverage,
synonym handling, sub-item splitting, etc.).

This deliberately does NOT prove real questions will work (a real
question uses the asker's words, not the clause's own words) -- but a
clause that fails even this weak, generous test is a near-certain miss on
real questions, and is worth fixing before demo day.

Usage:
    python tests/retrieval_audit.py           # k=5 (matches cli.py default)
    python tests/retrieval_audit.py --k 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import retrieve  # noqa: E402
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    chunks = json.loads(Path(__file__).parent.parent.joinpath("data/chunks.json").read_text())

    failures = []
    for c in chunks:
        # Use the first ~15 words of the clause as a stand-in "question" --
        # generous (it's literally the clause's own wording), which is
        # exactly why failing this is a strong signal, not a weak one.
        query_words = c["text"].split()[:15]
        query = " ".join(query_words)

        results = retrieve(query, k=args.k)
        found_ids = [r["id"] for r in results]
        rank = found_ids.index(c["id"]) + 1 if c["id"] in found_ids else None

        if rank is None:
            failures.append((c["id"], c["section"], "NOT IN TOP-K", query))

    print(f"Checked {len(chunks)} clauses at k={args.k}")
    print(f"{len(chunks) - len(failures)} retrieved themselves within top-{args.k}")
    print(f"{len(failures)} did NOT\n")

    if failures:
        print("Clauses that failed to self-retrieve (likely weak spots):")
        print("-" * 70)
        for clause_id, section, status, query in failures:
            print(f"  {clause_id}  ({section})")
            print(f"    query used: {query[:80]}...")
        print()
        print("These clauses are worth checking by hand -- try phrasing a real")
        print("question about each one and see if it actually fails, or if the")
        print("self-retrieval test was just being extra strict.")
    else:
        print("No failures -- every clause can retrieve itself using its own words.")


if __name__ == "__main__":
    main()
