"""
CLI for the Grounded Answer assistant.

Usage:
    python src/cli.py "Does an SSI recipient automatically qualify for benefits?"
    python src/cli.py --k 5                 # interactive loop
"""

import argparse
import sys

from answer import build_answer
from conflict_check import check_conflict, format_conflict_response
from refusal_gate import check_coverage
from retriever import retrieve


def answer_question(question: str, k: int = 5, verbose: bool = False) -> str:
    retrieved = retrieve(question, k=k)

    if verbose:
        print("--- retrieved clauses ---", file=sys.stderr)
        for r in retrieved:
            print(f"  {r['id']}  sim={r['similarity']:.3f}", file=sys.stderr)
        print("-------------------------", file=sys.stderr)

    # Conflict check runs BEFORE the refusal gate. A contradiction between
    # two clauses is a different situation from "not enough information" --
    # there's plenty of information, it just disagrees with itself. That
    # needs a human, not a best-effort answer picking one side.
    conflict = check_conflict(retrieved)
    if conflict.has_conflict:
        if verbose:
            print(f"--- conflict detected: {conflict.conflicting_ids} ---", file=sys.stderr)
        return format_conflict_response(conflict, retrieved)

    gate = check_coverage(question, retrieved)

    if not gate.sufficient:
        related = ", ".join(r["id"] for r in retrieved[:3]) if retrieved else "none"
        return (
            "This question is not settled by the manual with sufficient "
            "certainty to provide an answer.\n\n"
            f"Reason: {gate.reason}\n"
            f"Related but non-determinative provisions: {related}\n\n"
            "Recommended action: refer this question to a policy supervisor "
            "or an experienced caseworker rather than proceeding on an "
            "assumption."
        )

    result = build_answer(question, retrieved)
    out = result["answer"]

    if not result["fully_grounded"]:
        out += (
            "\n\nNote: this response includes a claim that could not be "
            "verified against the retrieved provisions. It should be "
            "confirmed manually before being relied upon."
        )

    return out


def main():
    parser = argparse.ArgumentParser(description="Grounded policy-manual assistant")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit to start interactive mode.")
    parser.add_argument("--k", type=int, default=5, help="Number of clauses to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved clauses and scores")
    args = parser.parse_args()

    if args.question:
        print(answer_question(args.question, k=args.k, verbose=args.verbose))
        return

    print("Grounded Answer — interactive mode. Ctrl+C or empty line to exit.\n")
    try:
        while True:
            q = input("question> ").strip()
            if not q:
                break
            print()
            print(answer_question(q, k=args.k, verbose=args.verbose))
            print()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
