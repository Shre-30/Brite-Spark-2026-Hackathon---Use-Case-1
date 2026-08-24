"""
Full pipeline CLI against the AFTER (pre-amendment) corpus.
Usage: python src/cli_before.py "your question"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from answer import build_answer
from conflict_check import check_conflict, format_conflict_response
from refusal_gate import check_coverage
import retriever_after as retriever


def answer_question(question: str, k: int = 5, verbose: bool = False) -> str:
    retrieved = retriever.retrieve(question, k=k)

    if verbose:
        print("--- [AFTER] retrieved clauses ---", file=sys.stderr)
        for r in retrieved:
            print(f"  {r['id']}  sim={r['similarity']:.3f}", file=sys.stderr)
        print("-------------------------", file=sys.stderr)

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
    parser = argparse.ArgumentParser(description="[AFTER] Grounded policy-manual assistant")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.question:
        print(answer_question(args.question, k=args.k, verbose=args.verbose))
        return

    print("[AFTER] Grounded Answer — interactive mode. Ctrl+C or empty line to exit.\n")
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
