"""
Runs tests/test_questions.json through the pipeline and prints results.

This does NOT auto-grade pass/fail against a rubric — the "expected" field
is a note to a human about what SHOULD happen. You read each output and
mark it yourself in RESULTS.md. That's intentional: this problem is
explicitly asking for a test set you evaluated honestly, not a green
checkmark script. Automating away that judgment call defeats the point.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli import answer_question  # noqa: E402


def main():
    questions = json.loads(
        (Path(__file__).parent / "test_questions.json").read_text()
    )

    for q in questions:
        print("=" * 80)
        print(f"{q['id']}  [{q['expected']}]  {q['question']}")
        print(f"note: {q['note']}")
        print("-" * 80)
        try:
            out = answer_question(q["question"], verbose=True)
        except Exception as e:
            out = f"ERROR: {e}"
        print(out)
        print()


if __name__ == "__main__":
    main()
