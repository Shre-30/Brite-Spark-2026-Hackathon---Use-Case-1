"""
Runs tests/amendment_questions.json through the pipeline and saves output
to a labeled file -- run once now (label "before"), run again after the
amendment is applied to data/manual.md (label "after"), then diff the two
files directly instead of relying on memory of what the terminal showed.

Usage:
    python tests/run_amendment_test.py before
    python tests/run_amendment_test.py after
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli import answer_question  # noqa: E402


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        print("Usage: python tests/run_amendment_test.py [before|after]")
        sys.exit(1)

    label = sys.argv[1]
    questions = json.loads(
        (Path(__file__).parent / "amendment_questions.json").read_text()
    )

    out_path = Path(__file__).parent / f"amendment_results_{label}.txt"
    lines = [f"Amendment 2026-01 test run -- label: {label} -- {datetime.now().isoformat()}", ""]

    for q in questions:
        lines.append("=" * 80)
        lines.append(f"{q['id']}  targets: {q['targets']}")
        lines.append(f"Q: {q['question']}")
        lines.append(f"expected ({label}): {q[f'{label}_expected']}")
        lines.append("-" * 80)
        try:
            result = answer_question(q["question"])
        except Exception as e:
            result = f"ERROR: {e}"
        lines.append(result)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
