"""
Runs tests/amendment_questions.json through BOTH the before and after
pipelines in one go, writing two separate files:
    tests/amendment_results_before.txt
    tests/amendment_results_after.txt

This replaces manually copying manual_amended.md over manual.md and
re-running -- cli_before.py and cli_after.py each point at their own
fixed chunk file (chunks_before.json / chunks_after.json), so both can
run back to back without touching each other's data.

Prerequisite: run these once first (or after editing either manual file)
    python src/chunk_before.py
    python src/chunk_after.py

Usage:
    python tests/run_before_after.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cli_before  # noqa: E402
import cli_after  # noqa: E402


def run(label: str, answer_fn, questions: list[dict]) -> str:
    lines = [f"Amendment 2026-01 test run -- label: {label} -- {datetime.now().isoformat()}", ""]
    for q in questions:
        lines.append("=" * 80)
        lines.append(f"{q['id']}  targets: {q['targets']}")
        lines.append(f"Q: {q['question']}")
        lines.append(f"expected ({label}): {q[f'{label}_expected']}")
        lines.append("-" * 80)
        try:
            result = answer_fn(q["question"])
        except Exception as e:
            result = f"ERROR: {e}"
        lines.append(result)
        lines.append("")
    return "\n".join(lines)


def main():
    tests_dir = Path(__file__).parent
    questions = json.loads((tests_dir / "amendment_questions.json").read_text())

    print("Running BEFORE pipeline...")
    before_text = run("before", cli_before.answer_question, questions)
    (tests_dir / "amendment_results_before.txt").write_text(before_text, encoding="utf-8")
    print(f"  saved to {tests_dir / 'amendment_results_before.txt'}")

    print("Running AFTER pipeline...")
    after_text = run("after", cli_after.answer_question, questions)
    (tests_dir / "amendment_results_after.txt").write_text(after_text, encoding="utf-8")
    print(f"  saved to {tests_dir / 'amendment_results_after.txt'}")

    print("\nDone. Both files are ready to commit to your repo.")


if __name__ == "__main__":
    main()
