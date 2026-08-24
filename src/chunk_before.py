"""
Chunks the ORIGINAL (pre-amendment) manual into data/chunks_before.json.
Run this once; re-run only if data/manual_before.md changes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chunk import chunk_manual  # reuses the same clause-parsing logic

SRC = "data/manual_before.md"
DST = "data/chunks_before.json"


def main():
    text = Path(SRC).read_text(encoding="utf-8")
    chunks = chunk_manual(text)
    Path(DST).write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"[BEFORE] Parsed {len(chunks)} clauses from {SRC} -> {DST}")


if __name__ == "__main__":
    main()
