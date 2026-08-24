"""
Splits a policy manual (markdown/plain text) into clause-level chunks.

Design choice: chunk by CLAUSE, not by fixed token window. A fixed window
chunker will happily cut a clause in half or merge two unrelated clauses,
which breaks clause-level citation before the pipeline even starts. Since
manuals like this are numbered (§1.1, §1.2, ...), we split on that pattern
and keep the section heading as context for each clause.

Output: data/chunks.json — a list of
    {"id": "§1.3", "section": "Section 1: Eligibility", "text": "..."}
"""

import json
import re
import sys
from pathlib import Path

# Matches clause markers like "**1.1.1**", "**4.3.2**", "**9.1.4**" —
# bold, three-level (Part.Section.Paragraph), which is Calder County's
# numbering convention. Some clauses also carry a bold defined-term right
# after the number (e.g. "**1.4.3 Household**") — that's captured too so
# it isn't lost from the clause text.
# Matches clause markers like "**1.1.1**", "**4.3.2**", "**9.1.4**" --
# bold, three-level (Part.Section.Paragraph), Calder County's numbering
# convention. Also matches amendment-inserted clauses with a trailing
# letter, like "**10.5.3A**" (inserted after §10.5.3 without renumbering
# everything after it -- standard legal drafting practice, and something
# the original regex missed, silently swallowing the new clause into its
# predecessor's text instead of citing it separately).
CLAUSE_PATTERN = re.compile(r"\*\*(\d+\.\d+\.\d+[A-Z]?)\*\*\s*(\**[A-Za-z][^*\n]*\**)?\s*", re.MULTILINE)

# Two heading levels in this manual: "# Part N — Title" (top level) and
# "## N.N Subtitle" (subsection). A clause ID like §9.1.4 is ambiguous
# without knowing which Part it's under, so we track both and combine them.
PART_PATTERN = re.compile(r"^#\s+Part\s+.*", re.MULTILINE)
SUBSECTION_PATTERN = re.compile(r"^##\s+(.*)", re.MULTILINE)


def chunk_manual(text: str) -> list[dict]:
    parts = [(m.start(), m.group(0).strip().lstrip("# ").strip()) for m in PART_PATTERN.finditer(text)]
    subsections = [(m.start(), m.group(1).strip()) for m in SUBSECTION_PATTERN.finditer(text)]

    def part_for(pos: int) -> str:
        current = "Unknown Part"
        for start, title in parts:
            if start <= pos:
                current = title
            else:
                break
        return current

    def subsection_for(pos: int) -> str:
        current = ""
        for start, title in subsections:
            if start <= pos:
                current = title
            else:
                break
        return current

    matches = list(CLAUSE_PATTERN.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        clause_id = "§" + m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Full clause text = this match through to just before the next clause marker
        full_text = text[start:end].strip()
        # Strip the leading "**X.X.X** " or "**X.X.XA** " marker so the stored text reads naturally
        full_text = re.sub(r"^\*\*\d+\.\d+\.\d+[A-Z]?\*\*\s*", "", full_text)
        # A heading (# Part.../## N.N) can fall between this clause and the
        # next clause marker (e.g. a clause is the last one in its
        # subsection) — strip any such trailing heading line so it doesn't
        # get glued onto the clause's own text.
        full_text = re.sub(r"\n#{1,2}\s+.*$", "", full_text, flags=re.MULTILINE).strip()

        part = part_for(start)
        subsection = subsection_for(start)
        section = f"{part} / {subsection}" if subsection else part

        chunks.append({
            "id": clause_id,
            "section": section,
            "text": full_text,
        })

    return chunks


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/manual.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/chunks.json")

    text = src.read_text(encoding="utf-8")
    chunks = chunk_manual(text)

    dst.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Parsed {len(chunks)} clauses from {src} -> {dst}")
    for c in chunks[:3]:
        preview = c["text"][:80].replace("\n", " ")
        print(f"  {c['id']} ({c['section']}): {preview}...")


if __name__ == "__main__":
    main()
