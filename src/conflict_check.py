"""
Detects when retrieved clauses disagree with each other on the same point,
so the pipeline can surface both sides instead of the answer-LLM silently
picking one (the failure mode this whole project exists to prevent).

This is deliberately a THIRD outcome, not folded into refusal_gate.py:
  - refusal_gate.py asks "is there enough here to answer AT ALL"
  - conflict_check.py asks "do the things we found agree with each other"
A question can pass the refusal gate (plenty of directly-relevant text
came back) and still need a conflict flag (that text contradicts itself).
Keeping them separate means each prompt does one narrow job, which is
easier to get right and easier to audit than one prompt doing both.

Pipeline order: retrieve -> conflict_check -> refusal_gate -> answer.
If a conflict is found, we short-circuit straight to a conflict response
and never reach the normal answer path at all.
"""

from dataclasses import dataclass, field

from llm_client import generate

CONFLICT_PROMPT = """You are checking whether any of the clauses below directly contradict each other on the SAME point -- not just different topics, but two clauses that cannot both be true or both be followed at once.

Clauses:
{clauses}

If you find a genuine contradiction, respond in exactly this format:
CONFLICT: YES
CLAUSES: the two (or more) clause IDs that conflict, e.g. §4.3.2, §9.1.4
EXPLANATION: one or two sentences on what they each say and why they conflict

If there is no genuine contradiction (clauses are consistent, or simply address different things), respond:
CONFLICT: NO
"""


@dataclass
class ConflictResult:
    has_conflict: bool
    conflicting_ids: list[str] = field(default_factory=list)
    explanation: str = ""


def _format_clauses(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def check_conflict(retrieved: list[dict]) -> ConflictResult:
    if len(retrieved) < 2:
        return ConflictResult(False)

    prompt = CONFLICT_PROMPT.format(clauses=_format_clauses(retrieved))
    raw = generate(prompt)

    if "CONFLICT: YES" not in raw.upper():
        return ConflictResult(False)

    conflicting_ids = []
    explanation = ""
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("CLAUSES:"):
            ids_part = line.split(":", 1)[1]
            conflicting_ids = [tok.strip() for tok in ids_part.replace(",", " ").split() if tok.strip().startswith("§")]
        if upper.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    # Sanity check: only trust clause IDs that were actually retrieved --
    # same discipline as the citation check in answer.py. If the model
    # names an ID that wasn't in the retrieved set, drop it rather than
    # surface a phantom clause.
    valid_ids = {c["id"] for c in retrieved}
    conflicting_ids = [cid for cid in conflicting_ids if cid in valid_ids]

    if len(conflicting_ids) < 2:
        # Model said YES but couldn't point to two real clauses -- treat
        # as no verified conflict rather than surfacing a vague warning.
        return ConflictResult(False)

    return ConflictResult(True, conflicting_ids, explanation)


def format_conflict_response(result: ConflictResult, retrieved: list[dict]) -> str:
    by_id = {c["id"]: c for c in retrieved}
    lines = [
        "The manual contains conflicting provisions on this question. No single "
        "answer can be given without resolving the discrepancy; both provisions "
        "are set out below.",
        "",
        f"Nature of the conflict: {result.explanation}",
        "",
    ]
    for cid in result.conflicting_ids:
        clause = by_id.get(cid)
        if clause:
            lines.append(f"[{cid}] {clause['text']}")
            lines.append("")
    lines.append(
        "Recommended action: refer this question to a policy supervisor for "
        "clarification. Neither provision should be relied upon in isolation "
        "until the conflict is resolved."
    )
    return "\n".join(lines)
