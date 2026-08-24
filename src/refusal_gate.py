"""
Decides SUFFICIENT vs INSUFFICIENT before any answer is generated.

Why this is a separate step from answer generation, not a flag the
answer-LLM sets itself: a single call that both writes a fluent answer
AND judges its own adequacy is exactly the failure mode this whole
problem is warning against (confident, fluent, wrong). Splitting
"can this be answered" from "here is the answer" means the judgment
happens on narrower, easier grounds, and a refusal decision can't be
argued out of by the model's urge to be helpful mid-generation.

Two independent signals, combined:
  1. Retrieval similarity threshold (catches "nothing relevant came back
     at all" — the pure out-of-scope case).
  2. LLM sufficiency judgment over the retrieved clauses (catches the
     harder case: clauses that LOOK relevant by similarity but don't
     actually settle the question — the "apparent gap" trap).

Threshold value and reasoning belong in DECISIONS.md, not just here.

CALIBRATION NOTE: the original SIMILARITY_FLOOR of 0.35 was set before
testing against the real embedding model and turned out to be too strict --
a real query ("What is the purpose of the Household Support Program?")
retrieved the exactly correct clause (§1.1.1) at similarity 0.24 and was
wrongly refused before the LLM sufficiency check even ran. Lowered to 0.15,
which only catches genuinely unrelated queries; the LLM sufficiency
judgment (not the similarity number) is the signal actually doing the
semantic work of deciding SUFFICIENT vs INSUFFICIENT.
"""

from dataclasses import dataclass

from llm_client import generate

SIMILARITY_FLOOR = 0.15  # see calibration note above and DECISIONS.md

SUFFICIENCY_PROMPT = """You are checking whether the policy clauses below are enough to answer the question with certainty. You are NOT answering the question yet.

Answer INSUFFICIENT if:
- the clauses do not directly address the question
- the clauses are only tangentially related
- the clauses conflict with each other on the point in question
- answering would require inference beyond what the clauses state

Answer SUFFICIENT only if a specific clause directly and unambiguously settles the question.

Clauses:
{clauses}

Question: {question}

Respond in exactly this format:
JUDGMENT: SUFFICIENT or INSUFFICIENT
REASON: one sentence
"""


@dataclass
class GateResult:
    sufficient: bool
    reason: str
    top_similarity: float
    conflicting: bool = False


def _format_clauses(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def check_coverage(question: str, retrieved: list[dict]) -> GateResult:
    if not retrieved:
        return GateResult(False, "No relevant clauses were retrieved at all.", 0.0)

    top_sim = retrieved[0]["similarity"]

    if top_sim < SIMILARITY_FLOOR:
        return GateResult(
            False,
            f"Nothing in the manual is closely related to this question "
            f"(best match similarity {top_sim:.2f}, below the {SIMILARITY_FLOOR} floor).",
            top_sim,
        )

    prompt = SUFFICIENCY_PROMPT.format(
        clauses=_format_clauses(retrieved), question=question
    )
    raw = generate(prompt)

    judgment = "INSUFFICIENT"
    reason = raw.strip() or "Model did not return a parseable judgment; defaulting to refusal."
    for line in raw.splitlines():
        if line.upper().startswith("JUDGMENT:"):
            judgment = line.split(":", 1)[1].strip().upper()
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return GateResult(
        sufficient=(judgment == "SUFFICIENT"),
        reason=reason,
        top_similarity=top_sim,
    )
