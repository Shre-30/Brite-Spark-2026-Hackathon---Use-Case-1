"""
Builds the final grounded answer. Only called after refusal_gate has
already said SUFFICIENT — this module does not make the answer/refuse
decision itself, it just writes the answer given that it's safe to.
"""

import re

from llm_client import generate

ANSWER_PROMPT = """Answer the question using ONLY the clauses below. Do not use outside knowledge and do not infer beyond what the clauses literally state.

Write in a formal, neutral register appropriate to a policy document -- as a caseworker reference, not a conversational assistant. Do not use contractions, first person, or casual phrasing.

Every factual claim in your answer must end with the clause ID it comes from, in square brackets, e.g. [§4.2]. If two clauses are both relevant, cite both.

Clauses:
{clauses}

Question: {question}

Answer (formal register, with inline clause citations):
"""

CITATION_PATTERN = re.compile(r"\[\s*§\s*(\d+(?:\.\d+)+)\s*(?:\([a-z]\))?\s*\]")


def _format_clauses(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def build_answer(question: str, retrieved: list[dict]) -> dict:
    prompt = ANSWER_PROMPT.format(
        clauses=_format_clauses(retrieved), question=question
    )
    raw = generate(prompt)

    valid_ids = {c["id"] for c in retrieved}
    cited = {f"§{m}" for m in CITATION_PATTERN.findall(raw)}
    unverifiable = cited - valid_ids  # cited but not actually in retrieved context

    return {
        "answer": raw,
        "cited_clauses": sorted(cited),
        "unverifiable_citations": sorted(unverifiable),
        "fully_grounded": len(unverifiable) == 0 and len(cited) > 0,
    }
