"""
Retrieves the top-k most relevant clauses for a query using TF-IDF +
cosine similarity, with Porter stemming and lettered-sub-item indexing --
no database, no persistent index, no server.

Why not a vector database (Chroma etc.) at this scale: the corpus is 137
short clauses. Everything fits in memory, TF-IDF over 137 documents fits
and searches in milliseconds, and there's no on-disk index that can go
stale or get built with the wrong settings baked in (which is exactly
what happened with Chroma's default distance metric during development --
see the git history if curious). Fewer moving parts, easier to debug: if
a query gives a wrong result, the fix is inspecting a plain Python list,
not diffing an index on disk.

Why stemming: plain TF-IDF treats "age" and "aged" as completely
different, unrelated tokens. A real test caught this -- the question
"What is the age requirement to be eligible for assistance?" scored an
exact 0.0 similarity against a clause literally stating "aged 18 or
over", purely because "age" != "aged" as strings. Porter stemming reduces
both to the same root before indexing. Needs no downloaded data --
PorterStemmer is a pure rule-based algorithm, not a trained model.

Why sub-item indexing: some clauses are one long enumerated list, e.g.
§2.1.2 packs six lettered conditions -- (a) residence, (b) age, (c)
income, (d) resources, (e) exclusions, (f) application -- into one
clause. A question about just one of those ("age requirement") gets
diluted across all 37 words of the full clause and can rank far outside
top-k even after stemming fixes the exact-word mismatch. Fix: each
lettered sub-item (a)/(b)/(c)... is *also* indexed as its own searchable
unit, scored independently, but still resolves back to and cites the
parent clause -- citation granularity doesn't change, only what gets
matched against a query.

Trade-off worth knowing: TF-IDF is still a lexical (word-overlap) method,
not a semantic embedding model. Stemming and sub-item indexing close two
real gaps found during testing, but a genuine vocabulary gap (e.g. "kid"
vs "dependent child" -- completely different words) isn't something
either of these fixes can bridge.
"""

import json
import re
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = "data/chunks.json"

_cache = {}  # lazy-loaded per process

_stemmer = PorterStemmer()
_token_pattern = re.compile(r"[A-Za-z]+")
_sub_item_pattern = re.compile(r"\([a-z]\)\s*(.+?)(?=\([a-z]\)|$)", re.DOTALL)

_STOPWORDS = frozenset("""
a an the of to in and or is are was were be been being for on at by with
this that these those it its as from into within not shall may must
""".split())


def _stem_analyzer(text: str) -> list[str]:
    """Tokenizes, lowercases, stems, drops stopwords, and builds bigrams
    from the resulting stream. Used as a custom `analyzer` for
    TfidfVectorizer so it replaces (not supplements) the default
    tokenization/stopword logic.
    """
    tokens = [
        _stemmer.stem(t.lower())
        for t in _token_pattern.findall(text)
        if t.lower() not in _STOPWORDS
    ]
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _build_search_entries(chunks: list[dict]) -> list[dict]:
    """Builds the list of searchable units. Each entry maps to a parent
    clause index in `chunks`, but a clause with lettered sub-items
    (a)/(b)/(c)... contributes one entry per sub-item IN ADDITION TO one
    entry for the full clause -- so a narrow question can match a single
    sub-item precisely, while a broad question can still match the full
    clause as a whole.
    """
    entries = []
    for i, c in enumerate(chunks):
        full_text = f"{c['section']} {c['text']}"
        entries.append({"chunk_index": i, "text": full_text})

        sub_items = _sub_item_pattern.findall(c["text"])
        if len(sub_items) >= 2:  # only worth splitting if it's an actual list
            for item in sub_items:
                entries.append({"chunk_index": i, "text": f"{c['section']} {item.strip()}"})

    return entries


def _load(chunks_path: str = CHUNKS_PATH):
    if "chunks" in _cache:
        return _cache["chunks"], _cache["vectorizer"], _cache["matrix"], _cache["entries"]

    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    entries = _build_search_entries(chunks)

    vectorizer = TfidfVectorizer(analyzer=_stem_analyzer)
    matrix = vectorizer.fit_transform([e["text"] for e in entries])

    _cache["chunks"] = chunks
    _cache["vectorizer"] = vectorizer
    _cache["matrix"] = matrix
    _cache["entries"] = entries
    return chunks, vectorizer, matrix, entries


def build_index(chunks_path: str = CHUNKS_PATH):
    """Kept for workflow consistency with the old pipeline -- there's no
    persistent index to build anymore, this just validates the chunks
    file loads correctly and reports how many clauses / search entries
    are available."""
    chunks, _, _, entries = _load(chunks_path)
    print(f"Loaded {len(chunks)} clauses ({len(entries)} searchable entries, "
          f"including lettered sub-items) from {chunks_path}")
    return chunks


def retrieve(query: str, k: int = 5, db_dir: str = None) -> list[dict]:
    """Returns top-k clauses as [{id, section, text, similarity}], best
    match first. Each clause's score is the MAX similarity across all of
    its searchable entries (full clause + any sub-items), so a clause
    scores well if EITHER the whole clause is broadly relevant OR any one
    of its sub-items is precisely relevant. `db_dir` is accepted and
    ignored -- kept only so this function's signature stays compatible
    with the old Chroma-based version, in case anything still passes it.
    """
    chunks, vectorizer, matrix, entries = _load()

    query_vec = vectorizer.transform([query])
    entry_scores = cosine_similarity(query_vec, matrix)[0]

    best_per_chunk = {}
    for entry, score in zip(entries, entry_scores):
        ci = entry["chunk_index"]
        if ci not in best_per_chunk or score > best_per_chunk[ci]:
            best_per_chunk[ci] = score

    ranked = sorted(best_per_chunk.items(), key=lambda kv: kv[1], reverse=True)[:k]

    out = []
    for chunk_index, score in ranked:
        c = chunks[chunk_index]
        out.append({
            "id": c["id"],
            "section": c["section"],
            "text": c["text"],
            "similarity": float(score),
            "distance": float(1 - score),
        })
    return out


if __name__ == "__main__":
    import sys
    build_index()
    print()
    test_q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the age requirement to be eligible for assistance?"
    print(f"Test query: {test_q}\n")
    for r in retrieve(test_q, k=5):
        print(f"  {r['id']}  sim={r['similarity']:.3f}  {r['text'][:70]}...")
