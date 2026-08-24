"""
Retrieval against the BEFORE (pre-amendment) corpus. Same TF-IDF + Porter
stemming + sub-item indexing logic as retriever.py, just pointed at
data/chunks_before.json instead of data/chunks.json.
"""

import json
import re
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = "data/chunks_before.json"

_cache = {}
_stemmer = PorterStemmer()
_token_pattern = re.compile(r"[A-Za-z]+")
_sub_item_pattern = re.compile(r"\([a-z]\)\s*(.+?)(?=\([a-z]\)|$)", re.DOTALL)
_STOPWORDS = frozenset("""
a an the of to in and or is are was were be been being for on at by with
this that these those it its as from into within not shall may must
""".split())


def _stem_analyzer(text):
    tokens = [_stemmer.stem(t.lower()) for t in _token_pattern.findall(text) if t.lower() not in _STOPWORDS]
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def _build_search_entries(chunks):
    entries = []
    for i, c in enumerate(chunks):
        entries.append({"chunk_index": i, "text": f"{c['section']} {c['text']}"})
        sub_items = _sub_item_pattern.findall(c["text"])
        if len(sub_items) >= 2:
            for item in sub_items:
                entries.append({"chunk_index": i, "text": f"{c['section']} {item.strip()}"})
    return entries


def _load():
    if "chunks" in _cache:
        return _cache["chunks"], _cache["vectorizer"], _cache["matrix"], _cache["entries"]
    chunks = json.loads(Path(CHUNKS_PATH).read_text(encoding="utf-8"))
    entries = _build_search_entries(chunks)
    vectorizer = TfidfVectorizer(analyzer=_stem_analyzer)
    matrix = vectorizer.fit_transform([e["text"] for e in entries])
    _cache.update(chunks=chunks, vectorizer=vectorizer, matrix=matrix, entries=entries)
    return chunks, vectorizer, matrix, entries


def build_index():
    chunks, _, _, entries = _load()
    print(f"[BEFORE] Loaded {len(chunks)} clauses ({len(entries)} searchable entries) from {CHUNKS_PATH}")
    return chunks


def retrieve(query, k=5, db_dir=None):
    chunks, vectorizer, matrix, entries = _load()
    query_vec = vectorizer.transform([query])
    entry_scores = cosine_similarity(query_vec, matrix)[0]
    best_per_chunk = {}
    for entry, score in zip(entries, entry_scores):
        ci = entry["chunk_index"]
        if ci not in best_per_chunk or score > best_per_chunk[ci]:
            best_per_chunk[ci] = score
    ranked = sorted(best_per_chunk.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        {"id": chunks[ci]["id"], "section": chunks[ci]["section"], "text": chunks[ci]["text"],
         "similarity": float(score), "distance": float(1 - score)}
        for ci, score in ranked
    ]


if __name__ == "__main__":
    import sys
    build_index()
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the disregard for household earnings?"
    print(f"\nTest query: {q}\n")
    for r in retrieve(q, k=5):
        print(f"  {r['id']}  sim={r['similarity']:.3f}  {r['text'][:70]}...")
