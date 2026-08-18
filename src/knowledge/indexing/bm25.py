"""
src/knowledge/indexing/bm25.py -- lexical index construction.

Migration Step 6: extracted verbatim from 04_vector_representation.py, which
keeps its CLI `main()` and re-exports these for compatibility.

Builds the BM25 index; it does not query it. Query-time lexical retrieval is
`src/retrieval/bm25.py` -- knowledge/ *produces* indexes, rag/ *consumes* them.
04_vector_representation.py previously also carried a duplicate query-time
`hybrid_search()`; see PROJECT_MEMORY.md for why it was removed.

NOTE: this tokenizer is intentionally identical to the one the query path uses
(`src/retrieval/search.py::_get_tokenizer`). Index and query tokenization must
match or BM25 scoring silently degrades; they are duplicated rather than
shared because a shared import would create a knowledge -> rag dependency.
Recorded as deferred debt: the right home is an infrastructure-level contract.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

__all__ = ["simple_tokenize", "build_bm25"]


def simple_tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Index Building
# ---------------------------------------------------------------------------

def _build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Build BM25 index from chunks."""
    tokenized = [simple_tokenize(c.get("search_text", c["text"])) for c in chunks]
    return BM25Okapi(tokenized)


# Public alias -- the underscore name is kept in 04_vector_representation.py
# for its existing callers/tests.
build_bm25 = _build_bm25
