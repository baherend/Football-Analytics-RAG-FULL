"""
src/knowledge/indexing/ -- turning chunks into queryable indexes.

    bm25.py          -- lexical index construction
    embeddings.py    -- chunk embedding generation
    vector_store.py  -- Chroma vector-store construction

These modules *produce* indexes. Query-time consumption lives in
`src/retrieval/` -- there is deliberately no import from this package to that
one, nor the reverse.
"""

from src.knowledge.indexing.bm25 import build_bm25, simple_tokenize
from src.knowledge.indexing.embeddings import build_embeddings
from src.knowledge.indexing.vector_store import create_vector_store, get_collection

__all__ = ["build_bm25", "simple_tokenize", "build_embeddings",
           "create_vector_store", "get_collection"]
