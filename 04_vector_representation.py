"""
04_vector_representation.py — Stage 4: Vector Representation

Builds sparse (BM25/TF-IDF) and dense (sentence embeddings) representations
for hybrid retrieval.

Input: chunks from 03_chunking.py
Output: BM25 index, TF-IDF matrix, sentence embeddings

The hybrid_search() function combines BM25 lexical matching with dense
embedding similarity using min-max normalization and weighted fusion.
"""

from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
ALPHA = 0.6  # weight for dense scores (1-ALPHA for BM25)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

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


def _build_embeddings(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Generate sentence embeddings for all chunks."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c.get("search_text", c["text"]) for c in chunks]
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

# Lazy-loaded globals
_chunks: list[dict] | None = None
_bm25: BM25Okapi | None = None
_chunk_embeddings: np.ndarray | None = None
_model = None


def _ensure_loaded():
    """Load indices lazily (once)."""
    global _chunks, _bm25, _chunk_embeddings, _model
    if _chunks is not None:
        return

    from sentence_transformers import SentenceTransformer

    # Try loading from pre-built chunks.json
    chunks_path = Path("output/chunks.json")
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            _chunks = json.load(f)
    else:
        _chunks = import_module("03_chunking").chunks

    # Try loading pre-built BM25
    import pickle
    bm25_path = Path("output/indices/bm25.pkl")
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            _bm25 = pickle.load(f)
    else:
        _bm25 = _build_bm25(_chunks)

    # Try loading pre-built embeddings
    embeddings_path = Path("output/embeddings/embeddings.npy")
    if embeddings_path.exists():
        _chunk_embeddings = np.load(embeddings_path)
    else:
        _chunk_embeddings = _build_embeddings(_chunks)

    _model = SentenceTransformer(MODEL_NAME)


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    Search using hybrid BM25 + dense embedding fusion.

    Returns top-k chunks with scores.
    """
    _ensure_loaded()

    clean_query = query  # preprocessing is optional for search

    # BM25 scores
    bm25_scores = _bm25.get_scores(simple_tokenize(clean_query))

    # Dense scores
    query_embedding = _model.encode(
        [clean_query], convert_to_numpy=True, normalize_embeddings=True
    )
    embedding_scores = np.dot(_chunk_embeddings, query_embedding.T).flatten()

    # Hybrid fusion
    hybrid_scores = (
        (1 - ALPHA) * min_max_normalize(bm25_scores)
        + ALPHA * min_max_normalize(embedding_scores)
    )

    ranking = np.argsort(hybrid_scores)[::-1][:k]
    return [
        {**_chunks[index], "score": float(hybrid_scores[index])}
        for index in ranking
        if hybrid_scores[index] > 0
    ]


if __name__ == "__main__":
    _ensure_loaded()
    print(f"Loaded {len(_chunks)} chunks")
    print(f"Embedding dim: {_chunk_embeddings.shape[1]}")
    results = hybrid_search("How many goals did Messi score?", k=3)
    for r in results:
        print(f"  [{r.get('level')}] score={r['score']:.4f} - {r['text'][:80]}...")
