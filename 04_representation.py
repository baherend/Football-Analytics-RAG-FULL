"""
04_representation.py — Phase 4: TF-IDF / BM25 Representation

Builds sparse representations for keyword-based retrieval.

Input: chunks.json
Output: tfidf_index.pkl, bm25_index.pkl, vocabulary.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    # Split on whitespace and punctuation, keep alphanumeric
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 1]  # skip single chars


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_tfidf_index(chunks: list[dict]) -> tuple:
    """
    Build TF-IDF index from chunks.

    Returns (vectorizer, tfidf_matrix, chunk_ids).
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix, chunk_ids


def build_bm25_index(chunks: list[dict]) -> tuple:
    """
    Build BM25 index from chunks.

    Returns (bm25, chunk_ids, tokenized_corpus).
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    tokenized_corpus = [simple_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, chunk_ids, tokenized_corpus


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def tfidf_search(query: str, vectorizer, tfidf_matrix, chunks: list[dict],
                 k: int = 5) -> list[dict]:
    """Search using TF-IDF similarity."""
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            })
    return results


def bm25_search(query: str, bm25, chunks: list[dict],
                k: int = 5) -> list[dict]:
    """Search using BM25."""
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            })
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_indices(vectorizer, tfidf_matrix, bm25, chunk_ids, output_dir: Path):
    """Save indices to disk."""
    output_dir.mkdir(exist_ok=True)

    # Save TF-IDF
    with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # Save BM25
    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Save chunk IDs
    with open(output_dir / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    print(f"Saved indices to {output_dir}")


def load_indices(output_dir: Path) -> tuple:
    """Load indices from disk."""
    with open(output_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(output_dir / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(output_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(output_dir / "chunk_ids.json") as f:
        chunk_ids = json.load(f)

    return vectorizer, tfidf_matrix, bm25, chunk_ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_representations(input_path: Path, output_dir: Path) -> dict:
    """Build and save TF-IDF and BM25 indices."""
    with open(input_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Building TF-IDF index for {len(chunks)} chunks...")
    vectorizer, tfidf_matrix, chunk_ids = build_tfidf_index(chunks)

    print(f"Building BM25 index...")
    bm25, _, _ = build_bm25_index(chunks)

    save_indices(vectorizer, tfidf_matrix, bm25, chunk_ids, output_dir)

    return {
        "total_chunks": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "tfidf_shape": tfidf_matrix.shape,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    input_path = Path("output/chunks.json")
    output_dir = Path("output/indices")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 03_chunking.py first.")
        return 1

    print(f"Building representations from {input_path}...")
    stats = build_representations(input_path, output_dir)

    print(f"\nRepresentation building complete:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  TF-IDF shape: {stats['tfidf_shape']}")
    print(f"\nOutput: {output_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
