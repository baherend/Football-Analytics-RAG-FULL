"""
02_preprocessing.py — Phase 4: Text Cleaning

Cleans and normalizes document text for embedding and retrieval.
Preserves original text alongside cleaned version.

Input: documents.json
Output: processed_documents.json
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (NFC form)."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space, strip ends."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and special punctuation."""
    replacements = {
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
        ' ': ' ',  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_control_chars(text: str) -> str:
    """Remove control characters except newline and tab."""
    return ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != 'C' or ch in '\n\t'
    )


def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to a document text."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = remove_control_chars(text)
    text = clean_whitespace(text)
    return text


def preprocess_for_tfidf(text: str) -> str:
    """Additional preprocessing for TF-IDF/BM25 (lowercase)."""
    return preprocess_text(text).lower()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess_documents(input_path: Path, output_path: Path) -> dict:
    """
    Preprocess all documents.

    Returns dict with original and cleaned text, plus metadata.
    """
    with open(input_path, encoding="utf-8") as f:
        documents = json.load(f)

    processed = []
    stats = {
        "total": len(documents),
        "by_level": {},
        "avg_original_length": 0,
        "avg_cleaned_length": 0,
    }

    total_orig, total_cleaned = 0, 0

    for doc in documents:
        original_text = doc["text"]
        cleaned_text = preprocess_text(original_text)
        tfidf_text = preprocess_for_tfidf(original_text)

        total_orig += len(original_text)
        total_cleaned += len(cleaned_text)

        level = doc["level"]
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

        processed.append({
            "document_id": doc["document_id"],
            "level": doc["level"],
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "tfidf_text": tfidf_text,
            "metadata": doc["metadata"],
        })

    stats["avg_original_length"] = total_orig / len(documents) if documents else 0
    stats["avg_cleaned_length"] = total_cleaned / len(documents) if documents else 0

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    input_path = Path("output/documents.json")
    output_path = Path("output/processed_documents.json")

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return 1

    print(f"Preprocessing {input_path}...")
    stats = preprocess_documents(input_path, output_path)

    print(f"\nPreprocessing complete:")
    print(f"  Total documents: {stats['total']}")
    print(f"  By level: {stats['by_level']}")
    print(f"  Avg original length: {stats['avg_original_length']:.0f} chars")
    print(f"  Avg cleaned length: {stats['avg_cleaned_length']:.0f} chars")
    print(f"\nOutput: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
