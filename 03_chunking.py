"""
03_chunking.py — Stage 3: Document Chunking

Splits documents into chunks for better retrieval.
Uses sentence-based chunking with overlap.

Input: documents from 01_documents.py
Output: list of chunk dicts with metadata
"""

from __future__ import annotations

import re
from importlib import import_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP = 50     # characters
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_document(doc: dict, max_size: int = MAX_CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split a document into chunks."""
    text = doc.get("cleaned_text") or doc.get("text", "")
    doc_id = doc["document_id"]
    level = doc.get("level", "unknown")

    # Short document — single chunk
    if len(text) <= max_size:
        return [{
            "chunk_id": f"{doc_id}-chunk-0",
            "document_id": doc_id,
            "level": level,
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": text,
            "search_text": text,
            "metadata": doc.get("metadata", {}),
        }]

    sentences = split_sentences(text)
    if not sentences:
        return [{
            "chunk_id": f"{doc_id}-chunk-0",
            "document_id": doc_id,
            "level": level,
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": text,
            "search_text": text,
            "metadata": doc.get("metadata", {}),
        }]

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
                "document_id": doc_id,
                "level": level,
                "match_id": doc.get("match_id"),
                "player_name": doc.get("player_name"),
                "team_name": doc.get("team_name"),
                "text": chunk_text,
                "search_text": chunk_text,
                "metadata": doc.get("metadata", {}),
            })
            chunk_idx += 1

            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
            "document_id": doc_id,
            "level": level,
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": chunk_text,
            "search_text": chunk_text,
            "metadata": doc.get("metadata", {}),
        })

    return chunks


def build_chunks(documents: list[dict] | None = None) -> list[dict]:
    """Build chunks from all documents."""
    if documents is None:
        documents = import_module("01_documents").documents

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# Module-level chunks
chunks = build_chunks()


if __name__ == "__main__":
    print(f"Built {len(chunks)} chunks from {len(import_module('01_documents').documents)} documents")
