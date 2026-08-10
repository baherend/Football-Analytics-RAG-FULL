"""
03_chunking.py — Stage 3: Document Chunking

Splits documents into chunks for better retrieval.
Uses sentence-based chunking with overlap.

Input: documents from 01_documents.py (build_chunks() as a library call)
Output: list of chunk dicts with metadata

Run directly, reads the selected competition/season's processed_documents
(falling back to documents.json if preprocessing hasn't run) and writes
its chunks.json (see src/artifacts.py's resolve_output_dir()). The WC2022
default (competition_id=43, season_id=106) reads/writes the legacy flat
output/ layout; any other competition/season uses its own namespaced
output/competitions/<id>/<id>/ directory. Pass --namespaced to explicitly
use the namespaced layout for a WC2022 build.

Usage:
    python 03_chunking.py [--competition-id ID] [--season-id ID] [--namespaced] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import re
from importlib import import_module
from pathlib import Path

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


def _make_chunk(doc: dict, chunk_idx: int, chunk_text: str) -> dict:
    """
    Build a single chunk dict for a document.

    This is the one place that defines what a "chunk" looks like — every
    call site below (short doc, no-sentences fallback, mid-loop, final
    leftover) goes through here instead of repeating the same dict literal.
    """
    doc_id = doc["document_id"]
    return {
        "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
        "document_id": doc_id,
        "level": doc.get("level", "unknown"),
        "match_id": doc.get("match_id"),
        "player_name": doc.get("player_name"),
        "team_name": doc.get("team_name"),
        "text": chunk_text,
        "search_text": chunk_text,
        "metadata": doc.get("metadata", {}),
    }


def chunk_document(doc: dict, max_size: int = MAX_CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split a document into chunks."""
    text = doc.get("cleaned_text") or doc.get("text", "")

    # Short document, or no sentence boundaries found — single chunk.
    sentences = split_sentences(text) if len(text) > max_size else []
    if len(text) <= max_size or not sentences:
        return [_make_chunk(doc, 0, text)]

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(_make_chunk(doc, chunk_idx, chunk_text))
            chunk_idx += 1

            if overlap > 0:
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
        chunks.append(_make_chunk(doc, chunk_idx, chunk_text))

    return chunks


def build_chunks(documents: list[dict] | None = None) -> list[dict]:
    """Build chunks from all documents."""
    if documents is None:
        documents = import_module("01_documents").documents

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# Module-level chunks for legacy library-import behavior.
# Avoid probing 01_documents.py when the legacy flat artifact does not exist;
# namespaced CLI builds load their selected dataset explicitly inside main().
chunks = build_chunks() if Path("output/documents.json").exists() else []


def main() -> int:
    from src.artifacts import resolve_output_dir
    from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    processed_path = output_dir / "processed_documents.json"
    documents_path = output_dir / "documents.json"

    # Prefer processed (cleaned) text; fall back to raw documents from the
    # SAME namespaced directory if preprocessing hasn't run yet -- never
    # crosses into another competition's or the legacy directory.
    if processed_path.exists():
        source_path = processed_path
    elif documents_path.exists():
        source_path = documents_path
    else:
        print(f"Error: neither {processed_path} nor {documents_path} found. "
              "Run generate_documents.py (and optionally 02_preprocessing.py) first.")
        return 1

    if not args.quiet:
        print(f"Loading documents from {source_path}")
    documents = json.loads(source_path.read_text(encoding="utf-8"))

    built_chunks = build_chunks(documents)

    output_path = output_dir / "chunks.json"
    output_path.write_text(json.dumps(built_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"Built {len(built_chunks)} chunks from {len(documents)} documents")
        print(f"Wrote chunks to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())