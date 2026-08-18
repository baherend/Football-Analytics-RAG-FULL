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

# Migration Step 6: the chunking logic moved to src/knowledge/chunking.py.
# This script keeps its CLI (argument parsing, competition/season path
# resolution, artifact I/O) and re-exports the moved symbols so existing
# importers and tests are unaffected.
from src.knowledge.chunking import (  # noqa: F401  (compatibility re-exports)
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    SENTENCE_END,
    chunk_document,
    split_sentences,
)
from src.knowledge.chunking import build_chunks as _build_chunks


def build_chunks(documents: list[dict] | None = None) -> list[dict]:
    """Build chunks from all documents.

    Thin wrapper over src.knowledge.chunking.build_chunks(). The `None`
    default is preserved here (and NOT in the knowledge module) because it
    reaches back into 01_documents.py, whose import-time filesystem read uses
    the hardcoded legacy output/ path -- legacy behavior this script's
    existing callers still rely on. See PROJECT_MEMORY.md's deferred debt.
    """
    if documents is None:
        documents = import_module("01_documents").documents
    return _build_chunks(documents)


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