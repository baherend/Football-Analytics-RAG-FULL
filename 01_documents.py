"""
01_documents.py — Stage 1: Document Loading

Thin wrapper that loads the pre-built FIFA World Cup 2022 documents produced
by the extraction/rendering pipeline (src/extraction/ + src/rendering/).
Documents cover five levels:
    Level 1     Match summaries (64)
    Level 2     Key events per match (64)
    Level 3     Player performance per match
    Level 4     Player tournament aggregates
    Team-level  Team analysis (32)

This module does NOT regenerate documents from raw StatsBomb data itself —
that logic lives in src/extraction/extract.py and src/rendering/render.py,
and is covered by the regression tests in tests/test_extraction.py. If
output/documents.json is missing, run that pipeline first (see README) rather
than expecting this file to rebuild it.
"""

from __future__ import annotations

import json
from pathlib import Path

DOCUMENTS_PATH = Path("output/documents.json")


def load_documents() -> list[dict]:
    """
    Load documents from the pre-built JSON file.

    Returns an empty list (with a clear warning) if the file doesn't exist
    yet, rather than silently pretending zero documents is a normal state.
    """
    if not DOCUMENTS_PATH.exists():
        print(
            f"[01_documents] WARNING: {DOCUMENTS_PATH} not found. "
            f"Run the extraction pipeline (src/extraction/extract.py -> "
            f"src/rendering/render.py) to generate it first."
        )
        return []

    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# Module-level documents list (loaded once at import)
documents = load_documents()


if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents from {DOCUMENTS_PATH}")
    if documents:
        levels = {}
        for d in documents:
            lvl = d.get("level", "unknown")
            levels[lvl] = levels.get(lvl, 0) + 1
        for lvl, count in sorted(levels.items(), key=str):
            print(f"  Level {lvl}: {count}")