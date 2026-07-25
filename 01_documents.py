"""
01_documents.py — Stage 1: Document Loading

Loads pre-generated FIFA World Cup 2022 documents from StatsBomb open data.
Documents cover five levels:
    Level 1     Match summaries (64)
    Level 2     Key events per match (64)
    Level 3     Player performance per match
    Level 4     Player tournament aggregates
    Team-level  Team analysis (32)

If documents.json exists in output/, it is loaded directly.
Otherwise, documents are generated from raw StatsBomb data in open-data-master/data/.
"""

from __future__ import annotations

import json
from pathlib import Path


DOCUMENTS_PATH = Path("output/documents.json")


def load_documents() -> list[dict]:
    """Load documents from pre-built JSON file."""
    if DOCUMENTS_PATH.exists():
        with open(DOCUMENTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# Module-level documents list (loaded once at import)
documents = load_documents()


if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents from {DOCUMENTS_PATH}")
    if documents:
        levels = {}
        for d in documents:
            lvl = d.get("level", "unknown")
            levels[lvl] = levels.get(lvl, 0) + 1
        for lvl, count in sorted(levels.items()):
            print(f"  Level {lvl}: {count}")
