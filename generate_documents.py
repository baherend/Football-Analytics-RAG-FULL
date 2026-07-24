"""
generate_documents.py — Phase 2 CLI Entry Point

Reads match_facts.json and produces documents.json via pure rendering.
No statistics computation. No raw event parsing.

Usage:
    python generate_documents.py [--quiet]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from src.rendering.render import render_all, persist


def main() -> int:
    verbose = "--quiet" not in sys.argv
    facts_path = Path("output/match_facts.json")
    output_path = Path("output/documents.json")

    if not facts_path.exists():
        print(f"Error: {facts_path} not found. Run extract.py first.")
        return 1

    print(f"Loading structured facts from {facts_path}")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))

    print(f"Rendering documents...")
    documents = render_all(facts)

    counts = Counter(d.level for d in documents)
    print(f"\nDocument counts by level")
    for level in ("1", "2", "3", "4", "team"):
        label = {"1": "Level 1  Match Summary", "2": "Level 2  Key Events",
                 "3": "Level 3  Player / match", "4": "Level 4  Player / tournament",
                 "team": "Team-level Analysis"}[level]
        print(f"  {label:<32} {counts.get(level, 0)}")
    print(f"  {'TOTAL':<32} {len(documents)}")

    persist(documents, output_path)
    print(f"\nWrote {len(documents)} documents to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
