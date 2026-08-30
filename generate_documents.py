"""
generate_documents.py — Phase 2 CLI Entry Point

Reads match_facts.json and produces documents.json via pure rendering.
No statistics computation. No raw event parsing.

The WC2022 default (competition_id=43, season_id=106) reads/writes the
legacy flat output/ layout; any other competition/season reads/writes its
own namespaced output/competitions/<id>/<id>/ directory (see
src/artifacts.py's resolve_output_dir()).

Pass --namespaced to explicitly read/write a WC2022 build at
output/competitions/43/106/ instead of the legacy flat layout.

Usage:
    python generate_documents.py [--competition-id ID] [--season-id ID] [--namespaced] [--quiet]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from src.artifacts import resolve_output_dir
from src.extraction.match_facts import COMPETITION_ID, SEASON_ID
from src.rendering.render import render_all, persist
from src.knowledge.enrichment.retrieval_enrichment import enrich_documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    facts_path = output_dir / "match_facts.json"
    output_path = output_dir / "documents.json"

    if not facts_path.exists():
        print(f"Error: {facts_path} not found. Run extract.py first.")
        return 1

    print(f"Loading structured facts from {facts_path}")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))

    print(f"Rendering documents...")
    documents = render_all(facts)
    documents = enrich_documents(documents)

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
