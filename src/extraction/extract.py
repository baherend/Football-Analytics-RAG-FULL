"""
extract.py — Phase 1 CLI Entry Point

Extracts structured facts from all 64 WC 2022 matches and persists them
as match_facts.json — the single source of truth for all downstream phases.

Usage:
    python extract.py [--quiet]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.extraction.match_facts import extract_all, persist, DATA_ROOT


def main() -> int:
    verbose = "--quiet" not in sys.argv
    print(f"Extracting structured facts from {DATA_ROOT.resolve()}")

    result = extract_all(DATA_ROOT, verbose=verbose)
    diag = result["diagnostics"]

    print(f"\nExtraction complete:")
    print(f"  Matches processed:     {diag['matches_processed']}")
    print(f"  Player-match facts:    {diag['total_player_facts']}")
    print(f"  Match facts:           {len(result['match_facts'])}")
    print(f"  Team-match facts:      {diag['total_team_facts']}")
    print(f"  Card parity failures:  {diag['card_count_mismatches'] or 'none'}")

    output_path = persist(result)
    print(f"\nPersisted to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
