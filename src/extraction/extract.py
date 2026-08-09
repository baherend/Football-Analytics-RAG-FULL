"""
extract.py — Phase 1 CLI Entry Point

Extracts structured facts for a requested competition/season and persists them
as match_facts.json — the single source of truth for all downstream phases.

The WC2022 default (competition_id=43, season_id=106) persists to the
legacy flat output/ layout; any other competition/season persists to its
own namespaced output/competitions/<id>/<id>/ directory (see
src/artifacts.py's resolve_output_dir()). Pass --namespaced to explicitly
persist a WC2022 build to output/competitions/43/106/ instead of the
legacy flat layout.

Usage:
    python extract.py [--competition-id ID] [--season-id ID] [--namespaced] [--quiet]
"""

from __future__ import annotations

import argparse

from src.artifacts import resolve_output_dir
from src.extraction.match_facts import (
    COMPETITION_ID,
    DATA_ROOT,
    SEASON_ID,
    extract_all,
    persist,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    print(f"Extracting structured facts from {DATA_ROOT.resolve()}")

    result = extract_all(
        DATA_ROOT,
        verbose=verbose,
        competition_id=args.competition_id,
        season_id=args.season_id,
    )
    diag = result["diagnostics"]

    print("\nExtraction complete:")
    print(f"  Matches processed:     {diag['matches_processed']}")
    print(f"  Player-match facts:    {diag['total_player_facts']}")
    print(f"  Match facts:           {len(result['match_facts'])}")
    print(f"  Team-match facts:      {diag['total_team_facts']}")
    print(f"  Card parity failures:  {diag['card_count_mismatches'] or 'none'}")

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    output_path = persist(
        result,
        output_dir=output_dir,
        competition_id=args.competition_id,
        season_id=args.season_id,
        dataset_identity=result.get("dataset_identity"),
    )
    print(f"\nPersisted to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
