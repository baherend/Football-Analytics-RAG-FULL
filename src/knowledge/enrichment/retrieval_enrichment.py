"""
src/knowledge/enrichment/retrieval_enrichment.py

Deterministic retrieval enrichment for rendered football documents.

This layer:
- consumes document objects without knowing rendering/retrieval details;
- adds retrieval-only metadata;
- never changes factual evidence text or original metadata fields.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.1"


def _build_match_enrichment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Build match retrieval patterns from structured match metadata only."""

    required_fields = (
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )

    if not all(field in metadata for field in required_fields):
        return None

    home = metadata.get("home_team")
    away = metadata.get("away_team")
    home_score = metadata.get("home_score")
    away_score = metadata.get("away_score")

    if (
        not home
        or not away
        or not isinstance(home_score, int)
        or not isinstance(away_score, int)
    ):
        return None

    score = f"{home_score}-{away_score}"

    if home_score == away_score:
        result_type = "draw"
        result_patterns = [
            f"{home} draw {away}",
            f"{home} {away} draw",
            f"{home} drew {away}",
            f"{home} and {away} shared points",
            f"{home} and {away} ended level",
            f"{home} vs {away} ended in a draw",
        ]
    elif home_score > away_score:
        result_type = "home_win"
        result_patterns = [
            f"{home} win {away}",
            f"{home} beat {away}",
            f"{home} defeated {away}",
            f"{home} won against {away}",
        ]
    else:
        result_type = "away_win"
        result_patterns = [
            f"{away} win {home}",
            f"{away} beat {home}",
            f"{away} defeated {home}",
            f"{away} won against {home}",
            f"{home} lost to {away}",
        ]

    return {
        "canonical_pair": f"{home} vs {away}",
        "score": score,
        "result_type": result_type,
        "score_semantics": {
            "home_goals": home_score,
            "away_goals": away_score,
            "is_draw": home_score == away_score,
            "has_extra_time": bool(metadata.get("went_to_extra_time", False)),
            "has_shootout": bool(metadata.get("went_to_shootout", False)),
            "score_scope": (
                "shootout" if metadata.get("went_to_shootout", False)
                else "extra_time" if metadata.get("went_to_extra_time", False)
                else "normal_time"
            ),
        },
        "search_patterns": {
            "score": [
                f"{home} {score} {away}",
            ],
            "result": result_patterns,
            "matchup": [
                f"{home} vs {away}",
                f"{home} {away} match",
            ],
        },
    }


def enrich_documents(documents: list[Any]) -> list[Any]:
    """
    Add retrieval enrichment metadata to rendered documents.

    The enrichment is additive and idempotent:
    running it multiple times produces the same metadata.
    """
    for document in documents:
        metadata = getattr(document, "metadata", None)

        if not isinstance(metadata, dict):
            continue

        enrichment = {
            "schema_version": SCHEMA_VERSION,
        }

        match_enrichment = _build_match_enrichment(metadata)
        if match_enrichment:
            enrichment["match"] = match_enrichment

        metadata["retrieval_enrichment"] = enrichment

    return documents


