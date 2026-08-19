"""
src/query/intent.py -- Query understanding: what the user is asking for.

Migration Step 3 (Query Understanding + Planning Split): mechanically
extracted from src/query/router.py's classification and comparison-detection
sections -- no logic changes. See src/query/router.py for the compatibility
re-exports existing callers keep using.

Understanding answers "what does the user want?" -- intent class, compared
entities, the metric being compared, and the shared structured entity type of
a comparison. It does NOT decide a retrieval strategy (see
src/query/planning.py), build a StructuredQuery (see src/query/parsing.py), or
execute anything (see src/query/router.py).
"""

from __future__ import annotations

import re
from pathlib import Path

# Phase 5 closed the understanding -> retrieval reverse dependency that used to
# sit here (this module imported _detect_team_style_query from
# src.retrieval.search). Team-style detection is pure text classification, not
# retrieval, so it now lives in the neutral shared module src/team_style.py --
# the same flat-shared-module convention as src/stage_taxonomy.py. Both the
# UNDERSTAND stage and src/retrieval/safeguards.py depend downward on it
# instead of sideways on each other.
from src.team_style import _detect_team_style_query
from src.query.resolver import resolve_entity_type
from src.query.vocab import resolve_metric


# ---------------------------------------------------------------------------
# Comparison Detection
# ---------------------------------------------------------------------------

# Comparison detection patterns (router version)
#
# The "X vs Y"/"X versus Y" entity groups are bounded to 60 characters. Left
# unbounded, they combined re.search()'s per-position retry (neither pattern
# has a fixed literal prefix to fast-scan for) with unbounded backtracking,
# giving O(N^2) behavior on any long run of word characters -- measured at a
# consistent 4x per doubling, reaching 16.5s on 20 KB of accented text ("é"
# is a \w character in Python 3, so this was never ASCII-only). That path is
# reachable from user query text on every query via classify_query().
#
# 60 is not arbitrary and is not merely copied from the equivalent fix in
# src/retrieval/safeguards.py::_detect_comparison_entities(): `\w` matches
# neither spaces nor hyphens, so these two patterns only ever captured a
# SINGLE token per side (a multi-word name like "Lionel Andres Messi" was
# never captured whole by them). The longest single `\w` token across all 713
# entity names in the production corpus is 14 characters
# ("Dayotchanculle"), so 60 leaves over 4x headroom for any realistic name
# while matching the bound safeguards.py already uses -- one convention, not
# two. For every token at or under the bound the bounded and unbounded
# patterns match identically; see
# tests/test_query_intent_security.py, which pins both the complexity
# property and the unchanged comparison behavior.
COMPARISON_PATTERNS = [
    r"compare\s+(.+?)\s+and\s+(.+?)(?:\s+in\b|\s+by\b|\s*$|\s*\?)",
    r"who\s+(?:performed|played|did)\s+better[,.]?\s*(.+?)\s+or\s+(.+?)(?:\s*$|\s*\?)",
    r"(\w{1,60})\s+vs\.?\s+(\w{1,60})",
    r"(\w{1,60})\s+versus\s+(\w{1,60})",
    r"who\s+(?:is|was)\s+better[,.]?\s*(.+?)\s+or\s+(.+?)(?:\s*$|\s*\?)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*$|\s*\?)",
    # "Who <verb> more <metric>, A or B?" -- e.g. "Who scored more goals,
    # Harry Kane or Jamie Vardy?". The metric clause is not captured; only
    # entity extraction is handled here.
    r"who\s+\w+\s+more\s+[\w\s]+?,\s*(.+?)\s+or\s+(.+?)(?:\s*$|\s*\?)",
]


def _detect_comparison(query: str) -> list[str]:
    """Detect if a query is comparing two entities. Returns entity names."""
    query_lower = query.lower().strip()
    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            entities = []
            for group in match.groups():
                if group:
                    entity = group.strip().rstrip("'s")
                    for suffix in ["'s tournament performance", "'s performance",
                                   "'s stats", "'s goals"]:
                        if entity.endswith(suffix):
                            entity = entity[:-len(suffix)]
                    # Strip trailing metric clauses: "in goals, assists, ..."
                    in_match = re.match(r"^(.+?)\s+in\s+[\w\s,]+$", entity)
                    if in_match:
                        entity = in_match.group(1)
                    if entity and len(entity) > 1:
                        entities.append(entity.title())
            if len(entities) >= 2:
                return entities[:2]
    return []


def _detect_comparison_metric(query: str) -> str:
    """
    Best-effort extraction of the metric a comparison query is asking
    about, reusing the existing metric vocabulary/resolver (no separate
    comparison-specific metric vocabulary).

    Two distinct cases, mirroring parse_structured_query()'s existing
    unknown-metric contract as closely as possible:

    - No metric phrase is mentioned at all (e.g. "Compare Messi and
      Mbappé's performance") -- falls back to "goals", the long-standing
      default for genuinely metric-less comparisons.
    - A metric phrase IS mentioned -- "who <verb> more <metric>, A or B"
      or "compare A and B by <metric>" -- but doesn't resolve (e.g.
      "corners") -- the raw, unresolved phrase is returned unchanged,
      exactly like parse_structured_query()'s "how many <metric> did
      <player> have" pattern already does when resolve_metric() fails.
      structured_resolve() then rejects it via validate_query(),
      producing status="empty" for that entity -- it must never be
      silently swapped for "goals".
    """
    query_lower = query.lower()
    match = (
        re.search(r"who\s+\w+\s+more\s+([\w\s]+?),", query_lower)
        or re.search(r"\bby\s+([\w\s]+?)[.?]?\s*$", query_lower)
    )
    if not match:
        return "goals"

    phrase = match.group(1).strip()
    return resolve_metric(phrase) or phrase


def _resolve_comparison_entity_type(
    entity_names: list[str],
    data_path: Path | None,
) -> str:
    """
    Determine the shared structured entity type ("player" or "team") for a
    two-entity comparison, reusing resolve_entity_type() (src/query/
    resolver.py) -- no separate team/player-name matching heuristic.

    Uses whichever type is unambiguously known when only one entity
    resolves (e.g. "Argentina vs Nonexististan": Argentina is a known
    team, the other name is neither a known player nor team -- treat both
    as "team" so Argentina resolves for real and the unknown side
    legitimately resolves empty, producing a "partial" comparison rather
    than failing both sides as players).

    Falls back to "player" -- the long-standing prior behavior -- when
    neither entity resolves to any known type at all (preserves existing
    behavior for names that don't exist in this dataset, which the
    structured resolver already reports as "no player found") and when
    the two entities resolve to genuinely DIFFERENT types (a real
    player-vs-team pair). In the mixed case this never silently coerces
    the comparison into one guessed type -- it simply falls through to
    the same "player" default an unresolvable pair already used before
    team support existed, so no fabricated same-type comparison is ever
    produced.
    """
    types = [resolve_entity_type(name, data_path=data_path) for name in entity_names]
    known_types = {t for t in types if t is not None}
    if len(known_types) == 1:
        return known_types.pop()
    return "player"


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

STRUCTURED_PATTERNS = [
    r"how\s+many\s+(\w+)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
    r"how\s+many\s+([\w\s]+?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
    r"how\s+many\s+(?:total\s+)?(\w+)\s+(?:were|was|are|is)?\s*(?:scored|made|conceded)?\s*(?:in|at|during)\s+(?:the\s+)?(?:tournament|world\s+cup|competition)",
    r"who\s+(?:has\s+the|had\s+the|scored\s+the|got\s+the)?\s*(most|highest|best|least|lowest|fewest)\s+(\w+)",
    r"who\s+(?:was|is|were|are)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+([\w\s]+?)(?:\s+player|\?|$)",
    r"which\s+(team|player)\s+(?:has|had|scored|got)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+(\w+)",
    r"(?:top|best|leading)\s+(scorer|goal\s*scorer|assists?|passer|tackler)",
    # Direct fouls/cards patterns
    r"(?:most|highest|fewest|least)\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?|cards?)",
    r"(?:which|who)\s+player\s+fouls?\s+(?:the\s+)?(?:most|least|often)",
    r"who\s+fouls?\s+(?:the\s+)?(?:most|least|often)",
    r"who\s+(?:has|had|committed|got)\s+(?:the\s+)?(?:most|highest|fewest|least)\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?|cards?)",
    r"who\s+(?:was|is)\s+(?:the\s+)?dirtiest\s+player",
    r"how\s+many\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+(?:get|have|commit|receive)|\s*$|\s*\?)",
    # Aggression-proxy patterns (route structured, but answer MUST use proxy disclaimer)
    r"(?:most|who\s+(?:was|is)\s+(?:the\s+)?)?\s*aggress(?:ive|ion)\s*(?:player)?",
    r"who\s+(?:plays?|played)\s+(?:the\s+)?(?:most\s+)?aggressively",
    r"(?:most|who\s+(?:was|is)\s+(?:the\s+)?)?\s*toug(?:h|est)\s*(?:player)?",
    r"who\s+fouls?\s+(?:a\s+lot|much|often|frequently)",
    # Goalkeeper patterns
    r"(?:most|highest|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*(?:saved?|saves?))",
    r"who\s+(?:has|had|made|got)\s+(?:the\s+)?(?:most|highest|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*saved?)",
    r"how\s+many\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*(?:saved?|saves?))\s+(?:did|does|has|have)\s+(.+?)(?:\s+(?:make|have|get|concede|save)|\s*$|\s*\?)",
    r"(?:which|who)\s+(?:goal\s*)?keeper\s+(?:has|had|made|got)\s+(?:the\s+)?(?:most|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?)",
    r"(?:clean\s*sheets?|shut\s*outs?)",
    r"who\s+(?:has|had)\s+(?:the\s+)?(?:most|fewest)\s+clean\s*sheets?",
    r"what\s+(?:is|was|are|were)\s+(.+?)(?:'s|'s)?\s+([\w\s]+?)(?:\s*$|\s*\?)",
    r"^(.+?)\s+(goals|assists|xg|shots|passes|minutes|tackles|interceptions)$",
]

SEMANTIC_PATTERNS = [
    r"how\s+did\s+(.+?)\s+play",
    r"tell\s+me\s+about\s+(.+)",
    r"describe\s+(.+)",
    r"what\s+happened\s+in\s+(.+)",
    r"explain\s+(.+)",
    r"compare\s+(.+?)\s+and\s+(.+)",
]

STRUCTURED_KEYWORDS = {
    "most", "highest", "best", "least", "lowest", "fewest", "top", "bottom",
    "average", "total", "sum", "count", "how many", "how much",
    "goals", "assists", "xg", "shots", "passes", "minutes", "tackles",
    "interceptions", "clearances", "pressures", "carries",
    "fouls", "foul", "yellow card", "yellow cards", "red card", "red cards",
    "bookings", "booked", "cards", "dirtiest", "aggressive", "aggression",
    "toughest", "tough",
    "saves", "save", "conceded", "claims", "punches", "penalty saves",
    "clean sheet", "clean sheets", "keeper", "goalkeeper",
}

SEMANTIC_KEYWORDS = {
    "how", "why", "explain", "describe", "tell me", "what happened",
    "play", "performance", "style", "strategy", "tactics", "formation",
    "compare", "difference", "similar", "better", "worse",
}


def classify_query(query: str) -> tuple[str, float]:
    """Classify a query as "structured", "semantic", or "hybrid"."""
    query_lower = query.lower().strip()

    if _detect_comparison(query):
        return "hybrid", 0.9

    # Team playing-style questions are qualitative. Route them through the
    # semantic path so hybrid retrieval and the team-style document safeguard
    # are executed instead of the broad numeric parser.
    if _detect_team_style_query(query):
        return "semantic", 0.9

    for pattern in STRUCTURED_PATTERNS:
        if re.search(pattern, query_lower):
            return "structured", 0.9

    for pattern in SEMANTIC_PATTERNS:
        if re.search(pattern, query_lower):
            return "semantic", 0.9

    structured_score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in query_lower)
    semantic_score = sum(1 for kw in SEMANTIC_KEYWORDS if kw in query_lower)

    total = structured_score + semantic_score
    if total == 0:
        return "semantic", 0.5

    structured_pct = structured_score / total
    semantic_pct = semantic_score / total

    if structured_pct > 0.7:
        return "structured", structured_pct
    elif semantic_pct > 0.7:
        return "semantic", semantic_pct
    else:
        return "hybrid", 0.6
