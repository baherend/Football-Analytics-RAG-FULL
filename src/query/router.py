"""
src/query/router.py -- Query Router: classification, structured-query
parsing, route selection, and route execution.

Structural Cleanup Phase A: mechanically extracted from
06_retrieve_context.py's router half (everything from its former "Query
Router" section onward, excluding the CLI entry point which stays in
06_retrieve_context.py) -- no logic changes. See 06_retrieve_context.py
for the temporary compatibility wrapper that re-exports these symbols for
existing callers.

Responsibilities: query classification, structured-query parsing, route
selection, comparison detection/metric/entity-type resolution for
structured comparison routing, and route execution -- dispatching between
structured resolution (src.query.resolver) and retrieval
(src.retrieval.search). Does NOT implement BM25/dense/hybrid search itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import json

from src.artifacts import ArtifactPaths
from src.retrieval.answerability import (
    AnswerabilityAssessment,
    assess_answerability,
)
from src.retrieval.search import (
    _detect_team_style_query,
    build_context,
    hybrid_search,
)
from src.query.query_schema import (
    StructuredQuery, StructuredResult, Filter, ComparisonResult, ComparisonValue,
)
from src.query.resolver import resolve as structured_resolve
from src.query.resolver import resolve_entity_type
from src.query.vocab import resolve_metric, resolve_aggregation, METRIC_SYNONYMS, AGGREGATION_SYNONYMS
from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY
from src.extraction.match_facts import WC2022_DATASET_IDENTITY


# Stage extraction patterns
STAGE_PATTERNS = [
    (r"in\s+the\s+(?:semi[\s-]*final)", "Semi-finals"),
    (r"in\s+the\s+(?:quarter[\s-]*final)", "Quarter-finals"),
    (r"in\s+the\s+(?:round\s+of\s+16)", "Round of 16"),
    (r"in\s+the\s+(?:group\s+stage)", "Group Stage"),
    (r"in\s+the\s+(?:final)", "Final"),
    (r"in\s+the\s+(?:3rd\s+place)", "3rd Place Final"),
    (r"in\s+knockout", None),
    (r"in\s+the\s+knockout", None),
    (r"during\s+the\s+(?:semi[\s-]*final)", "Semi-finals"),
    (r"during\s+the\s+(?:quarter[\s-]*final)", "Quarter-finals"),
    (r"during\s+the\s+(?:final)", "Final"),
]


def _extract_stage_filter(
    query: str,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
) -> Filter | None:
    """Extract a stage filter using the active dataset taxonomy."""
    query_lower = query.lower().strip()

    # "Knockout" is semantic intent backed by the universal is_knockout field,
    # not a literal competition-stage name.
    if re.search(r"\bin\s+(?:the\s+)?knockout\b", query_lower):
        return Filter("is_knockout", "eq", True)

    # Active competition vocabulary: literal, case-insensitive stage names.
    # Longest first prevents a shorter stage name from shadowing a longer one.
    for stage_name in sorted(stage_taxonomy.stages, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(stage_name.lower())}(?!\w)"
        if re.search(pattern, query_lower):
            return Filter("stage", "eq", stage_name)

    # Fixed aliases are legacy WC2022 compatibility only. They must never
    # silently impose WC2022 vocabulary on another competition.
    if stage_taxonomy == WC2022_STAGE_TAXONOMY:
        for pattern, stage_value in STAGE_PATTERNS:
            if re.search(pattern, query_lower):
                if stage_value is None:
                    return Filter("is_knockout", "eq", True)
                return Filter("stage", "eq", stage_value)

    return None


# Opponent extraction patterns
OPPONENT_PATTERNS = [
    r"against\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
    r"vs\.?\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
    r"versus\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
]


def _extract_opponent_filter(query: str) -> Filter | None:
    """Extract opponent filter from query text."""
    query_lower = query.lower().strip()
    for pattern in OPPONENT_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            opponent = match.group(1).strip()
            for word in ["the", "a", "an"]:
                if opponent.startswith(word + " "):
                    opponent = opponent[len(word) + 1:]
            if opponent and len(opponent) > 1:
                return Filter("opponent", "eq", opponent.title())
    return None


# Comparison detection patterns (router version)
COMPARISON_PATTERNS = [
    r"compare\s+(.+?)\s+and\s+(.+?)(?:\s+in\b|\s+by\b|\s*$|\s*\?)",
    r"who\s+(?:performed|played|did)\s+better[,.]?\s*(.+?)\s+or\s+(.+?)(?:\s*$|\s*\?)",
    r"(\w+)\s+vs\.?\s+(\w+)",
    r"(\w+)\s+versus\s+(\w+)",
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


# Route types
@dataclass
class Route:
    """Routing decision."""
    path: str  # "semantic" | "structured" | "hybrid"
    confidence: float
    reason: str
    structured_query: StructuredQuery | None = None
    semantic_query: str | None = None


@dataclass
class RoutedResult:
    """Result from routed execution."""
    route: Route
    structured_result: StructuredResult | None = None
    semantic_chunks: list[dict] | None = None
    answerability: AnswerabilityAssessment | None = None
    context: str = ""
    explanation: str = ""


# Query classification
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


def parse_structured_query(
    query: str,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
) -> StructuredQuery | None:
    """Parse a query into a StructuredQuery using the active stage vocabulary."""
    query_lower = query.lower().strip()

    stage_filter = _extract_stage_filter(query, stage_taxonomy=stage_taxonomy)
    opponent_filter = _extract_opponent_filter(query)
    filters = [f for f in [stage_filter, opponent_filter] if f is not None]

    # Pattern: "how many <metric> did <player> score/have"
    match = re.search(
        r"how\s+many\s+(\w+)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)
        else:
            return StructuredQuery(intent="numeric", entity="player", metric=metric_raw,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    # Pattern: "how many <multi-word metric> did <player> score/have"
    match = re.search(
        r"how\s+many\s+([\w\s]+?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)
        else:
            return StructuredQuery(intent="numeric", entity="player", metric=metric_raw.strip(),
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    # Pattern: "most aggressive player" / "toughest player" / "who fouls a lot"
    # MUST be checked BEFORE "who was the most <metric>" to avoid matching
    # "aggressive" as an unknown metric.
    # EXCLUDES queries with temporal/score-state qualifiers (e.g. "after Morocco's
    # first goal") — those can't be answered from structured data and must fall
    # through to semantic (honest refusal).
    _has_temporal = re.search(r"\b(after|before|during|when|following|since)\b", query_lower)
    if not _has_temporal and (
        re.search(r"aggress(?:ive|ion)", query_lower) or
        re.search(r"toug(?:h|est)\s*(?:player)?", query_lower) or
        re.search(r"who\s+fouls?\s+(?:a\s+lot|much|often|frequently)", query_lower)
    ):
        return StructuredQuery(intent="superlative", entity="player", metric="fouls_committed",
                               aggregation="max", limit=1, filters=filters)

    # Pattern: "who scored the most <metric>"
    match = re.search(
        r"who\s+(?:has\s+the|had\s+the|scored\s+the|got\s+the)?\s*(most|highest|best|least|lowest|fewest)\s+(\w+)",
        query_lower
    )
    if match:
        agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        agg = resolve_aggregation(agg_raw)
        if metric and agg:
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation=agg, limit=1, filters=filters)

    # Pattern: "who was the most <metric> player"
    match = re.search(
        r"who\s+(?:was|is|were|are)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+([\w\s]+?)(?:\s+player|\?|$)",
        query_lower
    )
    if match:
        agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        agg = resolve_aggregation(agg_raw)
        if metric and agg:
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation=agg, limit=1, filters=filters)
        else:
            return StructuredQuery(intent="superlative", entity="player",
                                   metric=metric_raw.strip() if metric_raw else "unknown",
                                   aggregation=agg or "max", limit=1, filters=filters)

    # Pattern: "which team had the highest <metric>"
    match = re.search(
        r"which\s+(team|player)\s+(?:has|had|scored|got|relied|used|played)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+(?:on\s+)?(\w+)",
        query_lower
    )
    if match:
        entity_raw, agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        agg = resolve_aggregation(agg_raw)
        entity = "team" if entity_raw == "team" else "player"
        if metric and agg:
            return StructuredQuery(intent="superlative", entity=entity, metric=metric,
                                   aggregation=agg, limit=1, filters=filters)

    # Pattern: "how many total <metric> in the tournament"
    match = re.search(
        r"how\s+many\s+(?:total\s+)?(\w+)\s+(?:were|was|are|is)?\s*(?:scored|made|conceded)?\s*(?:in|at|during)\s+(?:the\s+)?(?:tournament|world\s+cup|competition)",
        query_lower
    )
    if match:
        metric_raw = match.group(1)
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="numeric", entity="tournament", metric=metric,
                                   aggregation="sum", filters=filters)

    # Pattern: "top scorer" / "best passer" / "leading scorer"
    match = re.search(
        r"(?:top|best|leading)\s+(scorer|goal\s*scorer|assists?|passer|tackler)",
        query_lower
    )
    if match:
        metric_raw = match.group(1)
        metric_map = {
            "scorer": "goals", "goal scorer": "goals", "goals": "goals",
            "assist": "assists", "assists": "assists",
            "passer": "passes", "tackler": "tackles",
        }
        metric = metric_map.get(metric_raw, resolve_metric(metric_raw) or "goals")
        return StructuredQuery(intent="superlative", entity="player", metric=metric,
                               aggregation="max", limit=1, filters=filters)

    # Pattern: "most fouls" / "most yellow cards" / "most red cards"
    match = re.search(
        r"(?:most|highest|fewest|least)\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?|cards?)",
        query_lower
    )
    if match:
        metric_raw = match.group(1).replace("  ", " ")
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation="max", limit=1, filters=filters)

    # Pattern: "who committed the most fouls" / "who has the most yellow cards"
    match = re.search(
        r"who\s+(?:has|had|committed|got)\s+(?:the\s+)?(?:most|highest|fewest|least)\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?|cards?)",
        query_lower
    )
    if match:
        metric_raw = match.group(1)
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation="max", limit=1, filters=filters)

    # Pattern: "dirtiest player" → fouls_committed as proxy
    if re.search(r"dirtiest\s+player", query_lower):
        return StructuredQuery(intent="superlative", entity="player", metric="fouls_committed",
                               aggregation="max", limit=1, filters=filters)

    # Pattern: "which player fouls the most" / "who fouls the most"
    if re.search(r"(?:which|who)\s+(?:player\s+)?fouls?\s+(?:the\s+)?(?:most|often)", query_lower):
        return StructuredQuery(intent="superlative", entity="player", metric="fouls_committed",
                               aggregation="max", limit=1, filters=filters)

    # Pattern: "how many fouls did <player> have"
    match = re.search(
        r"how\s+many\s+(fouls?|yellow\s*cards?|red\s*cards?|bookings?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+(?:get|have|commit|receive)|\s*$|\s*\?)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    # Pattern: "most saves" / "fewest goals conceded" / "most punches"
    match = re.search(
        r"(?:most|highest|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*(?:saved?|saves?))",
        query_lower
    )
    if match:
        metric_raw = match.group(1).replace("  ", " ")
        metric = resolve_metric(metric_raw)
        if metric:
            agg = "min" if "fewest" in query_lower or "least" in query_lower else "max"
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation=agg, limit=1, filters=filters)

    # Pattern: "who has the most saves" / "who made the most saves"
    match = re.search(
        r"who\s+(?:has|had|made|got)\s+(?:the\s+)?(?:most|highest|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*saved?)",
        query_lower
    )
    if match:
        metric_raw = match.group(1)
        metric = resolve_metric(metric_raw)
        if metric:
            agg = "min" if "fewest" in query_lower or "least" in query_lower else "max"
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation=agg, limit=1, filters=filters)

    # Pattern: "which keeper has the most saves"
    match = re.search(
        r"(?:which|who)\s+(?:goal\s*)?keeper\s+(?:has|had|made|got)\s+(?:the\s+)?(?:most|fewest|least)\s+(saves?|goals?\s*conceded|claims?|punches?)",
        query_lower
    )
    if match:
        metric_raw = match.group(1)
        metric = resolve_metric(metric_raw)
        if metric:
            agg = "min" if "fewest" in query_lower or "least" in query_lower else "max"
            return StructuredQuery(intent="superlative", entity="player", metric=metric,
                                   aggregation=agg, limit=1, filters=filters)

    # Pattern: "who conceded the fewest/most goals"
    match = re.search(
        r"who\s+conceded\s+(?:the\s+)?(most|fewest|least|highest)\s+(goals?)",
        query_lower
    )
    if match:
        agg_raw, metric_raw = match.groups()
        agg = "min" if agg_raw in ("fewest", "least") else "max"
        return StructuredQuery(intent="superlative", entity="player", metric="goals_conceded",
                               aggregation=agg, limit=1, filters=filters)

    # Pattern: "how many saves did <player> make"
    match = re.search(
        r"how\s+many\s+(saves?|goals?\s*conceded|claims?|punches?|penalt(?:y|ies)\s*(?:saved?|saves?))\s+(?:did|does|has|have)\s+(.+?)(?:\s+(?:make|have|get|concede|save)|\s*$|\s*\?)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    # Pattern: "clean sheet" / "most clean sheets"
    if re.search(r"clean\s*sheet", query_lower):
        # Clean sheet = 0 goals conceded in a match. We can't directly query
        # "clean sheets" as a metric, but we can find keepers with fewest goals
        # conceded. Route as goals_conceded min (fewest).
        if re.search(r"most\s+clean\s*sheet", query_lower):
            return StructuredQuery(intent="superlative", entity="player", metric="goals_conceded",
                                   aggregation="min", limit=1, filters=filters)
        # For "did X keep a clean sheet", we'd need per-match filtering —
        # fall through to semantic for now
        return StructuredQuery(intent="superlative", entity="player", metric="goals_conceded",
                               aggregation="min", limit=1, filters=filters)

    # Pattern: "<player> <metric>"
    if match:
        player_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    # Pattern: "what is <player>'s <metric>"
    match = re.search(
        r"what\s+(?:is|was|are|were)\s+(.+?)(?:'s|'s)?\s+([\w\s]+?)(?:\s*$|\s*\?)",
        query_lower
    )
    if match:
        player_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        if metric:
            return StructuredQuery(intent="numeric", entity="player", metric=metric,
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)
        else:
            return StructuredQuery(intent="numeric", entity="player", metric=metric_raw.strip(),
                                   aggregation="sum", entity_name=player_raw.strip().title(),
                                   filters=filters)

    return None


def route_query(
    query: str,
    artifact_paths: ArtifactPaths | None = None,
) -> Route:
    """Determine routing using the selected dataset's stage vocabulary."""
    classification, confidence = classify_query(query)

    if classification == "structured":
        match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None
        stage_taxonomy = _load_active_stage_taxonomy(match_facts_path)
        structured_query = parse_structured_query(query, stage_taxonomy=stage_taxonomy)
        if structured_query:
            return Route(path="structured", confidence=confidence,
                         reason=f"Query matches structured pattern: {structured_query.intent}",
                         structured_query=structured_query)
        return Route(path="semantic", confidence=0.6,
                     reason="Query appears structured but couldn't be parsed or validated",
                     semantic_query=query)

    elif classification == "semantic":
        return Route(path="semantic", confidence=confidence,
                     reason="Query is descriptive/qualitative",
                     semantic_query=query)

    else:  # hybrid
        match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None
        stage_taxonomy = _load_active_stage_taxonomy(match_facts_path)
        structured_query = parse_structured_query(query, stage_taxonomy=stage_taxonomy)
        return Route(path="hybrid", confidence=confidence,
                     reason="Query has both structured and semantic components",
                     structured_query=structured_query,
                     semantic_query=query)


def _load_active_stage_taxonomy(
    match_facts_path: Path | None,
) -> StageTaxonomy:
    """Load the taxonomy persisted with the selected structured dataset."""
    if match_facts_path is None:
        return WC2022_STAGE_TAXONOMY

    with open(match_facts_path, encoding="utf-8") as handle:
        facts = json.load(handle)

    metadata = facts.get("metadata") or {}
    persisted = metadata.get("stage_taxonomy")
    if persisted is not None:
        return StageTaxonomy.from_dict(persisted)

    competition_id = metadata.get("competition_id")
    season_id = metadata.get("season_id")
    if (competition_id, season_id) == (
        WC2022_DATASET_IDENTITY.competition_id,
        WC2022_DATASET_IDENTITY.season_id,
    ):
        return WC2022_STAGE_TAXONOMY

    raise ValueError(
        f"{match_facts_path} is missing persisted stage_taxonomy for "
        "a non-WC2022 dataset; refusing to apply WC2022 stage semantics."
    )


def execute_route(
    route: Route,
    semantic_k: int = 3,
    original_query: str = "",
    artifact_paths: ArtifactPaths | None = None,
) -> RoutedResult:
    """
    Execute a routed query, returning structured and/or semantic results.

    `artifact_paths` selects a namespaced dataset (see src/artifacts.py):
    the structured path resolves against artifact_paths.match_facts and
    every semantic/hybrid retrieval call uses artifact_paths' BM25/chunks/
    Chroma artifacts. Defaults to None -- unchanged legacy WC2022 behavior.
    """
    structured_result = None
    semantic_chunks = None
    context = ""
    match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None
    stage_taxonomy = (
        _load_active_stage_taxonomy(match_facts_path)
        if route.path in ("structured", "hybrid")
        else None
    )

    # For hybrid comparison queries, run structured queries for each entity
    comparison_entities = _detect_comparison(route.semantic_query or "")
    if route.path == "hybrid" and comparison_entities:
        comparison_metric = _detect_comparison_metric(route.semantic_query or "")
        comparison_entity_type = _resolve_comparison_entity_type(
            comparison_entities, data_path=match_facts_path,
        )
        entity_results = []
        for entity_name in comparison_entities:
            sq = StructuredQuery(intent="numeric", entity=comparison_entity_type, metric=comparison_metric,
                                 aggregation="sum", entity_name=entity_name)
            try:
                result = structured_resolve(
                    sq,
                    data_path=match_facts_path,
                    stage_taxonomy=stage_taxonomy,
                )
                entity_results.append((entity_name, result))
            except Exception as e:
                print(f"Structured resolution failed for {entity_name}: {e}")

        if entity_results:
            parts = []
            values = []
            for name, result in entity_results:
                if result.status in ("resolved", "partial"):
                    parts.append(f"{name}: {result.explanation}")
                else:
                    parts.append(f"{name}: No data available")
                # Preserve the already-computed aggregated_value verbatim --
                # never reparsed from result.explanation. `None` for an
                # unresolved entity carries the same meaning it already has
                # on StructuredResult.aggregated_value.
                values.append(ComparisonValue(entity_name=name, value=result.aggregated_value))

            # Derive the combined status from the two original StructuredResult
            # objects -- never a hardcoded "resolved" -- mirroring the
            # resolver's own resolved/partial/empty meaning (see
            # src/query/resolver.py): "resolved" only when every requested
            # entity produced an unqualified value; "partial" when at least
            # one usable value exists but the comparison isn't fully honored
            # (an entity is missing, or any entity's own result was itself
            # "partial"); "empty" when no entity produced a usable value.
            usable_count = sum(1 for _, result in entity_results if result.aggregated_value is not None)
            has_partial_result = any(result.status == "partial" for _, result in entity_results)
            if usable_count == 0:
                comparison_status = "empty"
            elif usable_count < len(comparison_entities) or has_partial_result:
                comparison_status = "partial"
            else:
                comparison_status = "resolved"

            structured_result = ComparisonResult(
                status=comparison_status,
                metric=comparison_metric,
                values=values,
                explanation=" | ".join(parts),
            )

    elif route.path in ("structured", "hybrid") and route.structured_query:
        try:
            structured_result = structured_resolve(
                route.structured_query,
                data_path=match_facts_path,
                stage_taxonomy=stage_taxonomy,
            )
        except Exception as e:
            print(f"Structured resolution failed: {e}")

        if structured_result is not None and structured_result.status in ("resolved", "partial"):
            context = structured_result.explanation or ""
            # Aggression-proxy disclaimer: if the query was about "aggression" but
            # we resolved via fouls_committed, prepend a disclaimer that there's
            # no direct aggression metric.
            if structured_result.query and structured_result.query.metric == "fouls_committed" and \
               re.search(r"aggress|tough|dirtiest", original_query, re.IGNORECASE):
                context = (
                    "NOTE: There is no direct 'aggression' metric in the dataset. "
                    "The closest available signal is fouls committed and card counts. "
                    "Presenting fouls committed data as a proxy — this does not definitively "
                    "measure playing style or intent.\n\n" + context
                )
        elif structured_result is not None and structured_result.status == "empty":
            try:
                fallback_chunks = hybrid_search(
                    route.semantic_query or route.structured_query.entity_name or "",
                    k=semantic_k, artifact_paths=artifact_paths,
                )
                if fallback_chunks:
                    fallback_context = (
                        "NOTE: The structured data did not contain a direct answer to this question. "
                        "The following context is from related documents and may or may not be relevant. "
                        "Only answer if the context clearly and specifically addresses the original question. "
                        "If the context does not clearly answer the question, state that the data does not "
                        "contain a direct answer.\n\n"
                    )
                    context = fallback_context + build_context(fallback_chunks)
                    semantic_chunks = fallback_chunks
            except Exception as e:
                print(f"Semantic fallback failed: {e}")

    if route.path in ("semantic", "hybrid"):
        try:
            semantic_chunks = hybrid_search(route.semantic_query or "", k=semantic_k,
                                            artifact_paths=artifact_paths)
            context = build_context(semantic_chunks)
        except Exception as e:
            print(f"Semantic search failed: {e}")

    answerability = None
    if semantic_chunks is not None:
        answerability_query = (
            original_query
            or route.semantic_query
            or ""
        )
        answerability = assess_answerability(
            query=answerability_query,
            chunks=semantic_chunks,
        )

    explanation = f"Routed to {route.path} path (confidence: {route.confidence:.2f}). "
    if structured_result:
        explanation += f"Structured: {structured_result.explanation} "
    if semantic_chunks:
        explanation += f"Semantic: {len(semantic_chunks)} chunks retrieved."

    return RoutedResult(
        route=route,
        structured_result=structured_result,
        semantic_chunks=semantic_chunks,
        answerability=answerability,
        context=context,
        explanation=explanation,
    )


def route_and_execute(
    query: str, semantic_k: int = 3, artifact_paths: ArtifactPaths | None = None,
) -> RoutedResult:
    """
    Route and execute a query in one step.

    `artifact_paths` selects a namespaced dataset (see src/artifacts.py)
    and is threaded through both routing/parsing and execution. Defaults to
    None -- unchanged legacy WC2022 behavior.
    """
    route = route_query(query, artifact_paths=artifact_paths)
    return execute_route(route, semantic_k, original_query=query, artifact_paths=artifact_paths)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Structural Cleanup Phase B: relocated from 06_retrieve_context.py (now
# deleted) -- this is the same tested debug/inspection entry point
# (`python -m src.query.router "<query>"`), not new functionality. It
# exercises exactly route_query()/execute_route() above.


def main() -> int:
    import argparse

    from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

    parser = argparse.ArgumentParser(description="Query routing and retrieval")
    parser.add_argument("query", help="User query")
    parser.add_argument("--k", type=int, default=5, help="Number of semantic results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    args = parser.parse_args()

    is_legacy_wc2022 = (args.competition_id, args.season_id) == (COMPETITION_ID, SEASON_ID)
    if is_legacy_wc2022 and not args.namespaced:
        artifact_paths = None
    else:
        artifact_paths = ArtifactPaths(args.competition_id, args.season_id)

    print(f"Query: {args.query}")
    print()

    route = route_query(args.query, artifact_paths=artifact_paths)
    print(f"Route: {route.path} (confidence: {route.confidence:.2f})")
    print(f"Reason: {route.reason}")

    if args.verbose:
        if route.structured_query:
            print(f"Structured query: {route.structured_query.to_dict()}")
        if route.semantic_query:
            print(f"Semantic query: {route.semantic_query}")
    print()

    result = execute_route(route, args.k, artifact_paths=artifact_paths)
    print(result.explanation)

    if result.structured_result:
        print(f"\nStructured result:")
        print(f"  Status: {result.structured_result.status}")
        if result.structured_result.aggregated_value is not None:
            print(f"  Value: {result.structured_result.aggregated_value}")
        print(f"  Explanation: {result.structured_result.explanation}")

    if result.semantic_chunks:
        print(f"\nSemantic results ({len(result.semantic_chunks)} chunks):")
        for i, chunk in enumerate(result.semantic_chunks[:3]):
            score = chunk.get('rrf_score', chunk.get('score', 0))
            print(f"  {i+1}. [{chunk['metadata'].get('level')}] Score: {score:.4f}")
            print(f"     {chunk['text'][:100]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
