"""
src/query/parsing.py -- Query parsing: turning query text plus the active
dataset vocabulary into a StructuredQuery.

Migration Step 3 (Query Understanding + Planning Split): mechanically
extracted from src/query/router.py's filter-extraction, stage-taxonomy
loading, and structured-query parsing sections -- no logic changes. See
src/query/router.py for the compatibility re-exports existing callers keep
using.

Parsing answers "which structured query, if any, expresses this question?" --
including the filters it carries and which competition's stage vocabulary is
in force. It does NOT classify intent (see src/query/intent.py), select a
retrieval strategy (see src/query/planning.py), or execute anything (see
src/query/router.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.query.query_schema import Filter, StructuredQuery
from src.query.vocab import (
    METRIC_ALIASES,
    normalize_query_text,
    resolve_aggregation,
    resolve_metric,
)
from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY
from src.extraction.match_facts import WC2022_DATASET_IDENTITY


# ---------------------------------------------------------------------------
# Active Dataset Stage Vocabulary
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Filter Extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structured Query Parsing
# ---------------------------------------------------------------------------



def parse_compositional_dependency(
    query: str,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
) -> tuple[StructuredQuery, str] | None:
    """
    Detect one structured entity selector embedded inside a qualitative query.

    The dependency is resolved first, then its authoritative entity is used
    in the downstream semantic query. This intentionally supports one
    dependency hop only.
    """
    semantic_cue = re.search(
        r"\b(?:play|played|playing|perform|performed|performance|formation|"
        r"style|strategy|tactics|defend|defended|defending|attack|attacked|attacking)\b"
        r"|^\s*describe\b",
        query,
        re.IGNORECASE,
    )
    if not semantic_cue:
        return None

    # "team/player with the most/highest/... <metric>"
    selector = re.search(
        r"\b(?:the\s+)?(?P<entity>team|player)\s+with\s+(?:the\s+)?"
        r"(?P<agg>most|highest|best|least|lowest|fewest)\s+"
        r"(?P<tail>[^?.,]+)",
        query,
        re.IGNORECASE,
    )
    if selector:
        entity = selector.group("entity").lower()
        agg = resolve_aggregation(selector.group("agg").lower())
        tail = selector.group("tail")

        # Longest metric aliases first, so e.g. "goals conceded"
        # wins before the shorter "goals".
        for alias in sorted(METRIC_ALIASES, key=len, reverse=True):
            metric_match = re.match(
                rf"{re.escape(alias)}(?=\s|[?.!,]|$)",
                tail,
                re.IGNORECASE,
            )
            if not metric_match:
                continue

            metric = resolve_metric(metric_match.group(0))
            if metric and agg:
                phrase_end = selector.start("tail") + metric_match.end()
                phrase = query[selector.start():phrase_end]
                return (
                    StructuredQuery(
                        intent="superlative",
                        entity=entity,
                        metric=metric,
                        aggregation=agg,
                        limit=1,
                    ),
                    phrase,
                )

    # "highest-scoring team" / "lowest scoring team"
    scoring = re.search(
        r"\b(?:the\s+)?(?P<agg>highest|top|best|lowest|least)[-\s]+"
        r"scoring\s+(?P<entity>team|player)\b",
        query,
        re.IGNORECASE,
    )
    if scoring:
        raw_agg = scoring.group("agg").lower()
        aggregation = "min" if raw_agg in {"lowest", "least"} else "max"
        return (
            StructuredQuery(
                intent="superlative",
                entity=scoring.group("entity").lower(),
                metric="goals",
                aggregation=aggregation,
                limit=1,
            ),
            scoring.group(0),
        )

    # Reuse existing parsing for lexical selectors such as "top scorer".
    lexical = re.search(
        r"\b(?:the\s+)?(?:top|best|leading)\s+"
        r"(?:scorer|goal\s*scorer|assists?|passer|tackler)\b",
        query,
        re.IGNORECASE,
    )
    if lexical:
        dependency = parse_structured_query(
            lexical.group(0),
            stage_taxonomy=stage_taxonomy,
        )
        if dependency is not None and dependency.intent == "superlative":
            return dependency, lexical.group(0)

    return None


def parse_structured_query(
    query: str,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
) -> StructuredQuery | None:
    """Parse a query into a StructuredQuery using the active stage vocabulary."""
    query_lower = normalize_query_text(query)

    stage_filter = _extract_stage_filter(query, stage_taxonomy=stage_taxonomy)
    opponent_filter = _extract_opponent_filter(query)
    filters = [f for f in [stage_filter, opponent_filter] if f is not None]

    arabic_metric = resolve_metric("الاهداف")
    arabic_aggregation = resolve_aggregation("الاكثر")
    if arabic_metric and arabic_aggregation:
        if re.search(r"(?<!\w)(?:من\s+هو\s+)?(?:ال)?هداف(?!\w)", query_lower):
            return StructuredQuery(
                intent="superlative", entity="player", metric=arabic_metric,
                aggregation=arabic_aggregation, limit=1, filters=filters,
            )

        if re.search(
            r"من\s+(?:هو\s+)?(?:اللاعب\s+)?(?:الذي\s+)?(?:سجل|احرز)\s+"
            r"(?:اكبر\s+عدد\s+من|(?:ال)?اكثر)\s+(?:ال)?اهداف",
            query_lower,
        ):
            return StructuredQuery(
                intent="superlative", entity="player", metric=arabic_metric,
                aggregation=arabic_aggregation, limit=1, filters=filters,
            )

        if re.search(
            r"(?:ما|من)\s+(?:هو\s+)?الفريق\s+(?:"
            r"(?:ال)?اكثر\s+(?:تسجيلا|احرازا)\s+ل(?:ل)?(?:ال)?اهداف|"
            r"(?:الذي\s+)?(?:سجل|احرز)\s+"
            r"(?:اكبر\s+عدد\s+من|(?:ال)?اكثر)\s+(?:ال)?اهداف)",
            query_lower,
        ):
            return StructuredQuery(
                intent="superlative", entity="team", metric=arabic_metric,
                aggregation=arabic_aggregation, limit=1, filters=filters,
            )

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
