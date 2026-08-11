"""
resolver.py — Phase 3: Structured Query Executor

Executes StructuredQuery against match_facts.json and returns StructuredResult.
Handles numeric, superlative, slice, and aggregation queries.

HARD RULE: never silently drop a requested filter — if a filter can't be
honored, the result must say so explicitly (dropped_filters list).

Enhanced with MetricKind-aware resolution from the reference implementation:
- Period slicing via by_period data
- Validation via validate_query()
- FactStore for indexed lookups
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.query.query_schema import StructuredQuery, StructuredResult, Filter
from src.query.vocab import (
    ALL_PLAYER_METRICS, PLAYER_DIMENSIONS, MATCH_DIMENSIONS,
    resolve_metric, resolve_aggregation, resolve_stage,
    PERCENTAGE_METRICS,
    REGISTERED_METRICS, MetricKind, MetricSpec,
    validate_query, PERIOD_ALIASES, VALID_PERIODS,
)
from src.stage_taxonomy import StageTaxonomy

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path("output/match_facts.json")

# Cache for loaded data, isolated by resolved artifact path.
# Each entry stores ((mtime_ns, size), parsed_data).
_data_cache_by_path: dict[Path, tuple[tuple[int, int], dict]] = {}


def _load_data(path: Path = DATA_PATH) -> dict:
    """Load match_facts.json with path-aware, file-change-aware caching."""
    resolved_path = path.resolve()

    if resolved_path.exists():
        stat = resolved_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = _data_cache_by_path.get(resolved_path)

        if cached is not None and cached[0] == signature:
            return cached[1]

        data = json.loads(resolved_path.read_text(encoding="utf-8"))
        _data_cache_by_path[resolved_path] = (signature, data)
        return data

    raise FileNotFoundError(
        f"match_facts.json not found at {path}. "
        "Run 'python -m src.extraction.extract' to generate it."
    )


# ---------------------------------------------------------------------------
# FactStore — indexed access to match_facts.json
# ---------------------------------------------------------------------------


class FactStore:
    """In-memory index over match_facts.json for efficient lookups."""

    def __init__(self, data: dict):
        self.data = data
        self.player_match_facts = data.get("player_match_facts", [])
        self.match_facts = data.get("match_facts", [])
        self.team_match_facts = data.get("team_match_facts", [])

        # Build indexes
        self.by_player_id: dict[int, list[dict]] = defaultdict(list)
        self.by_team_name: dict[str, list[dict]] = defaultdict(list)
        self.match_index: dict[int, dict] = {}
        self.team_name_to_id: dict[str, int] = {}
        self.player_name_to_id: dict[str, int] = {}

        for mf in self.match_facts:
            self.match_index[mf["match_id"]] = mf
            self.team_name_to_id[mf["home_team"]] = mf.get("home_team_id", 0)
            self.team_name_to_id[mf["away_team"]] = mf.get("away_team_id", 0)

        for pmf in self.player_match_facts:
            pid = pmf.get("player_id")
            if pid:
                self.by_player_id[pid].append(pmf)
                self.player_name_to_id[pmf["player_name"]] = pid
            self.by_team_name[pmf.get("team_name", "")].append(pmf)

    def player_matches(self, player_id: int) -> list[dict]:
        return self.by_player_id.get(player_id, [])

    def team_matches(self, team_name: str) -> list[dict]:
        return self.by_team_name.get(team_name, [])


# ---------------------------------------------------------------------------
# Period-aware metric reading (MetricKind-aware)
# ---------------------------------------------------------------------------


def _read_metric_from_record(
    record: dict, metric: str, periods: tuple[int, ...] | None = None
) -> tuple[float | None, bool]:
    """
    Read a metric from a player_match_fact record, respecting MetricKind.

    Returns (value, period_applied):
        - For STORED metrics: sum across selected periods or total
        - For DERIVED_RATIO: compute from numerator/denominator
        - For MATCH_ONLY: return match value, period_applied=False if period requested

    period_applied is False when a period filter was requested but the metric
    is match-grain — the caller records that as a dropped filter.
    """
    spec = REGISTERED_METRICS.get(metric)
    if spec is None:
        # Fallback: try direct read
        return record.get(metric), True

    if spec.kind == MetricKind.MATCH_ONLY:
        # minutes, possession_share: no period breakdown
        return record.get(metric), (periods is None)

    # Check if record has by_period data
    by_period = record.get("by_period", {})

    if periods is not None and by_period:
        # Use period-specific data
        blocks = [by_period[str(p)] for p in periods if str(p) in by_period]
        if not blocks:
            blocks = [{}]
    else:
        # Use total
        blocks = [record]

    if spec.kind == MetricKind.STORED:
        return sum(b.get(metric, 0) for b in blocks), True

    if spec.kind == MetricKind.DERIVED_RATIO:
        num = sum(b.get(spec.numerator, 0) for b in blocks)
        den = sum(b.get(spec.denominator, 0) for b in blocks)
        return (num / den * 100.0 if den else None), True

    return None, True


def _resolve_periods_from_filters(filters: list[Filter]) -> tuple[int, ...] | None:
    """Extract period filter from Filter list, resolving aliases."""
    for f in filters:
        if f.dimension == "period":
            val = f.value
            if isinstance(val, str):
                # Check period aliases
                alias = PERIOD_ALIASES.get(val.lower())
                if alias:
                    return alias
                # Try as integer
                try:
                    return (int(val),)
                except ValueError:
                    pass
            elif isinstance(val, (list, tuple)):
                return tuple(val)
            elif isinstance(val, int):
                return (val,)
    return None


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------


def _apply_filter(records: list[dict], f: Filter) -> list[dict]:
    """Apply a single filter to a list of records."""
    dim = f.dimension
    op = f.operator
    val = f.value

    filtered = []
    for record in records:
        record_val = record.get(dim)
        if record_val is None:
            continue

        if op == "eq":
            # Case-insensitive string comparison
            if isinstance(record_val, str) and isinstance(val, str):
                if record_val.lower() == val.lower():
                    filtered.append(record)
            elif record_val == val:
                filtered.append(record)
        elif op == "neq":
            if isinstance(record_val, str) and isinstance(val, str):
                if record_val.lower() != val.lower():
                    filtered.append(record)
            elif record_val != val:
                filtered.append(record)
        elif op == "gt":
            if record_val > val:
                filtered.append(record)
        elif op == "lt":
            if record_val < val:
                filtered.append(record)
        elif op == "gte":
            if record_val >= val:
                filtered.append(record)
        elif op == "lte":
            if record_val <= val:
                filtered.append(record)
        elif op == "in":
            if isinstance(val, list):
                if isinstance(record_val, str):
                    if record_val.lower() in [v.lower() for v in val]:
                        filtered.append(record)
                elif record_val in val:
                    filtered.append(record)

    return filtered


def _validate_filters(filters: list[Filter], entity: str) -> tuple[list[Filter], list[str]]:
    """
    Validate filters against known dimensions.
    Returns (valid_filters, dropped_filter_names).
    """
    valid_dims = PLAYER_DIMENSIONS if entity == "player" else MATCH_DIMENSIONS
    valid = []
    dropped = []

    for f in filters:
        if f.dimension in valid_dims or f.dimension in {"team_name", "match_id", "stage", "is_knockout", "period"}:
            valid.append(f)
        else:
            dropped.append(f.dimension)

    return valid, dropped


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(records: list[dict], metric: str, aggregation: str) -> float | int | None:
    """Perform aggregation on a list of records."""
    if not records:
        return None

    values = [r.get(metric) for r in records if r.get(metric) is not None]
    if not values:
        return None

    if aggregation == "sum":
        return sum(values)
    elif aggregation in ("avg", "mean"):
        return sum(values) / len(values)
    elif aggregation == "max":
        return max(values)
    elif aggregation == "min":
        return min(values)
    elif aggregation == "count":
        return len(values)

    return None


def _aggregate_period_aware(
    records: list[dict],
    metric: str,
    aggregation: str,
    periods: tuple[int, ...] | None,
) -> float | int | None:
    """
    Aggregate a metric across records, MetricKind-aware.

    When `periods` is set, reads the per-period `by_period` blocks via
    _read_metric_from_record (period-sliceable metrics), computes ratios from
    their components, and falls back to the match total for match-only metrics.
    When `periods` is None this is equivalent to reading the stored total.
    """
    if not records:
        return None

    spec = REGISTERED_METRICS.get(metric)

    # Ratios aggregated across multiple records must be recomputed from the
    # combined numerator/denominator. Averaging per-record percentages gives
    # each record equal weight regardless of its denominator.
    if (
        spec is not None
        and spec.kind == MetricKind.DERIVED_RATIO
        and aggregation in ("avg", "mean")
    ):
        numerator_total = 0
        denominator_total = 0

        for record in records:
            by_period = record.get("by_period", {})
            if periods is not None and by_period:
                blocks = [
                    by_period[str(period)]
                    for period in periods
                    if str(period) in by_period
                ]
            else:
                blocks = [record]

            numerator_total += sum(
                block.get(spec.numerator, 0) or 0 for block in blocks
            )
            denominator_total += sum(
                block.get(spec.denominator, 0) or 0 for block in blocks
            )

        return (
            numerator_total / denominator_total * 100.0
            if denominator_total
            else None
        )

    values = []
    for record in records:
        value, _ = _read_metric_from_record(record, metric, periods)
        if value is not None:
            values.append(value)

    if not values:
        return None

    if aggregation == "sum":
        return sum(values)
    elif aggregation in ("avg", "mean"):
        return sum(values) / len(values)
    elif aggregation == "max":
        return max(values)
    elif aggregation == "min":
        return min(values)
    elif aggregation == "count":
        return len(values)

    return None


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def _resolve_player_name(name: str, data: dict) -> str | None:
    """
    Resolve a player name (possibly partial) to the canonical name.
    Case-insensitive matching, handles common variations and accents.
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # Build name lookup from player_match_facts
    name_counts: dict[str, int] = defaultdict(int)
    for pmf in data["player_match_facts"]:
        pname = pmf["player_name"]
        name_counts[pname] += 1

    # Exact match (case-insensitive)
    for pname in name_counts:
        if pname.lower() == name_lower:
            return pname

    # Partial match (last name or any word)
    for pname in name_counts:
        parts = pname.lower().split()
        if name_lower in parts:
            return pname

    # Substring match (for partial names like "Mbappe" -> "Mbappé")
    # Uses word-boundary matching: the query must match a complete word in the name.
    import unicodedata
    for pname in name_counts:
        pname_normalized = unicodedata.normalize('NFD', pname.lower())
        name_normalized = unicodedata.normalize('NFD', name_lower)
        # Remove combining characters (accents)
        pname_ascii = ''.join(c for c in pname_normalized if unicodedata.category(c) != 'Mn')
        name_ascii = ''.join(c for c in name_normalized if unicodedata.category(c) != 'Mn')
        # Word-boundary match: query must match a complete word in the player name
        pname_words = pname_ascii.split()
        name_words = name_ascii.split()
        if any(nw == pw for nw in name_words for pw in pname_words):
            return pname

    # Original substring match
    for pname in name_counts:
        if name_lower in pname.lower():
            return pname

    return None


def _resolve_team_name(name: str, data: dict) -> str | None:
    """Resolve a team name (possibly partial) to the canonical name."""
    if not name:
        return None

    name_lower = name.lower().strip()

    # Build name lookup
    team_names = set()
    for mf in data["match_facts"]:
        team_names.add(mf["home_team"])
        team_names.add(mf["away_team"])

    # Exact match (case-insensitive)
    for tname in team_names:
        if tname.lower() == name_lower:
            return tname

    # Partial match
    for tname in team_names:
        if name_lower in tname.lower():
            return tname

    return None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve(
    query: StructuredQuery,
    data: dict | None = None,
    stage_taxonomy: StageTaxonomy | None = None,
    data_path: Path | None = None,
) -> StructuredResult:
    """
    Execute a StructuredQuery against match_facts.json.

    Returns a StructuredResult with status:
        "resolved" — all filters honored
        "partial"  — some filters dropped
        "empty"    — no matching records

    Enhanced with MetricKind-aware validation: period filters on match-only
    metrics are detected and reported as dropped_filters rather than silently
    returning incorrect values.

    `stage_taxonomy` is forwarded to validate_query() for the "stage" filter
    check. Defaults to None, which falls back to the WC2022 vocabulary —
    unchanged behavior unless a caller supplies a different competition's
    taxonomy (see src/stage_taxonomy.py).

    `data_path` selects which match_facts.json to load and cache against
    when `data` is not supplied explicitly — e.g. a namespaced competition/
    season artifact (see src/artifacts.py). Defaults to None, which falls
    back to the module-level DATA_PATH (the legacy output/match_facts.json
    default) — unchanged behavior unless a caller supplies a different path.
    Ignored when `data` is supplied explicitly.

    Results are cached and automatically invalidated when the selected
    match_facts.json changes (see src/cache.py) — the cache entry is bound
    to `data_path`, so two different datasets never share a cache hit.
    """
    from src.cache import get_cached_structured_result, set_cached_structured_result

    # Generate cache key from the FULL query (including filters, limit, sort).
    # Serializing query.to_dict() ensures two queries that differ only by their
    # filters (e.g. Messi's goals overall vs. in the knockouts) get distinct
    # cache entries instead of colliding and returning each other's answers.
    # The stage taxonomy (when not the default) is folded in too, so the same
    # query validated under two different taxonomies never shares a cache hit.
    cache_key = json.dumps(query.to_dict(), sort_keys=True, default=str)
    if stage_taxonomy is not None:
        taxonomy_key = {
            "stages": sorted(stage_taxonomy.stages),
            "knockout_stages": sorted(stage_taxonomy.knockout_stages),
            "group_stages": sorted(stage_taxonomy.group_stages),
            "stage_order": list(stage_taxonomy.stage_order) if stage_taxonomy.stage_order is not None else None,
        }
        cache_key += ":" + json.dumps(taxonomy_key, sort_keys=True)

    # Explicit in-memory data must never share the file-backed cache.
    # Only the data=None path is tied to a match_facts.json file, and the
    # cache is bound to the actual path selected (data_path, defaulting to
    # DATA_PATH), never a fixed unrelated location.
    explicit_data = data is not None
    load_path = data_path if data_path is not None else DATA_PATH

    if not explicit_data:
        cached = get_cached_structured_result(cache_key, data_path=load_path)
        if cached is not None:
            return cached
        data = _load_data(load_path)

    # Validate query using MetricKind-aware validation
    validation = validate_query(query, stage_taxonomy=stage_taxonomy)
    extra_dropped = []
    if validation.droppable:
        for issue in validation.droppable:
            extra_dropped.append(issue.field)
    if validation.issues and not validation.ok:
        # Hard validation failure. Phrase unknown-metric failures with the
        # canonical "Unknown metric: X" wording so callers (and tests) get a
        # stable, recognizable message.
        metric_issues = [i for i in validation.issues if i.field == "metric"]
        if metric_issues:
            explanation = "; ".join(f"Unknown metric: {i.value}" for i in metric_issues)
        else:
            explanation = "; ".join(
                f"{i.field}={i.value}: {i.reason}" for i in validation.issues)
        return StructuredResult(
            status="empty",
            query=query,
            data=[],
            aggregated_value=None,
            dropped_filters=[],
            explanation=explanation,
        )

    # Validate filters
    valid_filters, dropped = _validate_filters(query.filters, query.entity)

    # Add period-related dropped filters from MetricKind validation
    dropped = list(set(dropped + extra_dropped))

    # Period is not a stored column on the records — it selects per-period
    # blocks at read time. Pull it out here and remove it from the filters that
    # are applied as row predicates (otherwise _apply_filter matches against a
    # missing "period" field and silently drops every record).
    periods = _resolve_periods_from_filters(valid_filters)
    valid_filters = [f for f in valid_filters if f.dimension != "period"]

    # The knockout filter (stage=None) is also special — _apply_filter would
    # match against a missing "stage" field and silently drop records. Pull it
    # out and apply it separately via the is_knockout boolean field.
    knockout_filter = None
    clean_filters = []
    for f in valid_filters:
        if f.dimension == "stage" and f.value is None:
            knockout_filter = f
        else:
            clean_filters.append(f)
    valid_filters = clean_filters

    # Resolve entity name if provided
    entity_name = query.entity_name
    if entity_name:
        if query.entity == "player":
            resolved = _resolve_player_name(entity_name, data)
            if resolved:
                entity_name = resolved
            else:
                return StructuredResult(
                    status="empty",
                    query=query,
                    explanation=f"No player found matching '{query.entity_name}'.",
                )
        elif query.entity == "team":
            resolved = _resolve_team_name(entity_name, data)
            if resolved:
                entity_name = resolved
            else:
                return StructuredResult(
                    status="empty",
                    query=query,
                    explanation=f"No team found matching '{query.entity_name}'.",
                )

    # Select records based on entity type
    if query.entity == "player":
        records = data["player_match_facts"]
    elif query.entity == "match":
        records = data["match_facts"]
    elif query.entity == "team":
        records = data["team_match_facts"]
    else:
        return StructuredResult(
            status="empty",
            query=query,
            explanation=f"Unknown entity type: {query.entity}",
        )

    # Apply entity name filter
    if entity_name:
        if query.entity == "player":
            records = [r for r in records if r.get("player_name", "").lower() == entity_name.lower()]
        elif query.entity == "team":
            records = [r for r in records if r.get("team_name", "").lower() == entity_name.lower()]

    # Apply additional filters
    for f in valid_filters:
        # Handle stage synonyms
        if f.dimension == "stage" and isinstance(f.value, str):
            resolved_stage = resolve_stage(f.value)
            if resolved_stage:
                f = Filter(f.dimension, f.operator, resolved_stage)
        records = _apply_filter(records, f)

    # Handle "knockout" special filter (extracted earlier)
    if knockout_filter:
        records = [r for r in records if r.get("is_knockout", False)]

    if not records:
        status = "partial" if dropped else "empty"
        return StructuredResult(
            status=status,
            query=query,
            data=[],
            aggregated_value=None,
            dropped_filters=dropped,
            explanation=f"No records found matching the query criteria.",
        )

    # Perform aggregation
    metric = resolve_metric(query.metric)
    if metric is None:
        return StructuredResult(
            status="empty",
            query=query,
            explanation=f"Unknown metric: {query.metric}",
        )

    agg = resolve_aggregation(query.aggregation)
    if agg is None:
        agg = "sum"

    # If a period slice was requested but it can't be honored — either the
    # metric isn't period-sliceable, or the persisted records carry no
    # per-period breakdown — drop the period filter honestly (→ partial) and
    # fall back to match totals, rather than silently returning an unsliced
    # value dressed up as the slice.
    if periods is not None:
        spec = REGISTERED_METRICS.get(metric)
        sliceable = spec.period_sliceable if spec else False
        has_period_data = any(r.get("by_period") for r in records)
        if not sliceable or not has_period_data:
            if "period" not in dropped:
                dropped.append("period")
            periods = None

    # For superlatives, aggregate by entity first, then sort
    if query.intent == "superlative" and query.limit:
        # Group by player/team and aggregate
        entity_groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            key = r.get("player_name") or r.get("team_name") or str(r.get("match_id"))
            entity_groups[key].append(r)

        # Build aggregated records
        # IMPORTANT: Use "sum" for cumulative metrics (goals, assists, xG),
        # but "avg" for percentage/average metrics (possession_share, pass_completion_pct)
        # The query's aggregation (e.g., "max") is for ranking, not per-entity aggregation
        per_entity_agg = "avg" if metric in PERCENTAGE_METRICS else "sum"

        agg_records = []
        for key, group in entity_groups.items():
            agg_val = _aggregate_period_aware(group, metric, per_entity_agg, periods)
            if agg_val is not None:
                # Use the first record as base, override the metric with aggregated value
                base = dict(group[0])
                base[metric] = agg_val
                base["_agg_count"] = len(group)
                agg_records.append(base)

        # Sort by aggregated value
        sorted_records = sorted(agg_records, key=lambda r: r.get(metric, 0) or 0,
                                reverse=(query.sort_order == "desc"))
        result_data = sorted_records[:query.limit]
        agg_value = result_data[0].get(metric) if result_data else None
    else:
        agg_value = _aggregate_period_aware(records, metric, agg, periods)
        result_data = records

    # Build explanation
    if entity_name:
        if agg_value is not None:
            if agg in ("avg", "mean"):
                explanation = f"{entity_name}'s average {metric} is {_format_value(agg_value, metric)}."
            elif agg == "sum":
                explanation = f"{entity_name}'s total {metric} is {_format_value(agg_value, metric)}."
            elif agg == "max":
                explanation = f"{entity_name}'s highest {metric} is {_format_value(agg_value, metric)}."
            elif agg == "min":
                explanation = f"{entity_name}'s lowest {metric} is {_format_value(agg_value, metric)}."
            else:
                explanation = f"{entity_name}'s {metric} ({agg}) is {_format_value(agg_value, metric)}."
        else:
            explanation = f"No {metric} data available for {entity_name}."
    else:
        if query.intent == "superlative" and result_data:
            top = result_data[0]
            name = top.get("player_name") or top.get("team_name") or str(top.get("match_id"))
            if agg == "min":
                explanation = f"The lowest {metric} is {_format_value(top.get(metric, 0), metric)} by {name}."
            else:
                explanation = f"The top {metric} is {_format_value(top.get(metric, 0), metric)} by {name}."
        elif agg_value is not None:
            explanation = f"The {agg} {metric} across all matching records is {_format_value(agg_value, metric)}."
        else:
            explanation = f"No {metric} data available."

    status = "partial" if dropped else "resolved"

    # Make partial answers self-explanatory: state which filters were dropped
    # right in the human-readable explanation the LLM ultimately sees.
    if dropped:
        explanation += (
            f" (Note: could not apply filter(s): "
            f"{', '.join(sorted(set(dropped)))}; value shown is unfiltered.)"
        )

    result = StructuredResult(
        status=status,
        query=query,
        data=[r for r in result_data],
        aggregated_value=agg_value,
        dropped_filters=dropped,
        explanation=explanation,
    )

    # Cache only results resolved from the file-backed dataset, bound to
    # the actual path that was loaded.
    if not explicit_data:
        set_cached_structured_result(cache_key, result, data_path=load_path)

    return result


def _format_value(value: float | int | None, metric: str) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    if metric in PERCENTAGE_METRICS:
        return f"{value:.1f}%"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def resolve_from_text(query_text: str) -> StructuredResult:
    """
    Attempt to resolve a natural language query using structured data.
    This is a simple pattern-matching approach — the router (Phase 5) will
    use a more sophisticated parser.
    """
    data = _load_data()
    query_text_lower = query_text.lower()

    # Simple pattern: "how many <metric> did <player> score/have"
    import re

    # Pattern: <player> <metric>
    player_metric_pattern = re.compile(
        r"(?:how many|what is|what's|what are)\s+(\w+)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?)?",
        re.IGNORECASE
    )
    match = player_metric_pattern.search(query_text)
    if match:
        metric_raw, player_raw = match.groups()
        from src.query.vocab import METRIC_SYNONYMS
        metric = METRIC_SYNONYMS.get(metric_raw.lower(), metric_raw.lower())
        player = _resolve_player_name(player_raw.strip(), data)
        if player and metric:
            q = StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric,
                aggregation="sum",
                entity_name=player,
            )
            return resolve(q)

    # Pattern: who <aggregation> <metric>
    who_pattern = re.compile(
        r"who\s+(has the|had the|scored the|got the)?\s*(most|highest|best|least|lowest|fewest)\s+(\w+)",
        re.IGNORECASE
    )
    match = who_pattern.search(query_text)
    if match:
        _, agg_raw, metric_raw = match.groups()
        from src.query.vocab import METRIC_SYNONYMS, AGGREGATION_SYNONYMS
        metric = METRIC_SYNONYMS.get(metric_raw.lower(), metric_raw.lower())
        agg = AGGREGATION_SYNONYMS.get(agg_raw.lower(), "max")
        q = StructuredQuery(
            intent="superlative",
            entity="player",
            metric=metric,
            aggregation=agg,
            limit=1,
        )
        return resolve(q)

    return StructuredResult(
        status="empty",
        query=StructuredQuery(intent="unknown", entity="unknown", metric="unknown"),
        explanation="Could not parse the query into a structured form.",
    )
