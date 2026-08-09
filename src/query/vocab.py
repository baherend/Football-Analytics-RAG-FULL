"""
vocab.py — Phase 3: Metric and Dimension Vocabularies

Defines the registered metrics, dimensions, and aggregations available
for structured queries against match_facts.json.

CRITICAL: These vocabularies are grounded in what match_facts.json ACTUALLY
persists. Do not add metrics or dimensions that don't exist in the data.

The MetricKind / MetricSpec system (from the reference implementation) makes
metric behavior explicit: which metrics are period-sliceable, which are
derived ratios (computed from numerator/denominator on read), and which exist
only at match grain. This prevents the class of bug where a period filter is
applied to a metric that has no per-period data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY


# ---------------------------------------------------------------------------
# MetricKind / MetricSpec — explicit metric typing
# ---------------------------------------------------------------------------


class MetricKind(str, Enum):
    STORED = "stored"                # a stored per-period + total count
    DERIVED_RATIO = "derived_ratio"  # computed from numerator/denominator on read
    MATCH_ONLY = "match_only"        # exists only at match grain (not per period)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    kind: MetricKind
    period_sliceable: bool
    numerator: str | None = None      # for DERIVED_RATIO
    denominator: str | None = None    # for DERIVED_RATIO
    aggregatable: bool = True         # can it be summed across matches for rank/aggregate?


# ---------------------------------------------------------------------------
# Registered metrics — the single source of truth for metric behavior
# ---------------------------------------------------------------------------

REGISTERED_METRICS: dict[str, MetricSpec] = {
    # --- 17 event-level stored counts: period-sliceable, summable ---
    "shots":                    MetricSpec("shots", MetricKind.STORED, True),
    "xg":                       MetricSpec("xg", MetricKind.STORED, True),
    "goals":                    MetricSpec("goals", MetricKind.STORED, True),
    "assists":                  MetricSpec("assists", MetricKind.STORED, True),
    "passes_attempted":         MetricSpec("passes_attempted", MetricKind.STORED, True),
    "passes_completed":         MetricSpec("passes_completed", MetricKind.STORED, True),
    "passes_under_pressure":    MetricSpec("passes_under_pressure", MetricKind.STORED, True),
    "passes_under_pressure_completed":
                                MetricSpec("passes_under_pressure_completed", MetricKind.STORED, True),
    "shots_inside_box":         MetricSpec("shots_inside_box", MetricKind.STORED, True),
    "shots_outside_box":        MetricSpec("shots_outside_box", MetricKind.STORED, True),
    "successful_tackles":       MetricSpec("successful_tackles", MetricKind.STORED, True),
    "successful_interceptions": MetricSpec("successful_interceptions", MetricKind.STORED, True),
    "clearances":               MetricSpec("clearances", MetricKind.STORED, True),
    "ball_losses":              MetricSpec("ball_losses", MetricKind.STORED, True),
    "carries":                  MetricSpec("carries", MetricKind.STORED, True),
    "carry_distance":           MetricSpec("carry_distance", MetricKind.STORED, True),
    "pressures":                MetricSpec("pressures", MetricKind.STORED, True),
    "final_third_passes":       MetricSpec("final_third_passes", MetricKind.STORED, True),
    "fouls_committed":          MetricSpec("fouls_committed", MetricKind.STORED, True),
    "yellow_cards":             MetricSpec("yellow_cards", MetricKind.STORED, True),
    "second_yellow_cards":      MetricSpec("second_yellow_cards", MetricKind.STORED, True),
    "red_cards":                MetricSpec("red_cards", MetricKind.STORED, True),
    "saves":                    MetricSpec("saves", MetricKind.STORED, True),
    "goals_conceded":           MetricSpec("goals_conceded", MetricKind.STORED, True),
    "penalties_saved":          MetricSpec("penalties_saved", MetricKind.STORED, True),
    "claims":                   MetricSpec("claims", MetricKind.STORED, True),
    "punches":                  MetricSpec("punches", MetricKind.STORED, True),
    "sweeper_actions":          MetricSpec("sweeper_actions", MetricKind.STORED, True),

    # --- 2 ratios: period-sliceable via their components, NOT summable ---
    "pass_pct": MetricSpec(
        "pass_pct", MetricKind.DERIVED_RATIO, True,
        numerator="passes_completed", denominator="passes_attempted",
        aggregatable=False),
    "pass_pct_under_pressure": MetricSpec(
        "pass_pct_under_pressure", MetricKind.DERIVED_RATIO, True,
        numerator="passes_under_pressure_completed",
        denominator="passes_under_pressure", aggregatable=False),

    # --- aliases for the existing pass_completion_pct / pass_completion_under_pressure_pct ---
    "pass_completion_pct": MetricSpec(
        "pass_completion_pct", MetricKind.DERIVED_RATIO, True,
        numerator="passes_completed", denominator="passes_attempted",
        aggregatable=False),
    "pass_completion_under_pressure_pct": MetricSpec(
        "pass_completion_under_pressure_pct", MetricKind.DERIVED_RATIO, True,
        numerator="passes_under_pressure_completed",
        denominator="passes_under_pressure", aggregatable=False),

    # --- minutes: match grain only, NOT period-sliceable ---
    "minutes": MetricSpec("minutes", MetricKind.MATCH_ONLY, False),

    # --- team-level metrics (match grain) ---
    "possession_share": MetricSpec("possession_share", MetricKind.MATCH_ONLY, False),
    "crosses": MetricSpec("crosses", MetricKind.STORED, True),
    "first_shot_minute": MetricSpec("first_shot_minute", MetricKind.MATCH_ONLY, False),
    "first_goal_minute": MetricSpec("first_goal_minute", MetricKind.MATCH_ONLY, False),
}


# ---------------------------------------------------------------------------
# Metric aliases — natural language -> canonical metric key
# ---------------------------------------------------------------------------

METRIC_ALIASES: dict[str, str] = {
    "expected goals": "xg", "xg": "xg",
    "goal": "goals", "goals": "goals", "scored": "goals",
    "assist": "assists", "assists": "assists",
    "shot": "shots", "shots": "shots",
    "pass": "passes_attempted", "passes": "passes_attempted",
    "completed passes": "passes_completed",
    "pass completion": "pass_pct", "pass accuracy": "pass_pct", "pass %": "pass_pct",
    "tackle": "successful_tackles", "tackles": "successful_tackles",
    "interception": "successful_interceptions", "interceptions": "successful_interceptions",
    "clearance": "clearances", "clearances": "clearances",
    "carry": "carries", "carries": "carries",
    "carry distance": "carry_distance",
    "pressure": "pressures", "pressures": "pressures",
    "foul": "fouls_committed", "fouls": "fouls_committed",
    "fouls committed": "fouls_committed", "committed fouls": "fouls_committed",
    "yellow card": "yellow_cards", "yellow cards": "yellow_cards",
    "booking": "yellow_cards", "bookings": "yellow_cards", "booked": "yellow_cards",
    "second yellow": "second_yellow_cards", "second yellow card": "second_yellow_cards",
    "second yellow cards": "second_yellow_cards",
    "red card": "red_cards", "red cards": "red_cards",
    "sent off": "red_cards", "dismissed": "red_cards", "dismissals": "red_cards",
    # Goalkeeper metrics
    "save": "saves", "saves": "saves", "stops": "saves", "shot saved": "saves",
    "goals conceded": "goals_conceded", "goals let in": "goals_conceded",
    "conceded": "goals_conceded",
    "penalty save": "penalties_saved", "penalty saves": "penalties_saved",
    "penalty saved": "penalties_saved", "saved penalty": "penalties_saved",
    "saved penalties": "penalties_saved",
    "claim": "claims", "claims": "claims", "collected": "claims",
    "punch": "punches", "punches": "punches",
    "sweeper": "sweeper_actions", "sweeper actions": "sweeper_actions",
    "sweeper action": "sweeper_actions", "keeper sweeper": "sweeper_actions",
    "final third pass": "final_third_passes", "final third passes": "final_third_passes",
    "ball loss": "ball_losses", "ball losses": "ball_losses", "turnovers": "ball_losses",
    "minute": "minutes", "minutes": "minutes",
    "possession": "possession_share", "possession share": "possession_share",
    "crosses": "crosses", "cross": "crosses",
}


# ---------------------------------------------------------------------------
# Sliceable dimensions
# ---------------------------------------------------------------------------


class Dimension(str, Enum):
    OPPONENT = "opponent"
    STAGE = "stage"
    MATCH = "match"
    PERIOD = "period"


SLICEABLE_DIMENSIONS: frozenset[Dimension] = frozenset({
    Dimension.OPPONENT, Dimension.STAGE, Dimension.MATCH, Dimension.PERIOD,
})

# Backward-compatible default stage vocabulary (WC2022). Sourced from the
# shared taxonomy module instead of being an independent literal — see
# validate_query()'s `stage_taxonomy` parameter for validating against a
# different competition's stage names.
VALID_STAGES = WC2022_STAGE_TAXONOMY.stages

VALID_PERIODS = frozenset({1, 2, 3, 4})
PERIOD_ALIASES: dict[str, tuple[int, ...]] = {
    "first half": (1,), "1st half": (1,), "h1": (1,),
    "second half": (2,), "2nd half": (2,), "h2": (2,),
    "extra time": (3, 4), "et": (3, 4),
    "full match": (1, 2, 3, 4), "full time": (1, 2, 3, 4),
}


# ---------------------------------------------------------------------------
# Operations and entity types (for structured query schema)
# ---------------------------------------------------------------------------


class Operation(str, Enum):
    LOOKUP = "lookup"
    RANK = "rank"
    COMPARE = "compare"
    AGGREGATE = "aggregate"


class EntityType(str, Enum):
    PLAYER = "player"
    TEAM = "team"


@dataclass
class EntityRef:
    type: EntityType
    id: int | None = None
    name: str | None = None


@dataclass
class Filters:
    opponent_id: int | None = None
    stage: str | None = None
    match_id: int | None = None
    periods: tuple[int, ...] | None = None


@dataclass
class StructuredQueryV2:
    """Structured query using the new MetricKind-aware schema."""
    operation: Operation
    metrics: list[str]
    entity: EntityRef
    filters: Filters = field(default_factory=Filters)
    order: str = "desc"
    limit: int = 1


@dataclass
class ValidationIssue:
    field: str
    value: object
    reason: str


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    droppable: list[ValidationIssue] = field(default_factory=list)


def validate_query(q, stage_taxonomy: StageTaxonomy | None = None) -> ValidationResult:
    """
    Fail-at-parse-time validation. Two outcomes:

    * hard failure (ok=False): unknown metric, unknown dimension
    * droppable (ok=True): well-formed but filter inexpressible for metric
      (e.g. period filter on match-only `minutes`)

    Accepts either StructuredQueryV2 or the legacy StructuredQuery from query_schema.

    `stage_taxonomy` controls which stage names pass the "stage" filter
    check below. Defaults to None, which falls back to the WC2022
    vocabulary (`VALID_STAGES`) for backward compatibility — pass an
    explicit `StageTaxonomy` (e.g. built via `StageTaxonomy.discover(...)`)
    to validate against a different competition's stage names.
    """
    issues: list[ValidationIssue] = []
    droppable: list[ValidationIssue] = []

    # Extract metrics — handle both old and new schema
    if hasattr(q, 'metrics'):
        metrics = q.metrics
    elif hasattr(q, 'metric'):
        metrics = [q.metric]
    else:
        metrics = []

    if not metrics:
        issues.append(ValidationIssue("metrics", None, "no metric specified"))

    unknown = [m for m in metrics if m not in REGISTERED_METRICS]
    for m in unknown:
        issues.append(ValidationIssue(
            "metric", m, f"'{m}' is not a registered metric"))

    known = [m for m in metrics if m in REGISTERED_METRICS]

    # Period filter validation
    periods = None
    if hasattr(q, 'filters'):
        if hasattr(q.filters, 'periods'):
            periods = q.filters.periods
        elif isinstance(q.filters, list):
            for f in q.filters:
                if hasattr(f, 'dimension') and f.dimension == 'period':
                    periods = f.value

    if periods is not None:
        bad_periods = [p for p in periods if p not in VALID_PERIODS]
        for p in bad_periods:
            issues.append(ValidationIssue(
                "period", p, f"period {p} is not a valid open-play period"))
        for m in known:
            spec = REGISTERED_METRICS[m]
            if not spec.period_sliceable:
                droppable.append(ValidationIssue(
                    "period", periods,
                    f"metric '{m}' is stored at match grain only and cannot be "
                    "sliced by period; period filter dropped for this metric"))

    # Stage validation
    stage = None
    if hasattr(q, 'filters'):
        if hasattr(q.filters, 'stage'):
            stage = q.filters.stage
        elif isinstance(q.filters, list):
            for f in q.filters:
                if hasattr(f, 'dimension') and f.dimension == 'stage':
                    stage = f.value

    valid_stages = stage_taxonomy.stages if stage_taxonomy is not None else VALID_STAGES
    if stage is not None and stage not in valid_stages:
        issues.append(ValidationIssue(
            "stage", stage, f"'{stage}' is not a known competition stage"))

    # Aggregatable check for rank/aggregate
    operation = q.operation if hasattr(q, 'operation') else getattr(q, 'intent', None)
    if operation in ("rank", "aggregate", Operation.RANK, Operation.AGGREGATE):
        for m in known:
            if not REGISTERED_METRICS[m].aggregatable:
                issues.append(ValidationIssue(
                    "metric", m,
                    f"metric '{m}' is a ratio and cannot be summed/ranked across "
                    "matches directly; rank on a component count instead"))

    return ValidationResult(ok=not issues, issues=issues, droppable=droppable)


# ---------------------------------------------------------------------------
# Metrics (from PlayerMatchFacts) — legacy dictionaries (backward compat)
# ---------------------------------------------------------------------------

# Numeric metrics that can be summed, averaged, maxed, etc.
PLAYER_NUMERIC_METRICS = {
    # Attacking
    "shots": {"description": "Total shots taken", "type": "int", "min": 0},
    "xg": {"description": "Expected goals", "type": "float", "min": 0.0},
    "goals": {"description": "Goals scored", "type": "int", "min": 0},
    "assists": {"description": "Assists provided", "type": "int", "min": 0},
    "shots_inside_box": {"description": "Shots from inside the penalty area", "type": "int", "min": 0},
    "shots_outside_box": {"description": "Shots from outside the penalty area", "type": "int", "min": 0},

    # Passing
    "passes_attempted": {"description": "Total passes attempted", "type": "int", "min": 0},
    "passes_completed": {"description": "Completed passes", "type": "int", "min": 0},
    "pass_completion_pct": {"description": "Pass completion percentage", "type": "float", "min": 0.0, "max": 100.0},
    "passes_under_pressure": {"description": "Passes attempted under pressure", "type": "int", "min": 0},
    "pass_completion_under_pressure_pct": {"description": "Pass completion under pressure percentage", "type": "float", "min": 0.0, "max": 100.0},
    "final_third_passes": {"description": "Passes into the final third", "type": "int", "min": 0},

    # Defensive
    "successful_tackles": {"description": "Successful tackles", "type": "int", "min": 0},
    "successful_interceptions": {"description": "Successful interceptions", "type": "int", "min": 0},
    "clearances": {"description": "Clearances", "type": "int", "min": 0},
    "ball_losses": {"description": "Ball losses (miscontrol + dispossessed)", "type": "int", "min": 0},
    "pressures": {"description": "Pressures applied", "type": "int", "min": 0},

    # Carrying
    "carries": {"description": "Ball carries", "type": "int", "min": 0},
    "carry_distance": {"description": "Total carry distance (yards)", "type": "float", "min": 0.0},

    # Discipline
    "fouls_committed": {"description": "Fouls committed", "type": "int", "min": 0},
    "yellow_cards": {"description": "Yellow cards", "type": "int", "min": 0},
    "second_yellow_cards": {"description": "Second yellow cards (dismissal)", "type": "int", "min": 0},
    "red_cards": {"description": "Red cards (direct)", "type": "int", "min": 0},

    # Goalkeeper
    "saves": {"description": "Goalkeeper saves", "type": "int", "min": 0},
    "goals_conceded": {"description": "Goals conceded while in goal", "type": "int", "min": 0},
    "penalties_saved": {"description": "Penalties saved", "type": "int", "min": 0},
    "claims": {"description": "High balls claimed/collected", "type": "int", "min": 0},
    "punches": {"description": "Punched clearances", "type": "int", "min": 0},
    "sweeper_actions": {"description": "Keeper sweeper actions outside box", "type": "int", "min": 0},

    # Playing time
    "minutes": {"description": "Minutes played", "type": "float", "min": 0.0},
}

# All player metrics (union of numeric)
ALL_PLAYER_METRICS = set(PLAYER_NUMERIC_METRICS.keys())

# Metrics that are percentages (0-100) — use "avg" not "sum" for aggregation
PERCENTAGE_METRICS = {"pass_completion_pct", "pass_completion_under_pressure_pct", "possession_share"}


# ---------------------------------------------------------------------------
# Team-level metrics (from TeamMatchFacts)
# ---------------------------------------------------------------------------

TEAM_NUMERIC_METRICS = {
    "possession_share": {"description": "Possession percentage (event-share proxy)", "type": "float", "min": 0.0, "max": 100.0},
    "crosses": {"description": "Total crosses", "type": "int", "min": 0},
    "first_shot_minute": {"description": "Minute of first shot", "type": "int", "min": 0},
    "first_goal_minute": {"description": "Minute of first goal", "type": "int", "min": 0},
}

ALL_TEAM_METRICS = set(TEAM_NUMERIC_METRICS.keys())

# Combined metrics for lookup
ALL_METRICS = ALL_PLAYER_METRICS | ALL_TEAM_METRICS


# ---------------------------------------------------------------------------
# Dimensions (filterable fields)
# ---------------------------------------------------------------------------

PLAYER_DIMENSIONS = {
    "player_name": {"description": "Player name (string)", "type": "str"},
    "player_id": {"description": "Player ID (int)", "type": "int"},
    "team_name": {"description": "Team name (string)", "type": "str"},
    "team_id": {"description": "Team ID (int)", "type": "int"},
    "match_id": {"description": "Match ID (int)", "type": "int"},
    "match_date": {"description": "Match date (YYYY-MM-DD)", "type": "str"},
    "stage": {"description": "Competition stage", "type": "str", "values": [
        "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals",
        "3rd Place Final", "Final"
    ]},
    "is_knockout": {"description": "Whether match is knockout round", "type": "bool"},
    "opponent": {"description": "Opponent team name", "type": "str"},
    "was_dismissed": {"description": "Whether player was dismissed", "type": "bool"},
}

MATCH_DIMENSIONS = {
    "match_id": {"description": "Match ID (int)", "type": "int"},
    "match_date": {"description": "Match date (YYYY-MM-DD)", "type": "str"},
    "stage": {"description": "Competition stage", "type": "str"},
    "is_knockout": {"description": "Whether match is knockout round", "type": "bool"},
    "home_team": {"description": "Home team name", "type": "str"},
    "away_team": {"description": "Away team name", "type": "str"},
    "home_score": {"description": "Home team score", "type": "int"},
    "away_score": {"description": "Away team score", "type": "int"},
    "went_to_extra_time": {"description": "Whether match went to extra time", "type": "bool"},
    "went_to_shootout": {"description": "Whether match went to shootout", "type": "bool"},
}

TEAM_DIMENSIONS = {
    "team_name": {"description": "Team name (string)", "type": "str"},
    "team_id": {"description": "Team ID (int)", "type": "int"},
    "match_id": {"description": "Match ID (int)", "type": "int"},
    "match_date": {"description": "Match date (YYYY-MM-DD)", "type": "str"},
    "stage": {"description": "Competition stage", "type": "str"},
    "is_knockout": {"description": "Whether match is knockout round", "type": "bool"},
    "opponent": {"description": "Opponent team name", "type": "str"},
}


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

AGGREGATIONS = {
    "sum": {"description": "Sum of values", "applicable": "numeric"},
    "avg": {"description": "Average of values", "applicable": "numeric"},
    "mean": {"description": "Average of values (alias for avg)", "applicable": "numeric"},
    "max": {"description": "Maximum value", "applicable": "numeric"},
    "min": {"description": "Minimum value", "applicable": "numeric"},
    "count": {"description": "Count of records", "applicable": "all"},
}


# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

INTENTS = {
    "numeric": "A specific numeric value for a metric (e.g., 'How many goals did Messi score?')",
    "superlative": "The top/bottom N entities for a metric (e.g., 'Who scored the most goals?')",
    "slice": "A metric for a specific entity with filters (e.g., 'Koundé's xG vs Morocco')",
    "aggregation": "An aggregated value across a group (e.g., 'Which team had the highest xG?')",
}


# ---------------------------------------------------------------------------
# Synonym mapping (for query parsing)
# ---------------------------------------------------------------------------

METRIC_SYNONYMS = {
    # Goals
    "goals": "goals",
    "goal": "goals",
    "scored": "goals",
    "scoring": "goals",
    # Assists
    "assists": "assists",
    "assist": "assists",
    # xG
    "xg": "xg",
    "expected goals": "xg",
    "expected_goals": "xg",
    "xG": "xg",
    # Shots
    "shots": "shots",
    "shot": "shots",
    "attempts": "shots",
    # Passes
    "passes": "passes_attempted",
    "pass": "passes_attempted",
    "passing": "passes_completed",
    "pass completion": "pass_completion_pct",
    "pass accuracy": "pass_completion_pct",
    # Tackles
    "tackles": "successful_tackles",
    "tackle": "successful_tackles",
    "successful tackles": "successful_tackles",
    # Interceptions
    "interceptions": "successful_interceptions",
    "interception": "successful_interceptions",
    # Minutes
    "minutes": "minutes",
    "playing time": "minutes",
    # Pressures
    "pressures": "pressures",
    "pressure": "pressures",
    # Clearances
    "clearances": "clearances",
    "clearance": "clearances",
    # Carries
    "carries": "carries",
    "carry": "carries",
    "dribbles": "carries",
    "dribble": "carries",
    # Team metrics
    "possession": "possession_share",
    "possession share": "possession_share",
    "possession average": "possession_share",
    "ball possession": "possession_share",
    "crosses": "crosses",
    "cross": "crosses",
    "crossing": "crosses",
    # Fouls
    "fouls": "fouls_committed",
    "foul": "fouls_committed",
    "fouls committed": "fouls_committed",
    "committed fouls": "fouls_committed",
    # Yellow cards
    "yellow cards": "yellow_cards",
    "yellow card": "yellow_cards",
    "bookings": "yellow_cards",
    "booking": "yellow_cards",
    "booked": "yellow_cards",
    # Second yellow
    "second yellow": "second_yellow_cards",
    "second yellow card": "second_yellow_cards",
    "second yellow cards": "second_yellow_cards",
    # Red cards
    "red cards": "red_cards",
    "red card": "red_cards",
    "sent off": "red_cards",
    "dismissals": "red_cards",
    "dismissed": "red_cards",
    # Goalkeeper
    "saves": "saves",
    "save": "saves",
    "stops": "saves",
    "shot saved": "saves",
    "goals conceded": "goals_conceded",
    "goals let in": "goals_conceded",
    "conceded": "goals_conceded",
    "penalty save": "penalties_saved",
    "penalty saves": "penalties_saved",
    "penalty saved": "penalties_saved",
    "saved penalty": "penalties_saved",
    "saved penalties": "penalties_saved",
    "penalties": "penalties_saved",
    "claims": "claims",
    "claim": "claims",
    "collected": "claims",
    "punches": "punches",
    "punch": "punches",
    "sweeper": "sweeper_actions",
    "sweeper actions": "sweeper_actions",
    "sweeper action": "sweeper_actions",
    "keeper sweeper": "sweeper_actions",
}

AGGREGATION_SYNONYMS = {
    "most": "max",
    "highest": "max",
    "best": "max",
    "top": "max",
    "least": "min",
    "lowest": "min",
    "worst": "min",
    "bottom": "min",
    "total": "sum",
    "sum": "sum",
    "average": "avg",
    "avg": "avg",
    "mean": "avg",
    "per match": "avg",
    "per game": "avg",
}

STAGE_SYNONYMS = {
    "group": "Group Stage",
    "group stage": "Group Stage",
    "groups": "Group Stage",
    "round of 16": "Round of 16",
    "r16": "Round of 16",
    "quarter": "Quarter-finals",
    "quarterfinals": "Quarter-finals",
    "quarter-finals": "Quarter-finals",
    "qf": "Quarter-finals",
    "semi": "Semi-finals",
    "semifinals": "Semi-finals",
    "semi-finals": "Semi-finals",
    "sf": "Semi-finals",
    "final": "Final",
    "3rd place": "3rd Place Final",
    "third place": "3rd Place Final",
    "knockout": None,  # special: means is_knockout=True
    "knockouts": None,
    "knockout stage": None,
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def resolve_metric(name: str) -> str | None:
    """Resolve a metric name or synonym to the canonical metric name."""
    name_lower = name.lower().strip()
    # Direct match (player or team metrics)
    if name_lower in ALL_METRICS:
        return name_lower
    # Synonym match
    return METRIC_SYNONYMS.get(name_lower)


def resolve_aggregation(name: str) -> str | None:
    """Resolve an aggregation name or synonym."""
    name_lower = name.lower().strip()
    if name_lower in AGGREGATIONS:
        return name_lower
    return AGGREGATION_SYNONYMS.get(name_lower)


def resolve_stage(name: str) -> str | None:
    """Resolve a stage name or synonym."""
    name_lower = name.lower().strip()
    if name_lower in {"Group Stage", "Round of 16", "Quarter-finals",
                       "Semi-finals", "3rd Place Final", "Final"}:
        return name_lower
    return STAGE_SYNONYMS.get(name_lower)


def is_metric(name: str) -> bool:
    """Check if a name is a valid metric."""
    return resolve_metric(name) is not None


def is_dimension(name: str) -> bool:
    """Check if a name is a valid dimension."""
    return name.lower().strip() in PLAYER_DIMENSIONS or name.lower().strip() in MATCH_DIMENSIONS
