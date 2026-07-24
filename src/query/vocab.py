"""
vocab.py — Phase 3: Metric and Dimension Vocabularies

Defines the registered metrics, dimensions, and aggregations available
for structured queries against match_facts.json.

CRITICAL: These vocabularies are grounded in what match_facts.json ACTUALLY
persists. Do not add metrics or dimensions that don't exist in the data.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Metrics (from PlayerMatchFacts)
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
