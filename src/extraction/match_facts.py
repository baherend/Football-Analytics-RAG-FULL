"""
match_facts.py — Phase 1: Structured Layer

Single source of truth for all structured data extracted from StatsBomb Open Data.
Produces three record types at clearly defined grains:

    PlayerMatchFacts  — one per player per match (minutes > 0)
    MatchFacts        — one per match
    TeamMatchFacts    — one per team per match

All downstream phases (document generation, structured queries) read from
these persisted records. No raw JSON parsing happens after this module.

CRITICAL: The extraction logic MUST be identical to what 01_documents.py does.
Do not reimplement any metric computation — reuse the same formulas.
"""

from __future__ import annotations

import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.extraction.minutes_played import build_match_context, minutes_played

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("open-data-master/data")
COMPETITION_ID = 43
SEASON_ID = 106

# ---------------------------------------------------------------------------
# §1 Constants (identical to 01_documents.py)
# ---------------------------------------------------------------------------

SHOOTOUT_PERIOD = 5
OPEN_PLAY_PERIODS = (1, 2, 3, 4)
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102, 18, 62
FINAL_THIRD_X = 80
SUCCESSFUL_OUTCOMES = frozenset({"Won", "Success", "Success In Play", "Success Out"})
DISMISSAL_CARDS = frozenset({"Red Card", "Second Yellow"})
ON_BALL_EVENT_TYPES = frozenset({"Pass", "Carry", "Dribble", "Shot"})

STAGE_ORDER = ["Group Stage", "Round of 16", "Quarter-finals",
               "Semi-finals", "3rd Place Final", "Final"]

# §6 Level 2 build-up narration
HIGH_XG_THRESHOLD = 0.30
BUILDUP_WALKBACK_CAP = 5
BUILDUP_EXCLUDED_TYPES = frozenset({
    "Ball Receipt*", "Pressure", "Carry", "Goal Keeper",
    "Half Start", "Half End", "Starting XI", "Tactical Shift",
})

# ---------------------------------------------------------------------------
# Schema: PlayerMatchFacts
# ---------------------------------------------------------------------------


@dataclass
class PlayerMatchFacts:
    """One record per player per match (minutes > 0). Grain: (player_id, match_id)."""

    # Identity
    player_id: int
    player_name: str
    team_name: str
    team_id: int | None
    match_id: int

    # Match context
    match_date: str
    stage: str
    opponent: str | None
    is_knockout: bool
    was_dismissed: bool
    roles: list[str]

    # Minutes (from minutes_played.py)
    minutes: float

    # Performance metrics (periods 1-4 only)
    shots: int = 0
    xg: float = 0.0
    goals: int = 0
    assists: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    pass_completion_pct: float | None = None
    passes_under_pressure: int = 0
    passes_under_pressure_completed: int = 0
    pass_completion_under_pressure_pct: float | None = None
    shots_inside_box: int = 0
    shots_outside_box: int = 0
    successful_tackles: int = 0
    successful_interceptions: int = 0
    clearances: int = 0
    ball_losses: int = 0
    carries: int = 0
    carry_distance: float = 0.0
    pressures: int = 0
    final_third_passes: int = 0

    # Discipline
    fouls_committed: int = 0
    yellow_cards: int = 0
    second_yellow_cards: int = 0
    red_cards: int = 0

    # Goalkeeper metrics
    saves: int = 0
    goals_conceded: int = 0
    penalties_saved: int = 0
    claims: int = 0
    punches: int = 0
    sweeper_actions: int = 0

    # Per-period breakdown (enables period-sliceable queries)
    # Key: period number (1-4) as string, Value: metric block
    by_period: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Schema: MatchFacts
# ---------------------------------------------------------------------------


@dataclass
class GoalFact:
    event_id: str
    player: str
    team: str
    period: int
    minute: int
    second: int
    xg: float
    outcome: str
    body_part: str | None
    shot_type: str | None
    play_pattern: str | None
    deflected: bool
    open_goal: bool
    first_time: bool
    follows_dribble: bool
    possession: int | None
    location: list[float] | None
    assist: str | None
    buildup_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OwnGoalFact:
    player: str | None
    conceded_by: str
    credited_to: str
    period: int
    minute: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CardFact:
    player: str
    team: str
    card: str
    period: int
    minute: int
    second: int
    source: str
    infringement: str | None
    dismissal: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubstitutionFact:
    off: str
    on: str
    team: str
    period: int
    minute: int
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShotFact:
    event_id: str
    player: str
    team: str
    period: int
    minute: int
    second: int
    xg: float
    outcome: str
    body_part: str | None
    shot_type: str | None
    play_pattern: str | None
    deflected: bool
    open_goal: bool
    first_time: bool
    follows_dribble: bool
    location: list[float] | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchFacts:
    """One record per match. Contains all match-level facts."""

    # Identity
    match_id: int
    match_date: str
    stage: str
    is_knockout: bool
    stadium: str

    # Teams
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int

    # Score (excludes shootout)
    home_score: int
    away_score: int
    score_excludes_shootout: bool = True

    # Match state
    went_to_extra_time: bool = False
    went_to_shootout: bool = False

    # Possession (event-share proxy, §7.4)
    possession: dict[str, float | None] = field(default_factory=dict)
    possession_is_proxy: bool = True

    # Events
    goals: list[GoalFact] = field(default_factory=list)
    own_goals: list[OwnGoalFact] = field(default_factory=list)
    cards: list[CardFact] = field(default_factory=list)
    substitutions: list[SubstitutionFact] = field(default_factory=list)
    shots: list[ShotFact] = field(default_factory=list)

    # Shootout
    shootout_goals: dict[str, int] = field(default_factory=dict)

    # Validation
    # NOTE: Only checks card COUNT parity, not card identity. Covers all 64
    # matches but lineup cards schema is undocumented (§7 limitation #1).
    card_count_matches: bool = True
    lineup_card_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Schema: TeamMatchFacts
# ---------------------------------------------------------------------------


@dataclass
class TeamMatchFacts:
    """One record per team per match."""

    team_name: str
    team_id: int | None
    match_id: int
    match_date: str
    stage: str
    opponent: str
    is_knockout: bool

    # Possession
    possession_share: float | None = None
    possession_is_proxy: bool = True

    # Style
    play_patterns: dict[str, int] = field(default_factory=dict)
    formations: dict[str, int] = field(default_factory=dict)
    pass_types: dict[str, int] = field(default_factory=dict)
    crosses: int = 0

    # Timing
    first_shot_minute: int | None = None
    first_goal_minute: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# §1 Global helpers (identical to 01_documents.py)
# ---------------------------------------------------------------------------


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def event_sort_key(event: dict) -> tuple:
    """§1: never sort by minute alone — periods overlap on the minute axis."""
    return (event.get("period", 0), event.get("minute", 0),
            event.get("second", 0), event.get("index", 0))


def in_play(event: dict) -> bool:
    """§1: exclude period 5 (penalty shootout) from aggregation."""
    return event.get("period") in OPEN_PLAY_PERIODS


def is_flag(container: dict, key: str) -> bool:
    """§1: boolean attributes are ABSENT when false, never present as false."""
    return bool(container.get(key, False))


def is_complete_pass(event: dict) -> bool:
    """§1: a pass with no outcome key is complete."""
    return "outcome" not in event.get("pass", {})


def card_of(event: dict) -> dict | None:
    """§5: cards live under foul_committed or bad_behaviour. Key on name only."""
    for block in ("foul_committed", "bad_behaviour"):
        card = (event.get(block) or {}).get("card")
        if card:
            return card
    return None


def euclidean(start, end) -> float:
    """§4 carry distance, in yards on the 120x80 pitch."""
    if not start or not end:
        return 0.0
    return math.hypot(end[0] - start[0], end[1] - start[1])


def shot_inside_box(location) -> bool:
    """§4: penalty area = x >= 102 and 18 <= y <= 62."""
    if not location:
        return False
    return location[0] >= BOX_X_MIN and BOX_Y_MIN <= location[1] <= BOX_Y_MAX


# ---------------------------------------------------------------------------
# Prose helpers (for build-up text pre-computation)
# ---------------------------------------------------------------------------


def _plural(word: str, count: int) -> str:
    if count == 1:
        return word
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _count_phrase(word: str, count: int) -> str:
    article = "an" if word[0] in "aeiou" else "a"
    return f"{article} {word}" if count == 1 else f"{count} {_plural(word, count)}"


def _oxford(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _describe_buildup(goal: dict, ordered_events: list[dict]) -> str:
    """
    §6: walk backwards through events sharing the goal's possession id and
    narrate the chain. Capped at BUILDUP_WALKBACK_CAP events.
    """
    possession_id = goal.get("possession")
    if possession_id is None:
        return ""
    goal_key = (goal["period"], goal["minute"], goal["second"], 10 ** 9)
    chain = [
        e for e in ordered_events
        if e.get("possession") == possession_id
        and e["type"]["name"] not in BUILDUP_EXCLUDED_TYPES
        and e.get("team", {}).get("name") == goal["team"]
        and e.get("id") != goal["event_id"]
        and event_sort_key(e) < goal_key
    ][-BUILDUP_WALKBACK_CAP:]
    if not chain:
        return "No build-up actions preceded the goal within the same possession."
    counts = Counter(e["type"]["name"] for e in chain)
    parts = [_count_phrase(name.lower(), n) for name, n in counts.items()]
    return (f"The goal followed {_oxford(parts)} within the same possession "
            f"(walk-back capped at {BUILDUP_WALKBACK_CAP} events).")


# ---------------------------------------------------------------------------
# Extraction: Match-level facts
# ---------------------------------------------------------------------------


def _extract_match_facts(match: dict, events: list[dict], lineups: list[dict]) -> MatchFacts:
    """Extract all match-level facts from one match."""
    ordered = sorted(events, key=event_sort_key)
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]

    goals, own_goals, cards, subs = [], [], [], []
    all_shots, shootout_goals = [], Counter()
    possession_events = Counter()
    assist_by_shot: dict[str, str] = {}
    extra_time = False

    for event in ordered:
        etype = event["type"]["name"]
        period = event.get("period")
        if period in (3, 4):
            extra_time = True

        if in_play(event) and etype in ON_BALL_EVENT_TYPES:
            possession_events[event["team"]["name"]] += 1

        if etype == "Pass" and is_flag(event["pass"], "goal_assist"):
            shot_id = event["pass"].get("assisted_shot_id")
            if shot_id:
                assist_by_shot[shot_id] = event["player"]["name"]

        if etype == "Shot":
            shot = event["shot"]
            if period == SHOOTOUT_PERIOD:
                if shot["outcome"]["name"] == "Goal":
                    shootout_goals[event["team"]["name"]] += 1
                continue
            if not in_play(event):
                continue
            record = ShotFact(
                event_id=event["id"],
                player=event["player"]["name"],
                team=event["team"]["name"],
                period=period,
                minute=event["minute"],
                second=event["second"],
                xg=shot.get("statsbomb_xg", 0.0),
                outcome=shot["outcome"]["name"],
                body_part=shot.get("body_part", {}).get("name"),
                shot_type=shot.get("type", {}).get("name"),
                play_pattern=event.get("play_pattern", {}).get("name"),
                deflected=is_flag(shot, "deflected"),
                open_goal=is_flag(shot, "open_goal"),
                first_time=is_flag(shot, "first_time"),
                follows_dribble=is_flag(shot, "follows_dribble"),
                location=event.get("location"),
            )
            all_shots.append(record)
            if record.outcome == "Goal":
                goal_dict = {
                    "event_id": record.event_id,
                    "player": record.player,
                    "team": record.team,
                    "period": record.period,
                    "minute": record.minute,
                    "second": record.second,
                }
                goals.append(GoalFact(
                    event_id=record.event_id,
                    player=record.player,
                    team=record.team,
                    period=record.period,
                    minute=record.minute,
                    second=record.second,
                    xg=record.xg,
                    outcome=record.outcome,
                    body_part=record.body_part,
                    shot_type=record.shot_type,
                    play_pattern=record.play_pattern,
                    deflected=record.deflected,
                    open_goal=record.open_goal,
                    first_time=record.first_time,
                    follows_dribble=record.follows_dribble,
                    possession=event.get("possession"),
                    location=record.location,
                    assist=assist_by_shot.get(record.event_id),
                    buildup_text=_describe_buildup(
                        {**goal_dict, "possession": event.get("possession")},
                        ordered),
                ))

        elif etype == "Own Goal Against":
            conceding = event["team"]["name"]
            own_goals.append(OwnGoalFact(
                player=event.get("player", {}).get("name"),
                conceded_by=conceding,
                credited_to=away if conceding == home else home,
                period=period,
                minute=event["minute"],
            ))

        card = card_of(event)
        if card:
            foul_type = (event.get("foul_committed") or {}).get("type", {}).get("name")
            cards.append(CardFact(
                player=event["player"]["name"],
                team=event["team"]["name"],
                card=card["name"],
                period=period,
                minute=event["minute"],
                second=event["second"],
                source=etype,
                infringement=foul_type,
                dismissal=card["name"] in DISMISSAL_CARDS,
            ))

        if etype == "Substitution":
            sub = event["substitution"]
            subs.append(SubstitutionFact(
                off=event["player"]["name"],
                on=sub["replacement"]["name"],
                team=event["team"]["name"],
                period=period,
                minute=event["minute"],
                reason=sub.get("outcome", {}).get("name"),
            ))

    total_on_ball = sum(possession_events.values())
    possession = {
        team: (count / total_on_ball * 100.0) if total_on_ball else None
        for team, count in possession_events.items()
    }

    lineup_card_count = sum(
        len(player.get("cards") or []) for team in lineups for player in team["lineup"]
    )

    return MatchFacts(
        match_id=match["match_id"],
        match_date=match["match_date"],
        stage=match["competition_stage"]["name"],
        is_knockout=match["competition_stage"]["name"] != "Group Stage",
        stadium=(match.get("stadium", {}).get("name") or "unspecified").strip(),
        home_team=home,
        away_team=away,
        home_team_id=match["home_team"]["home_team_id"],
        away_team_id=match["away_team"]["away_team_id"],
        home_score=match["home_score"],
        away_score=match["away_score"],
        went_to_extra_time=extra_time,
        went_to_shootout=bool(shootout_goals),
        possession=possession,
        goals=goals,
        own_goals=own_goals,
        cards=cards,
        substitutions=subs,
        shots=all_shots,
        shootout_goals=dict(shootout_goals),
        card_count_matches=lineup_card_count == len(cards),
        lineup_card_count=lineup_card_count,
    )


# ---------------------------------------------------------------------------
# Extraction: Player-match facts
# ---------------------------------------------------------------------------


def _lineup_player_ids(lineups: list[dict]) -> dict[tuple[str, str], int]:
    """Map (team_name, player_name) -> player_id from lineups."""
    return {(team["team_name"], player["player_name"]): player["player_id"]
            for team in lineups for player in team["lineup"]}


def _compute_player_stats(events: list[dict], minutes: dict,
                          player_ids: dict[tuple[str, str], int]) -> dict:
    """
    Build the metric block for every player with minutes > 0.
    All metrics use only periods 1-4 (see in_play).
    Identical logic to 01_documents.py compute_player_stats.

    Enhanced with per-period breakdowns (by_period) for period-sliceable queries.
    """
    blank = {
        "shots": 0, "xg": 0.0, "goals": 0, "assists": 0,
        "passes_attempted": 0, "passes_completed": 0,
        "passes_under_pressure": 0, "passes_under_pressure_completed": 0,
        "shots_inside_box": 0, "shots_outside_box": 0,
        "successful_tackles": 0, "successful_interceptions": 0,
        "clearances": 0, "ball_losses": 0,
        "carries": 0, "carry_distance": 0.0,
        "pressures": 0, "final_third_passes": 0,
        "fouls_committed": 0, "yellow_cards": 0, "second_yellow_cards": 0, "red_cards": 0,
        "saves": 0, "goals_conceded": 0, "penalties_saved": 0,
        "claims": 0, "punches": 0, "sweeper_actions": 0,
    }

    # Per-period metric keys (the 17 event-level stored counts)
    PER_PERIOD_KEYS = [
        "shots", "xg", "goals", "assists",
        "passes_attempted", "passes_completed",
        "passes_under_pressure", "passes_under_pressure_completed",
        "shots_inside_box", "shots_outside_box",
        "successful_tackles", "successful_interceptions",
        "clearances", "ball_losses",
        "carries", "carry_distance",
        "pressures", "final_third_passes",
        "fouls_committed", "yellow_cards", "second_yellow_cards", "red_cards",
        "saves", "goals_conceded", "penalties_saved",
        "claims", "punches", "sweeper_actions",
    ]

    def blank_period():
        return {k: 0 if k not in ("xg", "carry_distance") else 0.0
                for k in PER_PERIOD_KEYS}

    stats: dict[tuple[str, str], dict] = {}

    def slot(event):
        team = event["team"]["name"]
        player = event["player"]["name"]
        key = (team, player)
        if key not in stats:
            stats[key] = dict(blank, player_id=(event.get("player") or {}).get("id"),
                              team_id=(event.get("team") or {}).get("id"),
                              by_period={})
        return stats[key]

    def accumulate(block, event):
        """Add one event's contribution to a metric block."""
        etype = event["type"]["name"]

        if etype == "Shot":
            shot = event["shot"]
            block["shots"] += 1
            block["xg"] += shot.get("statsbomb_xg", 0.0)
            if shot["outcome"]["name"] == "Goal":
                block["goals"] += 1
            if shot_inside_box(event.get("location")):
                block["shots_inside_box"] += 1
            else:
                block["shots_outside_box"] += 1

        elif etype == "Pass":
            passd = event["pass"]
            block["passes_attempted"] += 1
            complete = is_complete_pass(event)
            if complete:
                block["passes_completed"] += 1
            if is_flag(event, "under_pressure"):
                block["passes_under_pressure"] += 1
                if complete:
                    block["passes_under_pressure_completed"] += 1
            end = passd.get("end_location")
            if end and end[0] >= FINAL_THIRD_X:
                block["final_third_passes"] += 1
            if is_flag(passd, "goal_assist"):
                block["assists"] += 1

        elif etype == "Duel":
            duel = event.get("duel", {})
            if duel.get("type", {}).get("name") == "Tackle" and \
                    duel.get("outcome", {}).get("name") in SUCCESSFUL_OUTCOMES:
                block["successful_tackles"] += 1

        elif etype == "Interception":
            outcome = event.get("interception", {}).get("outcome", {}).get("name")
            if outcome in SUCCESSFUL_OUTCOMES:
                block["successful_interceptions"] += 1

        elif etype == "Clearance":
            block["clearances"] += 1

        elif etype in ("Miscontrol", "Dispossessed"):
            block["ball_losses"] += 1

        elif etype == "Carry":
            block["carries"] += 1
            block["carry_distance"] += euclidean(
                event.get("location"), event.get("carry", {}).get("end_location"))

        elif etype == "Pressure":
            block["pressures"] += 1

        elif etype == "Goal Keeper":
            gk = event.get("goalkeeper", {})
            gk_type = gk.get("type", {}).get("name", "")
            gk_outcome = gk.get("outcome", {}).get("name", "")

            # Saves: Shot Saved (any outcome) + Penalty Saved
            if gk_type in ("Shot Saved", "Shot Saved to Post", "Penalty Saved"):
                block["saves"] += 1
                if gk_type == "Penalty Saved":
                    block["penalties_saved"] += 1

            # Goals conceded: Goal Conceded or Penalty Conceded (any outcome)
            if gk_type in ("Goal Conceded", "Penalty Conceded"):
                block["goals_conceded"] += 1

            # Claims: Collected or Keeper Sweeper with Claim outcome
            if gk_type == "Collected" or (gk_type == "Keeper Sweeper" and gk_outcome == "Claim"):
                block["claims"] += 1

            # Punches
            if gk_type == "Punch":
                block["punches"] += 1

            # Sweeper actions: Keeper Sweeper (any outcome)
            if gk_type == "Keeper Sweeper":
                block["sweeper_actions"] += 1

        elif etype == "Foul Committed":
            block["fouls_committed"] += 1

        # Cards (from Foul Committed or Bad Behaviour)
        if etype in ("Foul Committed", "Bad Behaviour"):
            card = None
            for cb in ("foul_committed", "bad_behaviour"):
                c = (event.get(cb) or {}).get("card")
                if c:
                    card = c
                    break
            if card:
                card_name = card.get("name", "")
                if card_name == "Yellow Card":
                    block["yellow_cards"] += 1
                elif card_name == "Second Yellow":
                    block["second_yellow_cards"] += 1
                elif card_name == "Red Card":
                    block["red_cards"] += 1

    for event in events:
        if not in_play(event) or "player" not in event:
            continue

        row = slot(event)
        period = event.get("period")

        # Accumulate to total
        accumulate(row, event)

        # Accumulate to period block (use string keys for JSON compatibility)
        if period in OPEN_PLAY_PERIODS:
            period_key = str(period)
            if period_key not in row["by_period"]:
                row["by_period"][period_key] = blank_period()
            accumulate(row["by_period"][period_key], event)

    # attach minutes and derive percentage metrics
    result = {}
    for (team, player), row in stats.items():
        mins = minutes.get((team, player), 0.0)
        if mins <= 0:
            continue
        row["minutes"] = mins
        row["player_id"] = player_ids.get((team, player), row.get("player_id"))
        row["pass_completion_pct"] = (
            row["passes_completed"] / row["passes_attempted"] * 100.0
            if row["passes_attempted"] else None)
        row["pass_completion_under_pressure_pct"] = (
            row["passes_under_pressure_completed"] / row["passes_under_pressure"] * 100.0
            if row["passes_under_pressure"] else None)
        # Round per-period blocks
        row["by_period"] = {
            str(p): {k: (round(v, 4) if k in ("xg", "carry_distance") else v)
                     for k, v in block.items()}
            for p, block in row.get("by_period", {}).items()
        }
        result[(team, player)] = row

    # players with minutes but no on-ball events still get a record
    for (team, player), mins in minutes.items():
        if mins > 0 and (team, player) not in result:
            result[(team, player)] = dict(
                blank, minutes=mins, player_id=player_ids.get((team, player)),
                team_id=None, pass_completion_pct=None,
                pass_completion_under_pressure_pct=None,
                by_period={})
    return result


def _player_roles(lineups: list[dict], team: str, player: str,
                  dismissal_second: int | None = None) -> list[str]:
    """
    §0/§6 tactical role from lineups.positions.
    Identical logic to 01_documents.py player_roles.
    """
    segments = []
    for entry in lineups:
        if entry["team_name"] != team:
            continue
        for row in entry["lineup"]:
            if row["player_name"] == player:
                segments = row.get("positions") or []

    roles = []
    for seg in segments:
        from_period, to_period = seg.get("from_period"), seg.get("to_period")
        if to_period is not None and from_period is not None and to_period < from_period:
            continue
        start = seg.get("from")
        if dismissal_second is not None and start is not None:
            parts = start.split(":")
            if len(parts) == 2:
                start_sec = int(parts[0]) * 60 + int(parts[1])
                if start_sec >= dismissal_second:
                    continue
        roles.append(seg["position"])
    return roles


def _extract_player_match_facts(
    match: dict,
    events: list[dict],
    lineups: list[dict],
    match_facts: MatchFacts,
) -> list[PlayerMatchFacts]:
    """Extract all player-match facts from one match."""
    context = build_match_context(events, lineups)
    minutes, _ = minutes_played(lineups, context, collect_anomalies=True)
    player_ids = _lineup_player_ids(lineups)
    stats = _compute_player_stats(events, minutes, player_ids)

    opponent_of = {match_facts.home_team: match_facts.away_team,
                   match_facts.away_team: match_facts.home_team}

    # Dismissal times for role filtering
    dismissal_seconds = {
        (c.team, c.player): c.minute * 60 + c.second
        for c in match_facts.cards
        if c.dismissal and c.period != SHOOTOUT_PERIOD
    }
    dismissed = set(dismissal_seconds)

    results = []
    for (team, player), row in sorted(stats.items()):
        roles = _player_roles(lineups, team, player,
                              dismissal_seconds.get((team, player)))
        pid = row.get("player_id") or player_ids.get((team, player))

        results.append(PlayerMatchFacts(
            player_id=pid,
            player_name=player,
            team_name=team,
            team_id=row.get("team_id"),
            match_id=match_facts.match_id,
            match_date=match_facts.match_date,
            stage=match_facts.stage,
            opponent=opponent_of.get(team),
            is_knockout=match_facts.is_knockout,
            was_dismissed=(team, player) in dismissed,
            roles=roles,
            minutes=row.get("minutes", 0.0),
            shots=row.get("shots", 0),
            xg=row.get("xg", 0.0),
            goals=row.get("goals", 0),
            assists=row.get("assists", 0),
            passes_attempted=row.get("passes_attempted", 0),
            passes_completed=row.get("passes_completed", 0),
            pass_completion_pct=row.get("pass_completion_pct"),
            passes_under_pressure=row.get("passes_under_pressure", 0),
            passes_under_pressure_completed=row.get("passes_under_pressure_completed", 0),
            pass_completion_under_pressure_pct=row.get("pass_completion_under_pressure_pct"),
            shots_inside_box=row.get("shots_inside_box", 0),
            shots_outside_box=row.get("shots_outside_box", 0),
            successful_tackles=row.get("successful_tackles", 0),
            successful_interceptions=row.get("successful_interceptions", 0),
            clearances=row.get("clearances", 0),
            ball_losses=row.get("ball_losses", 0),
            carries=row.get("carries", 0),
            carry_distance=row.get("carry_distance", 0.0),
            pressures=row.get("pressures", 0),
            final_third_passes=row.get("final_third_passes", 0),
            fouls_committed=row.get("fouls_committed", 0),
            yellow_cards=row.get("yellow_cards", 0),
            second_yellow_cards=row.get("second_yellow_cards", 0),
            red_cards=row.get("red_cards", 0),
            saves=row.get("saves", 0),
            goals_conceded=row.get("goals_conceded", 0),
            penalties_saved=row.get("penalties_saved", 0),
            claims=row.get("claims", 0),
            punches=row.get("punches", 0),
            sweeper_actions=row.get("sweeper_actions", 0),
            by_period=row.get("by_period", {}),
        ))
    return results


# ---------------------------------------------------------------------------
# Extraction: Team-match facts
# ---------------------------------------------------------------------------


def _extract_team_match_facts(
    events: list[dict],
    match_facts: MatchFacts,
) -> list[TeamMatchFacts]:
    """Extract team-match facts for both teams."""
    team_data: dict[str, dict] = {}

    for team_name in (match_facts.home_team, match_facts.away_team):
        team_data[team_name] = {
            "team_id": None,
            "possession_team_events": 0,
            "match_event_total": 0,
            "play_patterns": Counter(),
            "formations": Counter(),
            "pass_types": Counter(),
            "crosses": 0,
            "first_shot_minute": None,
            "first_goal_minute": None,
        }

    # First shot / first goal timing
    for shot in match_facts.shots:
        td = team_data.get(shot.team)
        if not td:
            continue
        if td["first_shot_minute"] is None or shot.minute < td["first_shot_minute"]:
            td["first_shot_minute"] = shot.minute
        if shot.outcome == "Goal":
            if td["first_goal_minute"] is None or shot.minute < td["first_goal_minute"]:
                td["first_goal_minute"] = shot.minute

    # Event-level accumulation
    for event in events:
        if not in_play(event):
            continue

        possession_team = event.get("possession_team", {}).get("name")
        for side in (match_facts.home_team, match_facts.away_team):
            team_data[side]["match_event_total"] += 1
        if possession_team in team_data:
            team_data[possession_team]["possession_team_events"] += 1

        team = event.get("team", {}).get("name")
        if team not in team_data:
            continue
        td = team_data[team]
        if td["team_id"] is None:
            td["team_id"] = (event.get("team") or {}).get("id")

        pattern = event.get("play_pattern", {}).get("name")
        if pattern:
            td["play_patterns"][pattern] += 1

        etype = event["type"]["name"]
        if etype == "Pass":
            passd = event["pass"]
            td["pass_types"][passd.get("type", {}).get("name", "Open Play")] += 1
            if is_flag(passd, "cross"):
                td["crosses"] += 1
        elif etype in ("Starting XI", "Tactical Shift"):
            formation = event.get("tactics", {}).get("formation")
            if formation:
                td["formations"][str(formation)] += 1

    results = []
    for team_name in (match_facts.home_team, match_facts.away_team):
        td = team_data[team_name]
        total_events = td["match_event_total"]
        possession_share = (
            td["possession_team_events"] / total_events * 100.0
            if total_events else None
        )

        opponent = (match_facts.away_team if team_name == match_facts.home_team
                    else match_facts.home_team)

        results.append(TeamMatchFacts(
            team_name=team_name,
            team_id=td["team_id"],
            match_id=match_facts.match_id,
            match_date=match_facts.match_date,
            stage=match_facts.stage,
            opponent=opponent,
            is_knockout=match_facts.is_knockout,
            possession_share=possession_share,
            play_patterns=dict(td["play_patterns"]),
            formations=dict(td["formations"]),
            pass_types=dict(td["pass_types"]),
            crosses=td["crosses"],
            first_shot_minute=td["first_shot_minute"],
            first_goal_minute=td["first_goal_minute"],
        ))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_all(data_root: Path = DATA_ROOT, verbose: bool = True, competition_id: int = COMPETITION_ID, season_id: int = SEASON_ID) -> dict:
    """
    Extract all structured facts for every match in the requested
    competition/season (defaults to WC 2022: competition_id=43, season_id=106).

    Returns:
        {
            "player_match_facts": list[PlayerMatchFacts],
            "match_facts": list[MatchFacts],
            "team_match_facts": list[TeamMatchFacts],
            "diagnostics": dict,
        }
    """
    matches = load_json(data_root / "matches" / str(competition_id) / f"{season_id}.json")

    all_player_facts: list[PlayerMatchFacts] = []
    all_match_facts: list[MatchFacts] = []
    all_team_facts: list[TeamMatchFacts] = []

    diagnostics = {
        "matches_processed": 0,
        "total_player_facts": 0,
        "total_team_facts": 0,
        "card_count_mismatches": [],
    }

    for match in sorted(matches, key=lambda m: m["match_date"]):
        match_id = match["match_id"]
        events = load_json(data_root / "events" / f"{match_id}.json")
        lineups = load_json(data_root / "lineups" / f"{match_id}.json")

        # Extract match-level facts
        mf = _extract_match_facts(match, events, lineups)
        all_match_facts.append(mf)

        if not mf.card_count_matches:
            diagnostics["card_count_mismatches"].append(match_id)

        # Extract player-match facts
        player_facts = _extract_player_match_facts(match, events, lineups, mf)
        all_player_facts.extend(player_facts)

        # Extract team-match facts
        team_facts = _extract_team_match_facts(events, mf)
        all_team_facts.extend(team_facts)

        diagnostics["matches_processed"] += 1
        if verbose:
            stage = match["competition_stage"]["name"]
            print(f"  extracted {match_id}  {mf.home_team} v {mf.away_team} ({stage})")

    diagnostics["total_player_facts"] = len(all_player_facts)
    diagnostics["total_team_facts"] = len(all_team_facts)

    return {
        "player_match_facts": all_player_facts,
        "match_facts": all_match_facts,
        "team_match_facts": all_team_facts,
        "diagnostics": diagnostics,
    }


def persist(result: dict, output_dir: Path = Path("output"), competition_id: int = COMPETITION_ID, season_id: int = SEASON_ID) -> Path:
    """
    Persist extracted facts to JSON files.

    Writes:
        <output_dir>/match_facts.json — all three record types

    Defaults to output/ so the artifact lands where every downstream phase
    (generate_documents.py, resolver.py, cache.py) expects to read it.

    Returns path to the output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "match_facts.json"

    data = {
        "metadata": {
            "competition_id": competition_id,
            "season_id": season_id,
            "extraction_date": None,  # filled by caller if needed
            "record_counts": {
                "player_match_facts": len(result["player_match_facts"]),
                "match_facts": len(result["match_facts"]),
                "team_match_facts": len(result["team_match_facts"]),
            },
        },
        "player_match_facts": [f.to_dict() for f in result["player_match_facts"]],
        "match_facts": [f.to_dict() for f in result["match_facts"]],
        "team_match_facts": [f.to_dict() for f in result["team_match_facts"]],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    return output_path
