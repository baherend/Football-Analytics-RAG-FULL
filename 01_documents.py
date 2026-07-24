"""
01_documents.py — Stage 1 of the modular RAG pipeline.

Loads raw StatsBomb Open Data for FIFA World Cup 2022 (competition_id=43,
season_id=106, 64 matches) and emits structured natural-language documents
across five categories, ready for 02_preprocessing.py.

Authoritative specification: rag_document_specs_v3.md. Section markers (§0-§8)
appear in comments beside any non-obvious rule so logic is traceable to the spec
during review. The minutes-played algorithm (§3) is imported from
minutes_played.py and is NOT reimplemented here.

Document counts:
    Level 1     one per match                        (64)
    Level 2     one per match                        (64)
    Level 3     one per player per match, minutes>0
    Level 4     one per player
    Team-level  one per team                         (32)

-----------------------------------------------------------------------------
§7 STATED LIMITATIONS — carried into every document's metadata under
`limitations`, and surfaced in the document text where the affected number
appears. These are declared, not hidden; the analysis is defensible only if
they are disclosed.

  1. `lineups.positions` and `lineups.cards` are undocumented. The official
     Open Data Lineups v2.0.0 spec (01 May 2019) documents only team_id,
     team_name and a lineup array of player_id / player_name /
     player_nickname / jersey_number / country. Both fields exist in the data
     but have no published schema, so no enumeration of `end_reason` values is
     guaranteed stable.
  2. `end_reason` does not reliably mark dismissals. "Foul Committed (Red
     Card)" occurs once in 64 matches; there were four sendings-off, three of
     them second yellows producing no red-card end_reason. Dismissals are
     therefore taken from events cards.
  3. Known source-data defects, handled and logged: Hennessey (3857273) has a
     phantom segment starting after his red card; Messi, Acuna and Montiel
     (3869685) have segments whose to_period precedes their from_period;
     Kounde and Varane (3869685) carry duplicate overlapping segments.
  4. Possession % is an event-share proxy, not StatsBomb's broadcast figure —
     matches.json carries no possession field.
  5. Card IDs in the published Events specification do not match this dataset.
     All card joins use `card.name`.
  6. Sampling caveat: positions/cards schema, end_reason distribution and
     card-time truncation were checked across all 64 lineups files. The
     events-vs-lineups card parity check covers 3 matches (19 cards) and is
     not claimed for all 64.
-----------------------------------------------------------------------------

NOTE ON IMPORTING THIS MODULE: the filename begins with a digit, so
`import 01_documents` is not valid Python. Downstream stages should load it via
importlib:

    import importlib.util, pathlib, sys
    spec = importlib.util.spec_from_file_location(
        "documents", pathlib.Path("01_documents.py"))
    documents = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = documents   # required BEFORE exec_module: this
                                         # module uses @dataclass with
                                         # `from __future__ import annotations`,
                                         # and dataclasses resolves the string
                                         # annotations via sys.modules
    spec.loader.exec_module(documents)
    docs = documents.generate_documents()

This is a consequence of the required numbered-file convention, not a design
choice.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# minutes_played.py implements §3 and is used as-is (spec §3, task constraint).
from src.extraction.minutes_played import build_match_context, minutes_played

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path("open-data-master/data")
COMPETITION_ID = 43
SEASON_ID = 106

# ---------------------------------------------------------------------------
# §1 Global parsing constants
# ---------------------------------------------------------------------------

SHOOTOUT_PERIOD = 5                     # §1: excluded from all aggregation
OPEN_PLAY_PERIODS = (1, 2, 3, 4)

# §1 pitch geometry: 120 x 80 yards
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102, 18, 62
FINAL_THIRD_X = 80

# §4 successful tackle / interception outcomes
SUCCESSFUL_OUTCOMES = frozenset({"Won", "Success", "Success In Play", "Success Out"})

# §1/§5 dismissals are keyed on card NAME, never card id
DISMISSAL_CARDS = frozenset({"Red Card", "Second Yellow"})

# §4 possession % — on-ball event types
ON_BALL_EVENT_TYPES = frozenset({"Pass", "Carry", "Dribble", "Shot"})

# §6 Level 2 thresholds
HIGH_XG_THRESHOLD = 0.30

# §6 Level 2 requires the build-up walk-back to be capped at "a fixed number of
# events" and the cap stated, but does not fix the number. AMBIGUITY: value
# chosen here and stated in the generated document text so a reader can see it.
BUILDUP_WALKBACK_CAP = 5

# §4 consistency score is emitted only at or above this many matches
CONSISTENCY_MIN_MATCHES = 3

LIMITATIONS_NOTE = (
    "possession % is an event-share proxy (§7.4); dismissals taken from events "
    "cards, not lineups end_reason (§7.2); lineups positions/cards are "
    "undocumented (§7.1); card joins use card.name (§7.5)"
)

# ---------------------------------------------------------------------------
# Document container
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """One retrievable document. `text` is what gets embedded downstream."""

    document_id: str
    level: str                       # "1" | "2" | "3" | "4" | "team"
    text: str
    metadata: dict = field(default_factory=dict)
    match_id: int | None = None
    player_name: str | None = None
    team_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# §1 Global helpers
# ---------------------------------------------------------------------------


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def event_sort_key(event: dict) -> tuple:
    """
    §1: never sort by `minute` alone — periods overlap on the minute axis
    (period 1 ran to 52' in the Final while period 2 begins at 45').
    """
    return (event.get("period", 0), event.get("minute", 0),
            event.get("second", 0), event.get("index", 0))


def in_play(event: dict) -> bool:
    """
    §1: exclude period 5 (penalty shootout) from aggregation.

    AMBIGUITY: §1 names "every shot, goal, xG, and minutes aggregation". It does
    not name passes, carries, pressures or defensive actions. This filter is
    applied to ALL Level 3/4 player metrics for consistency — a shootout should
    not contribute to any performance metric. In practice period 5 contains only
    Half Start, Shot, Goal Keeper, Bad Behaviour and Half End events, so the
    extension changes nothing but makes the intent explicit.
    """
    return event.get("period") in OPEN_PLAY_PERIODS


def is_flag(container: dict, key: str) -> bool:
    """§1: boolean attributes are ABSENT when false, never present as false."""
    return bool(container.get(key, False))


def is_complete_pass(event: dict) -> bool:
    """
    §1: a pass with no `outcome` key is complete. `pass.recipient` is populated
    even on incomplete passes and must never be used as a completion test.
    """
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
# Prose helpers — documents are full sentences, not stat dumps
# ---------------------------------------------------------------------------


def clock_to_seconds(clock: str | None) -> int | None:
    """'83:43' -> 5023. Mirrors the parser in minutes_played (§3)."""
    if not clock:
        return None
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + int(seconds)


def plural(word: str, count: int) -> str:
    if count == 1:
        return word
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def count_phrase(word: str, count: int) -> str:
    """1 -> 'a pass'; 3 -> '3 passes'."""
    article = "an" if word[0] in "aeiou" else "a"
    return f"{article} {word}" if count == 1 else f"{count} {plural(word, count)}"


def times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def oxford(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def pct(value: float | None, digits: int = 1) -> str:
    return "not available" if value is None else f"{value:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


# ---------------------------------------------------------------------------
# Per-match fact extraction
# ---------------------------------------------------------------------------


def extract_match_facts(match: dict, events: list[dict], lineups: list[dict]) -> dict:
    """Pull every §2/§6 fact needed by Levels 1 and 2 out of one match."""
    ordered = sorted(events, key=event_sort_key)          # §1
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
            extra_time = True                              # §2 extra time played

        # §4 possession % — on-ball event share, periods 1-4
        if in_play(event) and etype in ON_BALL_EVENT_TYPES:
            possession_events[event["team"]["name"]] += 1

        # §2/§6 assists: link the assisting pass to its shot
        if etype == "Pass" and is_flag(event["pass"], "goal_assist"):
            shot_id = event["pass"].get("assisted_shot_id")
            if shot_id:
                assist_by_shot[shot_id] = event["player"]["name"]

        if etype == "Shot":
            shot = event["shot"]
            if period == SHOOTOUT_PERIOD:
                # §2: shootout read ONLY for the shootout scoreline
                if shot["outcome"]["name"] == "Goal":
                    shootout_goals[event["team"]["name"]] += 1
                continue
            if not in_play(event):
                continue
            record = {
                "event_id": event["id"],
                "player": event["player"]["name"],
                "team": event["team"]["name"],
                "period": period,
                "minute": event["minute"],
                "second": event["second"],
                "xg": shot.get("statsbomb_xg", 0.0),
                "outcome": shot["outcome"]["name"],
                "body_part": shot.get("body_part", {}).get("name"),
                "shot_type": shot.get("type", {}).get("name"),        # §2 sole "how scored" source
                "play_pattern": event.get("play_pattern", {}).get("name"),
                "deflected": is_flag(shot, "deflected"),
                "open_goal": is_flag(shot, "open_goal"),
                "first_time": is_flag(shot, "first_time"),
                "follows_dribble": is_flag(shot, "follows_dribble"),
                "possession": event.get("possession"),
                "location": event.get("location"),
            }
            all_shots.append(record)
            if record["outcome"] == "Goal":
                goals.append(record)

        # §2 own goals: "Own Goal Against" carries the player and is used as the
        # canonical record; "Own Goal For" is its mirror on the benefiting team
        # and would double-count. Own goals have no shot object, so no xG and no
        # shot_type, and are excluded from every player's goal total.
        elif etype == "Own Goal Against":
            conceding = event["team"]["name"]
            own_goals.append({
                "player": event.get("player", {}).get("name"),
                "conceded_by": conceding,
                "credited_to": away if conceding == home else home,
                "period": period,
                "minute": event["minute"],
            })

        # §2/§5 cards — events only, keyed on card.name
        card = card_of(event)
        if card:
            foul_type = (event.get("foul_committed") or {}).get("type", {}).get("name")
            cards.append({
                "player": event["player"]["name"],
                "team": event["team"]["name"],
                "card": card["name"],
                "period": period,
                "minute": event["minute"],
                "second": event["second"],
                "source": etype,                 # "Foul Committed" or "Bad Behaviour"
                "infringement": foul_type,       # §2: optional, ~15% of cards
                "dismissal": card["name"] in DISMISSAL_CARDS,
            })

        if etype == "Substitution":
            sub = event["substitution"]
            subs.append({
                "off": event["player"]["name"],
                "on": sub["replacement"]["name"],
                "team": event["team"]["name"],
                "period": period,
                "minute": event["minute"],
                "reason": sub.get("outcome", {}).get("name"),   # §6: may be absent
            })

    # §4 possession % — event-share proxy (§7.4)
    total_on_ball = sum(possession_events.values())
    possession = {
        team: (count / total_on_ball * 100.0) if total_on_ball else None
        for team, count in possession_events.items()
    }

    # §5 card parity check against lineups.cards (count only, never timestamps)
    lineup_card_count = sum(
        len(player.get("cards") or []) for team in lineups for player in team["lineup"]
    )

    return {
        "home": home,
        "away": away,
        "ordered_events": ordered,
        "goals": goals,
        "own_goals": own_goals,
        "shots": all_shots,
        "cards": cards,
        "subs": subs,
        "assist_by_shot": assist_by_shot,
        "possession": possession,
        "shootout_goals": shootout_goals,
        "extra_time": extra_time,
        "card_parity_ok": lineup_card_count == len(cards),
        "lineup_card_count": lineup_card_count,
    }


# ---------------------------------------------------------------------------
# Level 1 — Match Summary (§2)
# ---------------------------------------------------------------------------


def build_level1(match: dict, facts: dict) -> Document:
    match_id = match["match_id"]
    home, away = facts["home"], facts["away"]
    home_score, away_score = match["home_score"], match["away_score"]
    stage = match["competition_stage"]["name"]
    # DATA QUALITY: two stadium names in matches.json carry trailing whitespace
    # ("Al Janoub Stadium   ", "Education City Stadium "), which would surface as
    # double spaces in the embedded text.
    stadium = (match.get("stadium", {}).get("name") or "an unspecified stadium").strip()

    lines = [
        f"{home} played {away} in the {stage} of the FIFA World Cup 2022 on "
        f"{match['match_date']} at {stadium}."
    ]

    # §2: score is end of normal/extra time and EXCLUDES any shootout
    period_note = "after extra time" if facts["extra_time"] else "in normal time"
    if home_score == away_score:
        lines.append(
            f"The match finished level at {home_score}-{away_score} {period_note}. "
            "This score excludes any penalty shootout."
        )
    else:
        winner, loser = (home, away) if home_score > away_score else (away, home)
        lines.append(
            f"{winner} beat {loser} {max(home_score, away_score)}-"
            f"{min(home_score, away_score)} {period_note}. "
            "This score excludes any penalty shootout."
        )

    # §2: shootout rendered as its own line, never merged into the score
    if facts["shootout_goals"]:
        ranked = facts["shootout_goals"].most_common()
        win_team, win_pens = ranked[0]
        lose_team, lose_pens = ranked[1] if len(ranked) > 1 else (
            away if win_team == home else home, 0)
        lines.append(
            f"{win_team} won the penalty shootout {win_pens}-{lose_pens} against {lose_team}."
        )

    # §2 goal scorers, with shot.type.name as the sole "how it was scored" source
    if facts["goals"]:
        for goal in facts["goals"]:
            part = f" with the {goal['body_part'].lower()}" if goal["body_part"] else ""
            how = goal["shot_type"] or "Open Play"
            if how == "Open Play":
                how_txt = "from open play"
            elif how == "Penalty":
                how_txt = "from a penalty"
            else:
                how_txt = f"directly from a {how.lower()}"
            assist = facts["assist_by_shot"].get(goal["event_id"])
            assist_txt = f", assisted by {assist}" if assist else ""
            deflect = ", and the shot was deflected" if goal["deflected"] else ""
            pattern = goal["play_pattern"]
            pattern_txt = ""
            # §2: shot.type marks DIRECT set-pieces only; a header from a corner
            # is Open Play, so play_pattern is reported separately as build-up.
            if pattern and pattern not in ("Regular Play", "Other"):
                pattern_txt = f" The build-up pattern was {pattern.lower()}."
            lines.append(
                f"{goal['player']} scored for {goal['team']} in the "
                f"{ordinal(goal['minute'])} minute{part}, {how_txt}{assist_txt}"
                f" (xG {num(goal['xg'])}){deflect}.{pattern_txt}"
            )
    else:
        lines.append("Neither side scored from open play or a set piece.")

    # §2 own goals listed separately, excluded from player goal totals
    for own in facts["own_goals"]:
        lines.append(
            f"{own['player']} of {own['conceded_by']} scored an own goal in the "
            f"{ordinal(own['minute'])} minute, credited to {own['credited_to']}. "
            "Own goals carry no expected-goals value and are excluded from player goal totals."
        )

    # §4 possession, explicitly labelled as a proxy (§7.4)
    if facts["possession"]:
        parts = [f"{team} {pct(share)}" for team, share in
                 sorted(facts["possession"].items(), key=lambda kv: -(kv[1] or 0))]
        lines.append(
            f"Possession was split {oxford(parts)}. This is an event-share proxy "
            "based on passes, carries, dribbles and shots, not StatsBomb's "
            "broadcast possession figure."
        )

    # §2 cards and dismissals
    if facts["cards"]:
        card_bits = []
        for card in facts["cards"]:
            detail = f" for {card['infringement'].lower()}" if card["infringement"] else ""
            when = (" after the match, during the penalty shootout"
                    if card["period"] == SHOOTOUT_PERIOD
                    else f" in the {ordinal(card['minute'])} minute")
            card_bits.append(
                f"{card['player']} ({card['team']}) received a "
                f"{card['card'].lower()}{when}{detail}"
            )
        lines.append(f"The referee showed {len(facts['cards'])} "
                     f"{plural('card', len(facts['cards']))}. " + oxford(card_bits) + ".")

        dismissals = [c for c in facts["cards"] if c["dismissal"]]
        for d in dismissals:
            kind = ("a second yellow card" if d["card"] == "Second Yellow"
                    else "a straight red card")
            lines.append(
                f"{d['player']} of {d['team']} was sent off for {kind} in the "
                f"{ordinal(d['minute'])} minute, leaving {d['team']} a player short."
            )
    else:
        lines.append("No cards were shown.")

    return Document(
        document_id=f"L1-match-{match_id}",
        level="1",
        match_id=match_id,
        team_name=None,
        text=" ".join(lines),
        metadata={
            "competition_id": COMPETITION_ID,
            "season_id": SEASON_ID,
            "match_date": match["match_date"],
            "stage": stage,
            "is_knockout": stage != "Group Stage",
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "score_excludes_shootout": True,
            "went_to_extra_time": facts["extra_time"],
            "went_to_shootout": bool(facts["shootout_goals"]),
            "goal_count": len(facts["goals"]),
            "own_goal_count": len(facts["own_goals"]),
            "card_count": len(facts["cards"]),
            "dismissal_count": sum(1 for c in facts["cards"] if c["dismissal"]),
            "possession_is_proxy": True,          # §7.4
            "card_parity_ok": facts["card_parity_ok"],   # §5 events vs lineups count
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Level 2 — Key Events (§6)
# ---------------------------------------------------------------------------


# Event types excluded from the build-up narration: bookkeeping events, the
# opponent's reactions, and the two highest-volume types (Carry at ~23% and
# Ball Receipt* at ~26% of all events), which would otherwise swamp the chain.
BUILDUP_EXCLUDED_TYPES = frozenset({
    "Ball Receipt*", "Pressure", "Carry", "Goal Keeper",
    "Half Start", "Half End", "Starting XI", "Tactical Shift",
})


def describe_buildup(goal: dict, ordered_events: list[dict]) -> str:
    """
    §6: walk backwards through events sharing the goal's `possession` id and
    narrate the chain. Capped at BUILDUP_WALKBACK_CAP events; the cap is stated
    in the returned sentence so a reader can see it. Only the scoring team's own
    actions count as build-up.
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
        and e.get("id") != goal["event_id"]              # not the goal itself
        and event_sort_key(e) < goal_key
    ][-BUILDUP_WALKBACK_CAP:]
    if not chain:
        return "No build-up actions preceded the goal within the same possession."
    counts = Counter(e["type"]["name"] for e in chain)
    parts = [count_phrase(name.lower(), n) for name, n in counts.items()]
    return (f"The goal followed {oxford(parts)} within the same possession "
            f"(walk-back capped at {BUILDUP_WALKBACK_CAP} events).")


def build_level2(match: dict, facts: dict) -> Document:
    match_id = match["match_id"]
    lines = [
        f"Key events from {facts['home']} versus {facts['away']}, "
        f"{match['competition_stage']['name']}, FIFA World Cup 2022 "
        f"({match['match_date']})."
    ]

    # §6 every goal
    for goal in facts["goals"]:
        extras = []
        if goal["first_time"]:
            extras.append("a first-time strike")
        if goal["follows_dribble"]:
            extras.append("taken after a successful dribble")
        if goal["deflected"]:
            extras.append("deflected on its way in")
        extra_txt = f" It was {oxford(extras)}." if extras else ""
        assist = facts["assist_by_shot"].get(goal["event_id"])
        assist_txt = f" {assist} provided the assist." if assist else ""
        lines.append(
            f"Goal: {goal['player']} ({goal['team']}) in period {goal['period']}, "
            f"{ordinal(goal['minute'])} minute, with the "
            f"{(goal['body_part'] or 'unspecified body part').lower()}, "
            f"shot type {goal['shot_type'] or 'unknown'}, xG {num(goal['xg'])}."
            f"{assist_txt}{extra_txt} "
            + describe_buildup(goal, facts["ordered_events"])
        )

    # §6 high-xG missed chances (threshold stated in the text)
    missed = [s for s in facts["shots"]
              if s["outcome"] != "Goal" and s["xg"] >= HIGH_XG_THRESHOLD]
    for shot in missed:
        lines.append(
            f"High-quality chance missed: {shot['player']} ({shot['team']}) in the "
            f"{ordinal(shot['minute'])} minute recorded {num(shot['xg'])} xG but the "
            f"shot ended {shot['outcome'].lower()} "
            f"(threshold for a high-quality chance is {HIGH_XG_THRESHOLD} xG)."
        )

    # §6 open-goal misses
    for shot in [s for s in facts["shots"] if s["open_goal"] and s["outcome"] != "Goal"]:
        lines.append(
            f"Open-goal miss: {shot['player']} ({shot['team']}) failed to score with "
            f"an open goal in the {ordinal(shot['minute'])} minute "
            f"(xG {num(shot['xg'])}, outcome {shot['outcome'].lower()})."
        )

    # §6 deflected shots — resolves the Level 1 / Level 2 asymmetry
    for shot in [s for s in facts["shots"] if s["deflected"]]:
        lines.append(
            f"Deflected shot: {shot['player']} ({shot['team']}) in the "
            f"{ordinal(shot['minute'])} minute; outcome {shot['outcome'].lower()}."
        )

    # §6 substitutions
    for sub in facts["subs"]:
        reason = f" The substitution was {sub['reason'].lower()}." if sub["reason"] else ""
        lines.append(
            f"Substitution: {sub['team']} brought on {sub['on']} for {sub['off']} in the "
            f"{ordinal(sub['minute'])} minute.{reason}"
        )

    if len(lines) == 1:
        lines.append("No goals, high-quality chances or substitutions were recorded.")

    return Document(
        document_id=f"L2-match-{match_id}",
        level="2",
        match_id=match_id,
        text=" ".join(lines),
        metadata={
            "competition_id": COMPETITION_ID,
            "season_id": SEASON_ID,
            "match_date": match["match_date"],
            "stage": match["competition_stage"]["name"],
            "is_knockout": match["competition_stage"]["name"] != "Group Stage",
            "home_team": facts["home"],
            "away_team": facts["away"],
            "goal_count": len(facts["goals"]),
            "high_xg_miss_count": len(missed),
            "substitution_count": len(facts["subs"]),
            "high_xg_threshold": HIGH_XG_THRESHOLD,
            "buildup_walkback_cap": BUILDUP_WALKBACK_CAP,
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Level 3 — Match-level Player Performance (§6)
# ---------------------------------------------------------------------------

# The exact Level 3 metric set (§6). Level 4 averages these and no others.
LEVEL3_METRICS = [
    "minutes", "shots", "xg", "goals", "assists",
    "passes_attempted", "passes_completed", "pass_completion_pct",
    "passes_under_pressure", "pass_completion_under_pressure_pct",
    "shots_inside_box", "shots_outside_box",
    "successful_tackles", "successful_interceptions",
    "clearances", "ball_losses",
    "carries", "carry_distance", "pressures", "final_third_passes",
]


def lineup_player_ids(lineups: list[dict]) -> dict[tuple[str, str], int]:
    """
    Map (team_name, player_name) -> player_id from lineups.

    player_id is the stable identifier and is used to key Level 4 aggregation.
    DATA QUALITY: StatsBomb records player_id 4354 under two different
    player_name values across this tournament ("Phil Foden" and "Philip
    Foden"), so aggregating on name alone splits one player into two.
    """
    return {(team["team_name"], player["player_name"]): player["player_id"]
            for team in lineups for player in team["lineup"]}


def compute_player_stats(events: list[dict], minutes: dict,
                         player_ids: dict[tuple[str, str], int]) -> dict:
    """
    Build the §6 Level 3 metric block for every player with minutes > 0.
    All metrics use only periods 1-4 (see `in_play`).
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
    }
    stats: dict[tuple[str, str], dict] = {}

    def slot(event):
        team = event["team"]["name"]
        player = event["player"]["name"]
        key = (team, player)
        if key not in stats:
            stats[key] = dict(blank, player_id=event["player"]["id"],
                              team_id=event["team"]["id"])
        return stats[key]

    for event in events:
        if not in_play(event) or "player" not in event:   # §1 period-5 excluded
            continue
        etype = event["type"]["name"]

        if etype == "Shot":
            shot = event["shot"]
            row = slot(event)
            row["shots"] += 1
            row["xg"] += shot.get("statsbomb_xg", 0.0)
            if shot["outcome"]["name"] == "Goal":
                row["goals"] += 1                          # §2 own goals excluded
            if shot_inside_box(event.get("location")):     # §4
                row["shots_inside_box"] += 1
            else:
                row["shots_outside_box"] += 1

        elif etype == "Pass":
            passd = event["pass"]
            row = slot(event)
            row["passes_attempted"] += 1
            complete = is_complete_pass(event)             # §1
            if complete:
                row["passes_completed"] += 1
            if is_flag(event, "under_pressure"):           # §1 absence = false
                row["passes_under_pressure"] += 1
                if complete:
                    row["passes_under_pressure_completed"] += 1
            end = passd.get("end_location")                # §4 final-third pass
            if end and end[0] >= FINAL_THIRD_X:
                row["final_third_passes"] += 1
            if is_flag(passd, "goal_assist"):
                row["assists"] += 1

        elif etype == "Duel":
            # §4 successful tackle: type must be Tackle AND outcome successful.
            # Not labelled "duels won" — duel.type only takes Aerial Lost and
            # Tackle, so aerial duels appear only when lost.
            duel = event.get("duel", {})
            if duel.get("type", {}).get("name") == "Tackle" and \
                    duel.get("outcome", {}).get("name") in SUCCESSFUL_OUTCOMES:
                slot(event)["successful_tackles"] += 1

        elif etype == "Interception":
            # §4: passes with pass.type.name == "Interception" are a different
            # thing (a one-touch pass off an interception) and are excluded.
            outcome = event.get("interception", {}).get("outcome", {}).get("name")
            if outcome in SUCCESSFUL_OUTCOMES:
                slot(event)["successful_interceptions"] += 1

        elif etype == "Clearance":
            # §4: raw count. No outcome filter — no outcome field exists.
            slot(event)["clearances"] += 1

        elif etype in ("Miscontrol", "Dispossessed"):      # §4 ball losses
            slot(event)["ball_losses"] += 1

        elif etype == "Carry":
            row = slot(event)
            row["carries"] += 1
            row["carry_distance"] += euclidean(
                event.get("location"), event.get("carry", {}).get("end_location"))

        elif etype == "Pressure":
            slot(event)["pressures"] += 1

    # attach minutes and derive the percentage metrics
    result = {}
    for (team, player), row in stats.items():
        mins = minutes.get((team, player), 0.0)
        if mins <= 0:                    # §1 unused subs get no Level 3 document
            continue
        row["minutes"] = mins
        # lineups is authoritative for player_id (see lineup_player_ids)
        row["player_id"] = player_ids.get((team, player), row.get("player_id"))
        row["pass_completion_pct"] = (
            row["passes_completed"] / row["passes_attempted"] * 100.0
            if row["passes_attempted"] else None)
        row["pass_completion_under_pressure_pct"] = (
            row["passes_under_pressure_completed"] / row["passes_under_pressure"] * 100.0
            if row["passes_under_pressure"] else None)
        result[(team, player)] = row

    # players with minutes but no on-ball events still get a document
    for (team, player), mins in minutes.items():
        if mins > 0 and (team, player) not in result:
            result[(team, player)] = dict(
                blank, minutes=mins, player_id=player_ids.get((team, player)),
                team_id=None, pass_completion_pct=None,
                pass_completion_under_pressure_pct=None)
    return result


def player_roles(lineups: list[dict], team: str, player: str,
                 dismissal_second: int | None = None) -> list[str]:
    """
    §0/§6 tactical role comes from lineups.positions, NOT events.position.

    Segments are filtered with the same validity rules the §3 minutes algorithm
    applies, so roles and minutes tell the same story. Without this, Hennessey
    (3857273) reads as having played left defensive midfield after his red card
    — that segment is the §7.3 phantom, and minutes already discard it.
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
            continue                                       # §7.3 inverted period
        start = clock_to_seconds(seg.get("from"))
        if dismissal_second is not None and start is not None and start >= dismissal_second:
            continue                                       # §7.3 phantom after dismissal
        roles.append(seg["position"])
    return roles


def build_level3(match: dict, facts: dict, lineups: list[dict], stats: dict) -> list[Document]:
    docs = []
    match_id = match["match_id"]
    stage = match["competition_stage"]["name"]
    opponent_of = {facts["home"]: facts["away"], facts["away"]: facts["home"]}

    # §3: a period-5 dismissal happened after the match and clamps nothing
    dismissal_seconds = {
        (c["team"], c["player"]): c["minute"] * 60 + c["second"]
        for c in facts["cards"]
        if c["dismissal"] and c["period"] != SHOOTOUT_PERIOD
    }
    dismissed = set(dismissal_seconds)

    for (team, player), row in sorted(stats.items()):
        roles = player_roles(lineups, team, player, dismissal_seconds.get((team, player)))
        # §6: list each spell if the player changed role
        if len(roles) > 1:
            role_txt = (f"played {oxford([r.lower() for r in dict.fromkeys(roles)])}, "
                        f"changing role {times(len(roles) - 1)} during the match")
        elif roles:
            role_txt = f"played as {roles[0].lower()}"
        else:
            role_txt = "played in an unrecorded position"

        lines = [
            f"{player} of {team} {role_txt} against {opponent_of.get(team, 'the opposition')} "
            f"in the {stage} on {match['match_date']}, and was on the pitch for "
            f"{num(row['minutes'], 1)} minutes."
        ]
        if (team, player) in dismissed:
            lines.append("He was sent off in this match, which is why his minutes stop short of full time.")

        # attacking output
        if row["shots"]:
            lines.append(
                f"He took {row['shots']} shot{'s' if row['shots'] != 1 else ''} worth "
                f"{num(row['xg'])} expected goals, {row['shots_inside_box']} from inside "
                f"the penalty area and {row['shots_outside_box']} from outside, and scored "
                f"{row['goals']}."
            )
        else:
            lines.append("He did not attempt a shot.")
        if row["assists"]:
            lines.append(f"He provided {row['assists']} assist{'s' if row['assists'] != 1 else ''}.")

        # passing, framed as decision quality under context
        if row["passes_attempted"]:
            under = ""
            if row["passes_under_pressure"]:
                under = (f" Under pressure he attempted {row['passes_under_pressure']} "
                         f"{plural('pass', row['passes_under_pressure'])} and completed "
                         f"{pct(row['pass_completion_under_pressure_pct'])}, "
                         "which is the more demanding measure of his decision-making.")
            lines.append(
                f"He attempted {row['passes_attempted']} "
                f"{plural('pass', row['passes_attempted'])} and completed "
                f"{pct(row['pass_completion_pct'])}, of which {row['final_third_passes']} "
                f"{'was' if row['final_third_passes'] == 1 else 'were'} delivered into the "
                f"final third.{under}"
            )

        # carrying and pressing
        if row["carries"]:
            lines.append(
                f"He carried the ball {times(row['carries'])} for a total of "
                f"{num(row['carry_distance'], 1)} "
                f"{plural('yard', round(row['carry_distance']))}."
            )
        if row["pressures"]:
            lines.append(f"He applied pressure to opponents {times(row['pressures'])}.")

        # defensive work — clearances deliberately unfiltered (§4)
        defensive = []
        if row["successful_tackles"]:
            defensive.append(f"{row['successful_tackles']} successful "
                             f"{plural('tackle', row['successful_tackles'])}")
        if row["successful_interceptions"]:
            defensive.append(f"{row['successful_interceptions']} successful "
                             f"{plural('interception', row['successful_interceptions'])}")
        if row["clearances"]:
            defensive.append(f"{row['clearances']} "
                             f"{plural('clearance', row['clearances'])}")
        if defensive:
            lines.append(
                "Defensively he recorded " + oxford(defensive) + ". Clearances are a raw "
                "count because StatsBomb records no success or failure outcome for them, "
                "unlike tackles and interceptions which are filtered to successful only."
            )
        if row["ball_losses"]:
            lines.append(
                f"He lost the ball {times(row['ball_losses'])} through miscontrol or "
                "being dispossessed."
            )

        pid = row.get("player_id")
        docs.append(Document(
            document_id=f"L3-match-{match_id}-player-{pid or player.replace(' ', '_')}",
            level="3",
            match_id=match_id,
            player_name=player,
            team_name=team,
            text=" ".join(lines),
            metadata={
                "competition_id": COMPETITION_ID,
                "season_id": SEASON_ID,
                "match_date": match["match_date"],
                "stage": stage,
                "is_knockout": stage != "Group Stage",
                "opponent": opponent_of.get(team),
                "was_dismissed": (team, player) in dismissed,
                "roles": roles,
                "player_id": pid,
                **{k: row.get(k) for k in LEVEL3_METRICS},
                "limitations": LIMITATIONS_NOTE,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Level 4 — Tournament-level Performance & Consistency (§6)
# ---------------------------------------------------------------------------

# §6: "per-match average of every Level 3 metric" and "consistency score per
# metric". AMBIGUITY: for the two percentage metrics, "per-match average" is
# read literally as the mean of the per-match percentages. The alternative
# (recomputing the ratio from tournament totals) gives a different number and is
# not what the spec asks for; matches where the metric is undefined are skipped.
LEVEL4_AVERAGED = [m for m in LEVEL3_METRICS]


def build_level4(player_matches: dict, match_index: dict) -> list[Document]:
    """
    `player_matches` is keyed on player_id, not player_name: the same player_id
    can appear under more than one spelling (see lineup_player_ids), and
    name-keyed aggregation would split one player across two documents.
    """
    docs = []
    for player_id, entry in sorted(player_matches.items(), key=lambda kv: str(kv[0])):
        rows = entry["rows"]
        team = entry["team"]
        # canonical display name: most frequent spelling, alphabetical on ties
        player = sorted(entry["names"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        name_variants = sorted(entry["names"])
        matches_played = len(rows)                       # §6: minutes > 0 only
        total_minutes = sum(r["minutes"] for r in rows)
        total_goals = sum(r["goals"] for r in rows)
        total_assists = sum(r["assists"] for r in rows)
        total_xg = sum(r["xg"] for r in rows)

        averages, consistency = {}, {}
        for metric in LEVEL4_AVERAGED:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            averages[metric] = statistics.fmean(values) if values else None
            # §4: sample stdev, only at or above CONSISTENCY_MIN_MATCHES
            consistency[metric] = (
                statistics.stdev(values)
                if len(values) >= CONSISTENCY_MIN_MATCHES else None)

        # §6 best/worst: total match xG, tie-broken by goals, then assists, then
        # earliest match_date — a fully deterministic total ordering.
        ranked = sorted(
            rows,
            key=lambda r: (-r["xg"], -r["goals"], -r["assists"],
                           match_index[r["match_id"]]["match_date"]))
        best, worst = ranked[0], ranked[-1]

        # §6 knockout vs group split
        group = [r for r in rows if r["stage"] == "Group Stage"]
        knockout = [r for r in rows if r["stage"] != "Group Stage"]

        # §4 goal contribution rate
        contribution = (total_goals + total_assists) / matches_played if matches_played else 0.0

        lines = [
            f"{player} of {team} appeared in {matches_played} "
            f"match{'es' if matches_played != 1 else ''} at the FIFA World Cup 2022, "
            f"playing {num(total_minutes, 1)} minutes in total.",
            f"Across the tournament he scored {total_goals} goal"
            f"{'s' if total_goals != 1 else ''} and provided {total_assists} assist"
            f"{'s' if total_assists != 1 else ''} from {num(total_xg)} expected goals, "
            f"a goal contribution rate of {num(contribution)} per match.",
        ]

        if averages["shots"]:
            # a player with no completed-pass sample gets no completion clause
            # rather than a "not available" reading mid-sentence
            passing = ""
            if averages["pass_completion_pct"] is not None:
                passing = (f", with {num(averages['passes_attempted'], 1)} passes "
                           f"attempted at {pct(averages['pass_completion_pct'])} completion")
            lines.append(
                f"He averaged {num(averages['shots'])} "
                f"{plural('shot', round(averages['shots']))} and "
                f"{num(averages['xg'])} expected goals per match{passing}."
            )
        if averages["pressures"] or averages["clearances"]:
            lines.append(
                "His average defensive workload was "
                f"{num(averages['pressures'], 1)} "
                f"{plural('pressure', round(averages['pressures']))}, "
                f"{num(averages['successful_tackles'])} successful "
                f"{plural('tackle', round(averages['successful_tackles']))}, "
                f"{num(averages['successful_interceptions'])} successful "
                f"{plural('interception', round(averages['successful_interceptions']))} and "
                f"{num(averages['clearances'])} "
                f"{plural('clearance', round(averages['clearances']))} per match."
            )

        # consistency, only where it is defined
        if matches_played >= CONSISTENCY_MIN_MATCHES and consistency["xg"] is not None:
            lines.append(
                f"His match-to-match consistency, measured as the sample standard "
                f"deviation across {matches_played} matches, was {num(consistency['xg'])} "
                f"for expected goals and {num(consistency['minutes'], 1)} for minutes played."
            )
        else:
            lines.append(
                f"A consistency score is not reported because he played fewer than "
                f"{CONSISTENCY_MIN_MATCHES} matches."
            )

        best_opp = match_index[best["match_id"]]
        worst_opp = match_index[worst["match_id"]]
        if matches_played == 1:
            # best and worst are the same fixture; ranking language would be absurd
            lines.append(
                f"His only appearance was against {best['opponent']} in the "
                f"{best['stage']} on {best_opp['match_date']}, with {num(best['xg'])} xG, "
                f"{best['goals']} {plural('goal', best['goals'])} and "
                f"{best['assists']} {plural('assist', best['assists'])}."
            )
        else:
            lines.append(
                f"His best match by expected goals was against {best['opponent']} in the "
                f"{best['stage']} on {best_opp['match_date']}, with {num(best['xg'])} xG, "
                f"{best['goals']} {plural('goal', best['goals'])} and "
                f"{best['assists']} {plural('assist', best['assists'])}. "
                f"His weakest was against "
                f"{worst['opponent']} in the {worst['stage']} on {worst_opp['match_date']}, "
                f"with {num(worst['xg'])} xG. Matches are ranked by total expected goals, "
                "tie-broken by goals then assists then date; no composite rating is used."
            )

        if group and knockout:
            g_xg = statistics.fmean([r["xg"] for r in group])
            k_xg = statistics.fmean([r["xg"] for r in knockout])
            # a rounding-level gap is not a real difference
            if abs(k_xg - g_xg) < 0.05:
                direction = "similarly"
            else:
                direction = "more" if k_xg > g_xg else "less"
            lines.append(
                f"He played {len(group)} group-stage "
                f"{plural('match', len(group))} and {len(knockout)} knockout "
                f"{plural('match', len(knockout))}, averaging {num(g_xg)} xG in the "
                f"group stage and {num(k_xg)} "
                f"in the knockout rounds, so he was {direction} threatening after the group "
                f"stage." if direction == "similarly" else
                f"in the knockout rounds, so he was {direction} threatening after the "
                f"group stage."
            )
        elif group:
            lines.append(
                "His single appearance came in the group stage." if len(group) == 1
                else f"All {len(group)} of his appearances came in the group stage.")
        elif knockout:
            lines.append(
                "His single appearance came in the knockout rounds." if len(knockout) == 1
                else f"All {len(knockout)} of his appearances came in the knockout rounds.")

        docs.append(Document(
            document_id=f"L4-player-{player_id}",
            level="4",
            player_name=player,
            team_name=team,
            text=" ".join(lines),
            metadata={
                "competition_id": COMPETITION_ID,
                "season_id": SEASON_ID,
                "player_id": player_id,
                # more than one spelling means the source data is inconsistent
                "name_variants": name_variants,
                "matches_played": matches_played,
                "total_minutes": round(total_minutes, 2),
                "total_goals": total_goals,
                "total_assists": total_assists,
                "total_xg": round(total_xg, 4),
                "goal_contribution_rate": round(contribution, 4),
                "group_matches": len(group),
                "knockout_matches": len(knockout),
                "best_match_id": best["match_id"],
                "worst_match_id": worst["match_id"],
                "averages": {k: (round(v, 4) if v is not None else None)
                             for k, v in averages.items()},
                "consistency": {k: (round(v, 4) if v is not None else None)
                                for k, v in consistency.items()},
                "consistency_min_matches": CONSISTENCY_MIN_MATCHES,
                "limitations": LIMITATIONS_NOTE,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Team-level Analysis (§6)
# ---------------------------------------------------------------------------


def build_team_documents(team_acc: dict, match_index: dict) -> list[Document]:
    docs = []
    for team, acc in sorted(team_acc.items()):
        matches = acc["matches"]
        first_shots = acc["first_shot_minutes"]
        first_goals = acc["first_goal_minutes"]

        lines = [
            f"{team} played {len(matches)} match"
            f"{'es' if len(matches) != 1 else ''} at the FIFA World Cup 2022."
        ]

        # §6 first shot / first goal timing
        if first_shots:
            lines.append(
                f"They took their first shot on average in the "
                f"{ordinal(round(statistics.fmean(first_shots)))} minute, "
                f"earliest in the {ordinal(min(first_shots))} minute."
            )
        if first_goals:
            lines.append(
                f"When they scored, their first goal came on average in the "
                f"{ordinal(round(statistics.fmean(first_goals)))} minute, across "
                f"{len(first_goals)} of their {len(matches)} matches."
            )
        else:
            lines.append("They did not score in any match.")

        # §6 playing style: possession share, play pattern, formations
        total_events = acc["match_event_total"]
        if total_events:
            share = acc["possession_team_events"] / total_events * 100.0
            lines.append(
                f"Across their matches they were the team in possession for "
                f"{pct(share)} of events, an event-share proxy rather than "
                "StatsBomb's broadcast possession figure."
            )
        if acc["play_patterns"]:
            top = acc["play_patterns"].most_common(4)
            total_pp = sum(acc["play_patterns"].values())
            bits = [f"{name.lower()} {pct(count / total_pp * 100)}" for name, count in top]
            lines.append(f"Their play patterns were dominated by {oxford(bits)}.")
        if acc["formations"]:
            forms = [f"{f} ({n} formation {plural('record', n)})"
                     for f, n in acc["formations"].most_common(3)]
            lines.append(
                "Their most common shapes, counted across Starting XI and Tactical "
                f"Shift events, were {oxford(forms)}.")

        # §6 set-piece reliance: missing pass.type means an open-play pass
        if acc["pass_types"]:
            total_passes = sum(acc["pass_types"].values())
            open_play = acc["pass_types"].get("Open Play", 0)
            set_pieces = {k: v for k, v in acc["pass_types"].items() if k != "Open Play"}
            sp_total = sum(set_pieces.values())
            bits = [f"{k.lower()} {pct(v / total_passes * 100)}"
                    for k, v in Counter(set_pieces).most_common(4)]
            lines.append(
                f"Of {total_passes} passes, {pct(open_play / total_passes * 100)} were "
                f"standard open-play passes and {pct(sp_total / total_passes * 100)} came "
                f"from set pieces or restarts, chiefly {oxford(bits)}. Passes carrying no "
                "pass type are standard open-play passes and are counted as such rather "
                "than discarded."
            )
            lines.append(
                f"They delivered {acc['crosses']} crosses, "
                f"{pct(acc['crosses'] / total_passes * 100)} of their passes."
            )

        docs.append(Document(
            document_id=f"TEAM-{acc['team_id']}",
            level="team",
            team_name=team,
            text=" ".join(lines),
            metadata={
                "competition_id": COMPETITION_ID,
                "season_id": SEASON_ID,
                "matches_played": len(matches),
                "match_ids": sorted(matches),
                "furthest_stage": acc["furthest_stage"],
                "avg_first_shot_minute": (round(statistics.fmean(first_shots), 2)
                                          if first_shots else None),
                "avg_first_goal_minute": (round(statistics.fmean(first_goals), 2)
                                          if first_goals else None),
                "matches_scored_in": len(first_goals),
                "formations": dict(acc["formations"]),
                "play_pattern_counts": dict(acc["play_patterns"]),
                "pass_type_counts": dict(acc["pass_types"]),
                "cross_count": acc["crosses"],
                "possession_is_proxy": True,       # §7.4
                "limitations": LIMITATIONS_NOTE,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

STAGE_ORDER = ["Group Stage", "Round of 16", "Quarter-finals",
               "Semi-finals", "3rd Place Final", "Final"]


def generate_documents(data_root: Path = DATA_ROOT, verbose: bool = True) -> dict:
    """
    Run the full pipeline over all 64 matches.

    Returns a dict with the document list plus the diagnostics the §8 regression
    tests assert against. Matches are processed one at a time so the ~193 MB of
    events JSON is never all in memory at once.
    """
    matches = load_json(data_root / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json")
    match_index = {m["match_id"]: m for m in matches}

    documents: list[Document] = []
    # keyed on player_id (see build_level4) -> {"team", "names", "rows"}
    player_matches: dict = {}
    team_acc: dict[str, dict] = {}
    diagnostics = {
        "minutes_by_match": {},
        "cards_by_match": {},
        "shootout_by_match": {},
        "empty_position_rows": 0,
        "segment_anomalies": 0,
        "stage_counts": Counter(),
        "card_parity_failures": [],
    }

    for match in sorted(matches, key=lambda m: m["match_date"]):
        match_id = match["match_id"]
        events = load_json(data_root / "events" / f"{match_id}.json")
        lineups = load_json(data_root / "lineups" / f"{match_id}.json")

        # §3 minutes played — imported, not reimplemented
        context = build_match_context(events, lineups)
        minutes, anomalies = minutes_played(lineups, context, collect_anomalies=True)
        diagnostics["segment_anomalies"] += len(anomalies)

        facts = extract_match_facts(match, events, lineups)
        stage = match["competition_stage"]["name"]
        diagnostics["stage_counts"][stage] += 1
        if not facts["card_parity_ok"]:
            diagnostics["card_parity_failures"].append(match_id)

        # §1: players with empty `positions` are unused substitutes
        diagnostics["empty_position_rows"] += sum(
            1 for t in lineups for p in t["lineup"] if not (p.get("positions") or []))

        documents.append(build_level1(match, facts))
        documents.append(build_level2(match, facts))

        player_ids = lineup_player_ids(lineups)
        stats = compute_player_stats(events, minutes, player_ids)
        documents.extend(build_level3(match, facts, lineups, stats))

        # accumulate for Level 4, keyed on the stable player_id
        opponent_of = {facts["home"]: facts["away"], facts["away"]: facts["home"]}
        for (team, player), row in stats.items():
            key = row.get("player_id") or f"name:{team}:{player}"
            entry = player_matches.setdefault(
                key, {"team": team, "names": Counter(), "rows": []})
            entry["names"][player] += 1
            entry["rows"].append(
                dict(row, match_id=match_id, stage=stage, opponent=opponent_of.get(team)))

        # accumulate for Team-level
        for team in (facts["home"], facts["away"]):
            acc = team_acc.setdefault(team, {
                "team_id": None, "matches": set(), "first_shot_minutes": [],
                "first_goal_minutes": [], "play_patterns": Counter(),
                "formations": Counter(), "pass_types": Counter(), "crosses": 0,
                "possession_team_events": 0, "match_event_total": 0,
                "furthest_stage": stage,
            })
            acc["matches"].add(match_id)
            if STAGE_ORDER.index(stage) > STAGE_ORDER.index(acc["furthest_stage"]):
                acc["furthest_stage"] = stage

        team_shots = defaultdict(list)
        team_goals = defaultdict(list)
        for shot in facts["shots"]:
            team_shots[shot["team"]].append(shot["minute"])
            if shot["outcome"] == "Goal":
                team_goals[shot["team"]].append(shot["minute"])
        for team, mins in team_shots.items():
            team_acc[team]["first_shot_minutes"].append(min(mins))
        for team, mins in team_goals.items():
            team_acc[team]["first_goal_minutes"].append(min(mins))

        for event in events:
            if not in_play(event):
                continue

            # §6 possession_team share. The denominator is EVERY event in the
            # match, not just this team's own events — scoring it against a
            # team's own events only would trivially approach 100%, because a
            # team's events mostly happen while it has the ball.
            possession_team = event.get("possession_team", {}).get("name")
            for side in (facts["home"], facts["away"]):
                team_acc[side]["match_event_total"] += 1
            if possession_team in team_acc:
                team_acc[possession_team]["possession_team_events"] += 1

            team = event.get("team", {}).get("name")
            if team not in team_acc:
                continue
            if team_acc[team]["team_id"] is None:
                team_acc[team]["team_id"] = event["team"]["id"]
            pattern = event.get("play_pattern", {}).get("name")
            if pattern:
                team_acc[team]["play_patterns"][pattern] += 1
            etype = event["type"]["name"]
            if etype == "Pass":
                passd = event["pass"]
                # §6: a missing pass.type IS a standard open-play pass
                team_acc[team]["pass_types"][
                    passd.get("type", {}).get("name", "Open Play")] += 1
                if is_flag(passd, "cross"):
                    team_acc[team]["crosses"] += 1
            elif etype in ("Starting XI", "Tactical Shift"):
                formation = event.get("tactics", {}).get("formation")
                if formation:
                    team_acc[team]["formations"][str(formation)] += 1

        diagnostics["minutes_by_match"][match_id] = minutes
        diagnostics["cards_by_match"][match_id] = facts["cards"]
        diagnostics["shootout_by_match"][match_id] = dict(facts["shootout_goals"])

        if verbose:
            print(f"  processed {match_id}  {facts['home']} v {facts['away']} "
                  f"({stage})", flush=True)

    documents.extend(build_level4(player_matches, match_index))
    documents.extend(build_team_documents(team_acc, match_index))

    return {"documents": documents, "diagnostics": diagnostics,
            "match_index": match_index, "player_matches": player_matches}


# ---------------------------------------------------------------------------
# §8 Regression tests
# ---------------------------------------------------------------------------


def run_regression_tests(result: dict) -> list[str]:
    """Assert the §8 table. Returns a list of failure strings (empty == pass)."""
    failures = []
    diag = result["diagnostics"]
    docs = result["documents"]

    def check(condition, message):
        if not condition:
            failures.append(message)

    def minutes_for(match_id, player):
        return next((v for (t, p), v in diag["minutes_by_match"][match_id].items()
                     if p == player), None)

    def team_minutes(match_id, team):
        return sum(v for (t, _), v in diag["minutes_by_match"][match_id].items() if t == team)

    # 1. Red card truncates minutes — 3857273
    hennessey = minutes_for(3857273, "Wayne Hennessey")
    check(hennessey is not None and abs(hennessey - 83.7) < 0.5,
          f"[§8 red card] Hennessey minutes = {hennessey}, expected ~83.7")

    # 2. Second yellow truncates minutes — 3857280
    aboubakar = minutes_for(3857280, "Vincent Paté Aboubakar")
    check(aboubakar is not None and abs(aboubakar - 92.9) < 0.5,
          f"[§8 second yellow] Aboubakar minutes = {aboubakar}, expected ~92.9")

    # 3. Extra time + duplicate segments — 3869685
    for team in ("Argentina", "France"):
        total = team_minutes(3869685, team)
        check(abs(total - 1365.0) < 5.0,
              f"[§8 extra time] {team} total minutes = {total:.1f}, expected ~1365")
    kounde = minutes_for(3869685, "Jules Koundé")
    varane = minutes_for(3869685, "Raphaël Varane")
    check(kounde is not None and abs(kounde - 120.6) < 0.5,
          f"[§8 duplicate segments] Koundé = {kounde}, expected ~120.6")
    check(varane is not None and abs(varane - 112.5) < 0.5,
          f"[§8 duplicate segments] Varane = {varane}, expected ~112.5")

    # 4. Shootout excluded from xG — no Final player carries shootout xG.
    # The 8 shootout penalties total 6.27 xG; the largest legitimate single-match
    # xG in the Final is well under 2, so any player above that means leakage.
    final_l3 = [d for d in docs if d.level == "3" and d.match_id == 3869685]
    check(final_l3, "[§8 shootout xG] no Level 3 documents found for the Final")
    max_xg = max((d.metadata["xg"] for d in final_l3), default=0.0)
    check(max_xg < 2.0,
          f"[§8 shootout xG] max player xG in the Final = {max_xg:.2f}; "
          "shootout penalties appear to be leaking in")

    # 5. Score vs shootout separated — 3869685
    final_l1 = next(d for d in docs if d.document_id == "L1-match-3869685")
    check(final_l1.metadata["home_score"] == 3 and final_l1.metadata["away_score"] == 3,
          "[§8 score] Final score should render 3-3")
    shootout = diag["shootout_by_match"][3869685]
    check(shootout.get("Argentina") == 4 and shootout.get("France") == 2,
          f"[§8 shootout] expected Argentina 4 - France 2, got {shootout}")
    check("4-2" in final_l1.text and "3-3" in final_l1.text,
          "[§8 score] Level 1 text must render the 3-3 score and the 4-2 shootout separately")

    # 6. Card timing — Paredes at 113:27, not the truncated lineups value 11:27
    paredes = next((c for c in diag["cards_by_match"][3869685]
                    if c["player"] == "Leandro Daniel Paredes"), None)
    check(paredes is not None and paredes["minute"] == 113 and paredes["second"] == 27,
          f"[§8 card timing] Paredes card = {paredes and (paredes['minute'], paredes['second'])}, "
          "expected (113, 27)")

    # 7. Stage coverage — six stages, 48/8/4/2/1/1, summing to 64
    expected_stages = {"Group Stage": 48, "Round of 16": 8, "Quarter-finals": 4,
                       "Semi-finals": 2, "3rd Place Final": 1, "Final": 1}
    check(dict(diag["stage_counts"]) == expected_stages,
          f"[§8 stages] got {dict(diag['stage_counts'])}, expected {expected_stages}")
    check(sum(diag["stage_counts"].values()) == 64,
          "[§8 stages] stage counts must sum to 64")

    # 8. Unused substitutes produce no Level 3 document
    check(diag["empty_position_rows"] == 1249,
          f"[§8 unused subs] {diag['empty_position_rows']} empty-position rows, expected 1249")
    l3_docs = [d for d in docs if d.level == "3"]
    check(all(d.metadata["minutes"] > 0 for d in l3_docs),
          "[§8 unused subs] every Level 3 document must have minutes > 0")

    # Structural checks: document counts and id uniqueness
    check(sum(1 for d in docs if d.level == "1") == 64, "[structure] expected 64 Level 1 documents")
    check(sum(1 for d in docs if d.level == "2") == 64, "[structure] expected 64 Level 2 documents")
    check(sum(1 for d in docs if d.level == "team") == 32, "[structure] expected 32 team documents")
    ids = [d.document_id for d in docs]
    check(len(ids) == len(set(ids)), "[structure] document_id values must be unique")
    check(all(d.text.strip() for d in docs), "[structure] every document needs non-empty text")

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Loading StatsBomb Open Data from {DATA_ROOT.resolve()}")
    result = generate_documents(DATA_ROOT, verbose=False)
    docs = result["documents"]

    counts = Counter(d.level for d in docs)
    print("\nDocument counts by level")
    for level in ("1", "2", "3", "4", "team"):
        label = {"1": "Level 1  Match Summary", "2": "Level 2  Key Events",
                 "3": "Level 3  Player / match", "4": "Level 4  Player / tournament",
                 "team": "Team-level Analysis"}[level]
        print(f"  {label:<32} {counts.get(level, 0)}")
    print(f"  {'TOTAL':<32} {len(docs)}")

    diag = result["diagnostics"]
    print(f"\nUnused-substitute rows skipped : {diag['empty_position_rows']}")
    print(f"Segment anomalies logged (§7.3): {diag['segment_anomalies']}")
    print(f"Card parity failures (§5)      : "
          f"{diag['card_parity_failures'] or 'none'}")

    print("\n§8 regression tests")
    failures = run_regression_tests(result)
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        print(f"\n{len(failures)} regression test(s) failed — do not proceed to "
              "02_preprocessing.py until these pass.")
        return 1
    print("  all regression tests passed")

    out = Path("documents.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump([d.to_dict() for d in docs], handle, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(docs)} documents to {out.resolve()} for 02_preprocessing.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
