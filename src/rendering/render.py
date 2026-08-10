"""
render.py — Phase 2: Pure Document Renderer

Reads structured facts from match_facts.json and produces natural-language
documents. No statistics computation. No raw event parsing. Pure rendering.

Output: documents.json (same format as 01_documents.py produces)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.extraction.match_facts import DatasetIdentity, WC2022_DATASET_IDENTITY
from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOOTOUT_PERIOD = 5
HIGH_XG_THRESHOLD = 0.30
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
    level: str
    text: str
    metadata: dict = field(default_factory=dict)
    match_id: int | None = None
    player_name: str | None = None
    team_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prose helpers
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


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _oxford(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "not available" if value is None else f"{value:.{digits}f}%"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


# ---------------------------------------------------------------------------
# Level 1 — Match Summary
# ---------------------------------------------------------------------------


def render_level1(mf: dict, dataset_identity: DatasetIdentity = WC2022_DATASET_IDENTITY) -> Document:
    """Render a Level 1 Match Summary document from MatchFacts."""
    match_id = mf["match_id"]
    home = mf["home_team"]
    away = mf["away_team"]
    home_score = mf["home_score"]
    away_score = mf["away_score"]
    stage = mf["stage"]
    stadium = mf.get("stadium") or "an unspecified stadium"

    lines = [
        f"The {dataset_identity.display_name} {stage} between {home} and {away} was played on "
        f"{mf['match_date']} at {stadium}."
    ]

    period_note = "after extra time" if mf["went_to_extra_time"] else "in normal time"
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

    shootout = mf.get("shootout_goals", {})
    if shootout:
        ranked = sorted(shootout.items(), key=lambda kv: -kv[1])
        win_team, win_pens = ranked[0]
        lose_team, lose_pens = ranked[1] if len(ranked) > 1 else (
            away if win_team == home else home, 0)
        lines.append(
            f"{win_team} won the penalty shootout {win_pens}-{lose_pens} against {lose_team}."
        )

    goals = mf.get("goals", [])
    if goals:
        for goal in goals:
            part = f" with the {goal['body_part'].lower()}" if goal.get("body_part") else ""
            how = goal.get("shot_type") or "Open Play"
            if how == "Open Play":
                how_txt = "from open play"
            elif how == "Penalty":
                how_txt = "from a penalty"
            else:
                how_txt = f"directly from a {how.lower()}"
            assist = goal.get("assist")
            assist_txt = f", assisted by {assist}" if assist else ""
            deflect = ", and the shot was deflected" if goal.get("deflected") else ""
            pattern = goal.get("play_pattern")
            pattern_txt = ""
            if pattern and pattern not in ("Regular Play", "Other"):
                pattern_txt = f" The build-up pattern was {pattern.lower()}."
            lines.append(
                f"{goal['player']} scored for {goal['team']} in the "
                f"{_ordinal(goal['minute'])} minute{part}, {how_txt}{assist_txt}"
                f" (xG {_num(goal['xg'])}){deflect}.{pattern_txt}"
            )
    else:
        lines.append("Neither side scored from open play or a set piece.")

    for own in mf.get("own_goals", []):
        lines.append(
            f"{own['player']} of {own['conceded_by']} scored an own goal in the "
            f"{_ordinal(own['minute'])} minute, credited to {own['credited_to']}. "
            "Own goals carry no expected-goals value and are excluded from player goal totals."
        )

    possession = mf.get("possession", {})
    if possession:
        parts = [f"{team} {_pct(share)}" for team, share in
                 sorted(possession.items(), key=lambda kv: -(kv[1] or 0))]
        lines.append(
            f"Possession was split {_oxford(parts)}. This is an event-share proxy "
            "based on passes, carries, dribbles and shots, not StatsBomb's "
            "broadcast possession figure."
        )

    cards = mf.get("cards", [])
    if cards:
        card_bits = []
        for card in cards:
            detail = f" for {card['infringement'].lower()}" if card.get("infringement") else ""
            when = (" after the match, during the penalty shootout"
                    if card["period"] == SHOOTOUT_PERIOD
                    else f" in the {_ordinal(card['minute'])} minute")
            card_bits.append(
                f"{card['player']} ({card['team']}) received a "
                f"{card['card'].lower()}{when}{detail}"
            )
        lines.append(f"The referee showed {len(cards)} "
                     f"{_plural('card', len(cards))}. " + _oxford(card_bits) + ".")

        dismissals = [c for c in cards if c["dismissal"]]
        for d in dismissals:
            kind = ("a second yellow card" if d["card"] == "Second Yellow"
                    else "a straight red card")
            lines.append(
                f"{d['player']} of {d['team']} was sent off for {kind} in the "
                f"{_ordinal(d['minute'])} minute, leaving {d['team']} a player short."
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
            "competition_id": dataset_identity.competition_id,
            "competition_name": dataset_identity.competition_name,
            "season_id": dataset_identity.season_id,
            "season_name": dataset_identity.season_name,
            "match_date": mf["match_date"],
            "stage": stage,
            "is_knockout": mf["is_knockout"],
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "score_excludes_shootout": True,
            "went_to_extra_time": mf["went_to_extra_time"],
            "went_to_shootout": bool(shootout),
            "goal_count": len(goals),
            "own_goal_count": len(mf.get("own_goals", [])),
            "card_count": len(cards),
            "dismissal_count": sum(1 for c in cards if c["dismissal"]),
            "possession_is_proxy": True,
            "card_parity_ok": mf.get("card_parity_ok", True),
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Level 2 — Key Events
# ---------------------------------------------------------------------------


def render_level2(mf: dict, dataset_identity: DatasetIdentity = WC2022_DATASET_IDENTITY) -> Document:
    """Render a Level 2 Key Events document from MatchFacts."""
    match_id = mf["match_id"]
    lines = [
        f"Key events from the {dataset_identity.display_name} {mf['stage']} between "
        f"{mf['home_team']} and {mf['away_team']} ({mf['match_date']})."
    ]

    goals = mf.get("goals", [])
    for goal in goals:
        extras = []
        if goal.get("first_time"):
            extras.append("a first-time strike")
        if goal.get("follows_dribble"):
            extras.append("taken after a successful dribble")
        if goal.get("deflected"):
            extras.append("deflected on its way in")
        extra_txt = f" It was {_oxford(extras)}." if extras else ""
        assist = goal.get("assist")
        assist_txt = f" {assist} provided the assist." if assist else ""
        lines.append(
            f"Goal: {goal['player']} ({goal['team']}) in period {goal['period']}, "
            f"{_ordinal(goal['minute'])} minute, with the "
            f"{(goal.get('body_part') or 'unspecified body part').lower()}, "
            f"shot type {goal.get('shot_type') or 'unknown'}, xG {_num(goal['xg'])}."
            f"{assist_txt}{extra_txt} "
            + goal.get("buildup_text", "")
        )

    shots = mf.get("shots", [])
    missed = [s for s in shots if s["outcome"] != "Goal" and s["xg"] >= HIGH_XG_THRESHOLD]
    for shot in missed:
        lines.append(
            f"High-quality chance missed: {shot['player']} ({shot['team']}) in the "
            f"{_ordinal(shot['minute'])} minute recorded {_num(shot['xg'])} xG but the "
            f"shot ended {shot['outcome'].lower()} "
            f"(threshold for a high-quality chance is {HIGH_XG_THRESHOLD} xG)."
        )

    for shot in [s for s in shots if s.get("open_goal") and s["outcome"] != "Goal"]:
        lines.append(
            f"Open-goal miss: {shot['player']} ({shot['team']}) failed to score with "
            f"an open goal in the {_ordinal(shot['minute'])} minute "
            f"(xG {_num(shot['xg'])}, outcome {shot['outcome'].lower()})."
        )

    for shot in [s for s in shots if s.get("deflected")]:
        lines.append(
            f"Deflected shot: {shot['player']} ({shot['team']}) in the "
            f"{_ordinal(shot['minute'])} minute; outcome {shot['outcome'].lower()}."
        )

    subs = mf.get("substitutions", [])
    for sub in subs:
        reason = f" The substitution was {sub['reason'].lower()}." if sub.get("reason") else ""
        lines.append(
            f"Substitution: {sub['team']} brought on {sub['on']} for {sub['off']} in the "
            f"{_ordinal(sub['minute'])} minute.{reason}"
        )

    if len(lines) == 1:
        lines.append("No goals, high-quality chances or substitutions were recorded.")

    return Document(
        document_id=f"L2-match-{match_id}",
        level="2",
        match_id=match_id,
        text=" ".join(lines),
        metadata={
            "competition_id": dataset_identity.competition_id,
            "competition_name": dataset_identity.competition_name,
            "season_id": dataset_identity.season_id,
            "season_name": dataset_identity.season_name,
            "match_date": mf["match_date"],
            "stage": mf["stage"],
            "is_knockout": mf["is_knockout"],
            "home_team": mf["home_team"],
            "away_team": mf["away_team"],
            "goal_count": len(goals),
            "high_xg_miss_count": len(missed),
            "substitution_count": len(subs),
            "high_xg_threshold": HIGH_XG_THRESHOLD,
            "buildup_walkback_cap": 5,
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Level 3 — Match-level Player Performance
# ---------------------------------------------------------------------------


def render_level3(pmf: dict, dataset_identity: DatasetIdentity = WC2022_DATASET_IDENTITY) -> Document:
    """Render a Level 3 Player Performance document from PlayerMatchFacts."""
    match_id = pmf["match_id"]
    player = pmf["player_name"]
    team = pmf["team_name"]
    stage = pmf["stage"]
    opponent = pmf.get("opponent", "the opposition")
    roles = pmf.get("roles", [])
    dismissed = pmf.get("was_dismissed", False)

    if len(roles) > 1:
        role_txt = (f"played {_oxford([r.lower() for r in dict.fromkeys(roles)])}, "
                    f"changing role {_times(len(roles) - 1)} during the match")
    elif roles:
        role_txt = f"played as {roles[0].lower()}"
    else:
        role_txt = "played in an unrecorded position"

    lines = [
        f"{player} of {team} {role_txt} against {opponent} "
        f"in the {stage} on {pmf['match_date']}, and was on the pitch for "
        f"{_num(pmf['minutes'], 1)} minutes."
    ]
    if dismissed:
        lines.append("He was sent off in this match, which is why his minutes stop short of full time.")

    shots = pmf.get("shots", 0)
    xg = pmf.get("xg", 0.0)
    goals = pmf.get("goals", 0)
    assists = pmf.get("assists", 0)
    shots_in = pmf.get("shots_inside_box", 0)
    shots_out = pmf.get("shots_outside_box", 0)

    if shots:
        lines.append(
            f"He took {shots} shot{'s' if shots != 1 else ''} worth "
            f"{_num(xg)} expected goals, {shots_in} from inside "
            f"the penalty area and {shots_out} from outside, and scored "
            f"{goals}."
        )
    else:
        lines.append("He did not attempt a shot.")
    if assists:
        lines.append(f"He provided {assists} assist{'s' if assists != 1 else ''}.")

    passes_attempted = pmf.get("passes_attempted", 0)
    passes_completed = pmf.get("passes_completed", 0)
    pass_pct = pmf.get("pass_completion_pct")
    passes_under_pressure = pmf.get("passes_under_pressure", 0)
    pass_pct_pressure = pmf.get("pass_completion_under_pressure_pct")
    final_third = pmf.get("final_third_passes", 0)

    if passes_attempted:
        under = ""
        if passes_under_pressure:
            under = (f" Under pressure he attempted {passes_under_pressure} "
                     f"{_plural('pass', passes_under_pressure)} and completed "
                     f"{_pct(pass_pct_pressure)}, "
                     "which is the more demanding measure of his decision-making.")
        lines.append(
            f"He attempted {passes_attempted} "
            f"{_plural('pass', passes_attempted)} and completed "
            f"{_pct(pass_pct)}, of which {final_third} "
            f"{'was' if final_third == 1 else 'were'} delivered into the "
            f"final third.{under}"
        )

    carries = pmf.get("carries", 0)
    carry_dist = pmf.get("carry_distance", 0.0)
    pressures = pmf.get("pressures", 0)

    if carries:
        lines.append(
            f"He carried the ball {_times(carries)} for a total of "
            f"{_num(carry_dist, 1)} "
            f"{_plural('yard', round(carry_dist))}."
        )
    if pressures:
        lines.append(f"He applied pressure to opponents {_times(pressures)}.")

    tackles = pmf.get("successful_tackles", 0)
    interceptions = pmf.get("successful_interceptions", 0)
    clearances = pmf.get("clearances", 0)
    ball_losses = pmf.get("ball_losses", 0)

    defensive = []
    if tackles:
        defensive.append(f"{tackles} successful {_plural('tackle', tackles)}")
    if interceptions:
        defensive.append(f"{interceptions} successful {_plural('interception', interceptions)}")
    if clearances:
        defensive.append(f"{clearances} {_plural('clearance', clearances)}")
    if defensive:
        lines.append(
            "Defensively he recorded " + _oxford(defensive) + ". Clearances are a raw "
            "count because StatsBomb records no success or failure outcome for them, "
            "unlike tackles and interceptions which are filtered to successful only."
        )
    if ball_losses:
        lines.append(
            f"He lost the ball {_times(ball_losses)} through miscontrol or "
            "being dispossessed."
        )

    pid = pmf.get("player_id")
    return Document(
        document_id=f"L3-match-{match_id}-player-{pid or player.replace(' ', '_')}",
        level="3",
        match_id=match_id,
        player_name=player,
        team_name=team,
        text=" ".join(lines),
        metadata={
            "competition_id": dataset_identity.competition_id,
            "competition_name": dataset_identity.competition_name,
            "season_id": dataset_identity.season_id,
            "season_name": dataset_identity.season_name,
            "match_date": pmf["match_date"],
            "stage": stage,
            "is_knockout": pmf["is_knockout"],
            "opponent": opponent,
            "was_dismissed": dismissed,
            "roles": roles,
            "player_id": pid,
            "minutes": pmf["minutes"],
            "shots": shots,
            "xg": xg,
            "goals": goals,
            "assists": assists,
            "passes_attempted": passes_attempted,
            "passes_completed": passes_completed,
            "pass_completion_pct": pass_pct,
            "passes_under_pressure": passes_under_pressure,
            "pass_completion_under_pressure_pct": pass_pct_pressure,
            "shots_inside_box": shots_in,
            "shots_outside_box": shots_out,
            "successful_tackles": tackles,
            "successful_interceptions": interceptions,
            "clearances": clearances,
            "ball_losses": ball_losses,
            "carries": carries,
            "carry_distance": carry_dist,
            "pressures": pressures,
            "final_third_passes": final_third,
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Level 4 — Tournament-level Player Performance
# ---------------------------------------------------------------------------

LEVEL3_METRICS = [
    "minutes", "shots", "xg", "goals", "assists",
    "passes_attempted", "passes_completed", "pass_completion_pct",
    "passes_under_pressure", "pass_completion_under_pressure_pct",
    "shots_inside_box", "shots_outside_box",
    "successful_tackles", "successful_interceptions",
    "clearances", "ball_losses",
    "carries", "carry_distance", "pressures", "final_third_passes",
]


def render_level4(
    player_id,
    player_facts: list[dict],
    match_index: dict,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
    dataset_identity: DatasetIdentity = WC2022_DATASET_IDENTITY,
) -> Document:
    """Render a Level 4 Player Tournament document from aggregated PlayerMatchFacts."""
    team = player_facts[0]["team_name"]
    # canonical display name: most frequent spelling
    name_counts = Counter(f["player_name"] for f in player_facts)
    player = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    name_variants = sorted(name_counts.keys())

    matches_played = len(player_facts)
    total_minutes = sum(f["minutes"] for f in player_facts)
    total_goals = sum(f["goals"] for f in player_facts)
    total_assists = sum(f["assists"] for f in player_facts)
    total_xg = sum(f["xg"] for f in player_facts)

    averages, consistency = {}, {}
    for metric in LEVEL3_METRICS:
        values = [f[metric] for f in player_facts if f.get(metric) is not None]
        averages[metric] = statistics.fmean(values) if values else None
        consistency[metric] = (
            statistics.stdev(values) if len(values) >= CONSISTENCY_MIN_MATCHES else None)

    # §6 best/worst ranking
    ranked = sorted(
        player_facts,
        key=lambda r: (-r["xg"], -r["goals"], -r["assists"],
                       match_index[r["match_id"]]["match_date"]))
    best, worst = ranked[0], ranked[-1]

    # Group and knockout are independent, explicit classifications from the
    # stage taxonomy — never derived from each other (a league stage is
    # neither "not knockout => group" nor "not group => knockout"; it can
    # legitimately be classified as neither). See src/stage_taxonomy.py.
    group = [f for f in player_facts if stage_taxonomy.is_group_stage(f["stage"])]
    knockout = [f for f in player_facts if stage_taxonomy.is_knockout(f["stage"])]
    contribution = (total_goals + total_assists) / matches_played if matches_played else 0.0

    lines = [
        f"{player} of {team} appeared in {matches_played} "
        f"match{'es' if matches_played != 1 else ''} at the {dataset_identity.display_name}, "
        f"playing {_num(total_minutes, 1)} minutes in total.",
        f"Across the tournament he scored {total_goals} goal"
        f"{'s' if total_goals != 1 else ''} and provided {total_assists} assist"
        f"{'s' if total_assists != 1 else ''} from {_num(total_xg)} expected goals, "
        f"a goal contribution rate of {_num(contribution)} per match.",
    ]

    if averages["shots"]:
        passing = ""
        if averages["pass_completion_pct"] is not None:
            passing = (f", with {_num(averages['passes_attempted'], 1)} passes "
                       f"attempted at {_pct(averages['pass_completion_pct'])} completion")
        lines.append(
            f"He averaged {_num(averages['shots'])} "
            f"{_plural('shot', round(averages['shots']))} and "
            f"{_num(averages['xg'])} expected goals per match{passing}."
        )
    if averages["pressures"] or averages["clearances"]:
        lines.append(
            "His average defensive workload was "
            f"{_num(averages['pressures'], 1)} "
            f"{_plural('pressure', round(averages['pressures']))}, "
            f"{_num(averages['successful_tackles'])} successful "
            f"{_plural('tackle', round(averages['successful_tackles']))}, "
            f"{_num(averages['successful_interceptions'])} successful "
            f"{_plural('interception', round(averages['successful_interceptions']))} and "
            f"{_num(averages['clearances'])} "
            f"{_plural('clearance', round(averages['clearances']))} per match."
        )

    if matches_played >= CONSISTENCY_MIN_MATCHES and consistency["xg"] is not None:
        lines.append(
            f"His match-to-match consistency, measured as the sample standard "
            f"deviation across {matches_played} matches, was {_num(consistency['xg'])} "
            f"for expected goals and {_num(consistency['minutes'], 1)} for minutes played."
        )
    else:
        lines.append(
            f"A consistency score is not reported because he played fewer than "
            f"{CONSISTENCY_MIN_MATCHES} matches."
        )

    best_opp = match_index[best["match_id"]]
    worst_opp = match_index[worst["match_id"]]
    if matches_played == 1:
        lines.append(
            f"His only appearance was against {best['opponent']} in the "
            f"{best['stage']} on {best_opp['match_date']}, with {_num(best['xg'])} xG, "
            f"{best['goals']} {_plural('goal', best['goals'])} and "
            f"{best['assists']} {_plural('assist', best['assists'])}."
        )
    else:
        lines.append(
            f"His best match by expected goals was against {best['opponent']} in the "
            f"{best['stage']} on {best_opp['match_date']}, with {_num(best['xg'])} xG, "
            f"{best['goals']} {_plural('goal', best['goals'])} and "
            f"{best['assists']} {_plural('assist', best['assists'])}. "
            f"His weakest was against "
            f"{worst['opponent']} in the {worst['stage']} on {worst_opp['match_date']}, "
            f"with {_num(worst['xg'])} xG. Matches are ranked by total expected goals, "
            "tie-broken by goals then assists then date; no composite rating is used."
        )

    if group and knockout:
        g_xg = statistics.fmean([f["xg"] for f in group])
        k_xg = statistics.fmean([f["xg"] for f in knockout])
        if abs(k_xg - g_xg) < 0.05:
            direction = "similarly"
        else:
            direction = "more" if k_xg > g_xg else "less"
        lines.append(
            f"He played {len(group)} group-stage "
            f"{_plural('match', len(group))} and {len(knockout)} knockout "
            f"{_plural('match', len(knockout))}, averaging {_num(g_xg)} xG in the "
            f"group stage and {_num(k_xg)} "
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

    return Document(
        document_id=f"L4-player-{player_id}",
        level="4",
        player_name=player,
        team_name=team,
        text=" ".join(lines),
        metadata={
            "competition_id": dataset_identity.competition_id,
            "competition_name": dataset_identity.competition_name,
            "season_id": dataset_identity.season_id,
            "season_name": dataset_identity.season_name,
            "player_id": player_id,
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
    )


# ---------------------------------------------------------------------------
# Team-level Analysis
# ---------------------------------------------------------------------------


def _furthest_stage(team_facts: list[dict], stage_order: tuple[str, ...]) -> str:
    """
    Pick the stage the team progressed furthest to.

    Uses `stage_order` (earliest -> latest) when every observed stage
    appears in it — this reproduces the WC2022 knockout-progression
    ranking exactly. Falls back to the stage of the chronologically last
    match otherwise (no fixed order defined, or a stage name outside it,
    e.g. a league's "Matchday N" labels), so rendering never crashes on an
    unrecognized stage name.
    """
    if stage_order:
        order_index = {stage: i for i, stage in enumerate(stage_order)}
        if all(tf["stage"] in order_index for tf in team_facts):
            return max(team_facts, key=lambda tf: order_index[tf["stage"]])["stage"]
    return max(team_facts, key=lambda tf: tf["match_date"])["stage"]


def render_team(
    team_name: str,
    team_facts: list[dict],
    match_index: dict,
    stage_taxonomy: StageTaxonomy = WC2022_STAGE_TAXONOMY,
    dataset_identity: DatasetIdentity = WC2022_DATASET_IDENTITY,
) -> Document:
    """Render a Team-level Analysis document from aggregated TeamMatchFacts."""
    matches = [f["match_id"] for f in team_facts]
    first_shots = [f["first_shot_minute"] for f in team_facts if f.get("first_shot_minute") is not None]
    first_goals = [f["first_goal_minute"] for f in team_facts if f.get("first_goal_minute") is not None]

    lines = [
        f"{team_name} played {len(matches)} match"
        f"{'es' if len(matches) != 1 else ''} at the {dataset_identity.display_name}."
    ]

    if first_shots:
        lines.append(
            f"They took their first shot on average in the "
            f"{_ordinal(round(statistics.fmean(first_shots)))} minute, "
            f"earliest in the {_ordinal(min(first_shots))} minute."
        )
    if first_goals:
        lines.append(
            f"When they scored, their first goal came on average in the "
            f"{_ordinal(round(statistics.fmean(first_goals)))} minute, across "
            f"{len(first_goals)} of their {len(matches)} matches."
        )
    else:
        lines.append("They did not score in any match.")

    # Aggregate play patterns, formations, pass types across matches
    all_play_patterns = Counter()
    all_formations = Counter()
    all_pass_types = Counter()
    total_crosses = 0
    total_possession_events = 0
    total_match_events = 0
    furthest_stage = _furthest_stage(team_facts, stage_taxonomy.stage_order)

    for tf in team_facts:
        all_play_patterns.update(tf.get("play_patterns", {}))
        all_formations.update(tf.get("formations", {}))
        all_pass_types.update(tf.get("pass_types", {}))
        total_crosses += tf.get("crosses", 0)
        if tf.get("possession_share") is not None:
            total_possession_events += tf["possession_share"]
            total_match_events += 100  # proxy

    if total_match_events:
        share = total_possession_events / len(team_facts)
        lines.append(
            f"Across their matches they were the team in possession for "
            f"{_pct(share)} of events, an event-share proxy rather than "
            "StatsBomb's broadcast possession figure."
        )
    if all_play_patterns:
        top = all_play_patterns.most_common(4)
        total_pp = sum(all_play_patterns.values())
        bits = [f"{name.lower()} {_pct(count / total_pp * 100)}" for name, count in top]
        lines.append(f"Their play patterns were dominated by {_oxford(bits)}.")
    if all_formations:
        forms = [f"{f} ({n} formation {_plural('record', n)})"
                 for f, n in all_formations.most_common(3)]
        lines.append(
            "Their most common shapes, counted across Starting XI and Tactical "
            f"Shift events, were {_oxford(forms)}.")

    if all_pass_types:
        total_passes = sum(all_pass_types.values())
        open_play = all_pass_types.get("Open Play", 0)
        set_pieces = {k: v for k, v in all_pass_types.items() if k != "Open Play"}
        sp_total = sum(set_pieces.values())
        bits = [f"{k.lower()} {_pct(v / total_passes * 100)}"
                for k, v in Counter(set_pieces).most_common(4)]
        lines.append(
            f"Of {total_passes} passes, {_pct(open_play / total_passes * 100)} were "
            f"standard open-play passes and {_pct(sp_total / total_passes * 100)} came "
            f"from set pieces or restarts, chiefly {_oxford(bits)}. Passes carrying no "
            "pass type are standard open-play passes and are counted as such rather "
            "than discarded."
        )
        lines.append(
            f"They delivered {total_crosses} crosses, "
            f"{_pct(total_crosses / total_passes * 100)} of their passes."
        )

    team_id = next((f.get("team_id") for f in team_facts if f.get("team_id")), None)

    return Document(
        document_id=f"TEAM-{team_id}",
        level="team",
        team_name=team_name,
        text=" ".join(lines),
        metadata={
            "competition_id": dataset_identity.competition_id,
            "competition_name": dataset_identity.competition_name,
            "season_id": dataset_identity.season_id,
            "season_name": dataset_identity.season_name,
            "matches_played": len(matches),
            "match_ids": sorted(matches),
            "furthest_stage": furthest_stage,
            "avg_first_shot_minute": (round(statistics.fmean(first_shots), 2)
                                      if first_shots else None),
            "avg_first_goal_minute": (round(statistics.fmean(first_goals), 2)
                                      if first_goals else None),
            "matches_scored_in": len(first_goals),
            "formations": dict(all_formations),
            "play_pattern_counts": dict(all_play_patterns),
            "pass_type_counts": dict(all_pass_types),
            "cross_count": total_crosses,
            "possession_is_proxy": True,
            "limitations": LIMITATIONS_NOTE,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _classify_facts_metadata(facts: dict) -> tuple[str, int | None, str | None, int | None, str | None]:
    """
    Classify the identity shape of facts["metadata"] without interpreting
    it — shared by _dataset_identity_from_facts() and
    _resolve_render_dataset_identity() so both apply the exact same rules.

    Returns (shape, competition_id, competition_name, season_id, season_name)
    where shape is one of:

        "absent"    — all four fields are None (no identity information at
                      all, e.g. a hand-built test fixture).
        "complete"  — all four fields are present.
        "ids_only"  — both IDs present, both names absent. This is the
                      shape of the real, already-shipped
                      output/match_facts.json artifact persisted before
                      this batch (competition_id=43, season_id=106).
        "partial"   — anything else: exactly one ID present, or names
                      present without both IDs, or one name present and
                      the other missing. Structurally inconsistent —
                      never silently guessed at.
    """
    meta = facts.get("metadata") or {}
    competition_id = meta.get("competition_id")
    competition_name = meta.get("competition_name")
    season_id = meta.get("season_id")
    season_name = meta.get("season_name")

    has_both_ids = competition_id is not None and season_id is not None
    has_no_ids = competition_id is None and season_id is None
    has_both_names = competition_name is not None and season_name is not None
    has_no_names = competition_name is None and season_name is None

    if has_no_ids and has_no_names:
        shape = "absent"
    elif has_both_ids and has_both_names:
        shape = "complete"
    elif has_both_ids and has_no_names:
        shape = "ids_only"
    else:
        shape = "partial"

    return shape, competition_id, competition_name, season_id, season_name


def _dataset_identity_from_facts(facts: dict) -> DatasetIdentity:
    """
    Derive the active DatasetIdentity from match_facts.json's own metadata
    block (persisted by src.extraction.match_facts.persist()) rather than
    a module-level constant. Used when no explicit dataset_identity is
    supplied to render_all() — see _classify_facts_metadata() for the
    shape rules.

    "absent"   -> legacy/default zero-argument WC2022 fallback.
    "complete" -> used as-is.
    "ids_only" and the IDs are the WC2022 default (43/106) -> safe
                  WC2022 fallback (the IDs prove it).
    "ids_only" with any other IDs -> refuses to guess; raises ValueError.
    "partial"  -> structurally inconsistent; raises ValueError.
    """
    shape, competition_id, competition_name, season_id, season_name = _classify_facts_metadata(facts)

    if shape == "absent":
        return WC2022_DATASET_IDENTITY

    if shape == "complete":
        return DatasetIdentity(
            competition_id=competition_id,
            competition_name=competition_name,
            season_id=season_id,
            season_name=season_name,
        )

    if shape == "ids_only":
        if (competition_id, season_id) == (
            WC2022_DATASET_IDENTITY.competition_id,
            WC2022_DATASET_IDENTITY.season_id,
        ):
            return WC2022_DATASET_IDENTITY
        raise ValueError(
            f"facts['metadata'] has competition_id={competition_id!r}, "
            f"season_id={season_id!r} but is missing competition_name/"
            "season_name, and these IDs do not match the WC2022 default "
            f"({WC2022_DATASET_IDENTITY.competition_id}/"
            f"{WC2022_DATASET_IDENTITY.season_id}) — refusing to guess the "
            "dataset identity. Pass an explicit dataset_identity instead."
        )

    raise ValueError(
        "facts['metadata'] has a structurally incomplete dataset identity "
        f"(competition_id={competition_id!r}, competition_name={competition_name!r}, "
        f"season_id={season_id!r}, season_name={season_name!r}) — expected either "
        "all four fields, none of them, or both IDs with both names absent. "
        "Refusing to guess the dataset identity."
    )


def _resolve_render_dataset_identity(
    facts: dict,
    dataset_identity: DatasetIdentity | None,
) -> DatasetIdentity:
    """
    Resolve the DatasetIdentity to render with, cross-checking an explicit
    override against facts["metadata"] so it can never silently
    contradict — or hide corruption in — what the data itself says. Uses
    the same shape classification as _dataset_identity_from_facts() (see
    _classify_facts_metadata()):

        "absent"   -> the explicit identity is used as given (nothing to
                      conflict with).
        "ids_only" -> the explicit identity may supply the missing names,
                      but its competition_id/season_id MUST equal
                      facts["metadata"]'s IDs — mismatch raises
                      ValueError. (This is the WC2022-legacy-artifact
                      shape, but the rule applies to any competition with
                      this shape, not only WC2022.)
        "complete" -> facts["metadata"] already has a full, authoritative
                      identity. The explicit identity must match it on
                      ALL FOUR fields (names are not disposable labels
                      once persisted) — any disagreement raises
                      ValueError.
        "partial"  -> structurally inconsistent metadata always raises,
                      even with an explicit override — an override must
                      never be used to paper over corrupt metadata.

    With no explicit `dataset_identity`, defers entirely to
    _dataset_identity_from_facts(facts).
    """
    if dataset_identity is None:
        return _dataset_identity_from_facts(facts)

    shape, meta_competition_id, meta_competition_name, meta_season_id, meta_season_name = (
        _classify_facts_metadata(facts)
    )

    if shape == "absent":
        return dataset_identity

    if shape == "ids_only":
        if (dataset_identity.competition_id, dataset_identity.season_id) != (
            meta_competition_id, meta_season_id,
        ):
            raise ValueError(
                f"Explicit dataset_identity (competition_id={dataset_identity.competition_id}, "
                f"season_id={dataset_identity.season_id}) conflicts with facts['metadata'] "
                f"(competition_id={meta_competition_id!r}, season_id={meta_season_id!r})."
            )
        return dataset_identity

    if shape == "complete":
        meta_identity = DatasetIdentity(
            competition_id=meta_competition_id,
            competition_name=meta_competition_name,
            season_id=meta_season_id,
            season_name=meta_season_name,
        )
        if dataset_identity != meta_identity:
            raise ValueError(
                f"Explicit dataset_identity ({dataset_identity}) conflicts with the "
                f"complete identity already in facts['metadata'] ({meta_identity})."
            )
        return dataset_identity

    raise ValueError(
        "facts['metadata'] has a structurally incomplete dataset identity "
        f"(competition_id={meta_competition_id!r}, competition_name={meta_competition_name!r}, "
        f"season_id={meta_season_id!r}, season_name={meta_season_name!r}) — an explicit "
        "dataset_identity cannot be used to override structurally inconsistent metadata."
    )


def _stage_taxonomy_from_facts(facts: dict) -> StageTaxonomy:
    """Resolve the persisted taxonomy without guessing non-WC semantics."""
    metadata = facts.get("metadata") or {}
    persisted = metadata.get("stage_taxonomy")

    if persisted is not None:
        return StageTaxonomy.from_dict(persisted)

    competition_id = metadata.get("competition_id")
    season_id = metadata.get("season_id")

    # Legacy zero-metadata rendering and already-shipped WC2022 artifacts.
    if not metadata or (
        competition_id == WC2022_DATASET_IDENTITY.competition_id
        and season_id == WC2022_DATASET_IDENTITY.season_id
    ):
        return WC2022_STAGE_TAXONOMY

    # A known non-WC dataset must carry the taxonomy used at extraction.
    if competition_id is not None or season_id is not None:
        raise ValueError(
            "facts['metadata'] is missing stage_taxonomy for a non-WC2022 "
            "dataset; refusing to silently apply WC2022 stage semantics."
        )

    # Metadata may contain unrelated legacy fields but no dataset identity.
    return WC2022_STAGE_TAXONOMY


def render_all(
    facts: dict,
    stage_taxonomy: StageTaxonomy | None = None,
    dataset_identity: DatasetIdentity | None = None,
) -> list[Document]:
    """
    Render all documents from match_facts.json.

    `stage_taxonomy` defaults to None, which loads the taxonomy persisted in
    facts["metadata"]. Legacy WC2022 artifacts without that field retain the
    explicit WC2022 taxonomy; known non-WC datasets must persist their taxonomy
    rather than silently inheriting WC2022 semantics.

    `dataset_identity` defaults to None, which auto-derives the identity
    from `facts["metadata"]` (see _dataset_identity_from_facts) — pass it
    explicitly to override, but an override that disagrees with what
    facts["metadata"] itself says raises ValueError rather than silently
    mislabeling the corpus (see _resolve_render_dataset_identity — when
    facts["metadata"] is complete, ALL FOUR identity fields must agree,
    not just the IDs). Either way, every rendered document's metadata and
    prose reflects the active competition/season, not a hardcoded one.
    """
    identity = _resolve_render_dataset_identity(facts, dataset_identity)
    if stage_taxonomy is None:
        stage_taxonomy = _stage_taxonomy_from_facts(facts)
    documents = []

    # Build match index for Level 4 best/worst ranking
    match_index = {mf["match_id"]: mf for mf in facts["match_facts"]}

    # Level 1 & 2: one per match
    for mf in facts["match_facts"]:
        documents.append(render_level1(mf, dataset_identity=identity))
        documents.append(render_level2(mf, dataset_identity=identity))

    # Level 3: one per player per match
    for pmf in facts["player_match_facts"]:
        documents.append(render_level3(pmf, dataset_identity=identity))

    # Level 4: aggregate player facts across matches
    player_groups: dict[int, list[dict]] = defaultdict(list)
    for pmf in facts["player_match_facts"]:
        pid = pmf.get("player_id")
        if pid is not None:
            player_groups[pid].append(pmf)

    for pid, pfacts in sorted(player_groups.items(), key=lambda kv: str(kv[0])):
        documents.append(render_level4(pid, pfacts, match_index, stage_taxonomy=stage_taxonomy,
                                        dataset_identity=identity))

    # Team-level: aggregate team facts across matches
    team_groups: dict[str, list[dict]] = defaultdict(list)
    for tf in facts["team_match_facts"]:
        team_groups[tf["team_name"]].append(tf)

    for team_name, tfacts in sorted(team_groups.items()):
        documents.append(render_team(team_name, tfacts, match_index, stage_taxonomy=stage_taxonomy,
                                      dataset_identity=identity))

    return documents


def persist(documents: list[Document], output_path: Path = Path("documents.json")):
    """Persist documents to JSON."""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump([d.to_dict() for d in documents], handle, ensure_ascii=False, indent=2)
    return output_path
