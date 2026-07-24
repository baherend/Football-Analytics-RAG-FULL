"""
minutes_played.py — §3 Minutes-Played Algorithm

Computes per-player minutes from lineups.json position segments.

Interface (used by 01_documents.py):
    context = build_match_context(events, lineups)
    minutes, anomalies = minutes_played(lineups, context, collect_anomalies=True)

Returns:
    minutes   — dict[(team_name, player_name) -> float minutes]
    anomalies — list of dicts describing handled data defects

CRITICAL: Clock times in position segments are MATCH-ABSOLUTE, not period-relative.
"83:43" with to_period=2 means minute 83:43 of the ENTIRE MATCH, not 83:43 into
the second half.

Algorithm:
    1. Determine match end time from the last event in non-shootout periods.
    2. Find substitution-off times and dismissal times from events.
    3. For each player, parse segments (from/to are match-absolute seconds).
    4. Fix inverted periods (§7.3) by swapping times and periods.
    5. For null `to`, substitute the match end time.
    6. Filter out segments starting at/after dismissal (phantom after red — §7.3).
    7. Merge overlapping segments via interval union (§7.3).
    8. Clip merged intervals at substitution-off time or dismissal time.
    9. Sum duration, return total minutes as float.

Known data defects handled (§7.3):
    - Hennessey (3857273): phantom segment starting after red card at 83:43
    - Messi, Acuna, Montiel (3869685): segments with to_period < from_period
    - Koundé, Varane (3869685): duplicate overlapping segments
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHOOTOUT_PERIOD = 5
DISMISSAL_CARDS = frozenset({"Red Card", "Second Yellow"})


# ---------------------------------------------------------------------------
# Clock parsing
# ---------------------------------------------------------------------------

def _clock_to_seconds(clock: str | None) -> float | None:
    """Parse '83:43' → 5023.0 seconds. Returns None for null/empty."""
    if not clock:
        return None
    parts = clock.split(":")
    if len(parts) != 2:
        return None
    return int(parts[0]) * 60 + int(parts[1])


# ---------------------------------------------------------------------------
# Match end time from events
# ---------------------------------------------------------------------------

def _find_match_end(events: list[dict]) -> float:
    """
    Find the actual match end time from events.
    Uses the maximum (minute*60 + second) across all events in non-shootout periods.
    """
    max_time = 0.0
    for event in events:
        if event.get("period") == SHOOTOUT_PERIOD:
            continue
        minute = event.get("minute", 0)
        second = event.get("second", 0)
        t = minute * 60 + second
        if t > max_time:
            max_time = t
    return max_time if max_time > 0 else 90.0 * 60


# ---------------------------------------------------------------------------
# Dismissal and substitution detection from events
# ---------------------------------------------------------------------------

def _find_dismissals(events: list[dict]) -> dict[tuple[str, str], float]:
    """
    Find dismissal times from events as match-absolute seconds.
    Keyed on (team_name, player_name).
    Period-5 dismissals (shootout) are excluded.
    """
    dismissals: dict[tuple[str, str], float] = {}
    for event in events:
        if event.get("type", {}).get("name") not in ("Foul Committed", "Bad Behaviour"):
            continue
        if event.get("period") == SHOOTOUT_PERIOD:
            continue
        card = None
        for block in ("foul_committed", "bad_behaviour"):
            c = (event.get(block) or {}).get("card")
            if c:
                card = c
                break
        if not card or card.get("name") not in DISMISSAL_CARDS:
            continue
        team = event.get("team", {}).get("name", "")
        player = event.get("player", {}).get("name", "")
        minute = event.get("minute", 0)
        second = event.get("second", 0)
        dismissals[(team, player)] = minute * 60 + second
    return dismissals


def _find_substitutions(events: list[dict]) -> dict[tuple[str, str], float]:
    """
    Find substitution-off times from events as match-absolute seconds.
    Keyed on (team_name, player_name) of the player being substituted OFF.
    """
    subs: dict[tuple[str, str], float] = {}
    for event in events:
        if event.get("type", {}).get("name") != "Substitution":
            continue
        if event.get("period") == SHOOTOUT_PERIOD:
            continue
        team = event.get("team", {}).get("name", "")
        player = event.get("player", {}).get("name", "")
        minute = event.get("minute", 0)
        second = event.get("second", 0)
        subs[(team, player)] = minute * 60 + second
    return subs


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def _parse_segments(
    positions: list[dict],
    match_end: float,
) -> list[tuple[float, float]]:
    """
    Parse position segments into (from_sec, to_sec).
    Times are MATCH-ABSOLUTE seconds.
    null `to` → match_end.
    Inverted periods (§7.3) are fixed by swapping times and periods.
    """
    segments = []
    for pos in positions:
        from_period = pos.get("from_period")
        to_period = pos.get("to_period")
        from_clock = pos.get("from")
        to_clock = pos.get("to")

        if from_period is None:
            continue

        from_sec = _clock_to_seconds(from_clock) or 0.0

        if to_period is None:
            # null to_period: player still on pitch at match end
            to_sec = _clock_to_seconds(to_clock)
            if to_sec is None:
                to_sec = match_end
        else:
            to_sec = _clock_to_seconds(to_clock)
            if to_sec is None:
                to_sec = match_end

        # §7.3 fix inverted periods: swap times and periods
        if to_period is not None and to_period < from_period:
            from_sec, to_sec = to_sec, from_sec

        segments.append((from_sec, to_sec))
    return segments


def _filter_phantoms(
    segments: list[tuple[float, float]],
    dismissal_sec: float | None,
    player: str,
    team: str,
    anomalies: list[dict],
) -> list[tuple[float, float]]:
    """Filter out segments starting at/after dismissal (phantom after red — §7.3)."""
    if dismissal_sec is None:
        return segments

    valid = []
    for from_sec, to_sec in segments:
        if from_sec >= dismissal_sec:
            anomalies.append({
                "type": "phantom_after_dismissal",
                "player": player, "team": team,
                "from_sec": from_sec, "dismissal_sec": dismissal_sec,
            })
            continue
        valid.append((from_sec, to_sec))
    return valid


def _merge_intervals(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Merge overlapping segments via interval union.
    §7.3: Koundé and Varane (3869685) carry duplicate overlapping segments.
    """
    if not segments:
        return []

    segments = sorted(segments)
    merged = [segments[0]]
    for start, end in segments[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clip_at_time(
    segments: list[tuple[float, float]],
    clip_time: float,
) -> list[tuple[float, float]]:
    """
    Clip all segments so none extends past clip_time.
    Used for substitution-off and dismissal clipping.
    """
    clipped = []
    for start, end in segments:
        if start >= clip_time:
            continue  # entire segment is past the clip point
        clipped.append((start, min(end, clip_time)))
    return clipped


def _compute_minutes(segments: list[tuple[float, float]]) -> float:
    """Sum minutes across merged segments."""
    total_seconds = sum(max(0, end - start) for start, end in segments)
    return total_seconds / 60.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_match_context(events: list[dict], lineups: list[dict]) -> dict:
    """
    Build per-match context needed by minutes_played.
    Extracts dismissal times, substitution times, and match end time from events.
    """
    dismissals = _find_dismissals(events)
    substitutions = _find_substitutions(events)
    match_end = _find_match_end(events)
    return {
        "dismissals": dismissals,
        "substitutions": substitutions,
        "match_end": match_end,
        "events": events,
        "lineups": lineups,
    }


def minutes_played(
    lineups: list[dict],
    context: dict,
    collect_anomalies: bool = False,
) -> tuple[dict[tuple[str, str], float], list[dict]]:
    """
    Compute minutes played for every player in the match.

    Parameters:
        lineups — the lineups JSON (list of team dicts)
        context — from build_match_context
        collect_anomalies — if True, return handled defect descriptions

    Returns:
        minutes — dict[(team_name, player_name) -> float minutes]
        anomalies — list of anomaly dicts (empty if collect_anomalies=False)
    """
    dismissals = context.get("dismissals", {})
    substitutions = context.get("substitutions", {})
    match_end = context.get("match_end", 90.0 * 60)
    anomalies: list[dict] = []
    minutes: dict[tuple[str, str], float] = {}

    for team_data in lineups:
        team = team_data["team_name"]
        for player_data in team_data["lineup"]:
            player = player_data["player_name"]
            positions = player_data.get("positions") or []

            if not positions:
                # Unused substitute
                minutes[(team, player)] = 0.0
                continue

            # Parse segments (fix inverted periods, null to → match_end)
            raw_segments = _parse_segments(positions, match_end)

            # Filter phantom-after-dismissal segments
            dismissal_sec = dismissals.get((team, player))
            valid_segments = _filter_phantoms(
                raw_segments, dismissal_sec, player, team, anomalies
            )

            # Merge overlapping segments
            merged = _merge_intervals(valid_segments)

            # Clip at substitution-off time (player subbed off before match end)
            sub_off_time = substitutions.get((team, player))
            if sub_off_time is not None:
                merged = _clip_at_time(merged, sub_off_time)

            # Clip at dismissal time (player sent off)
            if dismissal_sec is not None:
                merged = _clip_at_time(merged, dismissal_sec)

            # Compute total minutes
            mins = _compute_minutes(merged)
            minutes[(team, player)] = mins

    if not collect_anomalies:
        return minutes, []

    return minutes, anomalies
