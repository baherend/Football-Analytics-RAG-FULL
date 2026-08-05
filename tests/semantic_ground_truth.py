"""
semantic_ground_truth.py — Semantic Ground Truth Foundation

Versioned, tournament-agnostic semantic ground-truth dataset tied to an exact
chunks.json snapshot. Designed for reuse across future football tournaments.

This module contains:
- Schema version and dataset metadata
- Allowed-value constants
- Exactly twenty-four verified ground-truth cases (6 pilot + 6 expanded + 6 advanced + 6 extended)
- Pure validation helper functions (no retrieval, no API calls)

Usage:
    from tests.semantic_ground_truth import (
        SEMANTIC_GROUND_TRUTH_METADATA,
        SEMANTIC_GROUND_TRUTH,
        validate_semantic_ground_truth,
    )
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# A. Schema Version
# ---------------------------------------------------------------------------

SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# B. Dataset Metadata
# ---------------------------------------------------------------------------

SEMANTIC_GROUND_TRUTH_METADATA: dict = {
    "schema_version": SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION,
    "dataset_id": "statsbomb-fifa-world-cup-2022",
    "tournament_name": "FIFA World Cup 2022",
    "season_name": "2022",
    "source_name": "StatsBomb Open Data",
    "competition_id": 43,
    "season_id": 106,
    "chunks_path": "output/chunks.json",
    "chunks_sha256": "506f26be3ea4739d02212d7bda61559c0366d749e005182c65ab7651e4623fa6",
    "retrieval_unit": "document_id",
    "expected_case_count": 24,
    "expected_case_group_counts": {
        "l1": 4,
        "l2": 4,
        "l3": 4,
        "l4": 4,
        "team": 4,
        "multi": 4,
    },
    "notes": (
        "Ground truth is tied to the exact chunks.json snapshot identified by "
        "chunks_sha256. Rebuilding chunks will change the hash and may require "
        "re-verification of all document IDs and evidence snippets."
    ),
}

# ---------------------------------------------------------------------------
# C. Allowed Values
# ---------------------------------------------------------------------------

ALLOWED_ROUTES: tuple[str, ...] = ("semantic", "hybrid")
ALLOWED_ANSWERABILITY: tuple[str, ...] = (
    "answerable",
    "partially_answerable",
    "unanswerable",
)
ALLOWED_LEVELS: tuple[str, ...] = ("1", "2", "3", "4", "team")
ALLOWED_CASE_GROUPS: tuple[str, ...] = ("l1", "l2", "l3", "l4", "team", "multi")

# ---------------------------------------------------------------------------
# D. Original Pilot Case IDs and Canonical Hash
# ---------------------------------------------------------------------------

ORIGINAL_PILOT_CASE_IDS: tuple[str, ...] = (
    "gt-pilot-l1-01",
    "gt-pilot-l2-01",
    "gt-pilot-l3-01",
    "gt-pilot-l4-01",
    "gt-pilot-team-01",
    "gt-pilot-multi-01",
)

ORIGINAL_PILOT_CASES_SHA256: str = (
    "994f3bc6ddb10b53cdea8da1e91b91d7700b45681df96b6a93ad1db1c9915906"
)


def compute_canonical_case_hash(cases: list[dict]) -> str:
    """Compute SHA-256 of a canonical JSON serialization of cases.

    Cases are sorted by 'id' and serialized with ensure_ascii=False,
    sort_keys=True, and compact separators.
    """
    sorted_cases = sorted(cases, key=lambda c: c["id"])
    canonical = json.dumps(
        sorted_cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# E. Expected Case IDs (all 24)
# ---------------------------------------------------------------------------

EXPECTED_CASE_IDS: tuple[str, ...] = (
    # Original 6 pilot cases
    "gt-pilot-l1-01",
    "gt-pilot-l2-01",
    "gt-pilot-l3-01",
    "gt-pilot-l4-01",
    "gt-pilot-team-01",
    "gt-pilot-multi-01",
    # Expanded 6 cases
    "gt-l1-02",
    "gt-l2-02",
    "gt-l3-02",
    "gt-l4-02",
    "gt-team-02",
    "gt-multi-02",
    # Advanced 6 cases
    "gt-l1-03",
    "gt-l2-03",
    "gt-l3-03",
    "gt-l4-03",
    "gt-team-03",
    "gt-multi-03",
    # Extended 6 cases
    "gt-l1-04",
    "gt-l2-04",
    "gt-l3-04",
    "gt-l4-04",
    "gt-team-04",
    "gt-multi-04",
)

# ---------------------------------------------------------------------------
# F. Foundation Twelve-Case Hash
# ---------------------------------------------------------------------------

FOUNDATION_TWELVE_CASES_SHA256: str = (
    "ef68a6f83243ac3197a3d3da05824efa4f92d35d25f48de7f59b5d1e855adc91"
)

# ---------------------------------------------------------------------------
# G. Ground Truth Cases
# ---------------------------------------------------------------------------

SEMANTIC_GROUND_TRUTH: list[dict] = [
    # -----------------------------------------------------------------------
    # CASE 1 — L1 Pilot: Opening Match
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-l1-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l1",
        "query": "What happened in the opening match of the World Cup?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "1",
        "acceptable_levels": ["1"],
        "relevant_document_ids": ["L1-match-3857286"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Ecuador beat Qatar 2-0 in the opening match.",
                "source_document_ids": ["L1-match-3857286"],
                "evidence_snippets": [
                    "Ecuador beat Qatar 2-0 in normal time.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Enner Valencia scored both goals for Ecuador.",
                "source_document_ids": ["L1-match-3857286"],
                "evidence_snippets": [
                    "Enner Remberto Valencia Lastra scored for Ecuador in the 15th minute with the right foot, from a penalty (xG 0.78).",
                    "Enner Remberto Valencia Lastra scored for Ecuador in the 30th minute with the head, from open play",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "The match was played on 2022-11-20 at Al Bayt Stadium.",
                "source_document_ids": ["L1-match-3857286"],
                "evidence_snippets": [
                    "was played on 2022-11-20 at Al Bayt Stadium.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Qatar won the opening match.",
                "reason": "Ecuador won 2-0; claiming Qatar won contradicts the source.",
            },
            {
                "claim": "The opening match went to extra time.",
                "reason": "The match ended in normal time; no extra time occurred.",
            },
        ],
        "notes": (
            "Tests retrieval of a specific match by event description. "
            "The opening match is Qatar vs Ecuador, match_id 3857286, "
            "not identifiable by a metadata flag."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 2 — L2 Pilot: Argentina vs France Final Key Events
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-l2-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l2",
        "query": "What were the key events in the Argentina vs France Final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["1", "2"],
        "relevant_document_ids": ["L2-match-3869685"],
        "optional_relevant_document_ids": ["L1-match-3869685"],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Messi scored a penalty goal in the 22nd minute.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Lionel Andrés Messi Cuccittini (Argentina) in period 1, 22nd minute, with the left foot, shot type Penalty",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Di María scored in the 35th minute from open play, assisted by Mac Allister.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Ángel Fabián Di María Hernández (Argentina) in period 1, 35th minute, with the left foot, shot type Open Play",
                    "Alexis Mac Allister provided the assist.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Mbappé scored twice in quick succession in the 79th and 80th minutes.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Kylian Mbappé Lottin (France) in period 2, 79th minute, with the right foot, shot type Penalty",
                    "Goal: Kylian Mbappé Lottin (France) in period 2, 80th minute, with the right foot, shot type Open Play",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Messi scored again in the 107th minute in extra time.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Lionel Andrés Messi Cuccittini (Argentina) in period 4, 107th minute, with the right foot, shot type Open Play",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Mbappé completed his hat-trick with a penalty in the 117th minute.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Kylian Mbappé Lottin (France) in period 4, 117th minute, with the right foot, shot type Penalty",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Griezmann scored in the Final.",
                "reason": "The L2 document records no goal by Griezmann; he was substituted off.",
            },
            {
                "claim": "The Final ended 2-1 in normal time.",
                "reason": "The match was 3-3 after extra time; the L2 document records six goals.",
            },
        ],
        "notes": (
            "Tests retrieval of L2 key events for the Final. "
            "The true Final match_id is 3869685, NOT 3857270. "
            "L1-match-3869685 is an optional supporting document."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 3 — L3 Pilot: Messi vs Croatia Semi-final
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-l3-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l3",
        "query": "How did Messi perform against Croatia in the semi-final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "3",
        "acceptable_levels": ["3"],
        "relevant_document_ids": ["L3-match-3869519-player-5503"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Messi played 95 minutes against Croatia in the Semi-finals.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "was on the pitch for 95 minutes.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Messi scored 1 goal from 2 shots worth 0.89 expected goals.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "He took 2 shots worth 0.89 expected goals, 2 from inside the penalty area and 0 from outside, and scored 1.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Messi provided 1 assist in the match.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "He provided 1 assist.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Messi attempted 41 passes and completed 82.9%.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "He attempted 41 passes and completed 82.9%",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Messi scored a hat-trick against Croatia.",
                "reason": "The L3 document records 1 goal, not 3.",
            },
            {
                "claim": "Messi played 120 minutes against Croatia.",
                "reason": "The L3 document records 95 minutes.",
            },
        ],
        "notes": (
            "Tests retrieval of a specific L3 player-match document. "
            "The document must match on player (Messi), team (Argentina), "
            "opponent (Croatia), and stage (Semi-finals)."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 4 — L4 Pilot: Messi Tournament Summary
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-l4-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l4",
        "query": "Describe Messi's overall World Cup tournament performance.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "4",
        "acceptable_levels": ["4"],
        "relevant_document_ids": ["L4-player-5503"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Messi appeared in 7 matches and played 733.9 minutes in total.",
                "source_document_ids": ["L4-player-5503"],
                "evidence_snippets": [
                    "appeared in 7 matches at the FIFA World Cup 2022, playing 733.9 minutes in total.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Messi scored 7 goals and provided 3 assists from 6.03 expected goals.",
                "source_document_ids": ["L4-player-5503"],
                "evidence_snippets": [
                    "he scored 7 goals and provided 3 assists from 6.03 expected goals",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Messi's goal contribution rate was 1.43 per match.",
                "source_document_ids": ["L4-player-5503"],
                "evidence_snippets": [
                    "a goal contribution rate of 1.43 per match.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Messi's best match by expected goals was the Final against France with 1.46 xG and 2 goals.",
                "source_document_ids": ["L4-player-5503"],
                "evidence_snippets": [
                    "His best match by expected goals was against France in the Final on 2022-12-18, with 1.46 xG, 2 goals and 0 assists.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Messi averaged 4.57 shots and 0.86 expected goals per match.",
                "source_document_ids": ["L4-player-5503"],
                "evidence_snippets": [
                    "He averaged 4.57 shots and 0.86 expected goals per match",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Messi scored 10 goals in the tournament.",
                "reason": "The L4 document records 7 goals, not 10.",
            },
            {
                "claim": "Messi won the Golden Boot award.",
                "reason": "The L4 document does not mention any awards; this is external knowledge not in the source.",
            },
        ],
        "notes": (
            "Tests retrieval of the L4 tournament-summary document for Messi. "
            "This is a tournament-level summary, not a single-match document."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 5 — Team Pilot: Argentina Playing Style
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-team-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "team",
        "query": "What was Argentina's playing style and most common formation?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "team",
        "acceptable_levels": ["team"],
        "relevant_document_ids": ["TEAM-779"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Argentina's most common formations were 352, 433, and 442.",
                "source_document_ids": ["TEAM-779"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 352 (8 formation records), 433 (8 formation records), and 442 (5 formation records).",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Argentina had 58.6% possession as an event-share proxy, not StatsBomb broadcast possession.",
                "source_document_ids": ["TEAM-779"],
                "evidence_snippets": [
                    "they were the team in possession for 58.6% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Argentina's play patterns were dominated by regular play (40.0%), throw-ins (23.3%), and free kicks (20.3%).",
                "source_document_ids": ["TEAM-779"],
                "evidence_snippets": [
                    "Their play patterns were dominated by regular play 40.0%, from throw in 23.3%, from free kick 20.3%, and from goal kick 5.5%.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "84.3% of Argentina's 4615 passes were standard open-play passes.",
                "source_document_ids": ["TEAM-779"],
                "evidence_snippets": [
                    "Of 4615 passes, 84.3% were standard open-play passes and 15.7% came from set pieces or restarts",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Argentina's possession figure of 58.6% is the official StatsBomb broadcast possession.",
                "reason": (
                    "The document explicitly states this is an event-share proxy, "
                    "not StatsBomb's broadcast possession figure. Treating it as "
                    "official broadcast possession misrepresents the data limitation."
                ),
            },
            {
                "claim": "Argentina played exclusively in a 4-3-3 formation.",
                "reason": "The document lists three common formations (352, 433, 442), not a single fixed shape.",
            },
        ],
        "notes": (
            "Tests retrieval of the Team tournament document for Argentina. "
            "The possession limitation (event-share proxy) must be preserved "
            "in both required facts and forbidden claims."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 6 — Multi-level Pilot: Final Match + Player Performances
    # -----------------------------------------------------------------------
    {
        "id": "gt-pilot-multi-01",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "multi",
        "query": "How did the Argentina vs France Final unfold, and how did Messi and Mbappé perform?",
        "expected_route": "hybrid",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["1", "2", "3"],
        "relevant_document_ids": [
            "L1-match-3869685",
            "L2-match-3869685",
            "L3-match-3869685-player-5503",
            "L3-match-3869685-player-3009",
        ],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "The Final ended 3-3 after extra time and Argentina won the penalty shootout 4-2.",
                "source_document_ids": ["L1-match-3869685"],
                "evidence_snippets": [
                    "The match finished level at 3-3 after extra time. This score excludes any penalty shootout. Argentina won the penalty shootout 4-2 against France.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Di María scored Argentina's second goal in the 35th minute from open play.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Ángel Fabián Di María Hernández (Argentina) in period 1, 35th minute, with the left foot, shot type Open Play",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Mbappé scored twice in two minutes (79th and 80th) to force extra time.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Kylian Mbappé Lottin (France) in period 2, 79th minute, with the right foot, shot type Penalty",
                    "Goal: Kylian Mbappé Lottin (France) in period 2, 80th minute, with the right foot, shot type Open Play",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "In the Final, Messi scored 2 goals from 5 shots worth 1.46 expected goals in 124.1 minutes.",
                "source_document_ids": ["L3-match-3869685-player-5503"],
                "evidence_snippets": [
                    "was on the pitch for 124.1 minutes. He took 5 shots worth 1.46 expected goals, 4 from inside the penalty area and 1 from outside, and scored 2.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "In the Final, Mbappé scored 3 goals from 6 shots worth 1.78 expected goals.",
                "source_document_ids": ["L3-match-3869685-player-3009"],
                "evidence_snippets": [
                    "He took 6 shots worth 1.78 expected goals, 4 from inside the penalty area and 2 from outside, and scored 3.",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "Mbappé scored his third goal via a penalty in the 117th minute.",
                "source_document_ids": ["L2-match-3869685"],
                "evidence_snippets": [
                    "Goal: Kylian Mbappé Lottin (France) in period 4, 117th minute, with the right foot, shot type Penalty",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "France won the penalty shootout.",
                "reason": "Argentina won the shootout 4-2; claiming France won contradicts the L1 source.",
            },
            {
                "claim": "Messi scored a hat-trick in the Final.",
                "reason": "The L3 document records 2 goals for Messi, not 3. Mbappé scored the hat-trick.",
            },
            {
                "claim": "The Final ended 2-1 in normal time.",
                "reason": "The match was 3-3 after extra time with six goals scored.",
            },
        ],
        "notes": (
            "Tests multi-level retrieval: L1 match summary, L2 key events, "
            "and L3 player-match stats for both Messi and Mbappé. "
            "All documents share the same verified Final match_id 3869685. "
            "L4 tournament summaries are not included as the query asks about "
            "the specific Final match, not the overall tournament."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 7 — L1 Expanded: England vs Iran
    # -----------------------------------------------------------------------
    {
        "id": "gt-l1-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l1",
        "query": "Describe the match between England and Iran.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "1",
        "acceptable_levels": ["1", "2"],
        "relevant_document_ids": ["L1-match-3857271"],
        "optional_relevant_document_ids": ["L2-match-3857271"],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "England beat Iran 6-2 in normal time.",
                "source_document_ids": ["L1-match-3857271"],
                "evidence_snippets": [
                    "England beat Iran 6-2 in normal time.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "The match was played on 2022-11-21 at Sheikh Khalifa International Stadium.",
                "source_document_ids": ["L1-match-3857271"],
                "evidence_snippets": [
                    "was played on 2022-11-21 at Sheikh Khalifa International Stadium.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Bukayo Saka scored twice for England.",
                "source_document_ids": ["L1-match-3857271"],
                "evidence_snippets": [
                    "Bukayo Saka scored for England in the 42nd minute with the left foot, from open play",
                    "Bukayo Saka scored for England in the 61st minute with the left foot, from open play",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "The match did not require extra time or a penalty shootout.",
                "source_document_ids": ["L1-match-3857271"],
                "evidence_snippets": [
                    "England beat Iran 6-2 in normal time. This score excludes any penalty shootout.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Iran won the match against England.",
                "reason": "England won 6-2; claiming Iran won contradicts the source.",
            },
            {
                "claim": "The match ended 2-1.",
                "reason": "The final score was 6-2 to England, not 2-1.",
            },
        ],
        "notes": (
            "Tests retrieval of a high-scoring Group Stage match. "
            "The L1 document clearly records the score, venue, and multiple goalscorers."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 8 — L2 Expanded: Argentina vs Croatia Substitutions
    # -----------------------------------------------------------------------
    {
        "id": "gt-l2-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l2",
        "query": "What substitutions were made in the Argentina vs Croatia semi-final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["2"],
        "relevant_document_ids": ["L2-match-3869519"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Croatia brought on Mislav Oršić for Borna Sosa in the 45th minute.",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Substitution: Croatia brought on Mislav Oršić for Borna Sosa in the 45th minute.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Argentina brought on Lisandro Martínez for Leandro Daniel Paredes in the 61st minute.",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Substitution: Argentina brought on Lisandro Martínez for Leandro Daniel Paredes in the 61st minute.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Argentina brought on Paulo Bruno Exequiel Dybala for Julián Álvarez in the 73rd minute.",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Substitution: Argentina brought on Paulo Bruno Exequiel Dybala for Julián Álvarez in the 73rd minute.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Argentina won the Semi-finals match 3-0 against Croatia.",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Goal: Lionel Andrés Messi Cuccittini (Argentina) in period 1, 33rd minute, with the left foot, shot type Penalty",
                    "Goal: Julián Álvarez (Argentina) in period 1, 38th minute, with the right foot, shot type Open Play",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Messi was substituted off in the Semi-finals.",
                "reason": "The L2 document does not record a substitution involving Messi; he played the full match.",
            },
            {
                "claim": "Croatia made no substitutions in the match.",
                "reason": "The L2 document records multiple Croatian substitutions.",
            },
        ],
        "notes": (
            "Tests retrieval of L2 substitution events. "
            "The L2 document records 10 total substitutions across both teams. "
            "Substitution evidence includes team, player in/out, and minute."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 9 — L3 Expanded: Griezmann Passing in the Final
    # -----------------------------------------------------------------------
    {
        "id": "gt-l3-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l3",
        "query": "What were Antoine Griezmann's passing statistics in the Final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "3",
        "acceptable_levels": ["3"],
        "relevant_document_ids": ["L3-match-3869685-player-5487"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Griezmann played 70.5 minutes in the Final.",
                "source_document_ids": ["L3-match-3869685-player-5487"],
                "evidence_snippets": [
                    "was on the pitch for 70.5 minutes.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Griezmann attempted 30 passes and completed 70.0%.",
                "source_document_ids": ["L3-match-3869685-player-5487"],
                "evidence_snippets": [
                    "He attempted 30 passes and completed 70.0%",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "13 of Griezmann's passes were delivered into the final third.",
                "source_document_ids": ["L3-match-3869685-player-5487"],
                "evidence_snippets": [
                    "of which 13 were delivered into the final third.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Under pressure Griezmann attempted 4 passes and completed 75.0%.",
                "source_document_ids": ["L3-match-3869685-player-5487"],
                "evidence_snippets": [
                    "Under pressure he attempted 4 passes and completed 75.0%",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Griezmann played as center attacking midfield for France against Argentina.",
                "source_document_ids": ["L3-match-3869685-player-5487"],
                "evidence_snippets": [
                    "Antoine Griezmann of France played as center attacking midfield against Argentina in the Final",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Griezmann completed 90% of his passes in the Final.",
                "reason": "The L3 document records 70.0% pass completion, not 90%.",
            },
            {
                "claim": "Griezmann attempted 50 passes in the Final.",
                "reason": "The L3 document records 30 passes attempted, not 50.",
            },
        ],
        "notes": (
            "Tests retrieval of L3 passing statistics for a specific player in the Final. "
            "Griezmann's player_id is 5487. The document covers passing, "
            "final-third passes, and passing under pressure."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 10 — L4 Expanded: Mbappé Tournament Summary
    # -----------------------------------------------------------------------
    {
        "id": "gt-l4-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l4",
        "query": "Describe Kylian Mbappé's overall World Cup tournament performance.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "4",
        "acceptable_levels": ["4"],
        "relevant_document_ids": ["L4-player-3009"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Mbappé appeared in 7 matches and played 653.6 minutes in total.",
                "source_document_ids": ["L4-player-3009"],
                "evidence_snippets": [
                    "appeared in 7 matches at the FIFA World Cup 2022, playing 653.6 minutes in total.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Mbappé scored 8 goals and provided 2 assists from 4.23 expected goals.",
                "source_document_ids": ["L4-player-3009"],
                "evidence_snippets": [
                    "he scored 8 goals and provided 2 assists from 4.23 expected goals",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Mbappé's goal contribution rate was 1.43 per match.",
                "source_document_ids": ["L4-player-3009"],
                "evidence_snippets": [
                    "a goal contribution rate of 1.43 per match.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Mbappé's best match by expected goals was against Argentina in the Final with 1.78 xG and 3 goals.",
                "source_document_ids": ["L4-player-3009"],
                "evidence_snippets": [
                    "His best match by expected goals was against Argentina in the Final on 2022-12-18, with 1.78 xG, 3 goals and 0 assists.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Mbappé played 3 group-stage matches and 4 knockout matches.",
                "source_document_ids": ["L4-player-3009"],
                "evidence_snippets": [
                    "He played 3 group-stage matches and 4 knockout matches",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Mbappé scored 12 goals in the tournament.",
                "reason": "The L4 document records 8 goals, not 12.",
            },
            {
                "claim": "Mbappé provided 5 assists in the tournament.",
                "reason": "The L4 document records 2 assists, not 5.",
            },
        ],
        "notes": (
            "Tests retrieval of the L4 tournament-summary document for Mbappé. "
            "Mbappé's player_id is 3009. The document covers goals, assists, "
            "xG, match-to-match consistency, and group vs knockout breakdown."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 11 — Team Expanded: Morocco Playing Style
    # -----------------------------------------------------------------------
    {
        "id": "gt-team-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "team",
        "query": "How did Morocco play in the tournament, and which formations did they use most?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "team",
        "acceptable_levels": ["team"],
        "relevant_document_ids": ["TEAM-788"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Morocco played 7 matches at the FIFA World Cup 2022.",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "Morocco played 7 matches at the FIFA World Cup 2022.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Morocco's most common formations were 433 (10 records), 343 (7 records), and 4141 (4 records).",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 433 (10 formation records), 343 (7 formation records), and 4141 (4 formation records).",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Morocco had 38.8% possession as an event-share proxy, not StatsBomb broadcast possession.",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "they were the team in possession for 38.8% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Of Morocco's 2865 passes, 80.4% were standard open-play passes and 19.6% came from set pieces or restarts.",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "Of 2865 passes, 80.4% were standard open-play passes and 19.6% came from set pieces or restarts",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Morocco's possession figure of 38.8% is the official StatsBomb broadcast possession.",
                "reason": (
                    "The document explicitly states this is an event-share proxy, "
                    "not StatsBomb's broadcast possession figure. Treating it as "
                    "official broadcast possession misrepresents the data limitation."
                ),
            },
            {
                "claim": "Morocco played exclusively in a 4-3-3 formation.",
                "reason": "The document lists three common formations (433, 343, 4141), not a single fixed shape.",
            },
        ],
        "notes": (
            "Tests retrieval of the Team tournament document for Morocco. "
            "Morocco reached the Semi-finals, making this an interesting tactical case. "
            "The possession limitation (event-share proxy) must be preserved."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 12 — Multi-level Expanded: Argentina vs Croatia Semi-final
    # -----------------------------------------------------------------------
    {
        "id": "gt-multi-02",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "multi",
        "query": "How did Argentina beat Croatia in the semi-final, and how did Messi perform?",
        "expected_route": "hybrid",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["1", "2", "3"],
        "relevant_document_ids": [
            "L1-match-3869519",
            "L2-match-3869519",
            "L3-match-3869519-player-5503",
        ],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Argentina beat Croatia 3-0 in the Semi-finals on 2022-12-13 at Lusail Stadium.",
                "source_document_ids": ["L1-match-3869519"],
                "evidence_snippets": [
                    "Argentina beat Croatia 3-0 in normal time.",
                    "was played on 2022-12-13 at Lusail Stadium.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Possession was split Croatia 62.0% and Argentina 38.0% as an event-share proxy.",
                "source_document_ids": ["L1-match-3869519"],
                "evidence_snippets": [
                    "Possession was split Croatia 62.0% and Argentina 38.0%. This is an event-share proxy based on passes, carries, dribbles and shots, not StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Messi scored a penalty in the 33rd minute and Álvarez scored twice (38th and 68th minutes).",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Goal: Lionel Andrés Messi Cuccittini (Argentina) in period 1, 33rd minute, with the left foot, shot type Penalty",
                    "Goal: Julián Álvarez (Argentina) in period 1, 38th minute, with the right foot, shot type Open Play",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Messi provided the assist for Álvarez's second goal in the 68th minute.",
                "source_document_ids": ["L2-match-3869519"],
                "evidence_snippets": [
                    "Goal: Julián Álvarez (Argentina) in period 2, 68th minute, with the right foot, shot type Open Play, xG 0.33. Lionel Andrés Messi Cuccittini provided the assist.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Messi scored 1 goal from 2 shots worth 0.89 expected goals and provided 1 assist.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "He took 2 shots worth 0.89 expected goals, 2 from inside the penalty area and 0 from outside, and scored 1.",
                    "He provided 1 assist.",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "Messi attempted 41 passes at 82.9% completion and played 95 minutes.",
                "source_document_ids": ["L3-match-3869519-player-5503"],
                "evidence_snippets": [
                    "He attempted 41 passes and completed 82.9%",
                    "was on the pitch for 95 minutes.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Croatia won the Semi-final.",
                "reason": "Argentina won 3-0; claiming Croatia won contradicts the L1 source.",
            },
            {
                "claim": "The Semi-final went to extra time.",
                "reason": "The match ended 3-0 in normal time; no extra time was played.",
            },
            {
                "claim": "Messi scored a hat-trick against Croatia.",
                "reason": "The L3 document records 1 goal for Messi, not 3. Álvarez scored twice.",
            },
        ],
        "notes": (
            "Tests multi-level retrieval for the Argentina vs Croatia Semi-finals. "
            "L1 provides match-level result and possession, L2 provides goals and events, "
            "L3 provides Messi's individual match statistics. "
            "All three documents share match 3869519."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 13 — L1 Advanced: Knockout-Stage Penalty Shootout Matches
    # -----------------------------------------------------------------------
    {
        "id": "gt-l1-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l1",
        "query": "Which knockout-stage matches were decided by penalty shootouts?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "1",
        "acceptable_levels": ["1"],
        "relevant_document_ids": [
            "L1-match-3869219",
            "L1-match-3869220",
            "L1-match-3869321",
            "L1-match-3869420",
            "L1-match-3869685",
        ],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Japan vs Croatia (Round of 16) finished 1-1 after extra time and Croatia won the penalty shootout 3-1.",
                "source_document_ids": ["L1-match-3869219"],
                "evidence_snippets": [
                    "The match finished level at 1-1 after extra time.",
                    "Croatia won the penalty shootout 3-1 against Japan.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Morocco vs Spain (Round of 16) finished 0-0 after extra time and Morocco won the penalty shootout 3-0.",
                "source_document_ids": ["L1-match-3869220"],
                "evidence_snippets": [
                    "The match finished level at 0-0 after extra time.",
                    "Morocco won the penalty shootout 3-0 against Spain.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Netherlands vs Argentina (Quarter-finals) finished 2-2 after extra time and Argentina won the penalty shootout 4-3.",
                "source_document_ids": ["L1-match-3869321"],
                "evidence_snippets": [
                    "The match finished level at 2-2 after extra time.",
                    "Argentina won the penalty shootout 4-3 against Netherlands.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Croatia vs Brazil (Quarter-finals) finished 1-1 after extra time and Croatia won the penalty shootout 4-2.",
                "source_document_ids": ["L1-match-3869420"],
                "evidence_snippets": [
                    "The match finished level at 1-1 after extra time.",
                    "Croatia won the penalty shootout 4-2 against Brazil.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Argentina vs France (Final) finished 3-3 after extra time and Argentina won the penalty shootout 4-2.",
                "source_document_ids": ["L1-match-3869685"],
                "evidence_snippets": [
                    "The match finished level at 3-3 after extra time.",
                    "Argentina won the penalty shootout 4-2 against France.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "England vs France (Quarter-finals) was decided by a penalty shootout.",
                "reason": "France beat England 2-1 in normal time; no penalty shootout occurred.",
            },
            {
                "claim": "Japan won the penalty shootout against Croatia.",
                "reason": "Croatia won the shootout 3-1 against Japan.",
            },
            {
                "claim": "The Final score was 4-2, combining the shootout result with the match score.",
                "reason": "The match finished 3-3 after extra time; the 4-2 is only the shootout result and must not be conflated with the match score.",
            },
        ],
        "notes": (
            "Tests complete multi-document recall across all knockout-stage penalty shootout matches. "
            "Five L1 documents are required: Japan vs Croatia R16, Morocco vs Spain R16, "
            "Netherlands vs Argentina QF, Croatia vs Brazil QF, and Argentina vs France Final. "
            "Non-shootout knockout matches (e.g. England vs France QF) must not be included."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 14 — L2 Advanced: Morocco vs Portugal Quarter-final Second Half
    # -----------------------------------------------------------------------
    {
        "id": "gt-l2-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l2",
        "query": "What happened in the second half of the Morocco vs Portugal quarter-final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["2"],
        "relevant_document_ids": ["L2-match-3869486"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "No goals were scored in the second half; the only goal was En-Nesyri's 41st-minute header in the first half.",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "Goal: Youssef En-Nesyri (Morocco) in period 1, 41st minute, with the head, shot type Open Play, xG 0.2",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Portugal brought on Cristiano Ronaldo and Cancelo in the 50th minute as second-half tactical substitutions.",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "Substitution: Portugal brought on João Pedro Cavaco Cancelo for Raphaël Adelino José Guerreiro in the 50th minute. The substitution was tactical.",
                    "Substitution: Portugal brought on Cristiano Ronaldo dos Santos Aveiro for Rúben Diogo Da Silva Neves in the 50th minute. The substitution was tactical.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Morocco brought on Achraf Dari for Romain Saïss in the 56th minute due to injury.",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "Substitution: Morocco brought on Achraf Dari for Romain Saïss in the 56th minute. The substitution was injury.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Morocco made further tactical substitutions in the 63rd, 64th, and 80th minutes.",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "Substitution: Morocco brought on Walid Cheddira for Selim Amallah in the 63rd minute. The substitution was tactical.",
                    "Substitution: Morocco brought on Badr Banoun for Youssef En-Nesyri in the 64th minute. The substitution was tactical.",
                    "Substitution: Morocco brought on Zakaria Aboukhlal for Hakim Ziyech in the 80th minute. The substitution was tactical.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Zakaria Aboukhlal missed a high-quality chance in the 95th minute (0.39 xG, saved).",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "High-quality chance missed: Zakaria Aboukhlal (Morocco) in the 95th minute recorded 0.39 xG but the shot ended saved",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Morocco scored a second goal in the second half.",
                "reason": "The L2 document records only one goal (En-Nesyri, 41st minute, first half).",
            },
            {
                "claim": "Cristiano Ronaldo scored in the second half.",
                "reason": "The L2 document records no goal by Ronaldo; he was substituted on in the 50th minute.",
            },
        ],
        "notes": (
            "Tests period-specific retrieval from the L2 key-events document for "
            "Morocco vs Portugal Quarter-finals (match 3869486). "
            "Second-half evidence includes substitutions (50th-80th minutes) and "
            "a high-quality chance missed (95th minute, second-half stoppage time). "
            "The L2 document uses period 1 for the goal; substitutions are identified by minute."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 15 — L3 Advanced: Mbappé vs Poland Round of 16
    # -----------------------------------------------------------------------
    {
        "id": "gt-l3-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l3",
        "query": "How did Kylian Mbappé perform against Poland in the Round of 16?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "3",
        "acceptable_levels": ["3"],
        "relevant_document_ids": ["L3-match-3869152-player-3009"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Mbappé played 99.3 minutes against Poland in the Round of 16.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "was on the pitch for 99.3 minutes.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Mbappé scored 2 goals from 5 shots worth 0.26 expected goals.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "He took 5 shots worth 0.26 expected goals, 5 from inside the penalty area and 0 from outside, and scored 2.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Mbappé provided 1 assist in the match.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "He provided 1 assist.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Mbappé attempted 41 passes and completed 82.9%, with 28 delivered into the final third.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "He attempted 41 passes and completed 82.9%, of which 28 were delivered into the final third.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Mbappé made 7 pressures and 57 carries during the match.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "Kylian Mbappé Lottin of France played left wing and center forward, changing role once during the match against Poland in the Round of 16 on 2022-12-04",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "Mbappé played as left wing and center forward, changing role once during the match.",
                "source_document_ids": ["L3-match-3869152-player-3009"],
                "evidence_snippets": [
                    "played left wing and center forward, changing role once during the match against Poland in the Round of 16",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Mbappé scored 3 goals against Poland.",
                "reason": "The L3 document records 2 goals, not 3.",
            },
            {
                "claim": "Mbappé had 0.5 expected goals against Poland.",
                "reason": "The L3 document records 0.26 expected goals, not 0.5.",
            },
            {
                "claim": "Mbappé was awarded Man of the Match against Poland.",
                "reason": "The L3 document does not mention any awards; this is unsupported by the source.",
            },
        ],
        "notes": (
            "Tests retrieval of the exact L3 player-match document for Mbappé against Poland "
            "in the Round of 16 (match 3869152, player_id 3009). "
            "The document is uniquely identified by player + match combination. "
            "Covers minutes, goals, assists, shots, xG, passes, final-third passes, "
            "pressures, carries, and positional roles."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 16 — L4 Advanced: Griezmann Tournament Summary
    # -----------------------------------------------------------------------
    {
        "id": "gt-l4-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l4",
        "query": "Describe Antoine Griezmann's overall World Cup tournament performance.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "4",
        "acceptable_levels": ["4"],
        "relevant_document_ids": ["L4-player-5487"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Griezmann appeared in 7 matches and played 586.1 minutes in total.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "appeared in 7 matches at the FIFA World Cup 2022, playing 586.1 minutes in total.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Griezmann scored 0 goals and provided 3 assists from 0.5 expected goals.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "he scored 0 goals and provided 3 assists from 0.5 expected goals",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Griezmann's goal contribution rate was 0.43 per match.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "a goal contribution rate of 0.43 per match.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Griezmann averaged 46.1 passes attempted at 78.1% completion.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "with 46.1 passes attempted at 78.1% completion.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Griezmann averaged 0.86 shot and 0.07 expected goals per match.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "He averaged 0.86 shot and 0.07 expected goals per match",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "Griezmann's average defensive workload was 17.9 pressures, 0.86 successful tackle, 1.29 successful interception and 1.14 clearance per match.",
                "source_document_ids": ["L4-player-5487"],
                "evidence_snippets": [
                    "His average defensive workload was 17.9 pressures, 0.86 successful tackle, 1.29 successful interception and 1.14 clearance per match.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Griezmann scored 4 goals in the tournament.",
                "reason": "The L4 document records 0 goals, not 4.",
            },
            {
                "claim": "Griezmann provided 6 assists in the tournament.",
                "reason": "The L4 document records 3 assists, not 6.",
            },
            {
                "claim": "Griezmann played for Paris Saint-Germain at the World Cup.",
                "reason": "The L4 document identifies his team as France; club affiliation is not stated.",
            },
        ],
        "notes": (
            "Tests retrieval of the L4 tournament-summary document for Griezmann. "
            "Griezmann's player_id is 5487. The document covers goals, assists, xG, "
            "passing, shooting, defensive workload, and per-match averages. "
            "Unlike the Mbappé and Messi L4 cases, Griezmann scored 0 goals."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 17 — Team Advanced: France Playing Style and Formations
    # -----------------------------------------------------------------------
    {
        "id": "gt-team-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "team",
        "query": "What were France's passing patterns and most common formations?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "team",
        "acceptable_levels": ["team"],
        "relevant_document_ids": ["TEAM-771"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "France's most common formations were 4231 (10 formation records), 433 (4 formation records), and 442 (2 formation records).",
                "source_document_ids": ["TEAM-771"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 4231 (10 formation records), 433 (4 formation records), and 442 (2 formation records).",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "France's play patterns were dominated by regular play 38.8%, throw-ins 24.2%, free kicks 16.1%, and goal kicks 10.1%.",
                "source_document_ids": ["TEAM-771"],
                "evidence_snippets": [
                    "Their play patterns were dominated by regular play 38.8%, from throw in 24.2%, from free kick 16.1%, and from goal kick 10.1%.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Of France's 3926 passes, 83.2% were standard open-play passes and 16.8% came from set pieces or restarts.",
                "source_document_ids": ["TEAM-771"],
                "evidence_snippets": [
                    "Of 3926 passes, 83.2% were standard open-play passes and 16.8% came from set pieces or restarts",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "France delivered 94 crosses, 2.4% of their passes.",
                "source_document_ids": ["TEAM-771"],
                "evidence_snippets": [
                    "They delivered 94 crosses, 2.4% of their passes.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "France had 51.1% possession as an event-share proxy, not StatsBomb broadcast possession.",
                "source_document_ids": ["TEAM-771"],
                "evidence_snippets": [
                    "they were the team in possession for 51.1% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "France played exclusively in a 4-3-3 formation throughout the tournament.",
                "reason": "The document lists three common formations (4231, 433, 442), with 4231 being the most frequent.",
            },
            {
                "claim": "France completed 5000 passes in the tournament.",
                "reason": "The document records 3926 passes, not 5000.",
            },
            {
                "claim": "France's possession figure of 51.1% is the official StatsBomb broadcast possession.",
                "reason": (
                    "The document explicitly states this is an event-share proxy, "
                    "not StatsBomb's broadcast possession figure."
                ),
            },
        ],
        "notes": (
            "Tests retrieval of the Team tournament document for France (TEAM-771). "
            "Covers formations (4231, 433, 442), play patterns, pass types, "
            "crosses, and possession proxy limitation."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 18 — Multi-level Advanced: Morocco's Route to the Semi-finals
    # -----------------------------------------------------------------------
    {
        "id": "gt-multi-03",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "multi",
        "query": "How did Morocco reach the semi-finals, and what style did they use?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "1",
        "acceptable_levels": ["1", "2", "team"],
        "relevant_document_ids": [
            "L1-match-3869220",
            "L2-match-3869220",
            "L1-match-3869486",
            "L2-match-3869486",
            "TEAM-788",
        ],
        "optional_relevant_document_ids": [],
        "required_facts": [
            # --- Round of 16: Morocco vs Spain ---
            {
                "fact_id": "f1",
                "claim": "Morocco drew 0-0 with Spain after extra time in the Round of 16.",
                "source_document_ids": ["L1-match-3869220"],
                "evidence_snippets": [
                    "The match finished level at 0-0 after extra time.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Morocco won the penalty shootout 3-0 against Spain to advance to the Quarter-finals.",
                "source_document_ids": ["L1-match-3869220"],
                "evidence_snippets": [
                    "Morocco won the penalty shootout 3-0 against Spain.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "In the Round of 16 second half, Morocco brought on Ezzalzouli for Boufal in the 65th minute.",
                "source_document_ids": ["L2-match-3869220"],
                "evidence_snippets": [
                    "Substitution: Morocco brought on Abdessamad Ezzalzouli for Sofiane Boufal in the 65th minute. The substitution was tactical.",
                ],
            },
            # --- Quarter-final: Morocco vs Portugal ---
            {
                "fact_id": "f4",
                "claim": "Morocco beat Portugal 1-0 in normal time in the Quarter-finals.",
                "source_document_ids": ["L1-match-3869486"],
                "evidence_snippets": [
                    "Morocco beat Portugal 1-0 in normal time.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Youssef En-Nesyri scored Morocco's winner against Portugal in the 41st minute with a header.",
                "source_document_ids": ["L1-match-3869486"],
                "evidence_snippets": [
                    "Youssef En-Nesyri scored for Morocco in the 41st minute with the head, from open play, assisted by Yahia Attiyat allah (xG 0.2).",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "In the Quarter-final second half, Morocco brought on Achraf Dari for Romain Saïss in the 56th minute due to injury.",
                "source_document_ids": ["L2-match-3869486"],
                "evidence_snippets": [
                    "Substitution: Morocco brought on Achraf Dari for Romain Saïss in the 56th minute. The substitution was injury.",
                ],
            },
            # --- Team style ---
            {
                "fact_id": "f7",
                "claim": "Morocco's most common formations were 433 (10 records), 343 (7 records), and 4141 (4 records).",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 433 (10 formation records), 343 (7 formation records), and 4141 (4 formation records).",
                ],
            },
            {
                "fact_id": "f8",
                "claim": "Morocco had 38.8% possession as an event-share proxy, not StatsBomb broadcast possession.",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "they were the team in possession for 38.8% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f9",
                "claim": "Morocco's play patterns were dominated by regular play 41.2%, throw-ins 23.7%, and free kicks 16.8%.",
                "source_document_ids": ["TEAM-788"],
                "evidence_snippets": [
                    "Their play patterns were dominated by regular play 41.2%, from throw in 23.7%, from free kick 16.8%, and from goal kick 11.0%.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Morocco beat Spain in normal time in the Round of 16.",
                "reason": "The match finished 0-0 after extra time; Morocco won via penalty shootout, not normal time.",
            },
            {
                "claim": "Morocco beat Portugal via a penalty shootout.",
                "reason": "Morocco won 1-0 in normal time; no penalty shootout occurred in the Quarter-final.",
            },
            {
                "claim": "Morocco's possession figure of 38.8% is the official StatsBomb broadcast possession.",
                "reason": "The document explicitly states this is an event-share proxy.",
            },
            {
                "claim": "Morocco scored 3 goals against Portugal.",
                "reason": "Morocco scored 1 goal (En-Nesyri, 41st minute) and won 1-0.",
            },
        ],
        "notes": (
            "Tests multi-document recall, multi-level coverage, chronology, evidence "
            "separation between two matches, and Team-style grounding. "
            "Morocco's route: Round of 16 (vs Spain, 0-0, won shootout 3-0) then "
            "Quarter-finals (vs Portugal, 1-0 in normal time). "
            "L1 and L2 documents for each match share the same match ID. "
            "The Team document provides tournament-wide style context. "
            "Shootout and normal-score facts are not conflated."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 19 — L1 Extended: England vs France Quarter-final
    # -----------------------------------------------------------------------
    {
        "id": "gt-l1-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l1",
        "query": "Describe the quarter-final between England and France.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "1",
        "acceptable_levels": ["1"],
        "relevant_document_ids": ["L1-match-3869354"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "France beat England 2-1 in the Quarter-finals on 2022-12-10 at Al Bayt Stadium.",
                "source_document_ids": ["L1-match-3869354"],
                "evidence_snippets": [
                    "The FIFA World Cup 2022 Quarter-finals between England and France was played on 2022-12-10 at Al Bayt Stadium.",
                    "France beat England 2-1 in normal time.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Tchouaméni scored for France in the 16th minute from open play, assisted by Griezmann.",
                "source_document_ids": ["L1-match-3869354"],
                "evidence_snippets": [
                    "Aurélien Djani Tchouaméni scored for France in the 16th minute with the right foot, from open play, assisted by Antoine Griezmann (xG 0.03).",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Harry Kane scored for England from a penalty in the 53rd minute.",
                "source_document_ids": ["L1-match-3869354"],
                "evidence_snippets": [
                    "Harry Kane scored for England in the 53rd minute with the right foot, from a penalty (xG 0.78).",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Olivier Giroud scored the winner for France in the 77th minute with a header, assisted by Griezmann.",
                "source_document_ids": ["L1-match-3869354"],
                "evidence_snippets": [
                    "Olivier Giroud scored for France in the 77th minute with the head, from open play, assisted by Antoine Griezmann (xG 0.15).",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "The referee showed 4 yellow cards: Griezmann (42nd), Dembélé (45th), and Hernández (79th) for France, and Maguire (88th) for England.",
                "source_document_ids": ["L1-match-3869354"],
                "evidence_snippets": [
                    "Antoine Griezmann (France) received a yellow card in the 42nd minute, Ousmane Dembélé (France) received a yellow card in the 45th minute, Theo Bernard François Hernández (France) received a yellow card in the 79th minute, and Harry Maguire (England) received a yellow card in the 88th minute.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "The match ended in a draw or went to extra time.",
                "reason": "France won 2-1 in normal time; the match did not go to extra time.",
            },
            {
                "claim": "Kane scored from open play.",
                "reason": "Kane's goal was from a penalty, not open play.",
            },
            {
                "claim": "England won the quarter-final.",
                "reason": "France won 2-1; claiming England won contradicts the source.",
            },
        ],
        "notes": (
            "Tests retrieval of a knockout-stage match decided in normal time. "
            "The England vs France Quarter-final (match_id 3869354) features 3 goals, "
            "4 yellow cards, and possession data. "
            "Unlike gt-l1-03 (shootout matches), this match ended without extra time."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 20 — L2 Extended: England vs France Quarter-final Key Events
    # -----------------------------------------------------------------------
    {
        "id": "gt-l2-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l2",
        "query": "What were the key turning points in the England vs France quarter-final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "2",
        "acceptable_levels": ["2"],
        "relevant_document_ids": ["L2-match-3869354"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Tchouaméni opened the scoring for France in the 16th minute from open play, assisted by Griezmann.",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "Goal: Aurélien Djani Tchouaméni (France) in period 1, 16th minute, with the right foot, shot type Open Play, xG 0.03. Antoine Griezmann provided the assist.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Kane equalized for England from a penalty in the 53rd minute.",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "Goal: Harry Kane (England) in period 2, 53rd minute, with the right foot, shot type Penalty, xG 0.78.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Giroud restored France's lead in the 77th minute with a header, assisted by Griezmann.",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "Goal: Olivier Giroud (France) in period 2, 77th minute, with the head, shot type Open Play, xG 0.15. Antoine Griezmann provided the assist.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Kane missed a high-quality chance to equalize in the 83rd minute (0.78 xG).",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "High-quality chance missed: Harry Kane (England) in the 83rd minute recorded 0.78 xG but the shot ended off t (threshold for a high-quality chance is 0.3 xG).",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "France brought on Kingsley Coman for Dembélé in the 78th minute as a tactical substitution.",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "Substitution: France brought on Kingsley Coman for Ousmane Dembélé in the 78th minute. The substitution was tactical.",
                ],
            },
            {
                "fact_id": "f6",
                "claim": "England made a late injury substitution, bringing on Grealish for Stones in the 97th minute.",
                "source_document_ids": ["L2-match-3869354"],
                "evidence_snippets": [
                    "Substitution: England brought on Jack Grealish for John Stones in the 97th minute. The substitution was injury.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Kane scored the penalty to give England the lead.",
                "reason": "Kane's penalty equalized at 1-1; it did not give England the lead. France had scored first.",
            },
            {
                "claim": "The high-quality chance missed by Kane resulted in a goal.",
                "reason": "The shot ended off target; Kane did not score from this chance.",
            },
            {
                "claim": "All substitutions in the match were tactical.",
                "reason": "The Grealish-for-Stones substitution in the 97th minute was injury, not tactical.",
            },
        ],
        "notes": (
            "Tests retrieval of key turning points and decision-related events from an L2 document. "
            "Covers penalty decisions (equalizer), high-quality chance missed, and mixed substitution types "
            "(tactical vs injury). The England vs France QF (match_id 3869354) is not referenced by any existing case. "
            "Note: L2 documents do not contain card/disciplinary events; cards are in L1 documents only. "
            "Corpus note: The source text contains truncated 'off t' (should be 'off target'). "
            "This is a known corpus-level truncation preserved verbatim."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 21 — L3 Extended: Enzo Fernández Defensive Performance in Final
    # -----------------------------------------------------------------------
    {
        "id": "gt-l3-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l3",
        "query": "How did Enzo Fernández perform defensively in the World Cup Final?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "3",
        "acceptable_levels": ["3"],
        "relevant_document_ids": ["L3-match-3869685-player-38718"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Enzo Fernández played as center defensive midfield and right center midfield, changing role once, and was on the pitch for 124.1 minutes in the Final against France.",
                "source_document_ids": ["L3-match-3869685-player-38718"],
                "evidence_snippets": [
                    "Enzo Fernandez of Argentina played center defensive midfield and right center midfield, changing role once during the match against France in the Final on 2022-12-18, and was on the pitch for 124.1 minutes.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "He attempted 94 passes and completed 84.0%, with 23 delivered into the final third.",
                "source_document_ids": ["L3-match-3869685-player-38718"],
                "evidence_snippets": [
                    "He attempted 94 passes and completed 84.0%, of which 23 were delivered into the final third.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "He applied pressure to opponents 30 times during the match.",
                "source_document_ids": ["L3-match-3869685-player-38718"],
                "evidence_snippets": [
                    "He applied pressure to opponents 30 times.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Defensively he recorded 5 successful tackles, 1 successful interception, and 4 clearances.",
                "source_document_ids": ["L3-match-3869685-player-38718"],
                "evidence_snippets": [
                    "Defensively he recorded 5 successful tackles, 1 successful interception, and 4 clearances.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Enzo Fernández scored in the Final.",
                "reason": "The L3 document records 0 goals scored by Fernández in this match.",
            },
            {
                "claim": "Enzo Fernández played as a center back throughout the match.",
                "reason": "He played center defensive midfield and right center midfield, not center back.",
            },
            {
                "claim": "These are Enzo Fernández's tournament averages.",
                "reason": "These are his single-match statistics for the Final, not tournament averages. Tournament averages are in the L4 document.",
            },
        ],
        "notes": (
            "Tests retrieval of a defensive player's single-match performance from an L3 document. "
            "Enzo Fernández (player_id 38718) played as a defensive midfielder in the Final with high defensive output: "
            "5 tackles, 1 interception, 4 clearances, and 30 pressures. "
            "Unlike existing L3 cases (Messi, Griezmann, Mbappé), this case tests a primarily defensive role."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 22 — L4 Extended: Enzo Fernández Tournament Summary
    # -----------------------------------------------------------------------
    {
        "id": "gt-l4-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "l4",
        "query": "Describe Enzo Fernández's overall World Cup tournament performance.",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "4",
        "acceptable_levels": ["4"],
        "relevant_document_ids": ["L4-player-38718"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Enzo Fernández appeared in 7 matches and played 601.1 minutes in total.",
                "source_document_ids": ["L4-player-38718"],
                "evidence_snippets": [
                    "Enzo Fernandez of Argentina appeared in 7 matches at the FIFA World Cup 2022, playing 601.1 minutes in total.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "He scored 1 goal and provided 1 assist from 0.27 expected goals, a goal contribution rate of 0.29 per match.",
                "source_document_ids": ["L4-player-38718"],
                "evidence_snippets": [
                    "Across the tournament he scored 1 goal and provided 1 assist from 0.27 expected goals, a goal contribution rate of 0.29 per match.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "He averaged 67.1 passes attempted at 87.5% completion.",
                "source_document_ids": ["L4-player-38718"],
                "evidence_snippets": [
                    "He averaged 1.14 shot and 0.04 expected goals per match, with 67.1 passes attempted at 87.5% completion.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "His average defensive workload was 15.3 pressures, 1.86 successful tackles, 0.57 successful interceptions, and 1.86 clearances per match.",
                "source_document_ids": ["L4-player-38718"],
                "evidence_snippets": [
                    "His average defensive workload was 15.3 pressures, 1.86 successful tackles, 0.57 successful interception and 1.86 clearances per match.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "He played 3 group-stage matches and 4 knockout matches.",
                "source_document_ids": ["L4-player-38718"],
                "evidence_snippets": [
                    "He played 3 group-stage matches and 4 knockout matches",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Enzo Fernández scored 5 goals in the tournament.",
                "reason": "The L4 document records 1 goal, not 5.",
            },
            {
                "claim": "His defensive tackle average of 1.86 per match was from a single match.",
                "reason": "1.86 tackles per match is a tournament average across 7 matches, not a single-match figure.",
            },
            {
                "claim": "Enzo Fernández played for France at the World Cup.",
                "reason": "The L4 document identifies his team as Argentina.",
            },
        ],
        "notes": (
            "Tests retrieval of the L4 tournament-summary document for a defensive midfielder. "
            "Enzo Fernández (player_id 38718) has a clear defensive workload profile: 15.3 pressures, "
            "1.86 tackles, 0.57 interceptions, 1.86 clearances per match. "
            "Unlike existing L4 cases (Messi, Mbappé, Griezmann), this case tests a primarily defensive player."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 23 — Team Extended: Germany Group-Stage Elimination
    # -----------------------------------------------------------------------
    {
        "id": "gt-team-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "team",
        "query": "How did Germany play in the tournament, and which formations did they use?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "team",
        "acceptable_levels": ["team"],
        "relevant_document_ids": ["TEAM-770"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Germany played 3 matches at the FIFA World Cup 2022.",
                "source_document_ids": ["TEAM-770"],
                "evidence_snippets": [
                    "Germany played 3 matches at the FIFA World Cup 2022.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Germany had 60.0% possession as an event-share proxy, not StatsBomb broadcast possession.",
                "source_document_ids": ["TEAM-770"],
                "evidence_snippets": [
                    "Across their matches they were the team in possession for 60.0% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Germany's most common formations were 4231 (6 formation records), 442 (2 formation records), and 352 (1 formation record).",
                "source_document_ids": ["TEAM-770"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 4231 (6 formation records), 442 (2 formation records), and 352 (1 formation record).",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "Germany's play patterns were dominated by regular play 43.3%, throw-ins 26.7%, free kicks 13.3%, and goal kicks 6.9%.",
                "source_document_ids": ["TEAM-770"],
                "evidence_snippets": [
                    "Their play patterns were dominated by regular play 43.3%, from throw in 26.7%, from free kick 13.3%, and from goal kick 6.9%.",
                ],
            },
            {
                "fact_id": "f5",
                "claim": "Of Germany's 1963 passes, 84.3% were standard open-play passes and 15.7% came from set pieces or restarts.",
                "source_document_ids": ["TEAM-770"],
                "evidence_snippets": [
                    "Of 1963 passes, 84.3% were standard open-play passes and 15.7% came from set pieces or restarts",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Germany's possession figure of 60.0% is the official StatsBomb broadcast possession.",
                "reason": (
                    "The document explicitly states this is an event-share proxy, "
                    "not StatsBomb's broadcast possession figure. Treating it as "
                    "official broadcast possession misrepresents the data limitation."
                ),
            },
            {
                "claim": "Germany played 7 matches in the tournament.",
                "reason": "Germany played 3 matches and was eliminated in the group stage.",
            },
            {
                "claim": "Germany played exclusively in a 4-2-3-1 formation.",
                "reason": "The document lists three formations (4231, 442, 352), not a single fixed shape.",
            },
        ],
        "notes": (
            "Tests retrieval of a group-stage eliminated team's tournament document. "
            "Germany (TEAM-770) played only 3 matches, unlike Argentina, France, and Morocco in existing cases "
            "which all played 7 matches. The 60.0% possession proxy limitation must be preserved."
        ),
    },
    # -----------------------------------------------------------------------
    # CASE 24 — Multi Extended: Argentina vs France Playing Style Comparison
    # -----------------------------------------------------------------------
    {
        "id": "gt-multi-04",
        "dataset_id": "statsbomb-fifa-world-cup-2022",
        "case_group": "multi",
        "query": "How did Argentina and France differ in their playing styles at the World Cup?",
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": "team",
        "acceptable_levels": ["team"],
        "relevant_document_ids": ["TEAM-779", "TEAM-771"],
        "optional_relevant_document_ids": [],
        "required_facts": [
            {
                "fact_id": "f1",
                "claim": "Argentina had 58.6% possession as an event-share proxy, higher than France's 51.1%.",
                "source_document_ids": ["TEAM-779", "TEAM-771"],
                "evidence_snippets": [
                    "Across their matches they were the team in possession for 58.6% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                    "Across their matches they were the team in possession for 51.1% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            },
            {
                "fact_id": "f2",
                "claim": "Argentina's most common formations were 352, 433, and 442, while France's were 4231, 433, and 442.",
                "source_document_ids": ["TEAM-779", "TEAM-771"],
                "evidence_snippets": [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 352 (8 formation records), 433 (8 formation records), and 442 (5 formation records).",
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 4231 (10 formation records), 433 (4 formation records), and 442 (2 formation records).",
                ],
            },
            {
                "fact_id": "f3",
                "claim": "Argentina's play patterns included 40.0% regular play and 20.3% free kicks, while France's included 38.8% regular play and 10.1% goal kicks.",
                "source_document_ids": ["TEAM-779", "TEAM-771"],
                "evidence_snippets": [
                    "Their play patterns were dominated by regular play 40.0%, from throw in 23.3%, from free kick 20.3%, and from goal kick 5.5%.",
                    "Their play patterns were dominated by regular play 38.8%, from throw in 24.2%, from free kick 16.1%, and from goal kick 10.1%.",
                ],
            },
            {
                "fact_id": "f4",
                "claim": "France delivered more crosses (94, 2.4% of passes) than Argentina (82, 1.8% of passes).",
                "source_document_ids": ["TEAM-779", "TEAM-771"],
                "evidence_snippets": [
                    "They delivered 94 crosses, 2.4% of their passes.",
                    "They delivered 82 crosses, 1.8% of their passes.",
                ],
            },
        ],
        "forbidden_claims": [
            {
                "claim": "Argentina's possession figure of 58.6% is the official StatsBomb broadcast possession.",
                "reason": (
                    "Both documents explicitly state these are event-share proxies, "
                    "not StatsBomb's broadcast possession figures."
                ),
            },
            {
                "claim": "France had higher possession than Argentina.",
                "reason": "Argentina had 58.6% possession vs France's 51.1%; claiming France had higher possession contradicts the sources.",
            },
            {
                "claim": "Argentina and France used the same primary formation.",
                "reason": "Argentina's most-used formation was 352 (8 records) while France's was 4231 (10 records); they had different primary formations.",
            },
            {
                "claim": "France's higher cross count proves they played more attacking football.",
                "reason": "Cross count alone does not prove a qualitative tactical conclusion; the documents report statistics, not tactical judgments.",
            },
        ],
        "notes": (
            "Tests cross-team comparison requiring two TEAM documents from different teams. "
            "Neither TEAM-779 (Argentina) nor TEAM-771 (France) alone can answer the comparison query. "
            "Both documents are needed to compare possession, formations, play patterns, and crossing. "
            "Both possession figures must be labeled as event-share proxies. "
            "Unlike existing multi cases (which combine L1+L2+L3 or span multiple matches), "
            "this case tests same-level cross-team document retrieval."
        ),
    },
]


# ---------------------------------------------------------------------------
# H. Validation Functions
# ---------------------------------------------------------------------------


def load_chunks(chunks_path: str | Path) -> list[dict]:
    """Load chunks.json from disk."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_chunks_by_document_id(chunks: list[dict]) -> dict[str, list[dict]]:
    """Index chunks by their document_id."""
    index: dict[str, list[dict]] = {}
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if doc_id:
            index.setdefault(doc_id, []).append(chunk)
    return index


def _compute_sha256(chunks_path: str | Path) -> str:
    """Compute SHA-256 of the chunks file."""
    data = Path(chunks_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_metadata(
    metadata: dict,
    chunks_path: str | Path,
) -> list[str]:
    """Validate dataset metadata against the chunks snapshot."""
    errors: list[str] = []

    # Schema version
    sv = metadata.get("schema_version")
    if sv != SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION:
        errors.append(
            f"schema_version is '{sv}', expected '{SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION}'"
        )

    # Dataset ID
    if not metadata.get("dataset_id"):
        errors.append("dataset_id is empty")

    # Chunks path
    cp = metadata.get("chunks_path", "")
    if not cp:
        errors.append("chunks_path is empty")
    elif Path(cp).is_absolute():
        errors.append(f"chunks_path is absolute: '{cp}'")

    # File existence
    path = Path(chunks_path)
    if not path.exists():
        errors.append(f"chunks file does not exist: {path}")
        return errors  # Can't check further

    # SHA-256
    actual_sha = _compute_sha256(chunks_path)
    expected_sha = metadata.get("chunks_sha256", "")
    if actual_sha != expected_sha:
        errors.append(
            f"chunks_sha256 mismatch: stored '{expected_sha}', actual '{actual_sha}'"
        )

    # Competition and season IDs
    chunks = load_chunks(chunks_path)
    comp_ids: set[int] = set()
    season_ids: set[int] = set()
    for c in chunks:
        m = c.get("metadata", {})
        if "competition_id" in m:
            comp_ids.add(m["competition_id"])
        if "season_id" in m:
            season_ids.add(m["season_id"])

    meta_comp = metadata.get("competition_id")
    meta_season = metadata.get("season_id")

    if len(comp_ids) > 1:
        errors.append(f"chunks contain multiple competition_ids: {sorted(comp_ids)}")
    if len(season_ids) > 1:
        errors.append(f"chunks contain multiple season_ids: {sorted(season_ids)}")

    if meta_comp not in comp_ids:
        errors.append(
            f"metadata competition_id {meta_comp} not found in chunks {sorted(comp_ids)}"
        )
    if meta_season not in season_ids:
        errors.append(
            f"metadata season_id {meta_season} not found in chunks {sorted(season_ids)}"
        )

    return errors


def validate_case_schema(case: dict) -> list[str]:
    """Validate the schema of a single ground-truth case."""
    errors: list[str] = []
    case_id = case.get("id", "<no id>")

    # Required fields
    required_fields = [
        "id", "dataset_id", "case_group", "query", "expected_route",
        "answerability", "primary_level", "acceptable_levels",
        "relevant_document_ids", "optional_relevant_document_ids",
        "required_facts", "forbidden_claims", "notes",
    ]
    for field in required_fields:
        if field not in case:
            errors.append(f"[{case_id}] missing required field: {field}")

    # String fields must not be blank
    for field in ["id", "dataset_id", "query", "expected_route", "answerability",
                  "primary_level", "notes"]:
        val = case.get(field, "")
        if isinstance(val, str) and not val.strip():
            errors.append(f"[{case_id}] field '{field}' is blank")

    # Allowed values
    if case.get("expected_route") not in ALLOWED_ROUTES:
        errors.append(
            f"[{case_id}] expected_route '{case.get('expected_route')}' not in {ALLOWED_ROUTES}"
        )
    if case.get("answerability") not in ALLOWED_ANSWERABILITY:
        errors.append(
            f"[{case_id}] answerability '{case.get('answerability')}' not in {ALLOWED_ANSWERABILITY}"
        )
    if case.get("primary_level") not in ALLOWED_LEVELS:
        errors.append(
            f"[{case_id}] primary_level '{case.get('primary_level')}' not in {ALLOWED_LEVELS}"
        )
    if case.get("case_group") not in ALLOWED_CASE_GROUPS:
        errors.append(
            f"[{case_id}] case_group '{case.get('case_group')}' not in {ALLOWED_CASE_GROUPS}"
        )

    # Levels
    acceptable = case.get("acceptable_levels", [])
    if not isinstance(acceptable, list) or len(acceptable) == 0:
        errors.append(f"[{case_id}] acceptable_levels is empty or not a list")
    else:
        if case.get("primary_level") not in acceptable:
            errors.append(
                f"[{case_id}] primary_level '{case.get('primary_level')}' "
                f"not in acceptable_levels {acceptable}"
            )
        if len(acceptable) != len(set(acceptable)):
            errors.append(f"[{case_id}] acceptable_levels contains duplicates")

    # Document ID fields
    rel_docs = case.get("relevant_document_ids", [])
    opt_docs = case.get("optional_relevant_document_ids", [])
    if not isinstance(rel_docs, list) or len(rel_docs) == 0:
        errors.append(f"[{case_id}] relevant_document_ids is empty or not a list")
    if not isinstance(opt_docs, list):
        errors.append(f"[{case_id}] optional_relevant_document_ids is not a list")

    # Check for chunk IDs
    all_doc_ids = (rel_docs or []) + (opt_docs or [])
    for doc_id in all_doc_ids:
        if "-chunk-" in doc_id:
            errors.append(f"[{case_id}] document ID appears to be a chunk ID: '{doc_id}'")

    # Overlap
    if rel_docs and opt_docs:
        overlap = set(rel_docs) & set(opt_docs)
        if overlap:
            errors.append(
                f"[{case_id}] relevant and optional document IDs overlap: {overlap}"
            )

    # Required facts
    facts = case.get("required_facts", [])
    if not isinstance(facts, list) or len(facts) == 0:
        errors.append(f"[{case_id}] required_facts is empty or not a list")
    else:
        fact_ids = [f.get("fact_id") for f in facts]
        if len(fact_ids) != len(set(fact_ids)):
            errors.append(f"[{case_id}] required_facts contains duplicate fact_ids")
        for fact in facts:
            fid = fact.get("fact_id", "?")
            if not fact.get("claim"):
                errors.append(f"[{case_id}][{fid}] claim is empty")
            src_ids = fact.get("source_document_ids", [])
            if not isinstance(src_ids, list) or len(src_ids) == 0:
                errors.append(f"[{case_id}][{fid}] source_document_ids is empty")
            snippets = fact.get("evidence_snippets", [])
            if not isinstance(snippets, list) or len(snippets) == 0:
                errors.append(f"[{case_id}][{fid}] evidence_snippets is empty")

    # Forbidden claims
    forbidden = case.get("forbidden_claims", [])
    if not isinstance(forbidden, list) or len(forbidden) == 0:
        errors.append(f"[{case_id}] forbidden_claims is empty or not a list")
    else:
        claims_text = [fc.get("claim", "") for fc in forbidden]
        if len(claims_text) != len(set(claims_text)):
            errors.append(f"[{case_id}] forbidden_claims contains duplicate claim text")
        for fc in forbidden:
            if not fc.get("claim"):
                errors.append(f"[{case_id}] forbidden claim has empty 'claim'")
            if not fc.get("reason"):
                errors.append(f"[{case_id}] forbidden claim has empty 'reason'")

    return errors


def validate_case_evidence(
    case: dict,
    chunks_by_document: dict[str, list[dict]],
) -> list[str]:
    """Validate that evidence snippets exist verbatim in source documents."""
    errors: list[str] = []
    case_id = case.get("id", "<no id>")

    rel_docs = case.get("relevant_document_ids", [])
    opt_docs = case.get("optional_relevant_document_ids", [])
    all_valid_docs = set(rel_docs or []) | set(opt_docs or [])

    # Check document existence
    for doc_id in (rel_docs or []):
        if doc_id not in chunks_by_document:
            errors.append(
                f"[{case_id}] relevant document_id '{doc_id}' not found in chunks"
            )
    for doc_id in (opt_docs or []):
        if doc_id not in chunks_by_document:
            errors.append(
                f"[{case_id}] optional document_id '{doc_id}' not found in chunks"
            )

    # Check facts
    for fact in case.get("required_facts", []):
        fid = fact.get("fact_id", "?")
        src_ids = fact.get("source_document_ids", [])

        # Source IDs must be in relevant or optional
        for src_id in src_ids:
            if src_id not in all_valid_docs:
                errors.append(
                    f"[{case_id}][{fid}] source_document_id '{src_id}' "
                    f"not in relevant or optional document IDs"
                )
            if src_id not in chunks_by_document:
                errors.append(
                    f"[{case_id}][{fid}] source_document_id '{src_id}' "
                    f"not found in chunks"
                )

        # Evidence snippets must exist verbatim
        for snippet in fact.get("evidence_snippets", []):
            found = False
            for src_id in src_ids:
                doc_chunks = chunks_by_document.get(src_id, [])
                for chunk in doc_chunks:
                    if snippet in chunk.get("text", ""):
                        found = True
                        break
                if found:
                    break
            if not found:
                # Truncate snippet for error message
                display = snippet[:80] + "..." if len(snippet) > 80 else snippet
                errors.append(
                    f"[{case_id}][{fid}] evidence snippet not found verbatim: "
                    f"'{display}'"
                )

    return errors


def validate_semantic_ground_truth(
    metadata: dict,
    cases: list[dict],
    chunks_path: str | Path,
) -> list[str]:
    """Full validation of metadata and all cases."""
    errors: list[str] = []

    # Metadata validation
    errors.extend(validate_metadata(metadata, chunks_path))

    # Load chunks for evidence validation
    path = Path(chunks_path)
    if not path.exists():
        errors.append(f"Cannot validate evidence: chunks file not found at {path}")
        return errors

    chunks = load_chunks(chunks_path)
    chunks_by_doc = index_chunks_by_document_id(chunks)

    # Case count from metadata
    expected_count = metadata.get("expected_case_count")
    if expected_count is None:
        errors.append("metadata missing expected_case_count")
    elif len(cases) != expected_count:
        errors.append(
            f"Expected exactly {expected_count} cases, found {len(cases)}"
        )

    # Case IDs
    case_ids = [c.get("id") for c in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append(f"Duplicate case IDs found: {case_ids}")
    for expected_id in EXPECTED_CASE_IDS:
        if expected_id not in case_ids:
            errors.append(f"Missing required case ID: {expected_id}")

    # Case groups from metadata
    expected_group_counts = metadata.get("expected_case_group_counts")
    if expected_group_counts is None:
        errors.append("metadata missing expected_case_group_counts")
    else:
        actual_groups: dict[str, int] = {}
        for c in cases:
            g = c.get("case_group")
            actual_groups[g] = actual_groups.get(g, 0) + 1
        for group_name in ALLOWED_CASE_GROUPS:
            expected_gc = expected_group_counts.get(group_name)
            actual_gc = actual_groups.get(group_name, 0)
            if expected_gc is None:
                errors.append(
                    f"metadata expected_case_group_counts missing group '{group_name}'"
                )
            elif actual_gc != expected_gc:
                errors.append(
                    f"case_group '{group_name}' has {actual_gc} cases, "
                    f"expected {expected_gc}"
                )
        # Check for unexpected groups
        for group_name in actual_groups:
            if group_name not in ALLOWED_CASE_GROUPS:
                errors.append(f"Unexpected case_group: '{group_name}'")

    # Original pilot IDs must be present
    for pilot_id in ORIGINAL_PILOT_CASE_IDS:
        if pilot_id not in case_ids:
            errors.append(f"Original pilot case ID missing: {pilot_id}")

    # Original pilot canonical hash must be unchanged
    pilot_cases = [c for c in cases if c.get("id") in ORIGINAL_PILOT_CASE_IDS]
    if len(pilot_cases) == len(ORIGINAL_PILOT_CASE_IDS):
        actual_pilot_hash = compute_canonical_case_hash(pilot_cases)
        if actual_pilot_hash != ORIGINAL_PILOT_CASES_SHA256:
            errors.append(
                f"Original pilot cases canonical hash changed: "
                f"expected '{ORIGINAL_PILOT_CASES_SHA256}', "
                f"actual '{actual_pilot_hash}'"
            )

    # Foundation twelve-case hash must be unchanged
    foundation_ids = set(EXPECTED_CASE_IDS[:12])
    foundation_cases = [c for c in cases if c.get("id") in foundation_ids]
    if len(foundation_cases) == 12:
        actual_foundation_hash = compute_canonical_case_hash(foundation_cases)
        if actual_foundation_hash != FOUNDATION_TWELVE_CASES_SHA256:
            errors.append(
                f"Foundation twelve-case canonical hash changed: "
                f"expected '{FOUNDATION_TWELVE_CASES_SHA256}', "
                f"actual '{actual_foundation_hash}'"
            )

    # Dataset ID consistency
    dataset_id = metadata.get("dataset_id")
    for case in cases:
        if case.get("dataset_id") != dataset_id:
            errors.append(
                f"[{case.get('id')}] dataset_id '{case.get('dataset_id')}' "
                f"does not match metadata dataset_id '{dataset_id}'"
            )

    # Per-case validation
    for case in cases:
        errors.extend(validate_case_schema(case))
        errors.extend(validate_case_evidence(case, chunks_by_doc))

    # Level consistency
    for case in cases:
        case_id = case.get("id", "?")
        primary = case.get("primary_level")
        acceptable = case.get("acceptable_levels", [])
        rel_docs = case.get("relevant_document_ids", [])
        doc_levels = set()
        for doc_id in rel_docs:
            for chunk in chunks_by_doc.get(doc_id, []):
                doc_levels.add(chunk.get("level"))
        if primary not in doc_levels:
            errors.append(
                f"[{case_id}] primary_level '{primary}' not represented by "
                f"any relevant document (found levels: {doc_levels})"
            )
        for dl in doc_levels:
            if dl not in acceptable:
                errors.append(
                    f"[{case_id}] relevant document has level '{dl}' "
                    f"which is not in acceptable_levels {acceptable}"
                )

    return errors
