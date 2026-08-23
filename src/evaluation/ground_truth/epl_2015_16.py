"""Semantic Ground Truth benchmark for Premier League 2015/2016."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.ground_truth.semantic import (
    ALLOWED_CASE_GROUPS,
    load_chunks,
    index_chunks_by_document_id,
    validate_metadata,
    validate_case_schema,
    validate_case_evidence,
)


EPL_2015_16_GROUND_TRUTH_METADATA: dict = {
    "schema_version": "1.0",
    "dataset_id": "statsbomb-premier-league-2015-2016",
    "tournament_name": "Premier League",
    "season_name": "2015/2016",
    "source_name": "StatsBomb Open Data",
    "competition_id": 2,
    "season_id": 27,
    "chunks_path": "output/competitions/2/27/chunks.json",
    "chunks_sha256": "e4a16678b202ac94bf69970a981a109813d2dcabb1ad8d6b3fb3524f8c28ff90",
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
        "Ground truth is tied to the exact Premier League 2015/2016 "
        "competition-local chunks.json snapshot identified by chunks_sha256."
    ),
}


def _fact(
    fact_id: str,
    claim: str,
    source_document_id: str,
    evidence_snippets: list[str],
) -> dict:
    return {
        "fact_id": fact_id,
        "claim": claim,
        "source_document_ids": [source_document_id],
        "evidence_snippets": evidence_snippets,
    }


def _case(
    *,
    case_id: str,
    case_group: str,
    query: str,
    expected_route: str,
    primary_level: str,
    acceptable_levels: list[str],
    relevant_document_ids: list[str],
    required_facts: list[dict],
    forbidden_claim: str,
    forbidden_reason: str,
    notes: str,
    optional_relevant_document_ids: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "dataset_id": EPL_2015_16_GROUND_TRUTH_METADATA["dataset_id"],
        "case_group": case_group,
        "query": query,
        "expected_route": expected_route,
        "answerability": "answerable",
        "primary_level": primary_level,
        "acceptable_levels": acceptable_levels,
        "relevant_document_ids": relevant_document_ids,
        "optional_relevant_document_ids": optional_relevant_document_ids or [],
        "required_facts": required_facts,
        "forbidden_claims": [
            {
                "claim": forbidden_claim,
                "reason": forbidden_reason,
            }
        ],
        "notes": notes,
    }


EPL_2015_16_GROUND_TRUTH: list[dict] = [
    # ------------------------------------------------------------------
    # L1: match summaries
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-l1-01",
        case_group="l1",
        query="What happened in Norwich City's 4-5 match against Liverpool on 23 January 2016?",
        expected_route="semantic",
        primary_level="1",
        acceptable_levels=["1"],
        relevant_document_ids=["L1-match-3754348"],
        required_facts=[
            _fact(
                "f1",
                "Liverpool beat Norwich City 5-4 at Carrow Road on 2016-01-23.",
                "L1-match-3754348",
                [
                    "The Premier League 2015/2016 Regular Season between Norwich City and Liverpool was played on 2016-01-23 at Carrow Road.",
                    "Liverpool beat Norwich City 5-4 in normal time.",
                ],
            ),
            _fact(
                "f2",
                "Adam Lallana scored Liverpool's late winner in the 94th minute.",
                "L1-match-3754348",
                [
                    "Adam David Lallana scored for Liverpool in the 94th minute with the left foot, from open play (xG 0.08).",
                ],
            ),
        ],
        forbidden_claim="Norwich City won the match 5-4.",
        forbidden_reason="The L1 document states Liverpool won 5-4.",
        notes="High-scoring league match used to test exact match-summary retrieval.",
    ),
    _case(
        case_id="epl15-l1-02",
        case_group="l1",
        query="What happened when Newcastle United hosted Norwich City on 18 October 2015?",
        expected_route="semantic",
        primary_level="1",
        acceptable_levels=["1"],
        relevant_document_ids=["L1-match-3754102"],
        required_facts=[
            _fact(
                "f1",
                "Newcastle United beat Norwich City 6-2 at St. James' Park.",
                "L1-match-3754102",
                [
                    "The Premier League 2015/2016 Regular Season between Newcastle United and Norwich City was played on 2015-10-18 at St. James' Park.",
                    "Newcastle United beat Norwich City 6-2 in normal time.",
                ],
            ),
            _fact(
                "f2",
                "Georginio Wijnaldum scored four times for Newcastle United.",
                "L1-match-3754102",
                [
                    "Georginio Wijnaldum scored for Newcastle United in the 13th minute",
                    "Georginio Wijnaldum scored for Newcastle United in the 25th minute",
                    "Georginio Wijnaldum scored for Newcastle United in the 65th minute",
                    "Georginio Wijnaldum scored for Newcastle United in the 84th minute",
                ],
            ),
        ],
        forbidden_claim="Norwich City won 6-2.",
        forbidden_reason="The source states Newcastle United won 6-2.",
        notes="Tests retrieval of a high-scoring match and repeated scorer evidence.",
    ),
    _case(
        case_id="epl15-l1-03",
        case_group="l1",
        query="How did AFC Bournemouth beat West Ham United at the Boleyn Ground in August 2015?",
        expected_route="semantic",
        primary_level="1",
        acceptable_levels=["1"],
        relevant_document_ids=["L1-match-3754131"],
        required_facts=[
            _fact(
                "f1",
                "AFC Bournemouth beat West Ham United 4-3 on 2015-08-22.",
                "L1-match-3754131",
                [
                    "The Premier League 2015/2016 Regular Season between West Ham United and AFC Bournemouth was played on 2015-08-22 at Boleyn Ground.",
                    "AFC Bournemouth beat West Ham United 4-3 in normal time.",
                ],
            ),
            _fact(
                "f2",
                "Callum Wilson scored three goals for AFC Bournemouth.",
                "L1-match-3754131",
                [
                    "Callum Wilson scored for AFC Bournemouth in the 10th minute",
                    "Callum Wilson scored for AFC Bournemouth in the 27th minute",
                    "Callum Wilson scored for AFC Bournemouth in the 78th minute",
                ],
            ),
        ],
        forbidden_claim="West Ham United won the match 4-3.",
        forbidden_reason="The L1 source records a 4-3 AFC Bournemouth victory.",
        notes="Tests match retrieval with a hat-trick and seven total goals.",
    ),
    _case(
        case_id="epl15-l1-04",
        case_group="l1",
        query="What was the result of Leicester City versus Arsenal on 26 September 2015?",
        expected_route="semantic",
        primary_level="1",
        acceptable_levels=["1"],
        relevant_document_ids=["L1-match-3754174"],
        required_facts=[
            _fact(
                "f1",
                "Arsenal beat Leicester City 5-2 at King Power Stadium.",
                "L1-match-3754174",
                [
                    "The Premier League 2015/2016 Regular Season between Leicester City and Arsenal was played on 2015-09-26 at King Power Stadium.",
                    "Arsenal beat Leicester City 5-2 in normal time.",
                ],
            ),
            _fact(
                "f2",
                "Olivier Giroud scored Arsenal's fifth goal in the 92nd minute.",
                "L1-match-3754174",
                [
                    "Olivier Giroud scored for Arsenal in the 92nd minute with the left foot, from open play",
                ],
            ),
        ],
        forbidden_claim="Leicester City won 5-2.",
        forbidden_reason="The source states Arsenal won 5-2.",
        notes="Tests retrieval of a seven-goal Arsenal-Leicester league match.",
    ),

    # ------------------------------------------------------------------
    # L2: key-event documents
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-l2-01",
        case_group="l2",
        query="In Sunderland's 2-2 draw with West Ham United on 3 October 2015, how many goals, substitutions, and high-xG misses were recorded?",
        expected_route="semantic",
        primary_level="2",
        acceptable_levels=["1", "2"],
        relevant_document_ids=["L2-match-3754076"],
        optional_relevant_document_ids=["L1-match-3754076"],
        required_facts=[
            _fact(
                "f1",
                "The L2 record contains 4 goals, 6 substitutions, and 2 high-quality missed chances.",
                "L2-match-3754076",
                [
                    "Goal: Steven Fletcher (Sunderland) in period 1, 9th minute",
                    "Goal: Jeremain Lens (Sunderland) in period 1, 21st minute",
                    "Goal: Carl Jenkinson (West Ham United) in period 1, 45th minute",
                    "Goal: Dimitri Payet (West Ham United) in period 2, 59th minute",
                    "High-quality chance missed: Fabio Borini (Sunderland) in the 37th minute",
                    "(West Ham United) in the 62nd minute recorded 0.53 xG",
                    "for Victor Moses in the 57th minute",
                    "for Winston Reid in the 65th minute",
                    "for Mark Noble in the 78th minute",
                    "for Ola Toivonen in the 83rd minute",
                    "for Steven Fletcher in the 84th minute",
                    "in the 95th minute. The substitution was tactical.",
                ],
            ),
        ],
        forbidden_claim="The L2 record contains only three goals.",
        forbidden_reason="Four goal events are present in the L2 document.",
        notes="Tests event-level goal, substitution, and high-xG-miss coverage.",
    ),
    _case(
        case_id="epl15-l2-02",
        case_group="l2",
        query="In Arsenal's 2-1 win over Leicester City on 14 February 2016, how many goals, substitutions, and high-xG misses were recorded?",
        expected_route="semantic",
        primary_level="2",
        acceptable_levels=["1", "2"],
        relevant_document_ids=["L2-match-3754261"],
        optional_relevant_document_ids=["L1-match-3754261"],
        required_facts=[
            _fact(
                "f1",
                "The L2 record contains 3 goals, 6 substitutions, and 1 high-quality missed chance.",
                "L2-match-3754261",
                [
                    "Goal: Jamie Vardy (Leicester City) in period 1, 44th minute",
                    "Goal: Theo Walcott (Arsenal) in period 2, 69th minute",
                    "in period 2, 94th minute, with the head, shot type Open Play, xG 0.2",
                    "High-quality chance missed: Daniel Nii Tackie Mensah Welbeck (Arsenal) in the 86th minute",
                    "for Laurent Koscielny in the 45th minute",
                    "for Riyad Mahrez in the 57th minute",
                    "for Shinji Okazaki in the 60th minute",
                    "for Francis Joseph Coquelin in the 60th minute",
                    "for Marc Albrighton in the 82nd minute",
                    "for Alex Oxlade-Chamberlain in the 82nd minute",
                ],
            ),
        ],
        forbidden_claim="No high-quality chance was missed in the match.",
        forbidden_reason="The L2 source records a high-quality missed chance in the 86th minute.",
        notes="Tests event counts in an Arsenal-Leicester L2 document.",
    ),
    _case(
        case_id="epl15-l2-03",
        case_group="l2",
        query="In Everton's 2-3 loss to West Ham United on 5 March 2016, how many goals, substitutions, and high-xG misses were recorded?",
        expected_route="semantic",
        primary_level="2",
        acceptable_levels=["1", "2"],
        relevant_document_ids=["L2-match-3754277"],
        optional_relevant_document_ids=["L1-match-3754277"],
        required_facts=[
            _fact(
                "f1",
                "The L2 record contains 5 goals, 6 substitutions, and 1 high-quality missed chance.",
                "L2-match-3754277",
                [
                    "Goal: Romelu Lukaku Menama (Everton) in period 1, 12th minute",
                    "Goal: Aaron Lennon (Everton) in period 2, 55th minute",
                    "Goal: Michail Antonio (West Ham United) in period 2, 77th minute",
                    "Goal: Diafra Sakho (West Ham United) in period 2, 80th minute",
                    "Goal: Dimitri Payet (West Ham United) in period 2, 89th minute",
                    "High-quality chance missed: Romelu Lukaku Menama (Everton) in the 68th minute",
                    "for John Stones in the 45th minute",
                    "for Reece Oxford in the 45th minute",
                    "for Emmanuel Emenike in the 60th minute",
                    "for Pedro Mba Obiang Avomo in the 60th minute",
                    "for Aaron Lennon in the 75th minute",
                    "for Romelu Lukaku Menama in the 88th minute",
                ],
            ),
        ],
        forbidden_claim="Everton won the match 3-2.",
        forbidden_reason="The source records three West Ham goals and two Everton goals.",
        notes="Tests event-level retrieval for Everton-West Ham.",
    ),
    _case(
        case_id="epl15-l2-04",
        case_group="l2",
        query="In Chelsea's 2-2 draw with West Ham United on 19 March 2016, how many goals, substitutions, and high-xG misses were recorded?",
        expected_route="semantic",
        primary_level="2",
        acceptable_levels=["1", "2"],
        relevant_document_ids=["L2-match-3753973"],
        optional_relevant_document_ids=["L1-match-3753973"],
        required_facts=[
            _fact(
                "f1",
                "The L2 record contains 4 goals, 6 substitutions, and 1 high-quality missed chance.",
                "L2-match-3753973",
                [
                    "Goal: Manuel Lanzini (West Ham United) in period 1, 16th minute",
                    "in period 1, 47th minute, with the right foot, shot type Free Kick, xG 0.07",
                    "Goal: Andy Carroll (West Ham United) in period 2, 60th minute",
                    "in period 2, 88th minute, with the right foot, shot type Penalty, xG 0.78",
                    "High-quality chance missed: John Michael Nchekwube Obinna (Chelsea) in the 67th minute",
                    "for Robert Kenedy Nunes do Nascimento in the 45th minute",
                    "for Diafra Sakho in the 59th minute",
                    "Substitution: Chelsea brought on Bertrand Isidore",
                    "for Enner Remberto Valencia Lastra in the 74th minute",
                    "for Manuel Lanzini in the 81st minute",
                    "Substitution: Chelsea brought on Ruben Loftus-Cheek",
                ],
            ),
        ],
        forbidden_claim="The match contained only three goals.",
        forbidden_reason="Four goal events are present in the L2 document.",
        notes="Tests event counts in Chelsea-West Ham.",
    ),

    # ------------------------------------------------------------------
    # L3: player-match performance
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-l3-01",
        case_group="l3",
        query="How did Jermain Defoe perform for Sunderland against Swansea City on 13 January 2016?",
        expected_route="semantic",
        primary_level="3",
        acceptable_levels=["3"],
        relevant_document_ids=["L3-match-3754071-player-3337"],
        required_facts=[
            _fact(
                "f1",
                "Defoe played 94 minutes and scored 3 goals from 3 shots worth 1.67 xG.",
                "L3-match-3754071-player-3337",
                [
                    "was on the pitch for 94 minutes.",
                    "He took 3 shots worth 1.67 expected goals, 3 from inside the penalty area and 0 from outside, and scored 3.",
                ],
            ),
            _fact(
                "f2",
                "He attempted 12 passes and completed 91.7%.",
                "L3-match-3754071-player-3337",
                [
                    "He attempted 12 passes and completed 91.7%, of which 4 were delivered into the final third.",
                ],
            ),
        ],
        forbidden_claim="Defoe scored two goals.",
        forbidden_reason="The L3 document records three goals.",
        notes="Tests retrieval of a player-match hat-trick performance.",
    ),
    _case(
        case_id="epl15-l3-02",
        case_group="l3",
        query="How did Santiago Cazorla perform for Arsenal against Newcastle United on 29 August 2015?",
        expected_route="semantic",
        primary_level="3",
        acceptable_levels=["3"],
        relevant_document_ids=["L3-match-3754205-player-11386"],
        required_facts=[
            _fact(
                "f1",
                "Cazorla attempted 131 passes at 90.8% completion, with 71 delivered into the final third.",
                "L3-match-3754205-player-11386",
                [
                    "He attempted 131 passes and completed 90.8%, of which 71 were delivered into the final third.",
                ],
            ),
            _fact(
                "f2",
                "He took 3 shots worth 0.13 xG and scored 0 goals.",
                "L3-match-3754205-player-11386",
                [
                    "He took 3 shots worth 0.13 expected goals, 2 from inside the penalty area and 1 from outside, and scored 0.",
                ],
            ),
        ],
        forbidden_claim="Cazorla scored in the match.",
        forbidden_reason="The L3 document records zero goals.",
        notes="Tests a high-volume midfield passing performance.",
    ),
    _case(
        case_id="epl15-l3-03",
        case_group="l3",
        query="How did Jose Ramiro Funes Mori perform defensively for Everton against Tottenham Hotspur?",
        expected_route="semantic",
        primary_level="3",
        acceptable_levels=["3"],
        relevant_document_ids=["L3-match-3754140-player-4629"],
        required_facts=[
            _fact(
                "f1",
                "Funes Mori recorded 4 successful tackles, 4 successful interceptions, and 17 clearances.",
                "L3-match-3754140-player-4629",
                [
                    "Defensively he recorded 4 successful tackles, 4 successful interceptions, and 17 clearances.",
                ],
            ),
            _fact(
                "f2",
                "He attempted 42 passes and completed 66.7%.",
                "L3-match-3754140-player-4629",
                [
                    "He attempted 42 passes and completed 66.7%, of which 5 were delivered into the final third.",
                ],
            ),
        ],
        forbidden_claim="Funes Mori made no clearances.",
        forbidden_reason="The L3 document records 17 clearances.",
        notes="Tests retrieval of a defensive center-back match profile.",
    ),
    _case(
        case_id="epl15-l3-04",
        case_group="l3",
        query="How did Kevin De Bruyne perform against West Ham United on 19 September 2015?",
        expected_route="semantic",
        primary_level="3",
        acceptable_levels=["3"],
        relevant_document_ids=["L3-match-3754077-player-3089"],
        required_facts=[
            _fact(
                "f1",
                "De Bruyne scored 1 goal from 4 shots worth 0.2 xG.",
                "L3-match-3754077-player-3089",
                [
                    "He took 4 shots worth 0.2 expected goals, 1 from inside the penalty area and 3 from outside, and scored 1.",
                ],
            ),
            _fact(
                "f2",
                "He attempted 84 passes at 76.2% completion, with 70 delivered into the final third.",
                "L3-match-3754077-player-3089",
                [
                    "He attempted 84 passes and completed 76.2%, of which 70 were delivered into the final third.",
                ],
            ),
        ],
        forbidden_claim="De Bruyne failed to score.",
        forbidden_reason="The source records one goal.",
        notes="Tests an attacking-midfielder player-match profile.",
    ),

    # ------------------------------------------------------------------
    # L4: season player summaries
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-l4-01",
        case_group="l4",
        query="Describe Harry Kane's overall Premier League 2015/16 performance.",
        expected_route="semantic",
        primary_level="4",
        acceptable_levels=["4"],
        relevant_document_ids=["L4-player-10955"],
        required_facts=[
            _fact(
                "f1",
                "Harry Kane appeared in 38 matches, scored 25 goals, provided 1 assist, and produced 21.64 xG.",
                "L4-player-10955",
                [
                    "appeared in 38 matches at the Premier League 2015/2016, playing 3486.7 minutes in total.",
                    "Across the competition he scored 25 goals and provided 1 assist from 21.64 expected goals",
                ],
            ),
        ],
        forbidden_claim="Harry Kane scored 18 league goals.",
        forbidden_reason="The L4 document records 25 goals.",
        notes="Tests a season-level striker summary.",
    ),
    _case(
        case_id="epl15-l4-02",
        case_group="l4",
        query="Describe Mesut Ozil's overall Premier League 2015/16 performance.",
        expected_route="semantic",
        primary_level="4",
        acceptable_levels=["4"],
        relevant_document_ids=["L4-player-3496"],
        required_facts=[
            _fact(
                "f1",
                "Mesut Ozil appeared in 35 matches, scored 6 goals, provided 19 assists, and produced 5.97 xG.",
                "L4-player-3496",
                [
                    "appeared in 35 matches at the Premier League 2015/2016, playing 3132.8 minutes in total.",
                    "Across the competition he scored 6 goals and provided 19 assists from 5.97 expected goals",
                ],
            ),
            _fact(
                "f2",
                "He averaged 71.7 passes attempted at 81.5% completion.",
                "L4-player-3496",
                [
                    "with 71.7 passes attempted at 81.5% completion.",
                ],
            ),
        ],
        forbidden_claim="Ozil provided fewer than 10 assists.",
        forbidden_reason="The L4 source records 19 assists.",
        notes="Tests a season-level creative-midfielder summary.",
    ),
    _case(
        case_id="epl15-l4-03",
        case_group="l4",
        query="Describe Francesc Fabregas's overall Premier League 2015/16 performance.",
        expected_route="semantic",
        primary_level="4",
        acceptable_levels=["4"],
        relevant_document_ids=["L4-player-3478"],
        required_facts=[
            _fact(
                "f1",
                "Fabregas appeared in 37 matches, scored 5 goals, provided 7 assists, and produced 4.03 xG.",
                "L4-player-3478",
                [
                    "appeared in 37 matches at the Premier League 2015/2016, playing 3037.5 minutes in total.",
                    "Across the competition he scored 5 goals and provided 7 assists from 4.03 expected goals",
                ],
            ),
            _fact(
                "f2",
                "He averaged 79.6 passes attempted at 81.6% completion.",
                "L4-player-3478",
                [
                    "with 79.6 passes attempted at 81.6% completion.",
                ],
            ),
        ],
        forbidden_claim="Fabregas provided 11 assists.",
        forbidden_reason="The L4 source records 7 assists.",
        notes="Tests a season-level central-midfielder summary.",
    ),
    _case(
        case_id="epl15-l4-04",
        case_group="l4",
        query="Describe Ashley Williams's overall Premier League 2015/16 performance.",
        expected_route="semantic",
        primary_level="4",
        acceptable_levels=["4"],
        relevant_document_ids=["L4-player-3644"],
        required_facts=[
            _fact(
                "f1",
                "Ashley Williams appeared in 36 matches, scored 2 goals, provided 1 assist, and produced 2.04 xG.",
                "L4-player-3644",
                [
                    "appeared in 36 matches at the Premier League 2015/2016, playing 3398.1 minutes in total.",
                    "Across the competition he scored 2 goals and provided 1 assist from 2.04 expected goals",
                ],
            ),
            _fact(
                "f2",
                "Williams averaged 9 clearances per match.",
                "L4-player-3644",
                [
                    "1.14 successful interception and 9 clearances per match.",
                ],
            ),
        ],
        forbidden_claim="Ashley Williams scored 18 league goals.",
        forbidden_reason="The L4 document records 2 goals.",
        notes="Tests a season-level central-defender summary.",
    ),

    # ------------------------------------------------------------------
    # TEAM: season playing-style summaries
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-team-01",
        case_group="team",
        query="What does Arsenal's 2015/16 team profile show about formation usage and passing style?",
        expected_route="semantic",
        primary_level="team",
        acceptable_levels=["team"],
        relevant_document_ids=["TEAM-1"],
        required_facts=[
            _fact(
                "f1",
                "Arsenal's dominant recorded shape was 4231 with 70 formation records.",
                "TEAM-1",
                [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 4231 (70 formation records).",
                ],
            ),
            _fact(
                "f2",
                "82.7% of Arsenal's 22709 passes were standard open-play passes.",
                "TEAM-1",
                [
                    "Of 22709 passes, 82.7% were standard open-play passes",
                ],
            ),
        ],
        forbidden_claim="Arsenal's possession figure is official broadcast possession.",
        forbidden_reason="The team document explicitly describes possession as an event-share proxy.",
        notes="Tests a team formation and passing-style profile.",
    ),
    _case(
        case_id="epl15-team-02",
        case_group="team",
        query="What were Liverpool's most common shapes and passing profile in the 2015/16 Premier League?",
        expected_route="semantic",
        primary_level="team",
        acceptable_levels=["team"],
        relevant_document_ids=["TEAM-24"],
        required_facts=[
            _fact(
                "f1",
                "Liverpool's three most common recorded shapes were 4231 (27), 433 (19), and 4411 (9).",
                "TEAM-24",
                [
                    "Their most common shapes, counted across Starting XI and Tactical Shift events, were 4231 (27 formation records), 433 (19 formation records), and 4411 (9 formation records).",
                ],
            ),
            _fact(
                "f2",
                "79.4% of Liverpool's 21464 passes were standard open-play passes.",
                "TEAM-24",
                [
                    "Of 21464 passes, 79.4% were standard open-play passes",
                ],
            ),
        ],
        forbidden_claim="Liverpool used only one formation.",
        forbidden_reason="The team document lists multiple common shapes.",
        notes="Tests a diverse team formation profile without relying on non-rendered formation-count totals.",
    ),
    _case(
        case_id="epl15-team-03",
        case_group="team",
        query="What does Leicester City's 2015/16 team profile show about scoring frequency and recovery passes?",
        expected_route="semantic",
        primary_level="team",
        acceptable_levels=["team"],
        relevant_document_ids=["TEAM-22"],
        required_facts=[
            _fact(
                "f1",
                "Leicester City scored in 35 of their 38 matches.",
                "TEAM-22",
                [
                    "When they scored, their first goal came on average in the 41st minute, across 35 of their 38 matches.",
                ],
            ),
            _fact(
                "f2",
                "Recovery passes accounted for 14.9% of Leicester City's passes.",
                "TEAM-22",
                [
                    "chiefly recovery 14.9%, throw-in 6.0%, free kick 2.8%, and goal kick 2.2%.",
                ],
            ),
        ],
        forbidden_claim="Leicester City scored in all 38 matches.",
        forbidden_reason="The source records scoring in 35 of 38 matches.",
        notes="Tests team scoring frequency and restart/recovery passing profile.",
    ),
    _case(
        case_id="epl15-team-04",
        case_group="team",
        query="What does West Bromwich Albion's 2015/16 team profile show about possession and standard passing?",
        expected_route="semantic",
        primary_level="team",
        acceptable_levels=["team"],
        relevant_document_ids=["TEAM-27"],
        required_facts=[
            _fact(
                "f1",
                "West Bromwich Albion had a 40.5% event-share possession proxy.",
                "TEAM-27",
                [
                    "Across their matches they were the team in possession for 40.5% of events, an event-share proxy rather than StatsBomb's broadcast possession figure.",
                ],
            ),
            _fact(
                "f2",
                "71.1% of West Bromwich Albion's 13971 passes were standard open-play passes.",
                "TEAM-27",
                [
                    "Of 13971 passes, 71.1% were standard open-play passes",
                ],
            ),
        ],
        forbidden_claim="The 40.5% figure is StatsBomb broadcast possession.",
        forbidden_reason="The document explicitly labels it an event-share proxy.",
        notes="Tests low-possession team style and passing profile.",
    ),

    # ------------------------------------------------------------------
    # MULTI: L1 + L2 + L3 evidence
    # ------------------------------------------------------------------
    _case(
        case_id="epl15-multi-01",
        case_group="multi",
        query="How did Chelsea's 2-2 draw with Tottenham unfold, and how did Harry Kane perform?",
        expected_route="hybrid",
        primary_level="2",
        acceptable_levels=["1", "2", "3"],
        relevant_document_ids=[
            "L1-match-3754092",
            "L2-match-3754092",
            "L3-match-3754092-player-10955",
        ],
        required_facts=[
            _fact(
                "f1",
                "Chelsea and Tottenham Hotspur drew 2-2 at Stamford Bridge.",
                "L1-match-3754092",
                ["The match finished level at 2-2 in normal time."],
            ),
            _fact(
                "f2",
                "Chelsea brought Eden Hazard on at half-time.",
                "L2-match-3754092",
                [
                    "Substitution: Chelsea brought on Eden Hazard for Pedro Eliezer",
                ],
            ),
            _fact(
                "f3",
                "Harry Kane scored 1 goal from 5 shots worth 1.02 xG.",
                "L3-match-3754092-player-10955",
                [
                    "He took 5 shots worth 1.02 expected goals, 3 from inside the penalty area and 2 from outside, and scored 1.",
                ],
            ),
        ],
        forbidden_claim="Tottenham won the match 2-0.",
        forbidden_reason="The L1 document records a 2-2 draw.",
        notes="Tests multi-level evidence for a match plus a player performance.",
    ),
    _case(
        case_id="epl15-multi-02",
        case_group="multi",
        query="How did Manchester City's 6-1 win over Newcastle unfold, and how did Sergio Aguero perform?",
        expected_route="hybrid",
        primary_level="2",
        acceptable_levels=["1", "2", "3"],
        relevant_document_ids=[
            "L1-match-3754079",
            "L2-match-3754079",
            "L3-match-3754079-player-3237",
        ],
        required_facts=[
            _fact(
                "f1",
                "Manchester City beat Newcastle United 6-1.",
                "L1-match-3754079",
                ["Manchester City beat Newcastle United 6-1 in normal time."],
            ),
            _fact(
                "f2",
                "The key-event record includes a 0.85 xG missed chance by Fernando.",
                "L2-match-3754079",
                [
                    "in the 7th minute recorded 0.85 xG but the shot ended saved",
                ],
            ),
            _fact(
                "f3",
                "Sergio Aguero scored 5 goals from 8 shots worth 2.03 xG.",
                "L3-match-3754079-player-3237",
                [
                    "He took 8 shots worth 2.03 expected goals, 5 from inside the penalty area and 3 from outside, and scored 5.",
                ],
            ),
        ],
        forbidden_claim="Aguero scored three goals.",
        forbidden_reason="The L3 document records five goals.",
        notes="Tests multi-level evidence around a five-goal player performance.",
    ),
    _case(
        case_id="epl15-multi-03",
        case_group="multi",
        query="How did Everton's 6-2 win over Sunderland unfold, and how did Arouna Kone perform?",
        expected_route="hybrid",
        primary_level="2",
        acceptable_levels=["1", "2", "3"],
        relevant_document_ids=[
            "L1-match-3754082",
            "L2-match-3754082",
            "L3-match-3754082-player-26696",
        ],
        required_facts=[
            _fact(
                "f1",
                "Everton beat Sunderland 6-2.",
                "L1-match-3754082",
                ["Everton beat Sunderland 6-2 in normal time."],
            ),
            _fact(
                "f2",
                "Everton made an injury substitution in the 24th minute.",
                "L2-match-3754082",
                [
                    "Substitution: Everton brought on Brendan Joel Zibusiso Galloway for Bryan Oviedo in the 24th minute.",
                    "The substitution was injury.",
                ],
            ),
            _fact(
                "f3",
                "Arouna Kone scored 3 goals, provided 1 assist, and took 4 shots worth 0.63 xG.",
                "L3-match-3754082-player-26696",
                [
                    "He took 4 shots worth 0.63 expected goals, 3 from inside the penalty area and 1 from outside, and scored 3.",
                    "He provided 1 assist.",
                ],
            ),
        ],
        forbidden_claim="Sunderland won the match 6-2.",
        forbidden_reason="The L1 source records an Everton victory.",
        notes="Tests multi-level match and player evidence.",
    ),
    _case(
        case_id="epl15-multi-04",
        case_group="multi",
        query="How did Manchester United beat Arsenal 3-2, and how did Marcus Rashford perform?",
        expected_route="hybrid",
        primary_level="2",
        acceptable_levels=["1", "2", "3"],
        relevant_document_ids=[
            "L1-match-3754239",
            "L2-match-3754239",
            "L3-match-3754239-player-3318",
        ],
        required_facts=[
            _fact(
                "f1",
                "Manchester United beat Arsenal 3-2 at Old Trafford.",
                "L1-match-3754239",
                ["Manchester United beat Arsenal 3-2 in normal time."],
            ),
            _fact(
                "f2",
                "Marcus Rashford assisted Manchester United's third goal.",
                "L2-match-3754239",
                [
                    "Marcus Rashford provided the assist. It was a first-time strike and deflected on its way in.",
                ],
            ),
            _fact(
                "f3",
                "Rashford scored 2 goals, provided 1 assist, and took 2 shots worth 0.25 xG.",
                "L3-match-3754239-player-3318",
                [
                    "He took 2 shots worth 0.25 expected goals, 2 from inside the penalty area and 0 from outside, and scored 2.",
                    "He provided 1 assist.",
                ],
            ),
        ],
        forbidden_claim="Arsenal won the match 3-2.",
        forbidden_reason="The L1 document records a Manchester United victory.",
        notes="Tests multi-level evidence for Rashford's breakthrough Arsenal match.",
    ),
]


EXPECTED_EPL_2015_16_CASE_IDS: tuple[str, ...] = tuple(
    f"epl15-{group}-{number:02d}"
    for group in ("l1", "l2", "l3", "l4", "team", "multi")
    for number in range(1, 5)
)


def validate_epl_2015_16_ground_truth(
    metadata: dict,
    cases: list[dict],
    chunks_path: str | Path,
) -> list[str]:
    """Validate EPL 2015/16 cases without WC2022-specific pilot/hash rules."""
    errors: list[str] = []

    errors.extend(validate_metadata(metadata, chunks_path))

    path = Path(chunks_path)
    if not path.exists():
        return errors

    chunks = load_chunks(path)
    chunks_by_doc = index_chunks_by_document_id(chunks)

    expected_count = metadata.get("expected_case_count")
    if expected_count is None:
        errors.append("metadata missing expected_case_count")
    elif len(cases) != expected_count:
        errors.append(
            f"Expected exactly {expected_count} cases, found {len(cases)}"
        )

    case_ids = [case.get("id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append(f"Duplicate case IDs found: {case_ids}")

    expected_ids = set(EXPECTED_EPL_2015_16_CASE_IDS)
    actual_ids = set(case_ids)

    for missing_id in sorted(expected_ids - actual_ids):
        errors.append(f"Missing required case ID: {missing_id}")

    for unexpected_id in sorted(actual_ids - expected_ids):
        errors.append(f"Unexpected case ID: {unexpected_id}")

    expected_group_counts = metadata.get("expected_case_group_counts") or {}
    actual_groups: dict[str, int] = {}

    for case in cases:
        group = case.get("case_group")
        actual_groups[group] = actual_groups.get(group, 0) + 1

    for group in ALLOWED_CASE_GROUPS:
        expected_group_count = expected_group_counts.get(group)
        actual_group_count = actual_groups.get(group, 0)

        if expected_group_count is None:
            errors.append(
                f"metadata expected_case_group_counts missing group '{group}'"
            )
        elif actual_group_count != expected_group_count:
            errors.append(
                f"case_group '{group}' has {actual_group_count} cases, "
                f"expected {expected_group_count}"
            )

    dataset_id = metadata.get("dataset_id")

    for case in cases:
        case_id = case.get("id", "?")

        if case.get("dataset_id") != dataset_id:
            errors.append(
                f"[{case_id}] dataset_id '{case.get('dataset_id')}' "
                f"does not match metadata dataset_id '{dataset_id}'"
            )

        errors.extend(validate_case_schema(case))
        errors.extend(validate_case_evidence(case, chunks_by_doc))

        primary = case.get("primary_level")
        acceptable = case.get("acceptable_levels", [])
        relevant_docs = case.get("relevant_document_ids", [])

        document_levels: set[str] = set()

        for document_id in relevant_docs:
            for chunk in chunks_by_doc.get(document_id, []):
                document_levels.add(chunk.get("level"))

        if primary not in document_levels:
            errors.append(
                f"[{case_id}] primary_level '{primary}' not represented by "
                f"any relevant document (found levels: {document_levels})"
            )

        for document_level in document_levels:
            if document_level not in acceptable:
                errors.append(
                    f"[{case_id}] relevant document has level "
                    f"'{document_level}' which is not in acceptable_levels "
                    f"{acceptable}"
                )

    return errors
