from __future__ import annotations

from src.context.selection import select_relevant_chunks


FRANCE_TEAM_CHUNKS = [
    {
        "chunk_id": "TEAM-771-chunk-0",
        "text": (
            "France played 7 matches at the FIFA World Cup 2022. "
            "They took their first shot on average in the 16th minute. "
            "Their first goal came on average in the 38th minute. "
            "They were in possession for 51.1% of events."
        ),
        "metadata": {
            "document_id": "TEAM-771",
            "level": "team",
            "team_name": "France",
        },
        "score": 0.9,
        "rank": 1,
        "source": "hybrid",
    },
    {
        "chunk_id": "TEAM-771-chunk-1",
        "text": (
            "Their play patterns were dominated by regular play, throw ins, "
            "free kicks, and goal kicks. Their most common formations were "
            "4231, 433, and 442."
        ),
        "metadata": {
            "document_id": "TEAM-771",
            "level": "team",
            "team_name": "France",
        },
        "score": 0.7,
        "rank": 2,
        "source": "hybrid",
    },
    {
        "chunk_id": "TEAM-771-chunk-2",
        "text": (
            "Of 3926 passes, 83.2% were standard open-play passes and 16.8% "
            "came from set pieces or restarts. Their most common formations "
            "were 4231, 433, and 442. They delivered 94 crosses."
        ),
        "metadata": {
            "document_id": "TEAM-771",
            "level": "team",
            "team_name": "France",
        },
        "score": 0.6,
        "rank": 3,
        "source": "hybrid",
    },
]


def test_selects_chunks_that_jointly_cover_multi_facet_query():
    selected = select_relevant_chunks(
        query="What were France's passing patterns and most common formations?",
        chunks=FRANCE_TEAM_CHUNKS,
        max_chunks=2,
    )

    assert selected == [
        FRANCE_TEAM_CHUNKS[1],
        FRANCE_TEAM_CHUNKS[2],
    ]


def test_selects_single_answer_bearing_chunk():
    selected = select_relevant_chunks(
        query="When did France usually take their first shot and score?",
        chunks=FRANCE_TEAM_CHUNKS,
        max_chunks=1,
    )

    assert selected == [FRANCE_TEAM_CHUNKS[0]]


def test_selection_is_independent_of_entity_and_chunk_id_format():
    generic_chunks = [
        {
            "chunk_id": "DOC-X-part-7",
            "text": "Brazil averaged 55 percent possession.",
            "metadata": {
                "document_id": "DOC-X",
                "level": "team",
                "team_name": "Brazil",
            },
            "score": 0.8,
            "rank": 1,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-X-part-9",
            "text": "Brazil's common formations were 433 and 4231.",
            "metadata": {
                "document_id": "DOC-X",
                "level": "team",
                "team_name": "Brazil",
            },
            "score": 0.5,
            "rank": 2,
            "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="What formations did Brazil commonly use?",
        chunks=generic_chunks,
        max_chunks=1,
    )

    assert selected == [generic_chunks[1]]

def test_prefers_new_query_coverage_over_redundant_chunk():
    chunks = [
        {
            "chunk_id": "DOC-A-part-1",
            "text": (
                "Passing patterns included short passing and progressive passing."
            ),
            "metadata": {"document_id": "DOC-A", "level": "team"},
            "score": 0.9,
            "rank": 1,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-A-part-2",
            "text": (
                "Passing patterns relied on short passing and progressive passing."
            ),
            "metadata": {"document_id": "DOC-A", "level": "team"},
            "score": 0.8,
            "rank": 2,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-A-part-3",
            "text": "The most common formations were 433 and 4231.",
            "metadata": {"document_id": "DOC-A", "level": "team"},
            "score": 0.7,
            "rank": 3,
            "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="Which passing patterns and formations were most common?",
        chunks=chunks,
        max_chunks=2,
    )

    assert selected == [chunks[0], chunks[2]]

def test_hybrid_search_applies_chunk_selector_before_final_top_k(monkeypatch):
    """
    Structural Cleanup Phase A: hybrid_search() and its internal
    dependencies (bm25_search, dense_search, _ensure_*) now live in
    src.retrieval.search (moved from 06_retrieve_context.py) -- patched
    there directly, since hybrid_search() resolves them via that module's
    own globals, not the compatibility wrapper's.
    """
    from importlib import import_module

    retrieval = import_module("src.retrieval.search")

    monkeypatch.setattr(
        retrieval,
        "bm25_search",
        lambda query, k=20, artifact_paths=None: [
            {**FRANCE_TEAM_CHUNKS[0], "rank": 1, "source": "bm25"},
            {**FRANCE_TEAM_CHUNKS[1], "rank": 2, "source": "bm25"},
            {**FRANCE_TEAM_CHUNKS[2], "rank": 3, "source": "bm25"},
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "dense_search",
        lambda query, k=20, level_filter=None, artifact_paths=None: [],
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_comparison_entities",
        lambda query, results, k, artifact_paths=None: results,
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_team_style_doc",
        lambda query, results, k, artifact_paths=None: results,
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_match_summary",
        lambda query, results, k, artifact_paths=None: results,
    )

    selected = retrieval.hybrid_search(
        "What were France's passing patterns and most common formations?",
        k=2,
    )

    assert [chunk["chunk_id"] for chunk in selected] == [
        "TEAM-771-chunk-1",
        "TEAM-771-chunk-2",
    ]

def test_hybrid_search_expands_siblings_before_chunk_selection(monkeypatch):
    """
    Structural Cleanup Phase A: hybrid_search() and its internal
    dependencies now live in src.retrieval.search (moved from
    06_retrieve_context.py) -- patched there directly, for the same reason
    as test_hybrid_search_applies_chunk_selector_before_final_top_k above.
    """
    from importlib import import_module

    retrieval = import_module("src.retrieval.search")

    unrelated_chunk = {
        "chunk_id": "OTHER-DOC-part-1",
        "text": "Another team used passing patterns and common formations.",
        "metadata": {
            "document_id": "OTHER-DOC",
            "level": "team",
            "team_name": "Other Team",
        },
        "score": 0.8,
        "rank": 2,
        "source": "bm25",
    }

    monkeypatch.setattr(
        retrieval,
        "bm25_search",
        lambda query, k=20, artifact_paths=None: [
            {**FRANCE_TEAM_CHUNKS[0], "rank": 1, "source": "bm25"},
            unrelated_chunk,
        ],
    )
    monkeypatch.setattr(
        retrieval,
        "dense_search",
        lambda query, k=20, level_filter=None, artifact_paths=None: [],
    )
    monkeypatch.setattr(
        retrieval,
        "_load_chunks",
        lambda path=None: FRANCE_TEAM_CHUNKS + [unrelated_chunk],
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_comparison_entities",
        lambda query, results, k, artifact_paths=None: results,
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_team_style_doc",
        lambda query, results, k, artifact_paths=None: results,
    )
    monkeypatch.setattr(
        retrieval,
        "_ensure_match_summary",
        lambda query, results, k, artifact_paths=None: results,
    )

    selected = retrieval.hybrid_search(
        "What were France's passing patterns and most common formations?",
        k=2,
    )

    assert [chunk["chunk_id"] for chunk in selected] == [
        "TEAM-771-chunk-1",
        "TEAM-771-chunk-2",
    ]

def test_falls_back_to_original_ranking_when_no_lexical_evidence():
    chunks = [
        {
            "chunk_id": "DOC-1-part-1",
            "text": "The side controlled the tempo and circulated the ball patiently.",
            "metadata": {"document_id": "DOC-1", "level": "team"},
            "score": 0.9,
            "rank": 1,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-2-part-1",
            "text": "The opponent defended in a compact shape.",
            "metadata": {"document_id": "DOC-2", "level": "team"},
            "score": 0.8,
            "rank": 2,
            "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="How did they dominate proceedings?",
        chunks=chunks,
        max_chunks=1,
    )

    assert selected == [chunks[0]]


def test_backfills_ranked_chunks_after_query_coverage_is_exhausted():
    chunks = [
        {
            "chunk_id": "DOC-1",
            "text": "France used passing patterns and formations throughout the tournament.",
            "metadata": {"document_id": "DOC-1", "level": "team", "team_name": "France"},
            "score": 0.9,
            "rank": 1,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-2",
            "text": "France also controlled possession in several matches.",
            "metadata": {"document_id": "DOC-2", "level": "team", "team_name": "France"},
            "score": 0.8,
            "rank": 2,
            "source": "hybrid",
        },
        {
            "chunk_id": "DOC-3",
            "text": "France reached the final after seven matches.",
            "metadata": {"document_id": "DOC-3", "level": "team", "team_name": "France"},
            "score": 0.7,
            "rank": 3,
            "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="What were France's passing patterns and formations?",
        chunks=chunks,
        max_chunks=3,
    )

    assert selected == chunks


def test_keeps_candidate_with_missing_entity_metadata_when_query_evidence_matches():
    """Missing entity metadata must not be treated as a conflicting entity."""
    chunks = [
        {
            "chunk_id": "L2-correct",
            "text": "Substitution: Argentina made several tactical changes in the semi-final.",
            "metadata": {
                "document_id": "L2-match-correct",
                "level": "2",
            },
            "score": 0.9,
            "rank": 1,
            "source": "dense",
        },
        {
            "chunk_id": "ARG-player",
            "text": "Argentina player performance in the semi-final.",
            "metadata": {
                "document_id": "L3-arg",
                "level": "3",
                "team_name": "Argentina",
            },
            "score": 0.8,
            "rank": 2,
            "source": "bm25",
        },
        {
            "chunk_id": "CRO-player",
            "text": "Croatia player performance in the semi-final.",
            "metadata": {
                "document_id": "L3-cro",
                "level": "3",
                "team_name": "Croatia",
            },
            "score": 0.7,
            "rank": 3,
            "source": "bm25",
        },
    ]

    selected = select_relevant_chunks(
        query="What substitutions were made in Argentina vs Croatia semi-final?",
        chunks=chunks,
        max_chunks=2,
    )

    assert chunks[0] in selected

def test_selector_keeps_multiple_facets_from_same_document():
    chunks = [
        {
            "chunk_id": "L2-match-chunk-3",
            "text": "Goals high xG misses recorded in the match.",
            "metadata": {"document_id": "L2-match-1", "level": "2"},
            "rank": 1,
        },
        {
            "chunk_id": "L2-match-chunk-5",
            "text": "Six substitutions were recorded in the match.",
            "metadata": {"document_id": "L2-match-1", "level": "2"},
            "rank": 2,
        },
        {
            "chunk_id": "other",
            "text": "Unrelated match with goals.",
            "metadata": {"document_id": "other-doc", "level": "2"},
            "rank": 3,
        },
    ]

    selected = select_relevant_chunks(
        "How many goals substitutions and high xG misses were recorded?",
        chunks,
        max_chunks=2,
    )

    assert chunks[0] in selected
    assert chunks[1] in selected

def test_match_selection_prefers_exact_fixture_date_when_same_teams_play_twice():
    """Same team pair can meet more than once; fixture date must disambiguate selection."""
    wrong_fixture = {
        "chunk_id": "L2-old-fixture",
        "text": "Goals, substitutions, and high xG misses were recorded in the match.",
        "metadata": {
            "document_id": "L2-match-old",
            "level": "2",
            "home_team": "Leicester City",
            "away_team": "Arsenal",
            "match_date": "2015-09-26",
        },
        "rank": 1,
        "source": "bm25",
    }
    correct_fixture = {
        "chunk_id": "L2-correct-fixture",
        "text": "Goals, substitutions, and high xG misses were recorded in the match.",
        "metadata": {
            "document_id": "L2-match-correct",
            "level": "2",
            "home_team": "Arsenal",
            "away_team": "Leicester City",
            "match_date": "2016-02-14",
        },
        "rank": 2,
        "source": "bm25",
    }

    selected = select_relevant_chunks(
        "In Arsenal's 2-1 win over Leicester City on 14 February 2016, "
        "how many goals, substitutions, and high-xG misses were recorded?",
        [wrong_fixture, correct_fixture],
        max_chunks=1,
    )

    assert selected == [correct_fixture]

def test_selector_prioritizes_exact_fixture_expansion_after_match_pair_boost():
    match_boost = {
        "chunk_id": "L1-target",
        "text": "Manchester City beat Newcastle United 6-1.",
        "metadata": {"document_id": "L1-match-10", "level": "1"},
        "source": "match_pair_boost",
    }
    fixture_l2 = {
        "chunk_id": "L2-target",
        "text": "Goals substitutions and high xG misses were recorded in the match.",
        "metadata": {"document_id": "L2-match-10", "level": "2"},
        "source": "match_fixture_expansion",
    }
    fixture_l3 = {
        "chunk_id": "L3-target",
        "text": "Sergio Aguero scored five goals and produced the key player performance.",
        "metadata": {
            "document_id": "L3-match-10-player-1",
            "level": "3",
            "player_name": "Sergio Aguero",
        },
        "source": "match_fixture_expansion",
    }
    distractor = {
        "chunk_id": "wrong",
        "text": "Sergio Aguero scored goals in another Manchester City match.",
        "metadata": {
            "document_id": "L3-match-99-player-1",
            "level": "3",
            "player_name": "Sergio Aguero",
        },
        "source": "bm25",
    }

    selected = select_relevant_chunks(
        "How did Manchester City win 6-1, and how did Sergio Aguero perform with goals?",
        [match_boost, distractor, fixture_l2, fixture_l3],
        max_chunks=3,
    )

    assert selected[0] is match_boost
    assert fixture_l2 in selected
    assert fixture_l3 in selected
    assert distractor not in selected
