"""
test_router.py — Phase 5: Router Unit Tests

Tests routing decisions and execution.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

from src.query.query_schema import StructuredQuery, StructuredResult
from src.query.resolver import resolve as structured_resolve

# Import router with proper module setup
spec = importlib.util.spec_from_file_location("router", Path("06_retrieve_context.py"))
router = importlib.util.module_from_spec(spec)
sys.modules["router"] = router  # Required for dataclass resolution
spec.loader.exec_module(router)


# ---------------------------------------------------------------------------
# Tests: Classification
# ---------------------------------------------------------------------------


def test_structured_classification():
    """Numeric queries should be classified as structured."""
    route = router.route_query("How many goals did Messi score?")
    assert route.path == "structured", f"Path={route.path}"
    assert route.confidence >= 0.8, f"Confidence={route.confidence}"


def test_semantic_classification():
    """Descriptive queries should be classified as semantic."""
    route = router.route_query("How did France play in the final?")
    assert route.path == "semantic", f"Path={route.path}"
    assert route.confidence >= 0.8, f"Confidence={route.confidence}"


def test_superlative_classification():
    """Superlative queries should be classified as structured."""
    route = router.route_query("Who scored the most goals?")
    assert route.path == "structured", f"Path={route.path}"


def test_which_team_classification():
    """Which team queries should be classified as structured."""
    route = router.route_query("Which team had the highest xG?")
    assert route.path == "structured", f"Path={route.path}"


# ---------------------------------------------------------------------------
# Tests: Parsing
# ---------------------------------------------------------------------------


def test_parse_numeric():
    """Parse numeric query correctly."""
    query = router.parse_structured_query("How many goals did Messi score?")
    assert query is not None
    assert query.intent == "numeric"
    assert query.entity == "player"
    assert query.metric == "goals"
    assert query.aggregation == "sum"
    assert "Messi" in query.entity_name


def test_parse_superlative():
    """Parse superlative query correctly."""
    query = router.parse_structured_query("Who scored the most goals?")
    assert query is not None
    assert query.intent == "superlative"
    assert query.entity == "player"
    assert query.metric == "goals"
    assert query.limit == 1


def test_parse_which_team():
    """Parse which-team query correctly."""
    query = router.parse_structured_query("Which team had the highest xG?")
    assert query is not None
    assert query.intent == "superlative"
    assert query.entity == "team"
    assert query.metric == "xg"


# ---------------------------------------------------------------------------
# Tests: Execution
# ---------------------------------------------------------------------------


def test_structured_execution():
    """Structured query should return numeric result."""
    result = router.route_and_execute("How many goals did Messi score?")
    assert result.route.path == "structured"
    assert result.structured_result is not None
    assert result.structured_result.status == "resolved"
    assert result.structured_result.aggregated_value == 7


def test_structured_superlative_execution():
    """Superlative query should return top player."""
    result = router.route_and_execute("Who scored the most goals?")
    assert result.route.path == "structured"
    assert result.structured_result is not None
    assert result.structured_result.status == "resolved"
    assert result.structured_result.aggregated_value is not None


def test_semantic_execution():
    """Semantic query should return chunks."""
    result = router.route_and_execute("How did France play in the final?")
    assert result.route.path == "semantic"
    assert result.semantic_chunks is not None
    assert len(result.semantic_chunks) > 0


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


def test_ambiguous_query():
    """Ambiguous queries should default to semantic."""
    route = router.route_query("Tell me about the tournament")
    assert route.path in ("semantic", "hybrid")


def test_unknown_metric():
    """Unknown metric should return a StructuredQuery so resolver can report 'Unknown metric'."""
    query = router.parse_structured_query("How many widgets did Messi make?")
    assert query is not None
    assert query.metric == "widgets"
    assert query.entity_name == "Messi Make"


# ---------------------------------------------------------------------------
# Tests: Comparison Detection and Routing
# ---------------------------------------------------------------------------


def test_comparison_entity_extraction():
    """_detect_comparison() must return full multi-word entity names."""
    cases = [
        ("Compare Messi and Mbappé's performance", ["Messi", "Mbappé"]),
        ("Messi vs Mbappé", ["Messi", "Mbappé"]),
        ("Who was better, Lionel Messi or Kylian Mbappé?", ["Lionel Messi", "Kylian Mbappé"]),
        ("Compare Argentina and France", ["Argentina", "France"]),
        ("What is the difference between Lionel Messi and Julián Álvarez?", ["Lionel Messi", "Julián Álvarez"]),
        ("Compare Messi and Mbappé in goals, assists, xG, shots, and minutes", ["Messi", "Mbappé"]),
    ]
    for question, expected in cases:
        result = router._detect_comparison(question)
        assert result == expected, f"_detect_comparison({question!r}) = {result}, expected {expected}"


def test_comparison_classification_hybrid():
    """All six comparison questions must classify as hybrid."""
    questions = [
        "Compare Messi and Mbappé's performance",
        "Messi vs Mbappé",
        "Who was better, Lionel Messi or Kylian Mbappé?",
        "Compare Argentina and France",
        "What is the difference between Lionel Messi and Julián Álvarez?",
        "Compare Messi and Mbappé in goals, assists, xG, shots, and minutes",
    ]
    for question in questions:
        classification, confidence = router.classify_query(question)
        assert classification == "hybrid", f"classify_query({question!r}) = ({classification}, {confidence}), expected hybrid"


def test_comparison_routing_hybrid():
    """All six comparison questions must route to hybrid."""
    questions = [
        "Compare Messi and Mbappé's performance",
        "Messi vs Mbappé",
        "Who was better, Lionel Messi or Kylian Mbappé?",
        "Compare Argentina and France",
        "What is the difference between Lionel Messi and Julián Álvarez?",
        "Compare Messi and Mbappé in goals, assists, xG, shots, and minutes",
    ]
    for question in questions:
        route = router.route_query(question)
        assert route.path == "hybrid", f"route_query({question!r}).path = {route.path}, expected hybrid"



# ---------------------------------------------------------------------------
# Tests: Retrieval post-processing regressions
# ---------------------------------------------------------------------------


def test_team_style_detection_supports_passing_patterns():
    question = "What were France's passing patterns and most common formations?"
    assert router._detect_team_style_query(question) == "France"


def test_team_style_query_routes_to_semantic():
    question = "What were France's passing patterns and most common formations?"

    classification, confidence = router.classify_query(question)
    route = router.route_query(question)

    assert classification == "semantic"
    assert confidence == 0.9
    assert route.path == "semantic"
    assert route.semantic_query == question
    assert route.structured_query is None


def test_match_query_extracts_head_to_head_final():
    question = "What were the key events in the Argentina vs France Final?"
    assert router._detect_match_query(question) == ("Argentina", "Final")


def test_match_summary_uses_correct_final_and_preserves_first_result(monkeypatch):
    chunks = [
        {
            "chunk_id": "TEAM-779-chunk-0",
            "level": "team",
            "team_name": "Argentina",
            "metadata": {"team_name": "Argentina"},
        },
        {
            "chunk_id": "L1-match-3869684-chunk-0",
            "document_id": "L1-match-3869684",
            "level": "1",
            "text": "The 3rd Place Final between Croatia and Morocco.",
            "metadata": {"match_id": 3869684},
        },
        {
            "chunk_id": "L1-match-3869685-chunk-0",
            "document_id": "L1-match-3869685",
            "level": "1",
            "text": "The Final between Argentina and France.",
            "metadata": {"match_id": 3869685},
        },
    ]
    results = [
        {
            "chunk_id": "L2-match-3869685-chunk-0",
            "text": "Final key events.",
            "metadata": {"document_id": "L2-match-3869685", "level": "2"},
        },
        {
            "chunk_id": "other-1",
            "text": "Other result.",
            "metadata": {"document_id": "other-1", "level": "2"},
        },
        {
            "chunk_id": "other-2",
            "text": "Other result.",
            "metadata": {"document_id": "other-2", "level": "2"},
        },
    ]

    monkeypatch.setattr(router, "_load_chunks", lambda path=None: chunks)

    boosted = router._ensure_match_summary(
        "What were the key events in the Argentina vs France Final?",
        results,
        k=3,
    )

    assert boosted[0]["chunk_id"] == "L2-match-3869685-chunk-0"
    assert boosted[2]["chunk_id"] == "L1-match-3869685-chunk-0"
    assert all(
        item["chunk_id"] != "L1-match-3869684-chunk-0"
        for item in boosted
    )


def test_match_summary_skips_player_performance(monkeypatch):
    chunks = [
        {
            "chunk_id": "TEAM-779-chunk-0",
            "level": "team",
            "team_name": "Argentina",
            "metadata": {"team_name": "Argentina"},
        },
        {
            "chunk_id": "L1-match-3869519-chunk-0",
            "level": "1",
            "text": "The Semi-finals between Argentina and Croatia.",
            "metadata": {"match_id": 3869519},
        },
    ]
    results = [
        {
            "chunk_id": "L3-match-3869519-player-5503-chunk-0",
            "text": "Messi performance.",
            "metadata": {
                "document_id": "L3-match-3869519-player-5503",
                "level": "3",
            },
        }
    ]

    monkeypatch.setattr(router, "_load_chunks", lambda path=None: chunks)

    assert router._ensure_match_summary(
        "How did Messi perform against Croatia in the semi-final?",
        results,
        k=3,
    ) == results


def test_match_summary_skips_tournament_journey(monkeypatch):
    chunks = [
        {
            "chunk_id": "TEAM-788-chunk-0",
            "level": "team",
            "team_name": "Morocco",
            "metadata": {"team_name": "Morocco"},
        }
    ]
    results = [
        {
            "chunk_id": "TEAM-788-chunk-0",
            "text": "Morocco tournament analysis.",
            "metadata": {"document_id": "TEAM-788", "level": "team"},
        }
    ]

    monkeypatch.setattr(router, "_load_chunks", lambda path=None: chunks)

    assert router._ensure_match_summary(
        "How did Morocco reach the semi-finals, and what style did they use?",
        results,
        k=5,
    ) == results


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_structured_classification,
        test_semantic_classification,
        test_superlative_classification,
        test_which_team_classification,
        test_parse_numeric,
        test_parse_superlative,
        test_parse_which_team,
        test_structured_execution,
        test_structured_superlative_execution,
        test_semantic_execution,
        test_ambiguous_query,
        test_unknown_metric,
        test_comparison_entity_extraction,
        test_comparison_classification_hybrid,
        test_comparison_routing_hybrid,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    print("Running router unit tests...\n")
    failures = run_all_tests()
    raise SystemExit(1 if failures else 0)
