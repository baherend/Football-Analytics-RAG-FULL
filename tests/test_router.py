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
