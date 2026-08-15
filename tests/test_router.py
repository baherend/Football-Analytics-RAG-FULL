"""
test_router.py — Phase 5: Router Unit Tests

Tests routing decisions and execution.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_comparison_entity_extraction_who_scored_more_phrasing():
    """
    Comparison Engine audit (Step 1): _detect_comparison() only recognizes
    "compare X and Y", "X vs Y", "who was/is better X or Y", and
    "difference between X and Y" phrasing (see
    test_comparison_entity_extraction above). A natural comparison question
    phrased as "who <verb> more <metric>, A or B?" -- e.g. the question a
    user would actually ask to compare two players' goal totals -- is not
    recognized at all, so both entities are silently lost before any
    structured resolution is attempted. This is the same function
    classify_query() and execute_route() both rely on for comparison
    intent, so this single gap blocks comparison end-to-end for this
    phrasing regardless of downstream structured/aggregation behavior.

    "Harry Kane"/"Jamie Vardy" are used as realistic fixture names from the
    EPL 2015/16 portability dataset (competition_id=2, season_id=27) --
    this test exercises only the generic entity-extraction contract, not
    any player-specific production logic.
    """
    question = "Who scored more goals, Harry Kane or Jamie Vardy?"
    result = router._detect_comparison(question)
    assert result == ["Harry Kane", "Jamie Vardy"], (
        f"_detect_comparison({question!r}) = {result}, expected both full "
        "player names to be preserved so a structured comparison can be "
        "attempted for either entity."
    )


def test_comparison_preserves_requested_metric(monkeypatch):
    """
    Comparison Engine Step 2B: execute_route()'s comparison branch
    hardcodes metric="goals" for BOTH entities regardless of what the
    query actually asks about (06_retrieve_context.py, execute_route(),
    "for hybrid comparison queries, run structured queries for each
    entity"). A query explicitly comparing "assists" must send
    metric="assists" to both structured_resolve() calls, not "goals".
    """
    captured = []

    def fake_structured_resolve(query, data_path=None, stage_taxonomy=None):
        captured.append((query.entity_name, query.metric))
        return SimpleNamespace(
            status="resolved",
            explanation=f"{query.entity_name}: ok",
            aggregated_value=1,
            data=[],
            dropped_filters=[],
        )

    monkeypatch.setattr(router, "structured_resolve", fake_structured_resolve)

    route = router.Route(
        path="hybrid",
        confidence=0.9,
        reason="test",
        semantic_query="Who had more assists, Alpha Player or Beta Player?",
    )
    router.execute_route(route, semantic_k=0)

    assert captured == [
        ("Alpha Player", "assists"),
        ("Beta Player", "assists"),
    ], (
        f"execute_route() sent {captured} to structured_resolve() for an "
        "assists comparison -- expected both entities to receive the "
        "requested metric 'assists', not the hardcoded 'goals'."
    )


def test_comparison_unsupported_metric_does_not_silently_become_goals(monkeypatch):
    """
    Comparison Engine Step 2B safety check: the existing single-entity
    structured path already has an established contract for an explicitly
    requested but unsupported metric -- parse_structured_query() passes
    the raw, unresolved metric text through unchanged to StructuredQuery
    (see the "how many <metric> did <player> have" pattern's else branch),
    and structured_resolve() naturally rejects it via validate_query(),
    producing StructuredResult(status="empty", explanation="Unknown
    metric: <raw text>"). Verified live: "How many corners did Messi
    have?" -> StructuredResult(status='empty', explanation='Unknown
    metric: corners'). "corners" is confirmed absent from both
    METRIC_SYNONYMS and ALL_METRICS via resolve_metric("corners") is None.

    The comparison branch must mirror this exact contract, not fall back
    to its "no metric mentioned" default of "goals" -- an explicitly
    requested but unsupported metric must never silently become goals.
    """
    from src.query.vocab import resolve_metric
    assert resolve_metric("corners") is None, (
        "test fixture assumption broken: 'corners' must be an unsupported "
        "metric for this test to prove anything"
    )

    captured = []

    def fake_structured_resolve(query, data_path=None, stage_taxonomy=None):
        captured.append((query.entity_name, query.metric))
        return SimpleNamespace(
            status="resolved",
            explanation=f"{query.entity_name}: ok",
            aggregated_value=1,
            data=[],
            dropped_filters=[],
        )

    monkeypatch.setattr(router, "structured_resolve", fake_structured_resolve)

    route = router.Route(
        path="hybrid",
        confidence=0.9,
        reason="test",
        semantic_query="Who had more corners, Alpha Player or Beta Player?",
    )
    router.execute_route(route, semantic_k=0)

    assert captured == [
        ("Alpha Player", "corners"),
        ("Beta Player", "corners"),
    ], (
        f"execute_route() sent {captured} to structured_resolve() for an "
        "explicitly-requested but unsupported metric ('corners') -- it "
        "must be passed through unresolved, mirroring "
        "parse_structured_query()'s existing unknown-metric contract, "
        "not silently replaced with 'goals'."
    )


def test_comparison_entity_extraction_excludes_trailing_metric_clause():
    """
    Comparison Engine Step 2C: _detect_comparison()'s first pattern
    ("compare X and Y ...") only stops entity B's lazy capture at " in",
    end-of-string, or "?" (see COMPARISON_PATTERNS[0] and the existing "in
    goals, assists, ..." handling below it). A "by <metric>" trailing
    clause -- e.g. "Compare Harry Kane and Jamie Vardy by goals." -- has
    no terminator the pattern recognizes, so the lazy group swallows the
    whole trailing clause into entity B. Entity extraction must stay
    independent of metric extraction (_detect_comparison_metric() already
    correctly identifies "goals" for this query on its own).
    """
    question = "Compare Harry Kane and Jamie Vardy by goals."
    result = router._detect_comparison(question)
    assert result == ["Harry Kane", "Jamie Vardy"], (
        f"_detect_comparison({question!r}) = {result}, expected the trailing "
        "'by goals' metric clause to be excluded from entity B."
    )


def test_comparison_result_preserves_both_authoritative_values(monkeypatch):
    """
    Comparison Engine Step 2D/2E: execute_route()'s comparison branch
    already resolves a real, correct StructuredResult per entity (each
    with its own aggregated_value) -- Step 2D proved these were then
    discarded except for `.explanation`, flattened into a single prose
    string on the ad-hoc _CombinedResult (aggregated_value=None, data=[],
    verified live against real EPL artifacts). Step 2E replaces that with
    ComparisonResult (src/query/query_schema.py), whose `values` field
    preserves each entity's aggregated_value as ComparisonValue records --
    not overloading StructuredResult's single-query/single-value fields.

    This test proves both authoritative values, their entity identities,
    and the resolved metric are recoverable from structured (non-string)
    fields -- never by re-parsing `.explanation` prose.
    """
    def fake_structured_resolve(query, data_path=None, stage_taxonomy=None):
        value = 25 if query.entity_name == "Harry Kane" else 24
        return SimpleNamespace(
            status="resolved",
            explanation=f"{query.entity_name}'s total goals is {value}.",
            aggregated_value=value,
            data=[],
            dropped_filters=[],
        )

    monkeypatch.setattr(router, "structured_resolve", fake_structured_resolve)

    route = router.Route(
        path="hybrid",
        confidence=0.9,
        reason="test",
        semantic_query="Who scored more goals, Harry Kane or Jamie Vardy?",
    )
    result = router.execute_route(route, semantic_k=0)
    sr = result.structured_result

    # Both authoritative values, their entity identities.
    recovered = {v.entity_name: v.value for v in sr.values}
    assert recovered == {"Harry Kane": 25, "Jamie Vardy": 24}, (
        f"comparison structured_result.values = {sr.values!r} -- both authoritative "
        "entity values, keyed by entity identity, must be recoverable as structured "
        f"data, not only flattened into explanation text ({sr.explanation!r})."
    )

    # Metric preserved structurally (not re-derived from explanation text).
    assert sr.metric == "goals", (
        f"comparison structured_result.metric = {sr.metric!r}, expected 'goals' to be "
        "preserved structurally alongside the two values."
    )


def test_comparison_result_computes_deterministic_outcome(monkeypatch):
    """
    Comparison Engine Step 2F: ComparisonResult already preserves both
    entities' authoritative aggregated_value structurally (Step 2E,
    `.values`), but does not yet expose a deterministic difference or a
    machine-readable winner/tie outcome -- neither field exists yet, so a
    caller would have to recompute it from `.values` itself (or, worse,
    ask an LLM to eyeball two numbers in a prompt). The comparison result
    itself must carry this deterministic outcome, computed only from the
    already-resolved structured numeric values -- never from explanation
    prose, never via generation.
    """
    def fake_structured_resolve(query, data_path=None, stage_taxonomy=None):
        value = 25 if query.entity_name == "Harry Kane" else 24
        return SimpleNamespace(
            status="resolved",
            explanation=f"{query.entity_name}'s total goals is {value}.",
            aggregated_value=value,
            data=[],
            dropped_filters=[],
        )

    monkeypatch.setattr(router, "structured_resolve", fake_structured_resolve)

    route = router.Route(
        path="hybrid",
        confidence=0.9,
        reason="test",
        semantic_query="Who scored more goals, Harry Kane or Jamie Vardy?",
    )
    result = router.execute_route(route, semantic_k=0)
    sr = result.structured_result

    assert sr.difference == 1, (
        f"comparison structured_result.difference = {sr.difference!r}, expected the "
        "non-negative magnitude |25 - 24| = 1, computed only from the already-"
        "resolved structured values."
    )
    assert sr.outcome == "entity_a_higher", (
        f"comparison structured_result.outcome = {sr.outcome!r}, expected "
        "'entity_a_higher' since Harry Kane (values[0] = 25) is greater than "
        "Jamie Vardy (values[1] = 24)."
    )


def test_comparison_result_outcome_safety_cases():
    """
    Comparison Engine Step 2F regression: covers the outcome-derivation
    safety cases in one pass, constructing ComparisonResult directly
    (the same construction execute_route() already performs from
    structured_resolve() output) since the derivation itself is pure
    logic over already-resolved values -- no execute_route()/monkeypatch
    machinery needed to exercise it further per case.

    - A > B: guards the "first entity always wins" degenerate case.
    - B > A: proves direction isn't hardcoded to entity A.
    - A == B: tie, difference = 0.
    - A present, B missing: no fabricated winner, no invalid None
      comparison -- both difference and outcome stay None.
    """
    cases = [
        (25, 24, 1, "entity_a_higher"),
        (24, 25, 1, "entity_b_higher"),
        (10, 10, 0, "tie"),
        (10, None, None, None),
    ]
    for value_a, value_b, expected_difference, expected_outcome in cases:
        result = router.ComparisonResult(
            status="resolved",
            metric="goals",
            values=[
                router.ComparisonValue(entity_name="Entity A", value=value_a),
                router.ComparisonValue(entity_name="Entity B", value=value_b),
            ],
            explanation="Entity A: ... | Entity B: ...",
        )
        assert result.difference == expected_difference, (
            f"values=({value_a!r}, {value_b!r}): difference = {result.difference!r}, "
            f"expected {expected_difference!r}"
        )
        assert result.outcome == expected_outcome, (
            f"values=({value_a!r}, {value_b!r}): outcome = {result.outcome!r}, "
            f"expected {expected_outcome!r}"
        )


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
        test_comparison_entity_extraction_who_scored_more_phrasing,
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

def test_parse_structured_query_recognizes_active_taxonomy_stage_without_global_alias():
    taxonomy = router.StageTaxonomy.discover(
        stages=["League Phase"],
        knockout_stages=[],
        group_stages=[],
    )

    query = router.parse_structured_query(
        "How many goals did Messi score in the league phase?",
        stage_taxonomy=taxonomy,
    )

    assert query is not None
    assert query.entity_name == "Messi"
    assert any(
        f.dimension == "stage"
        and f.operator == "eq"
        and f.value == "League Phase"
        for f in query.filters
    )

def test_route_query_uses_selected_dataset_taxonomy_for_stage_parsing(tmp_path):
    import json
    from src.artifacts import ArtifactPaths
    from src.stage_taxonomy import StageTaxonomy

    paths = ArtifactPaths(competition_id=7, season_id=11, output_root=tmp_path)
    taxonomy = StageTaxonomy.discover(
        stages=["League Phase"],
        knockout_stages=[],
        group_stages=[],
    )

    paths.match_facts.parent.mkdir(parents=True, exist_ok=True)
    paths.match_facts.write_text(json.dumps({
        "metadata": {
            "competition_id": 7,
            "competition_name": "Synthetic League",
            "season_id": 11,
            "season_name": "2025",
            "stage_taxonomy": taxonomy.to_dict(),
        },
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }), encoding="utf-8")

    route = router.route_query(
        "How many goals did Messi score in the league phase?",
        artifact_paths=paths,
    )

    assert route.structured_query is not None
    assert any(
        f.dimension == "stage"
        and f.operator == "eq"
        and f.value == "League Phase"
        for f in route.structured_query.filters
    )

def test_route_and_execute_threads_selected_artifact_paths_to_routing(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.artifacts import ArtifactPaths

    selected = ArtifactPaths(competition_id=7, season_id=11, output_root=tmp_path)
    captured = {}

    fake_route = SimpleNamespace(
        path="semantic",
        confidence=1.0,
        reason="test",
        semantic_query="test question",
        structured_query=None,
    )

    def fake_route_query(query, artifact_paths=None):
        captured["route_artifact_paths"] = artifact_paths
        return fake_route

    def fake_execute_route(route, semantic_k=3, original_query="", artifact_paths=None):
        captured["execute_artifact_paths"] = artifact_paths
        return SimpleNamespace(route=route)

    monkeypatch.setattr(router, "route_query", fake_route_query)
    monkeypatch.setattr(router, "execute_route", fake_execute_route)

    router.route_and_execute("test question", artifact_paths=selected)

    assert captured["route_artifact_paths"] is selected
    assert captured["execute_artifact_paths"] is selected

def test_custom_taxonomy_does_not_inherit_wc2022_stage_aliases():
    taxonomy = router.StageTaxonomy.discover(
        stages=["League Phase"],
        knockout_stages=[],
        group_stages=[],
    )

    query = router.parse_structured_query(
        "How many goals did Messi score in the final?",
        stage_taxonomy=taxonomy,
    )

    assert query is not None
    assert not any(f.dimension == "stage" for f in query.filters)


def test_knockout_filter_remains_semantic_for_custom_taxonomy():
    taxonomy = router.StageTaxonomy.discover(
        stages=["League Phase"],
        knockout_stages=[],
        group_stages=[],
    )

    query = router.parse_structured_query(
        "How many goals did Messi score in the knockout?",
        stage_taxonomy=taxonomy,
    )

    assert query is not None
    assert any(
        f.dimension == "is_knockout"
        and f.operator == "eq"
        and f.value is True
        for f in query.filters
    )
