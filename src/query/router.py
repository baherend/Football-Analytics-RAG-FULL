"""
src/query/router.py -- Query execution, coordination, and the query package's
compatibility boundary.

Migration Step 3 (Query Understanding + Planning Split): this module used to
contain intent classification, structured-query parsing, filter extraction,
stage-taxonomy loading, route selection, and route execution together. The
understanding and planning halves are now split into focused sibling modules:

    src/query/intent.py    -- classification + comparison understanding
    src/query/parsing.py   -- StructuredQuery parsing, filters, stage vocabulary
    src/query/planning.py  -- Route (the plan) + route_query() strategy selection

What stays here is *execution* and *coordination*: running the structured
resolver and/or retrieval for a chosen Route, assembling RoutedResult, the
one-step route_and_execute() convenience, and the debug CLI.

This module also re-exports the extracted symbols so existing callers
(`chat.py`'s `router_mod.<name>`, `07_prompting.py`, `tests/`) keep working
unchanged -- it is a **transitional compatibility boundary, not final
architecture**, in the same sense as `src/retrieval/search.py` (see
PROJECT_MEMORY.md's Architecture Decisions for both, including the removal
plan).

Dependency direction is strictly one-way -- router.py -> planning.py ->
parsing.py + intent.py -- with no cycle and no lazy-import-back trick. That
is possible because the names tests monkeypatch on this module
(`structured_resolve`, `hybrid_search`, `assess_answerability`, plus
`route_query`/`execute_route` for `route_and_execute()`/`main()`) are all
either imported into *this* module and used by `execute_route()`, which stays
here, or resolved through this module's own globals by functions defined
here. Nothing that moved out carried a monkeypatch contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.artifacts import ArtifactPaths
# Context Engineering (Migration Step 4): the evidence subset shown
# downstream, and whether it suffices, now come from src/context/ rather than
# from inside the retrieval package.
from src.context.answerability import (
    AnswerabilityAssessment,
    assess_answerability,
)
from src.context.evidence import EvidencePack
from src.context.rendering import build_context
from src.retrieval.search import hybrid_search
# Phase 5: the team-style classifier moved out of retrieval into the neutral
# src/team_style.py. This module keeps importing hybrid_search from retrieval --
# that edge is execution calling the RETRIEVE stage, which follows the runtime
# flow -- but it no longer reaches into retrieval for classification.
from src.team_style import _detect_team_style_query
from src.query.query_schema import (
    StructuredQuery, StructuredResult, Filter, ComparisonResult, ComparisonValue,
)
from src.query.resolver import resolve as structured_resolve
from src.query.resolver import resolve_entity_type
from src.query.vocab import resolve_metric, resolve_aggregation, METRIC_SYNONYMS, AGGREGATION_SYNONYMS
from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY
from src.extraction.match_facts import WC2022_DATASET_IDENTITY

# ---------------------------------------------------------------------------
# Compatibility re-exports -- see module docstring. Existing callers
# (chat.py, 07_prompting.py, tests/) reach these through this module.
# ---------------------------------------------------------------------------

from src.query.intent import (
    COMPARISON_PATTERNS,
    SEMANTIC_KEYWORDS,
    SEMANTIC_PATTERNS,
    STRUCTURED_KEYWORDS,
    STRUCTURED_PATTERNS,
    _detect_comparison,
    _detect_comparison_metric,
    _resolve_comparison_entity_type,
    classify_query,
)
from src.query.parsing import (
    OPPONENT_PATTERNS,
    STAGE_PATTERNS,
    _extract_opponent_filter,
    _extract_stage_filter,
    _load_active_stage_taxonomy,
    parse_structured_query,
)
from src.query.planning import Route, route_query

__all__ = [
    "Route",
    "RoutedResult",
    "classify_query",
    "parse_structured_query",
    "route_query",
    "execute_route",
    "route_and_execute",
    "main",
]


@dataclass
class RoutedResult:
    """Result from routed execution."""
    route: Route
    structured_result: StructuredResult | None = None
    semantic_chunks: list[dict] | None = None
    answerability: AnswerabilityAssessment | None = None
    context: str = ""
    explanation: str = ""
    # Migration Step 4: the typed Context Engineering handoff for this result.
    # `semantic_chunks` remains the raw list every existing consumer already
    # reads (citations in 07_prompting.py, prompt text in chat.py) and stays
    # byte-identical -- `evidence.to_chunks()` returns those very objects.
    # The pack is additive: it carries the same evidence with guaranteed
    # provenance (chunk_id/document_id per item) and entity coverage.
    evidence: EvidencePack | None = None


def execute_route(
    route: Route,
    semantic_k: int = 3,
    original_query: str = "",
    artifact_paths: ArtifactPaths | None = None,
) -> RoutedResult:
    """
    Execute a routed query, returning structured and/or semantic results.

    `artifact_paths` selects a namespaced dataset (see src/artifacts.py):
    the structured path resolves against artifact_paths.match_facts and
    every semantic/hybrid retrieval call uses artifact_paths' BM25/chunks/
    Chroma artifacts. Defaults to None -- unchanged legacy WC2022 behavior.
    """
    structured_result = None
    semantic_chunks = None
    context = ""
    match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None

    dependency_query = getattr(route, "dependency_query", None)
    dependency_phrase = getattr(route, "dependency_phrase", None)
    dependency_resolved = dependency_query is None
    semantic_execution_query = route.semantic_query or ""

    stage_taxonomy = (
        _load_active_stage_taxonomy(match_facts_path)
        if route.path in ("structured", "hybrid")
        else None
    )

    # Compositional hybrid is sequential:
    # structured dependency -> authoritative entity -> semantic retrieval.
    if route.path == "hybrid" and dependency_query is not None:
        try:
            structured_result = structured_resolve(
                dependency_query,
                data_path=match_facts_path,
                stage_taxonomy=stage_taxonomy,
            )
        except Exception as e:
            print(f"Structured dependency resolution failed: {e}")
            structured_result = None

        if (
            structured_result is not None
            and structured_result.status in ("resolved", "partial")
            and structured_result.data
        ):
            top = structured_result.data[0]
            entity_key = (
                "team_name"
                if dependency_query.entity == "team"
                else "player_name"
            )
            resolved_entity = top.get(entity_key)

            if resolved_entity:
                dependency_resolved = True
                if dependency_phrase:
                    semantic_execution_query = semantic_execution_query.replace(
                        dependency_phrase,
                        str(resolved_entity),
                        1,
                    )
                context = structured_result.explanation or ""

    # Ordinary hybrid comparisons remain unchanged.
    comparison_entities = _detect_comparison(route.semantic_query or "")
    if (
        route.path == "hybrid"
        and dependency_query is None
        and comparison_entities
    ):
        comparison_metric = _detect_comparison_metric(route.semantic_query or "")
        comparison_entity_type = _resolve_comparison_entity_type(
            comparison_entities, data_path=match_facts_path,
        )
        entity_results = []
        for entity_name in comparison_entities:
            sq = StructuredQuery(intent="numeric", entity=comparison_entity_type, metric=comparison_metric,
                                 aggregation="sum", entity_name=entity_name)
            try:
                result = structured_resolve(
                    sq,
                    data_path=match_facts_path,
                    stage_taxonomy=stage_taxonomy,
                )
                entity_results.append((entity_name, result))
            except Exception as e:
                print(f"Structured resolution failed for {entity_name}: {e}")

        if entity_results:
            parts = []
            values = []
            for name, result in entity_results:
                if result.status in ("resolved", "partial"):
                    parts.append(f"{name}: {result.explanation}")
                else:
                    parts.append(f"{name}: No data available")
                # Preserve the already-computed aggregated_value verbatim --
                # never reparsed from result.explanation. `None` for an
                # unresolved entity carries the same meaning it already has
                # on StructuredResult.aggregated_value.
                values.append(ComparisonValue(entity_name=name, value=result.aggregated_value))

            # Derive the combined status from the two original StructuredResult
            # objects -- never a hardcoded "resolved" -- mirroring the
            # resolver's own resolved/partial/empty meaning (see
            # src/query/resolver.py): "resolved" only when every requested
            # entity produced an unqualified value; "partial" when at least
            # one usable value exists but the comparison isn't fully honored
            # (an entity is missing, or any entity's own result was itself
            # "partial"); "empty" when no entity produced a usable value.
            usable_count = sum(1 for _, result in entity_results if result.aggregated_value is not None)
            has_partial_result = any(result.status == "partial" for _, result in entity_results)
            if usable_count == 0:
                comparison_status = "empty"
            elif usable_count < len(comparison_entities) or has_partial_result:
                comparison_status = "partial"
            else:
                comparison_status = "resolved"

            structured_result = ComparisonResult(
                status=comparison_status,
                metric=comparison_metric,
                values=values,
                explanation=" | ".join(parts),
            )

    elif route.path in ("structured", "hybrid") and route.structured_query:
        try:
            structured_result = structured_resolve(
                route.structured_query,
                data_path=match_facts_path,
                stage_taxonomy=stage_taxonomy,
            )
        except Exception as e:
            print(f"Structured resolution failed: {e}")

        if structured_result is not None and structured_result.status in ("resolved", "partial"):
            context = structured_result.explanation or ""
            # Aggression-proxy disclaimer: if the query was about "aggression" but
            # we resolved via fouls_committed, prepend a disclaimer that there's
            # no direct aggression metric.
            if structured_result.query and structured_result.query.metric == "fouls_committed" and \
               re.search(r"aggress|tough|dirtiest", original_query, re.IGNORECASE):
                context = (
                    "NOTE: There is no direct 'aggression' metric in the dataset. "
                    "The closest available signal is fouls committed and card counts. "
                    "Presenting fouls committed data as a proxy — this does not definitively "
                    "measure playing style or intent.\n\n" + context
                )
        elif structured_result is not None and structured_result.status == "empty":
            try:
                fallback_chunks = hybrid_search(
                    route.semantic_query or route.structured_query.entity_name or "",
                    k=semantic_k, artifact_paths=artifact_paths,
                )
                if fallback_chunks:
                    fallback_context = (
                        "NOTE: The structured data did not contain a direct answer to this question. "
                        "The following context is from related documents and may or may not be relevant. "
                        "Only answer if the context clearly and specifically addresses the original question. "
                        "If the context does not clearly answer the question, state that the data does not "
                        "contain a direct answer.\n\n"
                    )
                    context = fallback_context + build_context(fallback_chunks)
                    semantic_chunks = fallback_chunks
            except Exception as e:
                print(f"Semantic fallback failed: {e}")

    if route.path in ("semantic", "hybrid") and dependency_resolved:
        try:
            semantic_chunks = hybrid_search(
                semantic_execution_query,
                k=semantic_k,
                artifact_paths=artifact_paths,
            )
            semantic_context = build_context(semantic_chunks)
            if context and semantic_context:
                context = context + "\n\n" + semantic_context
            elif semantic_context:
                context = semantic_context
        except Exception as e:
            print(f"Semantic search failed: {e}")

    # SELECT EVIDENCE -> ANSWERABILITY handoff. The pack wraps exactly the
    # chunks retrieval selected -- to_chunks() hands back those same objects,
    # so assess_answerability() sees byte-identical input to before.
    evidence = None
    answerability = None
    if semantic_chunks is not None:
        answerability_query = (
            original_query
            or route.semantic_query
            or ""
        )
        evidence = EvidencePack.from_chunks(answerability_query, semantic_chunks)
        answerability = assess_answerability(
            query=answerability_query,
            chunks=evidence.to_chunks(),
        )

    explanation = f"Routed to {route.path} path (confidence: {route.confidence:.2f}). "
    if structured_result:
        explanation += f"Structured: {structured_result.explanation} "
    if semantic_chunks:
        explanation += f"Semantic: {len(semantic_chunks)} chunks retrieved."

    return RoutedResult(
        route=route,
        structured_result=structured_result,
        semantic_chunks=semantic_chunks,
        answerability=answerability,
        context=context,
        explanation=explanation,
        evidence=evidence,
    )


def route_and_execute(
    query: str, semantic_k: int = 3, artifact_paths: ArtifactPaths | None = None,
) -> RoutedResult:
    """
    Route and execute a query in one step.

    `artifact_paths` selects a namespaced dataset (see src/artifacts.py)
    and is threaded through both routing/parsing and execution. Defaults to
    None -- unchanged legacy WC2022 behavior.

    Calls route_query()/execute_route() as bare names so they resolve through
    this module's own globals -- tests monkeypatch them here (see
    tests/test_router.py and tests/test_artifact_paths.py's CLI stubs).
    """
    route = route_query(query, artifact_paths=artifact_paths)
    return execute_route(route, semantic_k, original_query=query, artifact_paths=artifact_paths)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Structural Cleanup Phase B: relocated from 06_retrieve_context.py (now
# deleted) -- this is the same tested debug/inspection entry point
# (`python -m src.query.router "<query>"`), not new functionality. It
# exercises exactly route_query()/execute_route() above.


def main() -> int:
    import argparse

    from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

    parser = argparse.ArgumentParser(description="Query routing and retrieval")
    parser.add_argument("query", help="User query")
    parser.add_argument("--k", type=int, default=5, help="Number of semantic results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    args = parser.parse_args()

    is_legacy_wc2022 = (args.competition_id, args.season_id) == (COMPETITION_ID, SEASON_ID)
    if is_legacy_wc2022 and not args.namespaced:
        artifact_paths = None
    else:
        artifact_paths = ArtifactPaths(args.competition_id, args.season_id)

    print(f"Query: {args.query}")
    print()

    route = route_query(args.query, artifact_paths=artifact_paths)
    print(f"Route: {route.path} (confidence: {route.confidence:.2f})")
    print(f"Reason: {route.reason}")

    if args.verbose:
        if route.structured_query:
            print(f"Structured query: {route.structured_query.to_dict()}")
        if route.semantic_query:
            print(f"Semantic query: {route.semantic_query}")
    print()

    result = execute_route(route, args.k, artifact_paths=artifact_paths)
    print(result.explanation)

    if result.structured_result:
        print(f"\nStructured result:")
        print(f"  Status: {result.structured_result.status}")
        if result.structured_result.aggregated_value is not None:
            print(f"  Value: {result.structured_result.aggregated_value}")
        print(f"  Explanation: {result.structured_result.explanation}")

    if result.semantic_chunks:
        print(f"\nSemantic results ({len(result.semantic_chunks)} chunks):")
        for i, chunk in enumerate(result.semantic_chunks[:3]):
            score = chunk.get('rrf_score', chunk.get('score', 0))
            print(f"  {i+1}. [{chunk['metadata'].get('level')}] Score: {score:.4f}")
            print(f"     {chunk['text'][:100]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
