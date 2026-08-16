"""
06_retrieve_context.py — Temporary compatibility wrapper (Structural
Cleanup Phase A)

The retrieval and query-routing responsibilities previously combined in
this file have moved to:

    src/retrieval/search.py  -- BM25, dense, hybrid search, RRF, reranking,
                                 retrieval-side entity/team/match boosting,
                                 context building
    src/query/router.py      -- query classification, structured-query
                                 parsing, route selection, comparison
                                 routing detection, route execution

This file re-exports the public symbols existing callers (chat.py,
07_prompting.py, tests, this file's own CLI) still reach through the
"06_retrieve_context" module name, and keeps the CLI entry point (`python
06_retrieve_context.py "<query>"`) working unchanged. It is a temporary
compatibility layer -- direct imports will migrate to the new modules in a
later phase, after which this wrapper is removed.

No behavior changes: every re-exported name is the exact same object
(function/class/constant) defined in the new modules.
"""

from __future__ import annotations

from src.artifacts import ArtifactPaths

from src.retrieval.search import (
    INDICES_DIR,
    CHUNKS_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RRF_K,
    bm25_search,
    dense_search,
    reciprocal_rank_fusion,
    rerank,
    hybrid_search,
    semantic_search,
    build_context,
    retrieve_context,
    _load_bm25_index,
    _load_chunks,
    _detect_team_style_query,
    _detect_match_query,
    _ensure_match_summary,
    _expand_query_entity_siblings,
)

from src.query.router import (
    Route,
    RoutedResult,
    StructuredQuery,
    ComparisonResult,
    ComparisonValue,
    StageTaxonomy,
    classify_query,
    parse_structured_query,
    route_query,
    execute_route,
    route_and_execute,
    structured_resolve,
    assess_answerability,
    _detect_comparison,
    _load_active_stage_taxonomy,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
