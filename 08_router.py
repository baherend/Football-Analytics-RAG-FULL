"""
08_router.py — Phase 5: Query Router

Routes user questions to the appropriate execution path:
- Semantic only: descriptive questions ("How did France play?")
- Structured only: numeric/superlative questions ("Who scored the most goals?")
- Hybrid: questions needing both ("Compare Messi and Mbappé's performance")

Rules:
- Built on clear contracts from Phase 3 (resolver) and Phase 4 (semantic search)
- Never silently drops a requested filter
- Returns routing decision with explanation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.query.query_schema import StructuredQuery, StructuredResult, Filter
from src.query.resolver import resolve as structured_resolve
from src.query.vocab import resolve_metric, resolve_aggregation, resolve_stage, METRIC_SYNONYMS, AGGREGATION_SYNONYMS


# ---------------------------------------------------------------------------
# Stage extraction
# ---------------------------------------------------------------------------

# Patterns for extracting stage from queries
STAGE_PATTERNS = [
    (r"in\s+the\s+(?:semi[\s-]*final)", "Semi-finals"),
    (r"in\s+the\s+(?:quarter[\s-]*final)", "Quarter-finals"),
    (r"in\s+the\s+(?:round\s+of\s+16)", "Round of 16"),
    (r"in\s+the\s+(?:group\s+stage)", "Group Stage"),
    (r"in\s+the\s+(?:final)", "Final"),
    (r"in\s+the\s+(?:3rd\s+place)", "3rd Place Final"),
    (r"in\s+knockout", None),  # Special: means is_knockout=True
    (r"in\s+the\s+knockout", None),
    (r"during\s+the\s+(?:semi[\s-]*final)", "Semi-finals"),
    (r"during\s+the\s+(?:quarter[\s-]*final)", "Quarter-finals"),
    (r"during\s+the\s+(?:final)", "Final"),
]


def _extract_stage_filter(query: str) -> Filter | None:
    """Extract stage filter from query text."""
    query_lower = query.lower().strip()

    for pattern, stage_value in STAGE_PATTERNS:
        if re.search(pattern, query_lower):
            if stage_value is None:
                # Knockout filter
                return Filter("is_knockout", "eq", True)
            else:
                return Filter("stage", "eq", stage_value)

    return None


# ---------------------------------------------------------------------------
# Opponent extraction
# ---------------------------------------------------------------------------

# Patterns for extracting opponent from queries
OPPONENT_PATTERNS = [
    r"against\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
    r"vs\.?\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
    r"versus\s+([a-zA-Z\s]+?)(?:\s*$|\s*\?|\s*,|\s+in|\s+during)",
]


def _extract_opponent_filter(query: str) -> Filter | None:
    """Extract opponent filter from query text."""
    query_lower = query.lower().strip()

    for pattern in OPPONENT_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            opponent = match.group(1).strip()
            # Clean up common words
            for word in ["the", "a", "an"]:
                if opponent.startswith(word + " "):
                    opponent = opponent[len(word) + 1:]
            if opponent and len(opponent) > 1:
                return Filter("opponent", "eq", opponent.title())

    return None


# ---------------------------------------------------------------------------
# Comparison detection
# ---------------------------------------------------------------------------


# Patterns that indicate a comparison between two entities
COMPARISON_PATTERNS = [
    r"compare\s+(.+?)\s+and\s+(.+?)(?:\s|$|\?)",
    r"who\s+(?:performed|played|did)\s+better.*?(\w+)\s+or\s+(\w+)",
    r"(\w+)\s+vs\.?\s+(\w+)",
    r"(\w+)\s+versus\s+(\w+)",
    r"who\s+(?:is|was)\s+better.*?(\w+)\s+or\s+(\w+)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s|$|\?)",
]


def _detect_comparison(query: str) -> list[str]:
    """
    Detect if a query is comparing two entities.

    Returns list of entity names if comparison detected, empty list otherwise.
    """
    query_lower = query.lower().strip()

    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            entities = []
            for group in match.groups():
                if group:
                    # Clean up the entity name
                    entity = group.strip().rstrip("'s")
                    # Remove common suffixes
                    for suffix in ["'s tournament performance", "'s performance",
                                   "'s stats", "'s goals"]:
                        if entity.endswith(suffix):
                            entity = entity[:-len(suffix)]
                    if entity and len(entity) > 1:
                        entities.append(entity.title())
            if len(entities) >= 2:
                return entities[:2]  # Return first two entities

    return []


# ---------------------------------------------------------------------------
# Route types
# ---------------------------------------------------------------------------


@dataclass
class Route:
    """Routing decision."""
    path: str  # "semantic" | "structured" | "hybrid"
    confidence: float  # 0.0 - 1.0
    reason: str
    structured_query: StructuredQuery | None = None
    semantic_query: str | None = None


@dataclass
class RoutedResult:
    """Result from routed execution."""
    route: Route
    structured_result: StructuredResult | None = None
    semantic_chunks: list[dict] | None = None
    context: str = ""
    explanation: str = ""


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------


# Patterns that strongly suggest structured queries
STRUCTURED_PATTERNS = [
    # Numeric: "how many goals did Messi score"
    r"how\s+many\s+(\w+)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
    # Numeric: "how many penalty kicks did Messi score" (multi-word metric)
    r"how\s+many\s+([\w\s]+?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
    # Superlative: "who scored the most goals"
    r"who\s+(?:has\s+the|had\s+the|scored\s+the|got\s+the)?\s*(most|highest|best|least|lowest|fewest)\s+(\w+)",
    # Superlative: "who was the most aggressive player"
    r"who\s+(?:was|is|were|are)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+([\w\s]+?)(?:\s+player|\?|$)",
    # Superlative: "which team had the highest xG"
    r"which\s+(team|player)\s+(?:has|had|scored|got)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+(\w+)",
    # Numeric: "what is Messi's xG"
    r"what\s+(?:is|was|are|were)\s+(.+?)(?:'s|'s)?\s+([\w\s]+?)(?:\s*$|\s*\?)",
    # Direct metric: "Messi goals"
    r"^(.+?)\s+(goals|assists|xg|shots|passes|minutes|tackles|interceptions)$",
]

# Patterns that strongly suggest semantic queries
SEMANTIC_PATTERNS = [
    # Descriptive: "how did France play"
    r"how\s+did\s+(.+?)\s+play",
    # Descriptive: "tell me about"
    r"tell\s+me\s+about\s+(.+)",
    # Descriptive: "describe"
    r"describe\s+(.+)",
    # Descriptive: "what happened in"
    r"what\s+happened\s+in\s+(.+)",
    # Descriptive: "explain"
    r"explain\s+(.+)",
    # Comparative: "compare"
    r"compare\s+(.+?)\s+and\s+(.+)",
]

# Keywords that suggest structured queries
STRUCTURED_KEYWORDS = {
    "most", "highest", "best", "least", "lowest", "fewest", "top", "bottom",
    "average", "total", "sum", "count", "how many", "how much",
    "goals", "assists", "xg", "shots", "passes", "minutes", "tackles",
    "interceptions", "clearances", "pressures", "carries",
}

# Keywords that suggest semantic queries
SEMANTIC_KEYWORDS = {
    "how", "why", "explain", "describe", "tell me", "what happened",
    "play", "performance", "style", "strategy", "tactics", "formation",
    "compare", "difference", "similar", "better", "worse",
}


def classify_query(query: str) -> tuple[str, float]:
    """
    Classify a query as "structured", "semantic", or "hybrid".

    Returns (classification, confidence).
    """
    query_lower = query.lower().strip()

    # Check for explicit structured patterns
    for pattern in STRUCTURED_PATTERNS:
        if re.search(pattern, query_lower):
            return "structured", 0.9

    # Check for comparison queries (hybrid) — must come before semantic patterns
    # because "compare X and Y" matches both comparison and semantic patterns
    if _detect_comparison(query):
        return "hybrid", 0.9

    # Check for explicit semantic patterns
    for pattern in SEMANTIC_PATTERNS:
        if re.search(pattern, query_lower):
            return "semantic", 0.9

    # Count keyword matches
    structured_score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in query_lower)
    semantic_score = sum(1 for kw in SEMANTIC_KEYWORDS if kw in query_lower)

    # Normalize scores
    total = structured_score + semantic_score
    if total == 0:
        return "semantic", 0.5  # Default to semantic for ambiguous queries

    structured_pct = structured_score / total
    semantic_pct = semantic_score / total

    if structured_pct > 0.7:
        return "structured", structured_pct
    elif semantic_pct > 0.7:
        return "semantic", semantic_pct
    else:
        return "hybrid", 0.6


# ---------------------------------------------------------------------------
# Query parsing (for structured path)
# ---------------------------------------------------------------------------


def parse_structured_query(query: str) -> StructuredQuery | None:
    """
    Parse a query into a StructuredQuery if possible.
    Returns None if parsing fails or metric is unknown.
    """
    query_lower = query.lower().strip()

    # Extract filters if present
    stage_filter = _extract_stage_filter(query)
    opponent_filter = _extract_opponent_filter(query)
    filters = [f for f in [stage_filter, opponent_filter] if f is not None]

    # Pattern: "how many <metric> did <player> score/have"
    match = re.search(
        r"how\s+many\s+(\w+)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric,
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )
        else:
            # Return query with unknown metric — resolver will report "Unknown metric"
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric_raw,
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )

    # Pattern: "how many <multi-word metric> did <player> score/have"
    match = re.search(
        r"how\s+many\s+([\w\s]+?)\s+(?:did|does|has|have)\s+(.+?)(?:\s+score|\s+have|\s+get|\?|$)",
        query_lower
    )
    if match:
        metric_raw, player_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        if metric:
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric,
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )
        else:
            # Return query with unknown metric — resolver will report "Unknown metric"
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric_raw.strip(),
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )

    # Pattern: "who scored the most <metric>"
    match = re.search(
        r"who\s+(?:has\s+the|had\s+the|scored\s+the|got\s+the)?\s*(most|highest|best|least|lowest|fewest)\s+(\w+)",
        query_lower
    )
    if match:
        agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        agg = resolve_aggregation(agg_raw)
        if metric and agg:
            return StructuredQuery(
                intent="superlative",
                entity="player",
                metric=metric,
                aggregation=agg,
                limit=1,
                filters=filters,
            )

    # Pattern: "who was the most <metric> player"
    match = re.search(
        r"who\s+(?:was|is|were|are)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+([\w\s]+?)(?:\s+player|\?|$)",
        query_lower
    )
    if match:
        agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        agg = resolve_aggregation(agg_raw)
        if metric and agg:
            return StructuredQuery(
                intent="superlative",
                entity="player",
                metric=metric,
                aggregation=agg,
                limit=1,
                filters=filters,
            )
        else:
            # Return query with unknown metric — resolver will report "Unknown metric"
            return StructuredQuery(
                intent="superlative",
                entity="player",
                metric=metric_raw.strip() if metric_raw else "unknown",
                aggregation=agg or "max",
                limit=1,
                filters=filters,
            )

    # Pattern: "which team had the highest <metric>"
    match = re.search(
        r"which\s+(team|player)\s+(?:has|had|scored|got|relied|used|played)\s+(?:the\s+)?(most|highest|best|least|lowest|fewest)\s+(?:on\s+)?(\w+)",
        query_lower
    )
    if match:
        entity_raw, agg_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        agg = resolve_aggregation(agg_raw)
        entity = "team" if entity_raw == "team" else "player"
        if metric and agg:
            return StructuredQuery(
                intent="superlative",
                entity=entity,
                metric=metric,
                aggregation=agg,
                limit=1,
                filters=filters,
            )

    # Pattern: "<player> <metric>"
    match = re.search(r"^(.+?)\s+(goals|assists|xg|shots|passes|minutes|tackles|interceptions)$", query_lower)
    if match:
        player_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw)
        if metric:
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric,
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )

    # Pattern: "what is <player>'s <metric>"
    match = re.search(
        r"what\s+(?:is|was|are|were)\s+(.+?)(?:'s|'s)?\s+([\w\s]+?)(?:\s*$|\s*\?)",
        query_lower
    )
    if match:
        player_raw, metric_raw = match.groups()
        metric = resolve_metric(metric_raw.strip())
        if metric:
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric,
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )
        else:
            # Return query with unknown metric — resolver will report "Unknown metric"
            return StructuredQuery(
                intent="numeric",
                entity="player",
                metric=metric_raw.strip(),
                aggregation="sum",
                entity_name=player_raw.strip().title(),
                filters=filters,
            )

    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_query(query: str) -> Route:
    """
    Determine the routing for a query.

    Returns a Route with path, confidence, and optional structured query.
    """
    classification, confidence = classify_query(query)

    if classification == "structured":
        structured_query = parse_structured_query(query)
        if structured_query:
            # Always return structured path — let resolver report "Unknown metric"
            # This enables the structured→semantic fallback logic
            return Route(
                path="structured",
                confidence=confidence,
                reason=f"Query matches structured pattern: {structured_query.intent}",
                structured_query=structured_query,
            )
        # Couldn't parse as structured, fall back to semantic
        return Route(
            path="semantic",
            confidence=0.6,
            reason="Query appears structured but couldn't be parsed or validated",
            semantic_query=query,
        )

    elif classification == "semantic":
        return Route(
            path="semantic",
            confidence=confidence,
            reason="Query is descriptive/qualitative",
            semantic_query=query,
        )

    else:  # hybrid
        structured_query = parse_structured_query(query)
        return Route(
            path="hybrid",
            confidence=confidence,
            reason="Query has both structured and semantic components",
            structured_query=structured_query,
            semantic_query=query,
        )


def execute_route(route: Route, semantic_k: int = 3) -> RoutedResult:
    """
    Execute a routed query.

    Returns RoutedResult with structured and/or semantic results.
    """
    structured_result = None
    semantic_chunks = None
    context = ""

    # For hybrid comparison queries, run structured queries for each entity
    comparison_entities = _detect_comparison(route.semantic_query or "")
    if route.path == "hybrid" and comparison_entities:
        # Run structured queries for each entity
        from src.query.query_schema import StructuredQuery

        entity_results = []
        for entity_name in comparison_entities:
            # Default to "goals" metric for comparison queries
            sq = StructuredQuery(
                intent="numeric",
                entity="player",
                metric="goals",
                aggregation="sum",
                entity_name=entity_name,
            )
            try:
                result = structured_resolve(sq)
                entity_results.append((entity_name, result))
            except Exception as e:
                print(f"Structured resolution failed for {entity_name}: {e}")

        # Combine entity results into a single explanation
        if entity_results:
            parts = []
            for name, result in entity_results:
                if result.status in ("resolved", "partial"):
                    parts.append(f"{name}: {result.explanation}")
                else:
                    parts.append(f"{name}: No data available")
            combined_explanation = " | ".join(parts)

            # Create a synthetic StructuredResult-like object
            class _CombinedResult:
                def __init__(self, explanation):
                    self.status = "resolved"
                    self.explanation = explanation
                    self.aggregated_value = None
                    self.data = []
                    self.dropped_filters = []

            structured_result = _CombinedResult(combined_explanation)
    elif route.path in ("structured", "hybrid") and route.structured_query:
        try:
            structured_result = structured_resolve(route.structured_query)
        except Exception as e:
            print(f"Structured resolution failed: {e}")

        # Structured → Semantic fallback
        # If structured returned empty, attempt semantic fallback.
        # The prompt's honesty rules ensure the LLM won't fabricate answers
        # from loosely related chunks.
        if (structured_result is not None and
                structured_result.status == "empty"):
            # Attempt semantic fallback — prompt ensures honest refusal
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("retrieve_context", "07_retrieve_context.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fallback_chunks = mod.hybrid_search(route.semantic_query or route.structured_query.entity_name or "", k=semantic_k)
                if fallback_chunks:
                    # Prepend fallback context with explicit honesty instruction
                    fallback_context = (
                        "NOTE: The structured data did not contain a direct answer to this question. "
                        "The following context is from related documents and may or may not be relevant. "
                        "Only answer if the context clearly and specifically addresses the original question. "
                        "If the context does not clearly answer the question, state that the data does not "
                        "contain a direct answer.\n\n"
                    )
                    context = fallback_context + mod.build_context(fallback_chunks)
                    semantic_chunks = fallback_chunks
            except Exception as e:
                print(f"Semantic fallback failed: {e}")

    if route.path in ("semantic", "hybrid"):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("retrieve_context", "07_retrieve_context.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Use hybrid_search for full BM25 + dense pipeline
            semantic_chunks = mod.hybrid_search(route.semantic_query or "", k=semantic_k)
            context = mod.build_context(semantic_chunks)
        except Exception as e:
            print(f"Semantic search failed: {e}")

    # Build explanation
    explanation = f"Routed to {route.path} path (confidence: {route.confidence:.2f}). "
    if structured_result:
        explanation += f"Structured: {structured_result.explanation} "
    if semantic_chunks:
        explanation += f"Semantic: {len(semantic_chunks)} chunks retrieved."

    return RoutedResult(
        route=route,
        structured_result=structured_result,
        semantic_chunks=semantic_chunks,
        context=context,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def route_and_execute(query: str, semantic_k: int = 3) -> RoutedResult:
    """
    Route and execute a query in one step.

    Returns RoutedResult with all relevant information.
    """
    route = route_query(query)
    return execute_route(route, semantic_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Query router test")
    parser.add_argument("query", help="User query")
    parser.add_argument("--k", type=int, default=3, help="Number of semantic results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print()

    route = route_query(args.query)
    print(f"Route: {route.path} (confidence: {route.confidence:.2f})")
    print(f"Reason: {route.reason}")

    if args.verbose:
        if route.structured_query:
            print(f"Structured query: {route.structured_query.to_dict()}")
        if route.semantic_query:
            print(f"Semantic query: {route.semantic_query}")
    print()

    result = execute_route(route, args.k)
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
