"""
src/query/planning.py -- Query planning: choosing the retrieval strategy.

Migration Step 3 (Query Understanding + Planning Split): mechanically
extracted from src/query/router.py's Route dataclass and route_query() --
no logic changes. See src/query/router.py for the compatibility re-exports
existing callers keep using.

Planning answers "given what the user wants, which path should answer it?" --
structured, semantic, or hybrid, plus the parsed StructuredQuery and/or
semantic query text that path will need. It consumes understanding
(src/query/intent.py) and parsing (src/query/parsing.py); it does NOT execute
anything -- no structured resolution, no retrieval, no context building (see
src/query/router.py).

`Route` lives HERE, not in router.py, deliberately: it is the plan object
this module produces, and keeping it here means planning.py never has to
import router.py. That keeps the dependency strictly one-way
(router.py -> planning.py -> parsing.py + intent.py) with no cycle and no
reverse dependency -- the failure mode the Retrieval Split hit and corrected
(see PROJECT_MEMORY.md's Architecture Decisions). `RoutedResult`, the
*execution* outcome, stays with execute_route() in router.py.

A richer plan model (explicit evidence_requirements / retrieval_strategy /
coverage_requirements -- see docs/architecture/overview.md) is deliberately
NOT introduced here: nothing downstream consumes those fields today, so
adding them now would be an abstraction without a caller. Deferred to the
Context Engineering / Evidence Pack step.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.artifacts import ArtifactPaths
from src.query.query_schema import StructuredQuery
from src.query.intent import classify_query
from src.query.parsing import (
    _load_active_stage_taxonomy,
    parse_compositional_dependency,
    parse_structured_query,
)


@dataclass
class Route:
    """Routing decision."""
    path: str  # "semantic" | "structured" | "hybrid"
    confidence: float
    reason: str
    structured_query: StructuredQuery | None = None
    semantic_query: str | None = None
    dependency_query: StructuredQuery | None = None
    dependency_phrase: str | None = None


def route_query(
    query: str,
    artifact_paths: ArtifactPaths | None = None,
) -> Route:
    """Determine routing using the selected dataset's stage vocabulary."""

    # Detect an embedded structured selector before flat classification.
    dependency = parse_compositional_dependency(query)
    if dependency is not None:
        match_facts_path = (
            artifact_paths.match_facts if artifact_paths is not None else None
        )
        stage_taxonomy = _load_active_stage_taxonomy(match_facts_path)
        dependency = parse_compositional_dependency(
            query,
            stage_taxonomy=stage_taxonomy,
        )
        if dependency is not None:
            dependency_query, dependency_phrase = dependency
            return Route(
                path="hybrid",
                confidence=0.9,
                reason="Query has a structured dependency feeding a semantic continuation",
                semantic_query=query,
                dependency_query=dependency_query,
                dependency_phrase=dependency_phrase,
            )

    classification, confidence = classify_query(query)

    if classification == "structured":
        match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None
        stage_taxonomy = _load_active_stage_taxonomy(match_facts_path)
        structured_query = parse_structured_query(query, stage_taxonomy=stage_taxonomy)
        if structured_query:
            return Route(path="structured", confidence=confidence,
                         reason=f"Query matches structured pattern: {structured_query.intent}",
                         structured_query=structured_query)
        return Route(path="semantic", confidence=0.6,
                     reason="Query appears structured but couldn't be parsed or validated",
                     semantic_query=query)

    elif classification == "semantic":
        return Route(path="semantic", confidence=confidence,
                     reason="Query is descriptive/qualitative",
                     semantic_query=query)

    else:  # hybrid
        match_facts_path = artifact_paths.match_facts if artifact_paths is not None else None
        stage_taxonomy = _load_active_stage_taxonomy(match_facts_path)
        structured_query = parse_structured_query(query, stage_taxonomy=stage_taxonomy)
        return Route(path="hybrid", confidence=confidence,
                     reason="Query has both structured and semantic components",
                     structured_query=structured_query,
                     semantic_query=query)
