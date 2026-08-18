"""
src/orchestration/policy.py -- the answer policy shared by every runtime
interface.

Phase B1. `chat.py::process_query()` and `07_prompting.py::answer_question()`
independently implemented the same policy sequence. This module owns the parts
that are genuinely identical between them; the parts that legitimately differ
stay in the interface adapters (see below).

## What lives here (and why it is safe to share)

Each responsibility below was measured to have **zero monkeypatch contracts**
in the test suite, i.e. no test stubs it on an interface module, so moving it
changes no test seam:

    assemble_context()   -- authoritative structured block + rendered evidence,
                            then the non-authoritative conversation block
    should_refuse()      -- the answerability/structured refusal gate
    finalize_answer()    -- post-generation numeric verification + citations

## What deliberately does NOT live here

Routing, prompt construction, and generation are **not** extracted. They carry
15 monkeypatch contracts on the *interface* modules
(`route_and_execute` 5, `route_query` 3, `execute_route` 3, `build_prompt` 4
across four test files); tests stub them there on purpose. Pulling them in
would either break those contracts or force a pile of injected callables --
dependency-injection theatre rather than a real boundary. They stay
adapter-resolved.

Genuinely interface-specific behavior also stays in the adapters, because it
is a real product difference rather than duplication: `semantic_k` (CLI 5 vs
Streamlit 3, which measurably changes citation counts), the CLI mode override,
CLI debug state/history/`[LLM Error: ...]` fallback, and the differing
missing-API-key semantics.

## Dependency direction

orchestration -> generation, verification (+ the runtime result objects it is
handed). Nothing in `src/generation/`, `src/verification/`, `src/retrieval/`,
`src/query/` or `src/context/` imports this package, and neither does
`src/evaluation/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.generation.citations import build_user_citations
from src.generation.policy import is_unsupported_query, is_usable_structured_result
from src.generation.prompt import format_context_for_prompt
from src.verification.comparison import validate_structured_answer

__all__ = [
    "AssembledContext",
    "FinalizedAnswer",
    "assemble_context",
    "should_refuse",
    "finalize_answer",
]

# The authoritative-data banner. Both entry points emitted this byte-for-byte;
# it is defined once here so the two cannot drift apart.
AUTHORITATIVE_HEADER = (
    "## Authoritative Data (Verified from Match Facts)\n\n"
    "The following numbers are VERIFIED and must be used EXACTLY:\n\n"
)

NO_CONTEXT_MESSAGE = "No relevant context found."


@dataclass(frozen=True)
class AssembledContext:
    """Evidence text prepared for prompting, plus the structured verdict."""

    context: str            # retrieved evidence only (what the CLI shows via /context)
    full_context: str       # context with any conversation block prepended
    has_structured: bool


@dataclass(frozen=True)
class FinalizedAnswer:
    """Post-generation outcome: possibly corrected answer, plus its sources."""

    answer: str
    citations: list[dict]
    corrected: bool
    validation: Any = None   # the ValidationResult, when one was produced


def assemble_context(
    routed: Any,
    conversation_context: str = "",
    fallback_context: str | None = None,
) -> AssembledContext:
    """Build prompt-ready evidence text from a routed result.

    Structured facts are presented first and flagged authoritative; retrieved
    chunks follow as supporting narrative. A conversation block, when present,
    is prepended as clearly-labeled non-authoritative reference context -- it
    is never merged into the authoritative section.

    `fallback_context` is what to use when there is neither a usable structured
    result nor any retrieved chunk. This is a **real, measured** difference
    between the two interfaces, not duplication, so it is a parameter rather
    than something unified away: `07_prompting.answer_question()` falls back to
    the RoutedResult's own `context` (which `execute_route()` may have set to a
    structured explanation or to "No relevant documents found."), while
    `chat.py` uses its own "No relevant context found." Passing the caller's
    value keeps each entry point byte-identical to its previous behavior.

    Verified: with a real routed result, this function reproduces
    answer_question()'s assembled context on 8/8 representative queries
    (semantic, structured, comparison, MSA, EGY, insufficient, team-style,
    superlative) -- the empty-evidence branch is the only one that ever
    differed, which is exactly what `fallback_context` preserves.
    """
    structured_result = getattr(routed, "structured_result", None)
    has_structured = is_usable_structured_result(structured_result)

    parts: list[str] = []
    if has_structured:
        parts.append(AUTHORITATIVE_HEADER + structured_result.explanation)

    semantic_chunks = getattr(routed, "semantic_chunks", None)
    if semantic_chunks:
        parts.append(format_context_for_prompt(semantic_chunks))

    if parts:
        context = "\n\n".join(parts)
    elif fallback_context is not None:
        context = fallback_context
    else:
        context = NO_CONTEXT_MESSAGE

    full_context = (
        f"{conversation_context}\n\n{context}" if conversation_context else context
    )
    return AssembledContext(
        context=context, full_context=full_context, has_structured=has_structured
    )


def should_refuse(routed: Any) -> bool:
    """True when the deterministic refusal must be returned instead of generating.

    A usable structured result always wins: semantic-only answerability must
    never veto a valid structured answer (see src/generation/policy.py).
    """
    return is_unsupported_query(
        getattr(routed, "structured_result", None),
        getattr(routed, "answerability", None),
    )


def finalize_answer(
    answer: str,
    routed: Any,
    has_structured: bool,
) -> FinalizedAnswer:
    """Verify a generated answer against structured facts, then build citations.

    Verification is best-effort by design and must never break the pipeline --
    that contract is preserved from both original implementations. Citations
    come only from evidence that was actually retrieved.
    """
    structured_result = getattr(routed, "structured_result", None)
    corrected = False
    validation = None

    if has_structured:
        try:
            validation = validate_structured_answer(answer, structured_result)
            if not validation.is_valid:
                answer = validation.corrected_answer or answer
                corrected = True
        except Exception:
            # Best-effort: a validation failure must not take down the answer.
            pass

    citations = build_user_citations(
        structured_result if has_structured else None,
        getattr(routed, "semantic_chunks", None),
    )
    return FinalizedAnswer(
        answer=answer, citations=citations, corrected=corrected, validation=validation
    )
