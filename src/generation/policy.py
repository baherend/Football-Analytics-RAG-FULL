"""
src/generation/policy.py -- the gate that decides whether generation runs.

Migration Step 5: extracted verbatim from 07_prompting.py.

This is deliberately NOT verification. Keep the distinction:

    ANSWERABILITY  -- do we have enough evidence to answer?   (src/context/)
    POLICY (here)  -- given that assessment, do we generate,
                      or return the deterministic refusal?
    VERIFICATION   -- is the generated answer supported by
                      the evidence?                            (src/verification/)

It consumes an answerability assessment plus any structured result and makes a
pre-generation decision. It never inspects generated text.
"""

from __future__ import annotations

from src.generation.prompt import INSUFFICIENT_CONTEXT_MESSAGE

__all__ = ["is_usable_structured_result", "is_unsupported_query",
           "INSUFFICIENT_CONTEXT_MESSAGE"]


def is_usable_structured_result(structured_result) -> bool:
    """
    True when `structured_result` carries evidence the generation layer
    may present as authoritative.

    Ordinary StructuredResult: unchanged from the existing contract --
    status "resolved"/"partial" with a non-empty explanation is usable (a
    "partial" StructuredResult may still carry a real aggregated_value
    with a scope caveat, e.g. a dropped filter -- see
    src/query/resolver.py).

    ComparisonResult (duck-typed via its `values` list, so this module
    doesn't need to import the query schema): a comparison is only usable
    when it is actually complete -- both compared entities produced a
    real numeric value. "resolved" already implies this (Step 2G). A
    "partial" comparison with a missing side (Step 2F leaves
    difference/outcome both None in that case) must NOT be treated as a
    fully verified two-sided comparison merely because its status string
    passes the ordinary resolved/partial check -- but a "partial"
    comparison where both values ARE present (the caveat came from an
    underlying entity's own StructuredResult, not from a missing side)
    remains usable, with its partial status left intact.
    """
    if not (
        structured_result
        and getattr(structured_result, "status", None) in ("resolved", "partial")
        and getattr(structured_result, "explanation", None)
    ):
        return False

    values = getattr(structured_result, "values", None)
    if values is None:
        return True  # Ordinary StructuredResult -- existing contract.

    # ComparisonResult: complete only when both entities resolved to a
    # real numeric value.
    return len(values) == 2 and values[0].value is not None and values[1].value is not None


def is_unsupported_query(structured_result, answerability) -> bool:
    """
    True when there is no usable authoritative structured result AND the
    routed semantic evidence was explicitly assessed "unanswerable".

    A usable structured result always takes precedence -- semantic-only
    answerability must never veto a valid structured answer.
    """
    return (
        not is_usable_structured_result(structured_result)
        and answerability is not None
        and getattr(answerability, "status", None) == "unanswerable"
    )


