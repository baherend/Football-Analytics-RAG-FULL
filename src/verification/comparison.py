"""
src/verification/comparison.py -- comparison-answer grounding checks.

Migration Step 5: extracted verbatim from 07_prompting.py. Validates that a
generated comparison answer agrees with the authoritative ComparisonResult
(which entity is higher, the claimed values, the claimed difference), and
validates ordinary structured answers.

Verification only -- no model calls, no retrieval, no evidence selection.
"""

from __future__ import annotations

import re

from src.query.query_schema import ComparisonResult
from src.verification.validation import (
    ValidationResult,
    _generate_corrected_answer,
    extract_numeric_claims,
    validate_answer,
)


# ---------------------------------------------------------------------------

# Same metric-word vocabulary already used by extract_numeric_claims() above
# -- not a new vocabulary, just reused for entity-anchored matching.
_COMPARISON_METRIC_WORDS = {
    "goals": r"goals?",
    "assists": r"assists?",
    "shots": r"shots?",
    "passes_attempted": r"passes?",
    "minutes": r"minutes?",
    "successful_tackles": r"tackles?",
    "successful_interceptions": r"interceptions?",
    "xg": r"xG",
}


def _comparison_metric_word_pattern(metric: str) -> str:
    """Natural-language regex alternation for a canonical metric name."""
    return _COMPARISON_METRIC_WORDS.get(metric, re.escape(metric) + "s?")


def _claimed_value_near_entity(text: str, entity_name: str, metric_word: str) -> float | None:
    """
    Return the first numeric value `text` claims for `entity_name` near
    `metric_word`, or None if no such claim is made. Anchored on the
    exact (escaped) entity_name string rather than a generic \\w+ capture,
    so names with spaces, accents, apostrophes, or hyphens work safely.
    """
    escaped = re.escape(entity_name)
    patterns = [
        rf"{escaped}(?:'s)?\s*(?:scored|had|has|got|recorded|made)?\s*(\d+(?:\.\d+)?)\s+{metric_word}",
        rf"(\d+(?:\.\d+)?)\s+{metric_word}\s*(?:for|by|from)\s*{escaped}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


_NEGATION_PATTERN = re.compile(r"\b(?:not|n't|never|no\s+longer)\b", re.IGNORECASE)


def _claims_higher(text: str, name_x: str, name_y: str) -> bool:
    """
    True if `text` claims `name_x` scored/had more, or is higher, than
    `name_y` -- bounded to avoid matching across unrelated sentences.

    A negated claim ("X did not score more than Y") is deliberately NOT
    treated as an affirmative claim in either direction. Correctly
    interpreting a negated comparison's actual meaning would require real
    language understanding, which this deterministic validator
    intentionally does not attempt -- treating it as "no clear directional
    claim" (an omission, not a contradiction) is safer than guessing a
    polarity that may be wrong either way.
    """
    pattern = (
        rf"{re.escape(name_x)}\b([^.?!]{{0,60}}?)\b(?:more|higher)\b"
        rf"[^.?!]{{0,40}}?\bthan\b[^.?!]{{0,10}}?{re.escape(name_y)}"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return False
    if _NEGATION_PATTERN.search(match.group(1)):
        return False
    return True


def _claimed_difference(text: str, metric_word: str) -> float | None:
    """Return an explicitly stated numeric difference/margin, or None."""
    patterns = [
        rf"\bby\s+(\d+(?:\.\d+)?)\s+{metric_word}\b",
        rf"(?:difference|margin)\s+of\s+(\d+(?:\.\d+)?)\s*{metric_word}?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _generate_comparison_correction(
    comparison_result: ComparisonResult,
    response_language: str = "en",
) -> str:
    """Deterministic corrected comparison sentence -- never another LLM call."""
    name_a, value_a = comparison_result.values[0].entity_name, comparison_result.values[0].value
    name_b, value_b = comparison_result.values[1].entity_name, comparison_result.values[1].value
    metric = comparison_result.metric

    if comparison_result.outcome == "tie":
        if response_language == "ar":
            return f"تعادل {name_a} و{name_b} عند {value_a:g} {metric} لكل منهما."
        return f"{name_a} and {name_b} were tied at {value_a:g} {metric} each."

    if comparison_result.outcome == "entity_a_higher":
        higher_name, higher_value = name_a, value_a
        lower_name, lower_value = name_b, value_b
    else:
        higher_name, higher_value = name_b, value_b
        lower_name, lower_value = name_a, value_a

    if response_language == "ar":
        return (
            f"لدى {higher_name} {higher_value:g} {metric} مقابل "
            f"{lower_value:g} لدى {lower_name}، لذا يتفوق {higher_name} "
            f"بفارق {comparison_result.difference:g}."
        )

    return (
        f"{higher_name} had {higher_value:g} {metric} compared with {lower_name}'s {lower_value:g}, "
        f"so {higher_name} was higher by {comparison_result.difference:g}."
    )


def validate_comparison_answer(
    llm_answer: str,
    comparison_result: ComparisonResult,
    response_language: str = "en",
) -> ValidationResult:
    """
    Validate a generated comparison answer against ComparisonResult's
    deterministic structured facts (both entity values, difference,
    outcome) -- reusing the same ValidationResult/corrected_answer
    contract as validate_answer(), for a two-entity shape instead of a
    single scalar value.

    Only meaningful for a complete comparison (both values present,
    outcome/difference already computed by ComparisonResult.__post_init__)
    -- Step 2H already keeps an incomplete comparison from reaching
    generation as authoritative evidence, so the guard below is a
    defensive no-op, not a new gate.

    Detects only explicit contradictions -- never requires the answer to
    mention every fact, and never flags an answer merely for omitting
    values, the difference, or for paraphrasing "higher" as "more":
      - a claimed value for a named entity that disagrees with its
        authoritative aggregated_value
      - a claimed "X is higher/more than Y" ordering that disagrees with
        the authoritative outcome (including a false claim over a tie)
      - an explicitly stated numeric difference that disagrees with the
        authoritative difference

    Never re-derives values from explanation text or calls an LLM to
    judge or correct the answer -- both detection and correction use
    only comparison_result's already-computed fields.
    """
    result = ValidationResult(is_valid=True)

    values = comparison_result.values
    if len(values) != 2 or values[0].value is None or values[1].value is None:
        return result  # incomplete comparison -- nothing to validate here

    name_a, value_a = values[0].entity_name, values[0].value
    name_b, value_b = values[1].entity_name, values[1].value
    metric_word = _comparison_metric_word_pattern(comparison_result.metric)

    contradictions = []

    for name, expected_value in ((name_a, value_a), (name_b, value_b)):
        claimed = _claimed_value_near_entity(llm_answer, name, metric_word)
        if claimed is not None and claimed != expected_value:
            contradictions.append({
                "description": (
                    f"LLM claimed {name} has {claimed:g} {comparison_result.metric}, "
                    f"but structured data shows {expected_value:g}"
                ),
            })

    claims_a_higher = _claims_higher(llm_answer, name_a, name_b)
    claims_b_higher = _claims_higher(llm_answer, name_b, name_a)
    if claims_a_higher and comparison_result.outcome != "entity_a_higher":
        contradictions.append({
            "description": f"LLM claimed {name_a} is higher than {name_b}, contradicting the authoritative outcome",
        })
    elif claims_b_higher and comparison_result.outcome != "entity_b_higher":
        contradictions.append({
            "description": f"LLM claimed {name_b} is higher than {name_a}, contradicting the authoritative outcome",
        })

    claimed_diff = _claimed_difference(llm_answer, metric_word)
    if (
        claimed_diff is not None
        and comparison_result.difference is not None
        and claimed_diff != comparison_result.difference
    ):
        contradictions.append({
            "description": (
                f"LLM stated a difference of {claimed_diff:g}, but the authoritative "
                f"difference is {comparison_result.difference:g}"
            ),
        })

    if contradictions:
        result.is_valid = False
        result.contradictions = contradictions
        result.corrected_answer = _generate_comparison_correction(
            comparison_result,
            response_language=response_language,
        )

    return result


def validate_structured_answer(
    llm_answer: str,
    structured_result,
    response_language: str = "en",
) -> ValidationResult:
    """
    Shared validation dispatch used by both chat.py::process_query() and
    answer_question(): validate_comparison_answer() for a ComparisonResult
    (explicit isinstance check -- no circular import, see
    src/query/query_schema.py), otherwise the existing scalar
    validate_answer(), unchanged. Both return the same ValidationResult,
    so callers apply identical is_valid/corrected_answer handling
    regardless of which validator ran.
    """
    if isinstance(structured_result, ComparisonResult):
        return validate_comparison_answer(
            llm_answer,
            structured_result,
            response_language=response_language,
        )
    return validate_answer(
        llm_answer=llm_answer,
        structured_explanation=structured_result.explanation,
        structured_value=getattr(structured_result, "aggregated_value", None),
        structured_metric=getattr(getattr(structured_result, "query", None), "metric", None),
        response_language=response_language,
    )

