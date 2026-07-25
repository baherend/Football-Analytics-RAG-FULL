"""
validator.py — Answer Validation

Validates LLM-generated answers against structured facts to catch
contradictions before they reach the user.

Philosophy: structured data is ground truth. If the LLM contradicts it,
the structured data wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating an LLM answer against structured facts."""
    is_valid: bool
    contradictions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_answer: str | None = None

    def __str__(self):
        if self.is_valid:
            return "VALID"
        issues = [c["description"] for c in self.contradictions]
        return f"INVALID: {'; '.join(issues)}"


# ---------------------------------------------------------------------------
# Numeric Claim Extraction
# ---------------------------------------------------------------------------


def extract_numeric_claims(text: str) -> list[dict]:
    """
    Extract numeric claims from text.

    Returns list of {value: float, context: str, entity: str|None}
    where context is the surrounding text snippet.
    """
    claims = []

    # Pattern: "<entity> <number> <metric>"
    # e.g., "Messi scored 7 goals", "Mbappé had 8 goals"
    patterns = [
        # "X scored/has/had <number> goals/assists/etc."
        r"(\w[\w\s]*?)\s+(?:scored|has|had|made|achieved|recorded)\s+(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        # "<number> goals/assists by X"
        r"(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:by|from)\s+(\w[\w\s]*?)",
        # "X: <number> <metric>"
        r"(\w[\w\s]*?):\s*(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        # "<entity>'s <metric> is/was <number>"
        r"(\w[\w\s]*?)(?:'s|'s)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?)",
        # Standalone "<number> <metric>" (when entity is implied from context)
        r"(?:^|\s)(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|xG)(?:\s|$|[,.\])])",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            # Extract entity, value, metric based on pattern order
            if len(groups) == 3:
                if groups[0].replace(".", "").isdigit():
                    # Pattern 2: "<number> <metric> by X"
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = groups[2].strip()
                elif groups[2].replace(".", "").isdigit():
                    # Pattern 4: "X's metric is <number>"
                    entity = groups[0].strip()
                    metric = groups[1].lower()
                    value = float(groups[2])
                else:
                    # Pattern 1: "X scored <number> metric"
                    entity = groups[0].strip()
                    value = float(groups[1])
                    metric = groups[2].lower()
            elif len(groups) == 2:
                # Pattern 5: "<number> <metric>"
                if groups[0].replace(".", "").isdigit():
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = None
                else:
                    continue
            else:
                continue

            # Normalize metric name
            metric = metric.rstrip("s")  # "goals" -> "goal"
            metric_map = {
                "goal": "goals",
                "assist": "assists",
                "shot": "shots",
                "pass": "passes_attempted",
                "minute": "minutes",
                "tackle": "successful_tackles",
                "interception": "successful_interceptions",
                "xg": "xg",
            }
            normalized_metric = metric_map.get(metric, metric)

            claims.append({
                "value": value,
                "metric": normalized_metric,
                "entity": entity,
                "context": text[max(0, match.start() - 20):match.end() + 20],
            })

    return claims


# ---------------------------------------------------------------------------
# Validation Against Structured Facts
# ---------------------------------------------------------------------------


def validate_answer(
    llm_answer: str,
    structured_explanation: str,
    structured_value: float | int | None = None,
    structured_entity: str | None = None,
    structured_metric: str | None = None,
) -> ValidationResult:
    """
    Validate an LLM answer against structured facts.

    Parameters:
        llm_answer: The generated answer from the LLM
        structured_explanation: The structured result explanation
        structured_value: The aggregated value from structured resolution
        structured_entity: The entity name from the structured query
        structured_metric: The metric name from the structured query

    Returns:
        ValidationResult with is_valid, contradictions, and corrected_answer
    """
    result = ValidationResult(is_valid=True)

    # Extract numeric claims from LLM answer
    llm_claims = extract_numeric_claims(llm_answer)

    # Extract the expected value from structured explanation
    # Patterns: "total goals is 7" or "7 goals" or "goals: 7"
    expected_patterns = [
        r"(?:total|sum|count)?\s*(?:goals?|assists?|shots?|passes?|minutes?|xG)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:goals?|assists?|shots?|passes?|minutes?|xG)",
        r"(?:goals?|assists?|shots?|passes?|minutes?|xG)\s*:\s*(\d+(?:\.\d+)?)",
    ]

    expected_value = None
    for pattern in expected_patterns:
        expected_match = re.search(pattern, structured_explanation, re.IGNORECASE)
        if expected_match:
            expected_value = float(expected_match.group(1))
            break

    # Also use structured_value if provided and no value found in explanation
    if expected_value is None and structured_value is not None:
        expected_value = float(structured_value)

    if expected_value is not None:

        # Check each LLM claim against the expected value
        for claim in llm_claims:
            # If the claim is about the same metric and entity, check for contradiction
            if structured_metric and claim["metric"] != structured_metric:
                continue

            if structured_entity and claim["entity"]:
                # Check if entity matches (case-insensitive, partial match)
                entity_match = (
                    structured_entity.lower() in claim["entity"].lower() or
                    claim["entity"].lower() in structured_entity.lower()
                )
                if not entity_match:
                    continue

            # Check for contradiction
            if claim["value"] != expected_value:
                result.contradictions.append({
                    "llm_value": claim["value"],
                    "expected_value": expected_value,
                    "metric": claim["metric"],
                    "entity": claim["entity"],
                    "description": (
                        f"LLM claimed {claim['entity'] or 'entity'} has "
                        f"{claim['value']} {claim['metric']}, but structured "
                        f"data shows {expected_value}"
                    ),
                    "context": claim["context"],
                })

    # If contradictions found, generate corrected answer
    if result.contradictions:
        result.is_valid = False
        result.corrected_answer = _generate_corrected_answer(
            llm_answer, structured_explanation, result.contradictions
        )

    return result


def _generate_corrected_answer(
    llm_answer: str,
    structured_explanation: str,
    contradictions: list[dict],
) -> str:
    """
    Generate a corrected answer when contradictions are detected.

    Strategy: replace the incorrect number with the correct one from
    structured data, preserving the rest of the answer.
    """
    corrected = llm_answer

    for contradiction in contradictions:
        old_value = contradiction["llm_value"]
        new_value = contradiction["expected_value"]
        metric = contradiction["metric"]

        # Try to replace the incorrect value with the correct one
        # Pattern: "<old_value> <metric>"
        old_pattern = f"{old_value:g}\\s+{metric}"
        new_text = f"{new_value:g} {metric}"
        corrected = re.sub(old_pattern, new_text, corrected, flags=re.IGNORECASE)

    # If we couldn't correct automatically, fall back to structured facts
    if corrected == llm_answer:
        return (
            f"Based on the structured data:\n{structured_explanation}\n\n"
            f"(Note: The original answer contained incorrect numbers and was "
            f"replaced with verified data.)"
        )

    return (
        f"{corrected}\n\n"
        f"(Note: Numbers have been verified against structured data.)"
    )


# ---------------------------------------------------------------------------
# Convenience: Validate and Correct
# ---------------------------------------------------------------------------


def validate_and_correct(
    llm_answer: str,
    structured_result,
) -> tuple[str, ValidationResult]:
    """
    Validate an LLM answer and return corrected answer if needed.

    Parameters:
        llm_answer: The generated answer
        structured_result: StructuredResult from the resolver

    Returns:
        (answer, validation_result) — answer is either the original or corrected
    """
    if structured_result is None or not hasattr(structured_result, 'explanation'):
        return llm_answer, ValidationResult(is_valid=True)

    validation = validate_answer(
        llm_answer=llm_answer,
        structured_explanation=structured_result.explanation or "",
        structured_value=structured_result.aggregated_value,
        structured_entity=getattr(structured_result, 'entity_name', None),
        structured_metric=getattr(structured_result, 'metric', None),
    )

    if validation.is_valid:
        return llm_answer, validation
    else:
        # Return corrected answer
        answer = validation.corrected_answer or llm_answer
        return answer, validation
