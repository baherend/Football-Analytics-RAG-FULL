"""
src/verification/validation.py -- numeric-claim grounding checks.

Migration Step 5: extracted verbatim from 07_prompting.py.

VERIFICATION answers "is the generated answer supported by the evidence we
gave it?" -- distinct from ANSWERABILITY (src/context/answerability.py), which
asks "do we have enough evidence to answer at all?" and runs BEFORE
generation. Do not merge them.

This module reads generated text and compares its numeric claims against
authoritative structured values. It never calls a model, retrieves, or selects
evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating an LLM answer against structured facts."""
    is_valid: bool
    contradictions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_answer: str | None = None

    def __str__(self) -> str:
        if self.is_valid:
            return "VALID"
        issues = [c["description"] for c in self.contradictions]
        return f"INVALID: {'; '.join(issues)}"


# Maximum length of a captured entity span in the claim patterns below.
#
# SECURITY (ReDoS): three of these patterns capture the entity as a *lazy*
# span whose character class `[\w\s]` matches letters, digits, underscores and
# whitespace -- i.e. almost any prose. Left unbounded, `re.finditer()`'s
# per-position retry combined with that span expanding toward end-of-string
# (looking for a literal like "scored" or ":" that may never appear) gave
# O(N^2) behavior: measured a consistent ~4x per doubling, 15.8s on 16 KB and
# ~159s on 40 KB. The input is LLM output, which is attacker-influenceable
# (retrieved evidence can steer a model into emitting long number-dense text,
# and a hostile provider response is fully attacker-controlled), and this runs
# on every structured/hybrid answer via validate_structured_answer().
# Bounding the span makes the work per start position constant, so the scan is
# linear -- measured 2.0x per doubling, 0.15s at 16 KB, 0.4s at 40 KB.
#
# 200 is chosen on evidence, NOT copied from the 60 used in
# src/query/intent.py and src/retrieval/safeguards.py: those capture a single
# `\w` token, whereas this span can include a whole sentence prefix before the
# verb. The measured worst realistic capture -- the longest entity name in the
# corpus (46 chars) behind a long sentence prefix -- is 109 characters, so 200
# leaves ~83% headroom. Verified to produce zero output differences across
# 2570 realistic claim sentences generated from real corpus entity names; see
# tests/test_verification_security.py.
_MAX_ENTITY_SPAN = 200

# The two patterns without a leading unbounded entity capture (number-first
# "N metric by ENTITY", and the anchored bare "N metric") were measured linear
# already -- the first gates its trailing capture behind a literal prefix, the
# second captures no entity -- so they are left exactly as they were.


def extract_numeric_claims(text: str) -> list[dict]:
    """Extract supported numeric football-stat claims from generated text.

    Claim shape is unchanged: {value, metric, entity, context}. See
    _MAX_ENTITY_SPAN above for the entity-span bound and why it exists.
    """
    claims = []

    _E = rf"(\w[\w\s]{{0,{_MAX_ENTITY_SPAN}}}?)"   # bounded entity span

    patterns = [
        rf"{_E}\s+(?:scored|has|had|made|achieved|recorded)\s+(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        r"(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:by|from)\s+(\w[\w\s]*?)",
        rf"{_E}:\s*(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        rf"{_E}(?:'s|'s)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?)",
        r"(?:^|\s)(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|xG)(?:\s|$|[,.\])])",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()

            if len(groups) == 3:
                if groups[0].replace(".", "").isdigit():
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = groups[2].strip()
                elif groups[2].replace(".", "").isdigit():
                    entity = groups[0].strip()
                    metric = groups[1].lower()
                    value = float(groups[2])
                else:
                    entity = groups[0].strip()
                    value = float(groups[1])
                    metric = groups[2].lower()
            elif len(groups) == 2:
                if groups[0].replace(".", "").isdigit():
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = None
                else:
                    continue
            else:
                continue

            metric = metric.rstrip("s")
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

            claims.append(
                {
                    "value": value,
                    "metric": metric_map.get(metric, metric),
                    "entity": entity,
                    "context": text[max(0, match.start() - 20):match.end() + 20],
                }
            )

    return claims


def _generate_corrected_answer(
    llm_answer: str,
    structured_explanation: str,
    contradictions: list[dict],
    response_language: str = "en",
) -> str:
    """Correct contradicted numeric claims using verified structured values."""
    corrected = llm_answer

    for contradiction in contradictions:
        old_value = contradiction["llm_value"]
        new_value = contradiction["expected_value"]
        metric = contradiction["metric"]

        old_pattern = rf"{old_value:g}\s+{re.escape(metric)}"
        new_text = f"{new_value:g} {metric}"
        corrected = re.sub(
            old_pattern,
            new_text,
            corrected,
            flags=re.IGNORECASE,
        )

    corrected_arabic_narrative = corrected != llm_answer and any(
        unicodedata.category(ch).startswith("L")
        and (
            "\u0600" <= ch <= "\u06ff"
            or "\u0750" <= ch <= "\u077f"
            or "\u08a0" <= ch <= "\u08ff"
            or "\ufb50" <= ch <= "\ufdff"
            or "\ufe70" <= ch <= "\ufeff"
        )
        for ch in corrected
    )
    if response_language == "ar" and corrected_arabic_narrative:
        return (
            f"{corrected}\n\n"
            "(ملاحظة: تم التحقق من الأرقام باستخدام البيانات المنظمة.)"
        )

    if response_language == "ar":
        verified_values = []
        for contradiction in contradictions:
            fact = (
                f"القيمة الصحيحة لمقياس {contradiction['metric']} هي "
                f"{contradiction['expected_value']:g} {contradiction['metric']}."
            )
            if fact not in verified_values:
                verified_values.append(fact)
        verified_text = "\n".join(verified_values)
        return (
            "استنادًا إلى البيانات المنظمة:\n"
            f"{verified_text}\n\n"
            "(ملاحظة: تم التحقق من الأرقام باستخدام البيانات المنظمة.)"
        )

    if corrected == llm_answer:
        return (
            f"Based on the structured data:\n{structured_explanation}\n\n"
            "(Note: The original answer contained incorrect numbers and was "
            "replaced with verified data.)"
        )

    return f"{corrected}\n\n(Note: Numbers have been verified against structured data.)"


def validate_answer(
    llm_answer: str,
    structured_explanation: str,
    structured_value: float | int | None = None,
    structured_entity: str | None = None,
    structured_metric: str | None = None,
    response_language: str = "en",
) -> ValidationResult:
    """Validate generated numeric claims against authoritative structured facts."""
    result = ValidationResult(is_valid=True)
    llm_claims = extract_numeric_claims(llm_answer)

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

    if expected_value is None and structured_value is not None:
        expected_value = float(structured_value)

    if expected_value is not None:
        for claim in llm_claims:
            if structured_metric and claim["metric"] != structured_metric:
                continue

            if structured_entity and claim["entity"]:
                entity_match = (
                    structured_entity.lower() in claim["entity"].lower()
                    or claim["entity"].lower() in structured_entity.lower()
                )
                if not entity_match:
                    continue

            if claim["value"] != expected_value:
                result.contradictions.append(
                    {
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
                    }
                )

    if result.contradictions:
        result.is_valid = False
        result.corrected_answer = _generate_corrected_answer(
            llm_answer,
            structured_explanation,
            result.contradictions,
            response_language=response_language,
        )

    return result

