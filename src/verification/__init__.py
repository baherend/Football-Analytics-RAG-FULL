"""
src/verification/ -- post-generation grounding checks.

Migration Step 5. Receives a draft answer plus the evidence it was generated
from, and decides whether the answer is acceptable as-is or must be corrected.

    validation.py  -- ValidationResult, numeric-claim extraction,
                      validate_answer() against authoritative values
    comparison.py  -- comparison-answer and structured-answer validation

Kept deliberately distinct from ANSWERABILITY (src/context/answerability.py):

    ANSWERABILITY -- do we have enough evidence to answer?  (before generation)
    VERIFICATION  -- is the generated answer supported by
                     that evidence?                          (after generation)

Both concern grounding; they are not the same question and are not merged.

Scope is limited to what current behavior and tests already support: numeric
claims checked against structured values, and comparison direction/values/
difference checked against the authoritative ComparisonResult. No claim graph
and no LLM judge -- neither is justified by current evidence.
"""

from src.verification.comparison import (
    validate_comparison_answer,
    validate_structured_answer,
)
from src.verification.validation import (
    ValidationResult,
    extract_numeric_claims,
    validate_answer,
)

__all__ = [
    "ValidationResult",
    "extract_numeric_claims",
    "validate_answer",
    "validate_comparison_answer",
    "validate_structured_answer",
]
