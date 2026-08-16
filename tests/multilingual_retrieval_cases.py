"""
multilingual_retrieval_cases.py -- Multilingual Retrieval Baseline: English /
Modern Standard Arabic (MSA) / Egyptian Arabic (EGY) query variants for the
existing WC2022 Semantic Ground Truth (tests/semantic_ground_truth.py).

This module defines NO new relevance truth. Every variant below references
an existing semantic ground-truth case by `source_case_id`; the required/
optional relevant document IDs, levels, and all other relevance judgments
are resolved from `tests.semantic_ground_truth.SEMANTIC_GROUND_TRUTH` at
evaluation time (see `build_translated_cases()`), never duplicated or
re-authored here. Only the query TEXT changes between language variants --
the ground truth being measured against stays identical.

Entity-script policy (see repository task notes / final report for the
full rationale): the primary EN / MSA / EGY benchmark keeps player and team
names in their original Latin spelling embedded in the Arabic sentence
(e.g. "...قدام Croatia..."), isolating "does Arabic sentence structure hurt
retrieval" from "does Arabic transliteration of names hurt retrieval". A
small separate ENTITY_SCRIPT_DIAGNOSTIC_VARIANTS set (Egyptian Arabic with
Arabic-transliterated entity names, e.g. "كرواتيا" instead of "Croatia")
covers the second question for a handful of entity-heavy cases, kept
strictly separate from the primary benchmark's aggregates.

No production behavior depends on this file. It is benchmark/evaluation
data only.
"""

from __future__ import annotations

from dataclasses import dataclass

LANGUAGES: tuple[str, ...] = ("en", "ar_msa", "ar_egy")


@dataclass(frozen=True)
class MultilingualQueryVariant:
    """One language variant of an existing semantic ground-truth case's query."""
    source_case_id: str
    language: str  # "en" | "ar_msa" | "ar_egy" (or "ar_egy_translit" for the diagnostic set)
    query: str


# ---------------------------------------------------------------------------
# Primary benchmark: EN / MSA / EGY variants for all 24 semantic cases.
#
# English ("en") queries are the ORIGINAL semantic-case query text,
# unchanged. MSA and Egyptian Arabic ("ar_egy") variants preserve the
# semantic intent of each English query; named player/team entities are
# kept in their original Latin spelling embedded in the Arabic sentence
# (entity-script policy -- see module docstring) so the primary benchmark
# isolates Arabic sentence/lexical alignment from entity transliteration.
# ---------------------------------------------------------------------------

MULTILINGUAL_QUERY_VARIANTS: tuple[MultilingualQueryVariant, ...] = (
    # gt-pilot-l1-01 -- no named entities
    MultilingualQueryVariant("gt-pilot-l1-01", "en", "What happened in the opening match of the World Cup?"),
    MultilingualQueryVariant("gt-pilot-l1-01", "ar_msa", "ماذا حدث في مباراة افتتاح كأس العالم؟"),
    MultilingualQueryVariant("gt-pilot-l1-01", "ar_egy", "حصل ايه في ماتش افتتاح كأس العالم؟"),

    # gt-pilot-l2-01 -- Argentina, France
    MultilingualQueryVariant("gt-pilot-l2-01", "en", "What were the key events in the Argentina vs France Final?"),
    MultilingualQueryVariant("gt-pilot-l2-01", "ar_msa", "ما هي أهم الأحداث في نهائي Argentina ضد France؟"),
    MultilingualQueryVariant("gt-pilot-l2-01", "ar_egy", "ايه أهم الأحداث في نهائي Argentina وFrance؟"),

    # gt-pilot-l3-01 -- Messi, Croatia
    MultilingualQueryVariant("gt-pilot-l3-01", "en", "How did Messi perform against Croatia in the semi-final?"),
    MultilingualQueryVariant("gt-pilot-l3-01", "ar_msa", "كيف كان أداء Messi أمام Croatia في نصف النهائي؟"),
    MultilingualQueryVariant("gt-pilot-l3-01", "ar_egy", "Messi لعب ازاي قدام Croatia في نص النهائي؟"),

    # gt-pilot-l4-01 -- Messi
    MultilingualQueryVariant("gt-pilot-l4-01", "en", "Describe Messi's overall World Cup tournament performance."),
    MultilingualQueryVariant("gt-pilot-l4-01", "ar_msa", "صف أداء Messi العام في بطولة كأس العالم."),
    MultilingualQueryVariant("gt-pilot-l4-01", "ar_egy", "كلمني عن أداء Messi في البطولة كلها."),

    # gt-pilot-team-01 -- Argentina
    MultilingualQueryVariant("gt-pilot-team-01", "en", "What was Argentina's playing style and most common formation?"),
    MultilingualQueryVariant("gt-pilot-team-01", "ar_msa", "ما هو أسلوب لعب Argentina والتشكيل الأكثر استخدامًا؟"),
    MultilingualQueryVariant("gt-pilot-team-01", "ar_egy", "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟"),

    # gt-pilot-multi-01 -- Argentina, France, Messi, Mbappé
    MultilingualQueryVariant("gt-pilot-multi-01", "en", "How did the Argentina vs France Final unfold, and how did Messi and Mbappé perform?"),
    MultilingualQueryVariant("gt-pilot-multi-01", "ar_msa", "كيف جرت أحداث نهائي Argentina ضد France، وكيف كان أداء Messi وMbappé؟"),
    MultilingualQueryVariant("gt-pilot-multi-01", "ar_egy", "نهائي Argentina وFrance عدى ازاي، وMessi وMbappé لعبوا ازاي؟"),

    # gt-l1-02 -- England, Iran
    MultilingualQueryVariant("gt-l1-02", "en", "Describe the match between England and Iran."),
    MultilingualQueryVariant("gt-l1-02", "ar_msa", "صف المباراة بين England وIran."),
    MultilingualQueryVariant("gt-l1-02", "ar_egy", "احكيلي عن الماتش بين England وIran."),

    # gt-l2-02 -- Argentina, Croatia
    MultilingualQueryVariant("gt-l2-02", "en", "What substitutions were made in the Argentina vs Croatia semi-final?"),
    MultilingualQueryVariant("gt-l2-02", "ar_msa", "ما هي التبديلات التي تمت في نصف نهائي Argentina ضد Croatia؟"),
    MultilingualQueryVariant("gt-l2-02", "ar_egy", "التبديلات اللي حصلت في نص نهائي Argentina وCroatia كانت ايه؟"),

    # gt-l3-02 -- Antoine Griezmann
    MultilingualQueryVariant("gt-l3-02", "en", "What were Antoine Griezmann's passing statistics in the Final?"),
    MultilingualQueryVariant("gt-l3-02", "ar_msa", "ما هي إحصائيات التمرير الخاصة بـ Antoine Griezmann في النهائي؟"),
    MultilingualQueryVariant("gt-l3-02", "ar_egy", "Antoine Griezmann عمل كام باسة في النهائي؟"),

    # gt-l4-02 -- Kylian Mbappé
    MultilingualQueryVariant("gt-l4-02", "en", "Describe Kylian Mbappé's overall World Cup tournament performance."),
    MultilingualQueryVariant("gt-l4-02", "ar_msa", "صف أداء Kylian Mbappé العام في بطولة كأس العالم."),
    MultilingualQueryVariant("gt-l4-02", "ar_egy", "Kylian Mbappé لعب ازاي في البطولة كلها؟"),

    # gt-team-02 -- Morocco
    MultilingualQueryVariant("gt-team-02", "en", "How did Morocco play in the tournament, and which formations did they use most?"),
    MultilingualQueryVariant("gt-team-02", "ar_msa", "كيف لعبت Morocco في البطولة، وما هي التشكيلات التي استخدمتها أكثر؟"),
    MultilingualQueryVariant("gt-team-02", "ar_egy", "Morocco لعبت ازاي في البطولة، وايه أكتر تشكيل لعبوا بيه؟"),

    # gt-multi-02 -- Argentina, Croatia, Messi
    MultilingualQueryVariant("gt-multi-02", "en", "How did Argentina beat Croatia in the semi-final, and how did Messi perform?"),
    MultilingualQueryVariant("gt-multi-02", "ar_msa", "كيف فازت Argentina على Croatia في نصف النهائي، وكيف كان أداء Messi؟"),
    MultilingualQueryVariant("gt-multi-02", "ar_egy", "Argentina كسبت Croatia ازاي في نص النهائي، وMessi لعب ازاي؟"),

    # gt-l1-03 -- no named entities
    MultilingualQueryVariant("gt-l1-03", "en", "Which knockout-stage matches were decided by penalty shootouts?"),
    MultilingualQueryVariant("gt-l1-03", "ar_msa", "ما هي مباريات دور الإقصاء التي حُسمت بركلات الترجيح؟"),
    MultilingualQueryVariant("gt-l1-03", "ar_egy", "مباريات دور خروج المغلوب اتحسمت بضربات الترجيح كانت ايه؟"),

    # gt-l2-03 -- Morocco, Portugal
    MultilingualQueryVariant("gt-l2-03", "en", "What happened in the second half of the Morocco vs Portugal quarter-final?"),
    MultilingualQueryVariant("gt-l2-03", "ar_msa", "ماذا حدث في الشوط الثاني من ربع نهائي Morocco ضد Portugal؟"),
    MultilingualQueryVariant("gt-l2-03", "ar_egy", "حصل ايه في الشوط التاني من ربع نهائي Morocco وPortugal؟"),

    # gt-l3-03 -- Kylian Mbappé, Poland
    MultilingualQueryVariant("gt-l3-03", "en", "How did Kylian Mbappé perform against Poland in the Round of 16?"),
    MultilingualQueryVariant("gt-l3-03", "ar_msa", "كيف كان أداء Kylian Mbappé أمام Poland في دور الستة عشر؟"),
    MultilingualQueryVariant("gt-l3-03", "ar_egy", "Kylian Mbappé لعب ازاي قدام Poland في دور الـ16؟"),

    # gt-l4-03 -- Antoine Griezmann
    MultilingualQueryVariant("gt-l4-03", "en", "Describe Antoine Griezmann's overall World Cup tournament performance."),
    MultilingualQueryVariant("gt-l4-03", "ar_msa", "صف أداء Antoine Griezmann العام في بطولة كأس العالم."),
    MultilingualQueryVariant("gt-l4-03", "ar_egy", "Antoine Griezmann لعب ازاي في البطولة كلها؟"),

    # gt-team-03 -- France
    MultilingualQueryVariant("gt-team-03", "en", "What were France's passing patterns and most common formations?"),
    MultilingualQueryVariant("gt-team-03", "ar_msa", "ما هي أنماط التمرير لدى France والتشكيلات الأكثر استخدامًا؟"),
    MultilingualQueryVariant("gt-team-03", "ar_egy", "France كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟"),

    # gt-multi-03 -- Morocco
    MultilingualQueryVariant("gt-multi-03", "en", "How did Morocco reach the semi-finals, and what style did they use?"),
    MultilingualQueryVariant("gt-multi-03", "ar_msa", "كيف وصلت Morocco إلى نصف النهائي، وما هو الأسلوب الذي استخدمته؟"),
    MultilingualQueryVariant("gt-multi-03", "ar_egy", "Morocco وصلت ازاي لنص النهائي، وكانت بتلعب باسلوب ايه؟"),

    # gt-l1-04 -- England, France
    MultilingualQueryVariant("gt-l1-04", "en", "Describe the quarter-final between England and France."),
    MultilingualQueryVariant("gt-l1-04", "ar_msa", "صف ربع النهائي بين England وFrance."),
    MultilingualQueryVariant("gt-l1-04", "ar_egy", "احكيلي عن ربع النهائي بين England وFrance."),

    # gt-l2-04 -- England, France
    MultilingualQueryVariant("gt-l2-04", "en", "What were the key turning points in the England vs France quarter-final?"),
    MultilingualQueryVariant("gt-l2-04", "ar_msa", "ما هي نقاط التحول الرئيسية في ربع نهائي England ضد France؟"),
    MultilingualQueryVariant("gt-l2-04", "ar_egy", "أهم نقاط التحول في ربع نهائي England وFrance كانت ايه؟"),

    # gt-l3-04 -- Enzo Fernández
    MultilingualQueryVariant("gt-l3-04", "en", "How did Enzo Fernández perform defensively in the World Cup Final?"),
    MultilingualQueryVariant("gt-l3-04", "ar_msa", "كيف كان أداء Enzo Fernández الدفاعي في نهائي كأس العالم؟"),
    MultilingualQueryVariant("gt-l3-04", "ar_egy", "Enzo Fernández لعب دفاع ازاي في نهائي كأس العالم؟"),

    # gt-l4-04 -- Enzo Fernández
    MultilingualQueryVariant("gt-l4-04", "en", "Describe Enzo Fernández's overall World Cup tournament performance."),
    MultilingualQueryVariant("gt-l4-04", "ar_msa", "صف أداء Enzo Fernández العام في بطولة كأس العالم."),
    MultilingualQueryVariant("gt-l4-04", "ar_egy", "Enzo Fernández لعب ازاي في البطولة كلها؟"),

    # gt-team-04 -- Germany
    MultilingualQueryVariant("gt-team-04", "en", "How did Germany play in the tournament, and which formations did they use?"),
    MultilingualQueryVariant("gt-team-04", "ar_msa", "كيف لعبت Germany في البطولة، وما هي التشكيلات التي استخدمتها؟"),
    MultilingualQueryVariant("gt-team-04", "ar_egy", "Germany لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟"),

    # gt-multi-04 -- Argentina, France
    MultilingualQueryVariant("gt-multi-04", "en", "How did Argentina and France differ in their playing styles at the World Cup?"),
    MultilingualQueryVariant("gt-multi-04", "ar_msa", "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟"),
    MultilingualQueryVariant("gt-multi-04", "ar_egy", "Argentina وFrance كانوا بيختلفوا ازاي في أسلوب لعبهم في كأس العالم؟"),
)


# ---------------------------------------------------------------------------
# Entity-script diagnostic (small, separate from the primary benchmark --
# see module docstring). Egyptian Arabic with Arabic-transliterated named
# entities, for the same case IDs and semantic intent as their ar_egy
# primary-benchmark counterpart -- spans the l3/team/multi groups, where
# named entities are most prominent.
# ---------------------------------------------------------------------------

ENTITY_SCRIPT_DIAGNOSTIC_VARIANTS: tuple[MultilingualQueryVariant, ...] = (
    # vs gt-pilot-l3-01 ar_egy: "Messi لعب ازاي قدام Croatia في نص النهائي؟"
    MultilingualQueryVariant("gt-pilot-l3-01", "ar_egy_translit", "ميسي لعب ازاي قدام كرواتيا في نص النهائي؟"),
    # vs gt-pilot-team-01 ar_egy: "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟"
    MultilingualQueryVariant("gt-pilot-team-01", "ar_egy_translit", "الأرجنتين كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟"),
    # vs gt-l3-02 ar_egy: "Antoine Griezmann عمل كام باسة في النهائي؟"
    MultilingualQueryVariant("gt-l3-02", "ar_egy_translit", "أنطوان جريزمان عمل كام باسة في النهائي؟"),
    # vs gt-team-03 ar_egy: "France كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟"
    MultilingualQueryVariant("gt-team-03", "ar_egy_translit", "فرنسا كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟"),
    # vs gt-l3-04 ar_egy: "Enzo Fernández لعب دفاع ازاي في نهائي كأس العالم؟"
    MultilingualQueryVariant("gt-l3-04", "ar_egy_translit", "إنزو فرنانديز لعب دفاع ازاي في نهائي كأس العالم؟"),
    # vs gt-multi-04 ar_egy: "Argentina وFrance كانوا بيختلفوا ازاي في أسلوب لعبهم في كأس العالم؟"
    MultilingualQueryVariant("gt-multi-04", "ar_egy_translit", "الأرجنتين وفرنسا كانوا بيختلفوا ازاي في أسلوب لعبهم في كأس العالم؟"),
)


def build_translated_cases(
    language: str,
    variants: tuple[MultilingualQueryVariant, ...] = MULTILINGUAL_QUERY_VARIANTS,
) -> list[dict]:
    """
    Build a translated case list for `language` by taking each English
    semantic ground-truth case verbatim (from
    tests.semantic_ground_truth.SEMANTIC_GROUND_TRUTH) and replacing only
    its `query` field with the matching variant's translated text.

    Relevance truth (relevant_document_ids, optional_relevant_document_ids,
    primary_level, acceptable_levels, case_group, etc.) is never
    duplicated or re-authored -- it is resolved from the single English
    source of truth, so MSA/EGY variants can never silently drift from
    what the English case actually verified.

    Cases with no variant for `language` are skipped (not silently given
    a fabricated query).
    """
    from tests.semantic_ground_truth import SEMANTIC_GROUND_TRUTH

    query_by_case_id = {
        v.source_case_id: v.query for v in variants if v.language == language
    }

    translated = []
    for case in SEMANTIC_GROUND_TRUTH:
        case_id = case["id"]
        if case_id not in query_by_case_id:
            continue
        new_case = dict(case)
        new_case["query"] = query_by_case_id[case_id]
        translated.append(new_case)
    return translated


def _validate_translated_cases(metadata: dict, cases: list, chunks_path) -> list[str]:
    """
    Structural validation for a translated case bundle, passed as
    GroundTruthBundle.validate_fn.

    Deliberately NOT the same checks as
    tests.semantic_ground_truth.validate_semantic_ground_truth: that
    validator pins an immutable content hash over the English cases
    (compute_canonical_case_hash), which exists to protect the English
    ground truth's wording from silent drift. A translated variant's query
    text is *supposed* to differ from the English source -- pinning that
    hash here would reject every legitimate translation. Document-ID-
    exists-in-chunks checks still happen unconditionally inside
    validate_ground_truth_and_chunks() regardless of this function, so
    they are intentionally not duplicated here.
    """
    errors: list[str] = []

    expected_count = metadata.get("expected_case_count")
    if expected_count is not None and len(cases) != expected_count:
        errors.append(
            f"Expected exactly {expected_count} cases, found {len(cases)}"
        )

    case_ids = [c.get("id") for c in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append(f"Duplicate case IDs found: {case_ids}")

    for case in cases:
        case_id = case.get("id", "?")
        if not case.get("query", "").strip():
            errors.append(f"[{case_id}] empty query")
        if not case.get("relevant_document_ids"):
            errors.append(f"[{case_id}] no required relevant documents")

    return errors


def build_ground_truth_bundle(language: str):
    """
    Build a GroundTruthBundle (see tests.retrieval_evaluator) for `language`,
    reusing the WC2022 Semantic Ground Truth's own metadata (same
    dataset_id, same chunks_sha256 -- this is the SAME chunks.json, only the
    query text differs) so run_retrieval_baseline()'s existing chunks-hash
    integrity check still applies unchanged.
    """
    from tests.retrieval_evaluator import GroundTruthBundle
    from tests.semantic_ground_truth import SEMANTIC_GROUND_TRUTH_METADATA

    cases = build_translated_cases(language)
    metadata = dict(SEMANTIC_GROUND_TRUTH_METADATA)
    metadata["expected_case_count"] = len(cases)
    metadata["language_variant"] = language

    return GroundTruthBundle(
        metadata=metadata,
        cases=cases,
        validate_fn=_validate_translated_cases,
    )
