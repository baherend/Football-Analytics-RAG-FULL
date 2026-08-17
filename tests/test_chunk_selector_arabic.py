"""
test_chunk_selector_arabic.py -- Arabic-Aware Chunk Selector phase.

Proves src.context.selection's tokenizer recognizes Arabic-script
query/content terms (previously ASCII-only: `[a-z0-9]+`), while preserving
exact existing English behavior and staying bounded/safe against
adversarial input.

Baseline evidence (see the phase report) established two distinct things:
1. The ASCII-only tokenizer genuinely discards all Arabic content from
   `_content_terms()` -- a real, confirmed defect (H1).
2. Because every chunk's `text` field in this corpus is English-only (and
   entity metadata is always Latin-script), fixing the tokenizer does NOT
   change `select_relevant_chunks()`'s final output for the real
   MSA/Egyptian benchmark queries -- content-term coverage against an
   English-only corpus is structurally empty regardless of tokenizer
   quality (verified empirically, not assumed).
3. The fix's real, demonstrable value is narrower: (a) it prevents the
   catastrophic `return []` early-exit for a hypothetical Arabic query
   with zero Latin/numeric content (previously: totally empty
   `raw_query_terms` -> discards every candidate; after: falls back
   gracefully like English's own no-lexical-evidence case), and (b) it
   makes the coverage-selection mechanism itself correct and usable
   whenever Arabic content genuinely exists on both the query and chunk
   side (proven with synthetic Arabic chunk text below, since the real
   corpus has none).
"""

from __future__ import annotations

import time

import pytest

from src.context.selection import (
    _TOKEN_PATTERN,
    _content_terms,
    select_relevant_chunks,
)


# ---------------------------------------------------------------------------
# Tokenizer: Arabic/MSA/Egyptian term recognition
# ---------------------------------------------------------------------------


def test_english_tokenization_unchanged():
    """Regression anchor -- must keep working exactly as before."""
    terms = _content_terms("What was Argentina's playing style?")
    assert terms == {"argentina", "s", "play", "style"}
    assert "what" not in terms  # stopword
    assert "was" not in terms  # stopword


def test_msa_semantic_terms_are_recognized():
    """MSA content words must produce non-empty tokens -- the core bug:
    previously _content_terms() returned an empty set for any Arabic-only
    text."""
    terms = _content_terms("ما هو أسلوب لعب الفريق والتشكيل الأكثر استخدامًا؟")
    assert terms, "MSA query produced zero tokens -- Arabic content is invisible to the tokenizer"
    assert any("اسلوب" in t or "أسلوب" in t for t in terms)


def test_egyptian_semantic_terms_are_recognized():
    terms = _content_terms("الفريق كان بيلعب ازاي وايه أكتر تشكيل استخدموه؟")
    assert terms, "Egyptian query produced zero tokens"
    assert any("تشكيل" in t for t in terms)


def test_mixed_arabic_latin_entity_extraction():
    """A query mixing a Latin entity name with Arabic content must
    tokenize BOTH -- the entity must remain intact and the Arabic content
    must no longer be silently dropped."""
    terms = _content_terms("Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟")
    assert "argentina" in terms
    assert any("تشكيل" in t for t in terms)
    assert any("بتلعب" in t or "لعب" in t for t in terms)


def test_arabic_terms_dedup_via_set_semantics():
    """The same Arabic word repeated must collapse to one set entry,
    exactly like English already does."""
    terms_once = _content_terms("تشكيل الفريق")
    terms_twice = _content_terms("تشكيل الفريق تشكيل")
    assert terms_once == terms_twice


def test_arabic_numeric_stat_query():
    """Numbers alongside Arabic text must still tokenize -- e.g. a stat
    question mixing a number with Arabic words."""
    terms = _content_terms("سجل 3 اهداف في البطولة")
    assert "3" in terms
    assert any(t for t in terms if t != "3")  # some Arabic content term too


# ---------------------------------------------------------------------------
# Punctuation: Arabic punctuation must not attach to word tokens
# ---------------------------------------------------------------------------


def test_arabic_punctuation_is_excluded_from_tokens():
    """Arabic question mark / comma / semicolon must act as separators,
    not become part of the adjacent word token -- otherwise "تشكيل؟" and
    "تشكيل" would never be recognized as the same term."""
    terms = _content_terms("تشكيل؟")
    assert "تشكيل" in terms
    assert "تشكيل؟" not in terms

    terms_comma = _content_terms("تشكيل، أسلوب؛ لعب")
    assert "تشكيل" in terms_comma
    assert "تشكيل،" not in terms_comma
    assert "أسلوب؛" not in terms_comma


def test_english_punctuation_still_excluded():
    """Regression: English punctuation handling must stay exactly as
    before."""
    terms = _content_terms("formation? style, tactics;")
    assert "formation" in terms
    assert "formation?" not in terms


# ---------------------------------------------------------------------------
# Unicode / RTL / malformed input safety
# ---------------------------------------------------------------------------


def test_mixed_rtl_ltr_does_not_crash():
    terms = _content_terms("Argentina تشكيل France أسلوب")
    assert "argentina" in terms
    assert "france" in terms


def test_empty_and_whitespace_strings_are_safe():
    assert _content_terms("") == set()
    assert _content_terms(" ") == set()
    assert _content_terms(None) == set()  # _content_terms does `text or ""`


def test_extremely_long_arabic_input_is_bounded():
    payload = "تشكيل " * 50_000
    start = time.perf_counter()
    terms = _content_terms(payload)
    elapsed = time.perf_counter() - start
    assert terms == {"تشكيل"}
    assert elapsed < 2.0, f"took {elapsed:.2f}s on a 50000-word repeated payload"


def test_extremely_long_mixed_input_is_bounded():
    payload = ("Argentina تشكيل " * 25_000) + "ازاي؟"
    start = time.perf_counter()
    terms = _content_terms(payload)
    elapsed = time.perf_counter() - start
    assert "argentina" in terms
    assert elapsed < 2.0, f"took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# select_relevant_chunks(): the catastrophic early-exit bug
# ---------------------------------------------------------------------------


def test_pure_arabic_query_no_longer_returns_empty_when_candidates_exist():
    """The confirmed bug: a query with ZERO Latin/numeric content (a
    legitimate, if rare, Arabic query naming no entity) previously
    produced raw_query_terms = set() -> select_relevant_chunks()
    returned [] unconditionally, discarding every candidate regardless of
    relevance. After the fix, Arabic content terms make raw_query_terms
    non-empty, so the function no longer takes that catastrophic
    early-exit -- it falls back gracefully (matching English's own
    no-lexical-evidence-overlap fallback), same as
    test_falls_back_to_original_ranking_when_no_lexical_evidence already
    established for English."""
    chunks = [
        {
            "chunk_id": "TEAM-1-chunk-0",
            "text": "The team controlled possession and passed patiently.",
            "metadata": {"document_id": "TEAM-1", "level": "team"},
            "score": 0.9, "rank": 1, "source": "hybrid",
        },
    ]
    selected = select_relevant_chunks(
        query="ما هو الأسلوب الأكثر استخدامًا في البطولة؟",  # no entity, no Latin content at all
        chunks=chunks,
        max_chunks=1,
    )
    assert selected == chunks[:1], (
        "a genuinely Arabic-only query (no Latin content) must fall back to the "
        "existing ranking, not discard every candidate outright"
    )


# ---------------------------------------------------------------------------
# Coverage-selection mechanism: proven correct with synthetic Arabic
# content on BOTH sides (the real corpus has none -- see module docstring)
# ---------------------------------------------------------------------------


def test_coverage_selection_prefers_arabic_facet_coverage_over_redundant_chunk():
    """Integration-style proof that the coverage-maximizing selection
    algorithm itself works correctly for Arabic once content exists on
    both the query and chunk side -- mirrors
    test_prefers_new_query_coverage_over_redundant_chunk's English
    version exactly, with Arabic substituted throughout."""
    chunks = [
        {
            "chunk_id": "DOC-1-part-1",
            "text": "الفريق حافظ على الاستحواذ ومرر الكرة بصبر. أسلوب اللعب كان هجومي.",
            "metadata": {"document_id": "DOC-1", "level": "team"},
            "score": 0.9, "rank": 1, "source": "hybrid",
        },
        {
            "chunk_id": "DOC-2-part-1",
            "text": "التشكيل الذي استخدمه الفريق كان 4-3-3 طوال البطولة.",
            "metadata": {"document_id": "DOC-2", "level": "team"},
            "score": 0.7, "rank": 2, "source": "hybrid",
        },
        {
            "chunk_id": "DOC-1-part-2",
            "text": "أسلوب اللعب كان هجومي مع التركيز على الاستحواذ.",  # redundant with DOC-1-part-1
            "metadata": {"document_id": "DOC-1", "level": "team"},
            "score": 0.6, "rank": 3, "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="ما هو أسلوب اللعب والتشكيل الذي استخدمه الفريق؟",
        chunks=chunks,
        max_chunks=2,
    )

    selected_ids = [c["chunk_id"] for c in selected]
    # Must jointly cover both facets (أسلوب/style, تشكيل/formation) using
    # the two chunks that each add NEW coverage, not two chunks that both
    # cover the same facet redundantly.
    assert "DOC-1-part-1" in selected_ids
    assert "DOC-2-part-1" in selected_ids
    assert "DOC-1-part-2" not in selected_ids


def test_coverage_selection_entity_filtering_works_for_arabic_entities_in_content():
    """The entity-grounding filter (discard chunks about an unmentioned
    entity) must also work once Arabic content makes query_evidence_terms
    non-empty and the coverage loop is actually reached."""
    chunks = [
        {
            "chunk_id": "TEAM-A-chunk-0",
            "text": "أسلوب اللعب لدى الفريق أ كان هجوميا وسريعا.",
            "metadata": {"document_id": "TEAM-A", "level": "team", "team_name": "TeamA"},
            "score": 0.9, "rank": 1, "source": "hybrid",
        },
        {
            "chunk_id": "TEAM-B-chunk-0",
            "text": "أسلوب اللعب لدى الفريق ب كان دفاعيا وبطيئا.",
            "metadata": {"document_id": "TEAM-B", "level": "team", "team_name": "TeamB"},
            "score": 0.8, "rank": 2, "source": "hybrid",
        },
    ]

    selected = select_relevant_chunks(
        query="ما هو أسلوب لعب TeamA؟",
        chunks=chunks,
        max_chunks=2,
    )

    selected_ids = {c["chunk_id"] for c in selected}
    assert "TEAM-A-chunk-0" in selected_ids
    assert "TEAM-B-chunk-0" not in selected_ids, (
        "entity grounding must exclude the unmentioned team's chunk once "
        "Arabic content makes the coverage loop reachable"
    )


# ---------------------------------------------------------------------------
# Security: ReDoS, injection payloads, uncontrolled growth
# ---------------------------------------------------------------------------


def test_tokenizer_is_redos_safe_on_adversarial_arabic_payloads():
    """The two alternation branches ([a-z0-9]+ and the Arabic range) are
    disjoint character classes -- no character can match both -- so there
    is no ambiguous-overlap backtracking shape to exploit, unlike the
    ReDoS patterns found and fixed in the Arabic Safeguards phase."""
    payloads = [
        "\u062a\u0634\u0643\u064a\u0644" * 200_000,           # repeated Arabic word, 1M chars
        ("\u0627" + "a") * 50_000,                             # alternating Arabic/Latin, 100k chars
        ("\u062a\u0634\u0643\u064a\u0644\u061f\u060c\u061b" * 20_000),  # Arabic word + punctuation, repeated
    ]
    for payload in payloads:
        start = time.perf_counter()
        _content_terms(payload)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"took {elapsed:.3f}s on a {len(payload)}-char payload"


def test_injection_payloads_remain_inert_text():
    """SQL/shell/path/Chroma-filter-operator payloads must tokenize as
    ordinary (harmless) words -- never interpreted as anything else, and
    must never crash select_relevant_chunks()."""
    payloads = [
        "'; DROP TABLE embeddings; --",
        '" OR 1=1 --',
        "'); DELETE FROM collections; --",
        "\u0645\u064a\u0633\u064a'; DROP TABLE embeddings; --",
        "Messi & del C:\\*",
        "Messi; rm -rf /",
        "$(whoami)",
        "../../output/chroma_db",
        "..\\..\\output\\competitions",
        '{"$where": "true"}',
        '{"$ne": ""}',
    ]
    chunks = [{
        "chunk_id": "X-1", "text": "some content",
        "metadata": {"document_id": "X-1", "level": "team"},
        "score": 0.5, "rank": 1, "source": "hybrid",
    }]
    for payload in payloads:
        _content_terms(payload)  # must not raise
        result = select_relevant_chunks(payload, chunks, max_chunks=3)
        assert isinstance(result, list)


def test_many_distinct_arabic_tokens_stay_bounded_and_fast():
    """Uncontrolled token/list growth check: thousands of genuinely
    distinct Arabic-looking words must not cause quadratic or unbounded
    behavior -- extraction is linear in input length regardless of how
    many distinct tokens result."""
    import itertools

    arabic_letters = "\u0627\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u064a"
    distinct_words = ["".join(p) for p in itertools.islice(
        itertools.product(arabic_letters, repeat=4), 5000,
    )]
    payload = " ".join(distinct_words)

    start = time.perf_counter()
    terms = _content_terms(payload)
    elapsed = time.perf_counter() - start

    assert len(terms) <= 5000
    assert elapsed < 2.0, f"took {elapsed:.3f}s on {len(distinct_words)} distinct tokens"
