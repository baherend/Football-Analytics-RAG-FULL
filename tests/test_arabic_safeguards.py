"""
test_arabic_safeguards.py -- Arabic Retrieval Safeguards phase.

Proves the retrieval-safeguard trigger/detection layer in
src.retrieval.search (_detect_team_style_query, _detect_comparison_entities,
and the _ensure_*_doc() functions that consume them) recognizes equivalent
MSA/Egyptian retrieval intent, not just English phrasing -- while
preserving exact existing English behavior. Deliberately does NOT touch
Dense/BM25/RRF, the chunk selector, entity normalization, or stage/match
detection (see the phase's causal-isolation boundary).

Entity-script control: per this phase's explicit scope, all Arabic test
queries below keep player/team entities in their canonical Latin form
(e.g. "Argentina لعبت ازاي؟") -- Arabic-transliterated entities (e.g.
"الأرجنتين") are a separate, deliberately deferred failure mode.
"""

from __future__ import annotations

import time

import pytest

import src.retrieval.search as search


# ---------------------------------------------------------------------------
# Team-style / formation / tactics detection
# ---------------------------------------------------------------------------


def test_team_style_english_playing_style_unchanged():
    """English regression anchor -- must keep working exactly as before."""
    q = "What was Argentina's playing style and most common formation?"
    assert search._detect_team_style_query(q) == "Argentina"


def test_team_style_english_passing_patterns_unchanged():
    """Pre-existing test (test_router.py) restated here as an explicit
    regression anchor for this phase."""
    q = "What were France's passing patterns and most common formations?"
    assert search._detect_team_style_query(q) == "France"


def test_team_style_msa_style_and_formation():
    q = "ما هو أسلوب لعب Argentina والتشكيل الأكثر استخدامًا؟"
    assert search._detect_team_style_query(q) == "Argentina"


def test_team_style_egy_playing_how_and_formation():
    q = "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟"
    assert search._detect_team_style_query(q) == "Argentina"


def test_team_style_msa_formation_only():
    q = "كيف لعبت Morocco في البطولة، وما هي التشكيلات التي استخدمتها أكثر؟"
    assert search._detect_team_style_query(q) == "Morocco"


def test_team_style_egy_formation_only():
    q = "Morocco لعبت ازاي في البطولة، وايه أكتر تشكيل لعبوا بيه؟"
    assert search._detect_team_style_query(q) == "Morocco"


def test_team_style_msa_passing_patterns():
    q = "ما هي أنماط التمرير لدى France والتشكيلات الأكثر استخدامًا؟"
    assert search._detect_team_style_query(q) == "France"


def test_team_style_egy_passing():
    q = "France كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟"
    assert search._detect_team_style_query(q) == "France"


def test_team_style_msa_germany_formations():
    q = "كيف لعبت Germany في البطولة، وما هي التشكيلات التي استخدمتها؟"
    assert search._detect_team_style_query(q) == "Germany"


def test_team_style_egy_germany_formations():
    q = "Germany لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟"
    assert search._detect_team_style_query(q) == "Germany"


def test_team_style_egy_played_how_without_formation_word():
    """Task's own illustrative phrasing (spelled with hamza -- إزاي) must
    also trigger, exercising the "played how" phrase alone, without any
    formation/style keyword present, and proving alef-hamza spelling
    variants (إزاي vs ازاي) are handled."""
    q = "Argentina لعبهم كان عامل إزاي في البطولة؟"
    assert search._detect_team_style_query(q) == "Argentina"


def test_team_style_egy_hamza_spelling_variant_of_how():
    """Same intent, alternate common spelling of 'how' without the hamza
    (ازاي) -- both spellings must resolve to the same detection."""
    q = "Argentina كانت بتلعب ازاي في البطولة؟"
    assert search._detect_team_style_query(q) == "Argentina"


# ---------------------------------------------------------------------------
# Comparison detection
# ---------------------------------------------------------------------------


def test_comparison_english_compare_and_unchanged():
    """English regression anchor."""
    assert search._detect_comparison_entities("Compare Messi and Mbappe") == ["messi", "mbappe"]


def test_comparison_english_vs_unchanged():
    assert search._detect_comparison_entities("Messi vs Mbappe") == ["messi", "mbappe"]


def test_comparison_english_who_better_unchanged():
    assert search._detect_comparison_entities("Who was better, Messi or Mbappe?") == ["messi", "mbappe"]


def test_comparison_msa_qaran_bayn():
    q = "قارن بين Messi و Mbappe"
    assert search._detect_comparison_entities(q) == ["messi", "mbappe"]


def test_comparison_egy_meen_ahsan_wala():
    q = "مين كان أحسن Messi ولا Mbappe؟"
    assert search._detect_comparison_entities(q) == ["messi", "mbappe"]


def test_comparison_egy_reversed_order():
    """Egyptian word order can put both entities before the question word --
    a structure with no English equivalent regex to reuse."""
    q = "Messi ولا Mbappe مين كان أفضل؟"
    assert search._detect_comparison_entities(q) == ["messi", "mbappe"]


def test_comparison_msa_al_afdal_bayn():
    q = "مين الأفضل بين Messi و Mbappe؟"
    assert search._detect_comparison_entities(q) == ["messi", "mbappe"]


# ---------------------------------------------------------------------------
# False-positive controls
# ---------------------------------------------------------------------------


def test_team_style_no_false_positive_on_numeric_arabic_query():
    """A structured numeric question in Egyptian Arabic must not be
    misclassified as team-style intent -- it contains no style/formation
    trigger phrase at all."""
    q = "كام هدف سجل Messi؟"
    assert search._detect_team_style_query(q) is None


def test_team_style_no_false_positive_pure_arabic_no_entity():
    """A style-triggering phrase with no Latin-script entity present must
    not fabricate a team name -- the safeguard should decline to fire
    rather than guess."""
    q = "أسلوب اللعب كان عامل ازاي في المباراة؟"
    assert search._detect_team_style_query(q) is None


def test_comparison_no_false_positive_on_and_without_comparison_words():
    """Arabic 'و' (and) is an extremely common conjunction -- two Latin
    entities joined by 'و' without an actual comparison trigger word
    (قارن/أحسن/أفضل) must NOT be misread as comparison intent. This is a
    deliberate precision choice: unlike English's broad 'X or Y' fallback,
    no bare 'X و Y' Arabic fallback is implemented (see design decision in
    the phase report) specifically to avoid this false-positive class."""
    q = "Argentina و France لعبوا كويس جدا في البطولة"
    assert search._detect_comparison_entities(q) == []


def test_comparison_no_false_positive_single_entity_superlative():
    """A single-entity superlative claim (no second entity, no 'بين'
    structure) must not be misread as a two-entity comparison."""
    q = "Messi هو الأفضل في الفريق"
    assert search._detect_comparison_entities(q) == []


def test_team_style_no_false_positive_on_english_unrelated_query():
    """Sanity: unrelated English query with no trigger word must still
    return None (regression, not a new behavior)."""
    assert search._detect_team_style_query("How many goals did Messi score?") is None


def test_comparison_no_false_positive_on_english_unrelated_query():
    assert search._detect_comparison_entities("How many goals did Messi score?") == []


# ---------------------------------------------------------------------------
# Adversarial audit additions -- found via active red-teaming, not just
# incremental coverage. Each test below protects a specific, concretely
# demonstrated failure mode discovered during the audit.
# ---------------------------------------------------------------------------


# --- ReDoS / regex-complexity regressions -----------------------------------
#
# Both detectors previously exhibited O(N^2)-or-worse backtracking on long,
# adversarial, ultimately-non-matching input: _detect_comparison_entities's
# pre-existing English "X vs Y"/"X or Y" patterns, and this phase's own new
# "X ولا Y مين كان أفضل" Arabic pattern, all shared the same vulnerable
# shape -- an unbounded quantified group with no fixed literal prefix for
# re.search() to fast-scan for, so long non-matching input forced exhaustive
# backtracking across every retry position. Confirmed via subprocess-isolated
# timeout testing: all 4 payloads below hung indefinitely (>60s) before the
# fix (bounding every entity-capturing quantifier to 60 chars -- more than
# enough for any real name) and complete in well under 1 second after.
#
# 2-second bound chosen deliberately generous (fixed code completes these in
# well under 0.5s) to avoid flakiness while still catching a real regression
# back to unbounded/catastrophic behavior.

_REDOS_TIMEOUT_SECONDS = 2.0

_REDOS_PAYLOADS = {
    "pure_arabic_100k": "\u0627" * 100_000,
    "repeated_latin_word_60k": "Messi " * 10_000,
    "long_latin_run_100k": "A" * 100_000 + " تشكيل ازاي؟",
    "alternating_latin_arabic_40k": ("A" + "\u0627") * 20_000,
}


@pytest.mark.parametrize("payload", _REDOS_PAYLOADS.values(), ids=_REDOS_PAYLOADS.keys())
def test_team_style_detector_is_redos_safe(payload):
    start = time.perf_counter()
    search._detect_team_style_query(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_SECONDS, (
        f"_detect_team_style_query took {elapsed:.2f}s on a {len(payload)}-char "
        f"adversarial payload -- possible regex-complexity regression."
    )


@pytest.mark.parametrize("payload", _REDOS_PAYLOADS.values(), ids=_REDOS_PAYLOADS.keys())
def test_comparison_detector_is_redos_safe(payload):
    start = time.perf_counter()
    search._detect_comparison_entities(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_SECONDS, (
        f"_detect_comparison_entities took {elapsed:.2f}s on a {len(payload)}-char "
        f"adversarial payload -- possible regex-complexity regression."
    )


# --- Comparison-entity extraction correctness (found via adversarial testing) --


def test_comparison_msa_qaran_maa():
    """"قارن X مع Y" (compare X with Y, using مع instead of بين...و) --
    a natural MSA phrasing found missing during adversarial testing."""
    assert search._detect_comparison_entities("قارن Griezmann مع Rabiot") == ["griezmann", "rabiot"]


def test_comparison_tolerates_comma_before_connector():
    """A comma directly after the first entity (a natural typed pause)
    must not break extraction -- found broken during adversarial testing
    ("قارن بين Messi, و Mbappe" previously returned [])."""
    assert search._detect_comparison_entities("قارن بين Messi, و Mbappe") == ["messi", "mbappe"]


def test_comparison_preserves_full_multiword_entity_names():
    """The trailing entity in "قارن بين X و Y" must capture the FULL
    multi-word name, not just its first word. Found broken during
    adversarial testing: the previous non-greedy trailing capture stopped
    at the first internal space, since bare whitespace already satisfied
    its generic end-of-phrase terminator."""
    result = search._detect_comparison_entities(
        "قارن بين Kylian Mbappe Lottin و Lionel Andres Messi"
    )
    assert result == ["kylian mbappe lottin", "lionel andres messi"]


def test_comparison_three_entities_keeps_only_first_two():
    """Matches the pre-existing English "compare X and Y" contract exactly
    -- a third mentioned entity is dropped, not an error."""
    result = search._detect_comparison_entities("قارن بين Messi و Mbappe و Neymar")
    assert result == ["messi", "mbappe"]
    assert "neymar" not in result


def test_comparison_hyphenated_and_apostrophe_names():
    assert search._detect_comparison_entities("قارن بين Al-Dawsari و Messi") == ["al-dawsari", "messi"]
    assert search._detect_comparison_entities("قارن بين O'Brien و Messi") == ["o'brien", "messi"]


def test_comparison_trigger_works_inside_surrounding_arabic_text():
    """The comparison trigger need not be at the start of the query."""
    result = search._detect_comparison_entities(
        "في المباراة النهائية قارن بين Messi و Mbappe من فضلك"
    )
    assert result == ["messi", "mbappe"]


# --- Team-style detector correctness (found via adversarial testing) --------


def test_team_style_msa_khutta_plan_synonym():
    """"الخطة" (the [tactical] plan) is a formation synonym explicitly
    listed as an example concept for this phase, missing from the
    original keyword set -- found via adversarial testing."""
    assert search._detect_team_style_query("Argentina الخطة كانت ايه؟") == "Argentina"


def test_team_style_egy_khutta_plan_synonym():
    assert search._detect_team_style_query("Argentina كانوا بيلعبوا بخطة ايه؟") == "Argentina"


def test_team_style_tolerates_irregular_whitespace_in_trigger_phrase():
    """Double/triple spacing inside a multi-word Arabic trigger phrase
    (e.g. "بتلعب   ازاي") previously broke the literal substring check --
    found via adversarial testing."""
    assert search._detect_team_style_query("Argentina  كانت   بتلعب   ازاي؟") == "Argentina"


def test_team_style_entity_placement_start_middle_end():
    """The Latin entity span must be found regardless of its position
    relative to the Arabic trigger phrase."""
    assert search._detect_team_style_query("Argentina تشكيل ايه؟") == "Argentina"
    assert search._detect_team_style_query("تشكيل Argentina ايه؟") == "Argentina"
    assert search._detect_team_style_query("تشكيل ايه Argentina؟") == "Argentina"


# --- Malformed / out-of-contract input --------------------------------------
#
# _detect_team_style_query/_detect_comparison_entities are typed to accept
# `str`. Traced the full call chain (hybrid_search -> _ensure_team_style_doc/
# _ensure_comparison_entities -> these detectors) back to every real caller
# (chat.py's input()-sourced question, src/query/router.py's CLI argv,
# route.semantic_query): query is always a str by construction before
# reaching this layer, so None/int/list/dict are not reachable from the
# public API in practice. These tests document that the failure mode for
# such (unreachable) inputs is a clean, immediate AttributeError -- not a
# silent wrong-answer, hang, or security-relevant behavior -- rather than
# silently forcing support for types the caller contract excludes.


@pytest.mark.parametrize("bad_input", [None, 123, [], {}])
def test_team_style_detector_fails_cleanly_on_non_string_input(bad_input):
    with pytest.raises(AttributeError):
        search._detect_team_style_query(bad_input)


@pytest.mark.parametrize("bad_input", [None, 123, [], {}])
def test_comparison_detector_fails_cleanly_on_non_string_input(bad_input):
    with pytest.raises(AttributeError):
        search._detect_comparison_entities(bad_input)


def test_team_style_detector_handles_extreme_but_valid_strings_without_crashing():
    """Very long strings, emoji, mixed RTL/LTR, null characters, and other
    control characters must never crash the detector -- only str inputs
    that are merely unusual, not out-of-contract."""
    weird_but_valid = [
        "\u0627" * 10_000 + " تشكيل Argentina",
        "Argentina \U0001F1E6\U0001F1F7\u26bd تشكيل ايه؟",
        "Argentina تشكيل France ايه؟",
        "Argentina تشكيل\x00 ايه؟",
        "Argentina\x01\x02 تشكيل ايه؟",
        "",
        " ",
    ]
    for q in weird_but_valid:
        search._detect_team_style_query(q)  # must not raise


def test_comparison_detector_handles_extreme_but_valid_strings_without_crashing():
    weird_but_valid = [
        "\u0627" * 10_000 + " قارن بين Messi و Mbappe",
        "قارن بين Messi\x00 و Mbappe",
        "",
        " ",
    ]
    for q in weird_but_valid:
        search._detect_comparison_entities(q)  # must not raise


# --- Confirmed pre-existing, out-of-scope defects (documented, not fixed) ---


def test_confirmed_deferred_english_formation_regex_bug():
    """CONFIRMED DEFERRED ISSUE: _detect_team_style_query's pre-existing
    English "(.+?)\\s+(?:formations?|...)" pattern mis-extracts "What" as
    the team name for "What formation did Germany use?" -- the non-greedy
    group stops at the shortest possible match ("what"), not "Germany".
    This is a pre-existing English regex defect, unrelated to and not
    introduced by Arabic-safeguard work, and out of scope for this phase
    (English behavior must not be silently redesigned here). Documented so
    this known limitation is never mistaken for an Arabic-specific gap."""
    assert search._detect_team_style_query("What formation did Germany use?") == "What"


def test_confirmed_deferred_accent_matching_limitation_in_comparison_injection():
    """CONFIRMED DEFERRED ISSUE: _ensure_comparison_entities's L4-document
    lookup uses plain substring matching (`entity_lower in player_name`),
    so an extracted entity spelled without diacritics ("mbappe") never
    matches a chunk store player_name spelled with them ("Kylian Mbappé
    Lottin"). Verified language-independent -- affects English, MSA, and
    Egyptian queries identically (see the phase report's real feature
    test). Detection itself is correct (both entities are extracted); the
    defect is downstream, in the pre-existing, unmodified injection
    lookup. Out of scope for this phase (not Arabic-specific, not a
    regression, requires entity-normalization work explicitly deferred to
    a separate phase) -- documented here so it is never mistaken for an
    Arabic-parity failure."""
    entities = search._detect_comparison_entities("Compare Messi and Mbappe")
    assert entities == ["messi", "mbappe"], "detection itself is correct"
    # The accented chunk-store name never matches the unaccented extracted
    # entity via plain substring containment -- the documented defect.
    assert "mbappe" not in "kylian mbappé lottin"


# --- Retrieval-consequence tests: the injection functions themselves -------
#
# Every test above only checks the DETECTOR's return value. None of them
# would catch a bug in _ensure_team_style_doc/_ensure_comparison_entities
# (the functions that actually inject a document into the results) -- e.g.
# injecting the wrong document, injecting into the wrong position, or a
# broken dedup. These tests close that gap by exercising the injection
# functions directly against synthetic chunk data, asserting on the
# RESULTING candidate list, not just the detector's classification.

_FIXTURE_CHUNKS = [
    {
        "chunk_id": "TEAM-argentina-chunk-0", "document_id": "TEAM-argentina-doc",
        "level": "team", "team_name": "Argentina",
        "text": "Argentina's tactical analysis: 4-3-3 formation.",
    },
    {
        "chunk_id": "TEAM-france-chunk-0", "document_id": "TEAM-france-doc",
        "level": "team", "team_name": "France",
        "text": "France's tactical analysis: 4-2-3-1 formation.",
    },
    {
        "chunk_id": "L4-messi-chunk-0", "document_id": "L4-messi-doc",
        "level": "4", "player_name": "Messi",
        "text": "Messi's tournament summary.",
    },
    {
        "chunk_id": "L4-mbappe-chunk-0", "document_id": "L4-mbappe-doc",
        "level": "4", "player_name": "Mbappe",
        "text": "Mbappe's tournament summary.",
    },
]


def test_ensure_team_style_doc_injects_correct_team_not_wrong_one(monkeypatch):
    """Retrieval-consequence test: an Arabic team-style query for Argentina
    must inject ARGENTINA's document, never France's, even though both
    exist in the chunk store."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    empty_results = []
    boosted = search._ensure_team_style_doc(
        "Argentina كانت بتلعب ازاي؟", empty_results, k=5,
    )

    assert len(boosted) == 1
    assert boosted[0]["chunk_id"] == "TEAM-argentina-chunk-0"
    assert boosted[0]["metadata"]["team_name"] == "Argentina"


def test_ensure_team_style_doc_does_not_fire_without_arabic_trigger(monkeypatch):
    """Negative retrieval-consequence test: a query that merely mentions a
    team name (no style/formation trigger) must not inject anything."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    empty_results = []
    result = search._ensure_team_style_doc(
        "Argentina سجلت كام هدف؟", empty_results, k=5,
    )
    assert result == []


def test_ensure_team_style_doc_no_op_when_team_already_present(monkeypatch):
    """If the team's document is already in the top-k, the safeguard must
    not duplicate it."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    already_present = [{
        "chunk_id": "TEAM-argentina-chunk-0",
        "metadata": {"level": "team", "team_name": "Argentina"},
    }]
    result = search._ensure_team_style_doc(
        "Argentina كانت بتلعب ازاي؟", already_present, k=5,
    )
    assert result == already_present


def test_ensure_comparison_entities_injects_both_real_documents(monkeypatch):
    """Retrieval-consequence test: an Arabic comparison query must inject
    BOTH real players' L4 documents, not just one, and not a wrong one."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    empty_results = []
    boosted = search._ensure_comparison_entities(
        "قارن بين Messi و Mbappe", empty_results, k=5,
    )

    chunk_ids = {r["chunk_id"] for r in boosted}
    assert chunk_ids == {"L4-messi-chunk-0", "L4-mbappe-chunk-0"}


def test_ensure_comparison_entities_does_not_fire_on_single_entity(monkeypatch):
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    result = search._ensure_comparison_entities(
        "أفضل هدف سجله Messi", [], k=5,
    )
    assert result == []


# --- SQL / shell / path injection: retrieval-level proof --------------------
#
# _detect_team_style_query/_detect_comparison_entities never construct SQL,
# shell commands, or filesystem paths from query text (confirmed by source
# inspection -- see the phase report's Parts 10-13). These tests provide
# empirical, executable proof at the retrieval-consequence level: malicious
# payloads must be treated as ordinary (non-matching) text, causing no
# injection/crash/mutation, through the actual injection functions.

_INJECTION_PAYLOADS = [
    "'; DROP TABLE embeddings; --",
    '" OR 1=1 --',
    "'); DELETE FROM collections; --",
    "Messi'; DROP TABLE chunks; --",
    "ميسي'; DROP TABLE embeddings; --",
    "Messi & del C:\\*",
    "Messi; rm -rf /",
    "$(whoami)",
    "`whoami`",
    "../../output/chroma_db",
    "..\\..\\output\\competitions",
    '{"$where": "true"}',
    '{"$ne": ""}',
]


@pytest.mark.parametrize("payload", _INJECTION_PAYLOADS)
def test_injection_payloads_are_treated_as_ordinary_text(payload, monkeypatch):
    """No malicious payload may crash the detectors or the injection
    functions, and none may cause a document to be injected (none of
    these payloads name a real team/player), proving the payload was
    never interpreted as anything other than literal query text."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    # Must not raise.
    style = search._detect_team_style_query(payload)
    entities = search._detect_comparison_entities(payload)

    style_result = search._ensure_team_style_doc(payload, [], k=5)
    comparison_result = search._ensure_comparison_entities(payload, [], k=5)

    # None of these payloads name a real team/player, so nothing should
    # ever be injected -- proves the payload stayed inert text throughout.
    assert style_result == []
    assert comparison_result == []


# ---------------------------------------------------------------------------
# Multi-entity team-style collision fix (found in the adversarial audit):
# _detect_team_style_query()'s Arabic branch called _extract_latin_entity_span
# (singular, first-match-only), so a genuinely two-team style-comparison
# query (e.g. gt-multi-04: "كيف اختلفت Argentina وFrance في أسلوب
# لعبهما...") silently collapsed to just the first-mentioned team
# (Argentina), never injecting France's document even though the
# committed Ground Truth (gt-multi-04.relevant_document_ids = ['TEAM-779',
# 'TEAM-771']) requires BOTH. gt-multi-03 ("...Morocco...الأسلوب...") is
# the important contrasting case: it genuinely names only ONE team, and
# Ground Truth confirms that team's document (TEAM-788) IS required -- so
# the fix must keep injecting for single-entity queries, not merely
# suppress the safeguard whenever "multi" is in the case name.
# ---------------------------------------------------------------------------


def test_team_style_single_entity_still_returns_that_entity():
    """Regression anchor matching gt-multi-03's shape: exactly one Latin
    team name in an otherwise-Arabic style query must still resolve to
    that team, unchanged."""
    q = "كيف وصلت Morocco إلى نصف النهائي، وما هو الأسلوب الذي استخدمته؟"
    assert search._detect_team_style_query(q) == "Morocco"


def test_team_style_two_entities_is_no_longer_silently_first_one():
    """The core bug: a genuinely two-team Arabic style query must not
    silently resolve to whichever team happens to be mentioned first.
    matches gt-multi-04's exact shape (two teams, "أسلوب" style trigger)."""
    q = "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟"
    assert search._detect_team_style_query(q) != "Argentina"
    assert search._detect_team_style_query(q) is None


def test_team_style_two_entities_egy_is_no_longer_silently_first_one():
    q = "Argentina وFrance كانوا بيختلفوا ازاي في أسلوب لعبهم في كأس العالم؟"
    assert search._detect_team_style_query(q) is None


def test_detect_team_style_entities_returns_both_for_two_team_query():
    """The new plural detector must surface BOTH teams -- not silently
    pick one -- exactly matching what gt-multi-04's Ground Truth requires
    (both TEAM-779 and TEAM-771)."""
    q = "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟"
    assert search._detect_team_style_entities(q) == ["Argentina", "France"]


def test_detect_team_style_entities_single_team_unchanged():
    q = "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟"
    assert search._detect_team_style_entities(q) == ["Argentina"]


def test_detect_team_style_entities_no_trigger_returns_empty():
    assert search._detect_team_style_entities("كام هدف سجل Messi؟") == []


def test_team_style_english_two_entities_behavior_is_unchanged_by_this_fix():
    """Protect existing English behavior: this fix only changes the
    Arabic branch. Whatever the pre-existing (imperfect) English
    two-entity behavior was, it must be bit-for-bit identical after this
    patch -- English redesign is explicitly out of scope."""
    q = "What was Argentina and France's playing style?"
    assert search._detect_team_style_query(q) == "Argentina And France"


def test_team_style_three_entities_does_not_crash_and_is_bounded():
    """More than two named teams must never crash or silently pick one
    at random -- either all (bounded) get returned or none do, but the
    function must complete and return a list."""
    q = "قارن أسلوب لعب Argentina وFrance وGermany في البطولة"
    result = search._detect_team_style_entities(q)
    assert isinstance(result, list)
    assert len(result) <= 5  # bounded, see _extract_latin_entity_spans


def test_ensure_team_style_doc_injects_both_teams_for_two_team_query(monkeypatch):
    """Retrieval-consequence test, the actual bug: for a genuine two-team
    style query, BOTH teams' documents must be injected -- matching
    gt-multi-04's Ground Truth (['TEAM-779', 'TEAM-771']) exactly, not
    just the first-mentioned team's."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    boosted = search._ensure_team_style_doc(
        "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟", [], k=5,
    )

    chunk_ids = {r["chunk_id"] for r in boosted}
    assert chunk_ids == {"TEAM-argentina-chunk-0", "TEAM-france-chunk-0"}


def test_ensure_team_style_doc_single_team_still_injects_exactly_one(monkeypatch):
    """Regression anchor: the single-team case (gt-multi-03-shaped, and
    all 8 real team-group benchmark cases) must inject exactly one
    document, unchanged."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    boosted = search._ensure_team_style_doc(
        "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟", [], k=5,
    )

    assert len(boosted) == 1
    assert boosted[0]["chunk_id"] == "TEAM-argentina-chunk-0"


def test_ensure_team_style_doc_two_teams_one_already_present(monkeypatch):
    """If one of the two named teams' documents is already in the top-k,
    only the missing one should be injected -- no duplicate."""
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: _FIXTURE_CHUNKS)

    already_has_argentina = [{
        "chunk_id": "TEAM-argentina-chunk-0",
        "metadata": {"level": "team", "team_name": "Argentina"},
    }]
    boosted = search._ensure_team_style_doc(
        "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟",
        already_has_argentina, k=5,
    )

    chunk_ids = [r["chunk_id"] for r in boosted]
    assert chunk_ids.count("TEAM-argentina-chunk-0") == 1
    assert "TEAM-france-chunk-0" in chunk_ids


# --- Mutation-style regression: "always pick first entity" must be caught ---


def test_mutation_always_first_entity_is_now_caught():
    """The specific mutation the previous audit found could survive
    ("_extract_latin_entity_span always returns the first Latin entity")
    must now be caught: for a two-entity query, the detector must not
    return the first-mentioned entity as if it were the only one."""
    q = "كيف اختلفت Argentina وFrance في أسلوب لعبهما في كأس العالم؟"
    result = search._detect_team_style_query(q)
    assert result != "Argentina", (
        "detector silently collapsed a two-team query to the first-mentioned "
        "team -- the exact mutation this test exists to catch"
    )


# --- Multi-entity extraction: bounded, no quadratic/list-growth DoS -------


def test_extract_latin_entity_spans_is_bounded_under_many_distinct_entities():
    """Security design review: the multi-entity extraction added for the
    collision fix must stay bounded (both in result size and time) even
    when a query contains thousands of distinct-looking Latin words, not
    just when it contains one repeated word (which dedup alone would
    bound). Uses genuinely distinct 4-letter combinations so
    de-duplication cannot trivially collapse them to one entity."""
    import itertools
    import string
    import time

    names = ["".join(p) for p in itertools.islice(
        itertools.product(string.ascii_uppercase, repeat=4), 5000,
    )]
    payload = " و ".join(names) + " تشكيل ازاي؟"

    start = time.perf_counter()
    spans = search._extract_latin_entity_spans(payload)
    elapsed = time.perf_counter() - start

    assert len(spans) <= 5, "must stay bounded regardless of how many distinct Latin words appear"
    assert elapsed < 1.0, f"took {elapsed:.3f}s on a 5000-distinct-entity payload -- possible unbounded growth"


def test_ensure_team_style_doc_bounded_under_many_distinct_entities(monkeypatch):
    """Same stress scenario at the injection-function level, against a
    production-scale (6600-chunk) fixture -- confirms no O(entities *
    chunks) blowup beyond the small fixed bound on entity count."""
    import itertools
    import string
    import time

    large_chunks = [
        {"chunk_id": f"TEAM-{i}-chunk-0", "document_id": f"TEAM-{i}-doc", "level": "team",
         "team_name": f"RealTeam{i}", "text": f"RealTeam{i} tactical analysis."}
        for i in range(6600)
    ]
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: large_chunks)

    names = ["".join(p) for p in itertools.islice(
        itertools.product(string.ascii_uppercase, repeat=4), 5000,
    )]
    payload = " و ".join(names) + " تشكيل ازاي؟"

    start = time.perf_counter()
    result = search._ensure_team_style_doc(payload, [], k=5)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"took {elapsed:.3f}s -- possible unbounded growth"
    # None of the fake entities match a real team name, so nothing is
    # injected -- safe by the same "no match, no injection" construction
    # used throughout this module.
    assert result == []



# ---------------------------------------------------------------------------
# Match pair boosting regression
# ---------------------------------------------------------------------------


def test_match_pair_boost_resolves_exact_fixture():
    """Exact match query must resolve the correct fixture, not another match
    containing the same team name.
    """
    from src.artifacts import ArtifactPaths
    from src.retrieval.search import hybrid_search

    artifact_paths = ArtifactPaths(2, 27)

    results = hybrid_search(
        "In Sunderland's 2-2 draw with West Ham United on 3 October 2015, "
        "how many goals were recorded?",
        k=10,
        artifact_paths=artifact_paths,
    )

    assert results
    assert results[0]["chunk_id"] == "L1-match-3754076-chunk-0"
    assert results[0]["source"] == "match_pair_boost"

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "How did Chelsea's 2-2 draw with Tottenham unfold, and how did Harry Kane perform?",
            ("Chelsea", "Tottenham"),
        ),
        (
            "How did Manchester City's 6-1 win over Newcastle unfold, and how did Sergio Aguero perform?",
            ("Manchester City", "Newcastle"),
        ),
        (
            "How did Everton's 6-2 win over Sunderland unfold, and how did Arouna Kone perform?",
            ("Everton", "Sunderland"),
        ),
        (
            "How did Manchester United beat Arsenal 3-2, and how did Marcus Rashford perform?",
            ("Manchester United", "Arsenal"),
        ),
    ],
)
def test_detect_match_teams_handles_multi_level_match_phrasings(query, expected):
    from src.retrieval.safeguards import _detect_match_teams

    assert _detect_match_teams(query) == expected

@pytest.mark.parametrize(
    ("query", "expected_document_id"),
    [
        (
            "How did Chelsea's 2-2 draw with Tottenham unfold, and how did Harry Kane perform?",
            "L1-match-3754092",
        ),
        (
            "How did Manchester City's 6-1 win over Newcastle unfold, and how did Sergio Aguero perform?",
            "L1-match-3754079",
        ),
        (
            "How did Everton's 6-2 win over Sunderland unfold, and how did Arouna Kone perform?",
            "L1-match-3754082",
        ),
        (
            "How did Manchester United beat Arsenal 3-2, and how did Marcus Rashford perform?",
            "L1-match-3754239",
        ),
    ],
)
def test_match_pair_boost_uses_canonical_teams_and_score(query, expected_document_id):
    from src.artifacts import ArtifactPaths
    from src.retrieval.safeguards import _boost_match_pair_candidates

    results = _boost_match_pair_candidates(
        query,
        [],
        artifact_paths=ArtifactPaths(2, 27),
    )

    assert results
    assert (
        results[0].get("metadata", {}).get("document_id")
        or results[0].get("document_id")
    ) == expected_document_id

def test_exact_fixture_expansion_adds_same_match_l2_and_requested_player_l3(monkeypatch):
    from src.retrieval import safeguards

    chunks = [
        {
            "chunk_id": "L1-match-10-chunk-0",
            "document_id": "L1-match-10",
            "level": "1",
            "match_id": 10,
            "text": "Manchester City 6-1 Newcastle United.",
            "metadata": {
                "home_team": "Manchester City",
                "away_team": "Newcastle United",
            },
        },
        {
            "chunk_id": "L2-match-10-chunk-0",
            "document_id": "L2-match-10",
            "level": "2",
            "match_id": 10,
            "text": "Match event summary.",
            "metadata": {},
        },
        {
            "chunk_id": "L3-match-10-player-1-chunk-0",
            "document_id": "L3-match-10-player-1",
            "level": "3",
            "match_id": 10,
            "player_name": "Sergio Leonel Ag\u00fcero del Castillo",
            "text": "Sergio Aguero player-match summary.",
            "metadata": {"player_name": "Sergio Leonel Ag\u00fcero del Castillo"},
        },
        {
            "chunk_id": "L3-match-10-player-2-chunk-0",
            "document_id": "L3-match-10-player-2",
            "level": "3",
            "match_id": 10,
            "player_name": "David Josu\u00e9 Jim\u00e9nez Silva",
            "text": "David Silva player-match summary.",
            "metadata": {"player_name": "David Josu\u00e9 Jim\u00e9nez Silva"},
        },
    ]

    monkeypatch.setattr(safeguards, "_get_chunks", lambda artifact_paths=None: chunks)

    results = [
        {
            "chunk_id": "L1-match-10-chunk-0",
            "document_id": "L1-match-10",
            "text": "Manchester City 6-1 Newcastle United.",
            "metadata": {
                "document_id": "L1-match-10",
                "level": "1",
                "match_id": 10,
            },
            "source": "match_pair_boost",
        }
    ]

    expanded = safeguards._expand_exact_fixture_candidates(
        "How did Manchester City's 6-1 win over Newcastle unfold, and how did Sergio Aguero perform?",
        results,
    )

    docs = {
        item.get("metadata", {}).get("document_id") or item.get("document_id")
        for item in expanded
    }

    assert "L2-match-10" in docs
    assert "L3-match-10-player-1" in docs
    assert "L3-match-10-player-2" not in docs

def test_exact_fixture_expansion_promotes_existing_same_match_candidates(monkeypatch):
    from src.retrieval import safeguards

    chunks = [
        {
            "chunk_id": "L1-match-10-chunk-0",
            "document_id": "L1-match-10",
            "level": "1",
            "match_id": 10,
            "text": "Manchester City 6-1 Newcastle United.",
            "metadata": {},
        },
        {
            "chunk_id": "L2-match-10-chunk-0",
            "document_id": "L2-match-10",
            "level": "2",
            "match_id": 10,
            "text": "Match event summary.",
            "metadata": {},
        },
        {
            "chunk_id": "L3-match-10-player-1-chunk-0",
            "document_id": "L3-match-10-player-1",
            "level": "3",
            "match_id": 10,
            "player_name": "Sergio Leonel Ag\u00fcero del Castillo",
            "text": "Sergio Aguero player-match summary.",
            "metadata": {"player_name": "Sergio Leonel Ag\u00fcero del Castillo"},
        },
    ]

    monkeypatch.setattr(safeguards, "_get_chunks", lambda artifact_paths=None: chunks)

    results = [
        {
            "chunk_id": "L1-match-10-chunk-0",
            "document_id": "L1-match-10",
            "metadata": {"document_id": "L1-match-10", "level": "1", "match_id": 10},
            "source": "match_pair_boost",
        },
        {
            "chunk_id": "L2-match-10-chunk-0",
            "document_id": "L2-match-10",
            "metadata": {"document_id": "L2-match-10", "level": "2", "match_id": 10},
            "source": "bm25",
        },
        {
            "chunk_id": "L3-match-10-player-1-chunk-0",
            "document_id": "L3-match-10-player-1",
            "metadata": {
                "document_id": "L3-match-10-player-1",
                "level": "3",
                "match_id": 10,
                "player_name": "Sergio Leonel Ag\u00fcero del Castillo",
            },
            "source": "dense",
        },
    ]

    expanded = safeguards._expand_exact_fixture_candidates(
        "How did Manchester City's 6-1 win over Newcastle unfold, and how did Sergio Aguero perform?",
        results,
    )

    by_doc = {
        item.get("metadata", {}).get("document_id") or item.get("document_id"): item
        for item in expanded
    }

    assert by_doc["L1-match-10"]["source"] == "match_pair_boost"
    assert by_doc["L2-match-10"]["source"] == "match_fixture_expansion"
    assert by_doc["L3-match-10-player-1"]["source"] == "match_fixture_expansion"

def test_detect_match_teams_handles_match_between_phrasing():
    from src.retrieval.safeguards import _detect_match_teams

    assert _detect_match_teams(
        "Describe the match between England and Iran."
    ) == ("England", "Iran")
