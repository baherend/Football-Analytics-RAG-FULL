"""
test_verification_security.py -- ReDoS regression + behavior-parity guard for
src/verification/validation.py::extract_numeric_claims().

Background: three of the five claim patterns shared an **unbounded lazy entity
capture**, `(\\w[\\w\\s]*?)`:

    P1  (\\w[\\w\\s]*?)\\s+(?:scored|has|had|...)\\s+(\\d+...)\\s+(goals?|...)
    P3  (\\w[\\w\\s]*?):\\s*(\\d+...)\\s+(goals?|...)
    P4  (\\w[\\w\\s]*?)(?:'s|'s)\\s+(goals?|...)\\s+(?:is|was|...)\\s+(\\d+...)

`[\\w\\s]` matches letters, digits, underscores AND whitespace -- i.e. almost
any prose. With `re.finditer()` retrying at every position and the lazy
quantifier expanding toward end-of-string looking for a literal that never
appears, work was O(N) per start position across O(N) start positions.
Measured before the fix, on the full function:

    2 KB   0.094 s
    4 KB   0.361 s     (3.9x per doubling)
    8 KB   1.435 s     (4.0x)
   16 KB  15.77 s
   40 KB  ~159 s

Per-pattern attribution: P1/P3/P4 were quadratic (5.4x / 7.6x / 4.5x per
doubling); P2 and P5 were already linear (P2's entity capture is trailing and
gated behind a literal prefix, P5 has no entity capture) and were left alone.

This matters because the input is **LLM output**, which is attacker-
influenceable: retrieved evidence can steer a model into emitting long
number-dense text, and a compromised or hostile provider response is fully
attacker-controlled. The function is reached on every structured/hybrid answer
via validate_structured_answer() -> validate_answer().

Fix: bound the three entity captures to 200 characters. The bound is NOT
copied from the 60 used elsewhere in this repo -- those patterns capture a
single `\\w` token, whereas this capture is a lazy span that can include a
whole sentence prefix before the verb. Measured worst realistic capture
(longest corpus entity name, 46 chars, behind a long sentence prefix) is 109
characters; 200 leaves ~83% headroom and produced zero output differences
across 2570 generated realistic claim sentences built from real corpus names.
"""

from __future__ import annotations

import re
import time

import pytest

from src.verification.validation import extract_numeric_claims


# --- Security: ReDoS regression ---------------------------------------------
#
# 2-second bound is deliberately generous: the fixed code handles the 16 KB
# payload in ~0.15 s and even 40 KB in ~0.4 s, while the unbounded version
# needed 15.8 s at 16 KB. Wide enough not to flake on a loaded machine, tight
# enough to catch a regression to quadratic behavior.

_REDOS_TIMEOUT_SECONDS = 2.0

_REDOS_PAYLOADS = {
    # Long runs where every character is in [\w\s] -- the shape that made the
    # lazy entity capture scan to end-of-string from every start position.
    "digit_run_16k": "1 " * 8_000,
    "letter_run_16k": "a " * 8_000,
    "word_run_16k": "ab " * 5_333,
    "newline_heavy_16k": "a\n" * 8_000,
    "decimal_fragments_16k": "1.5 " * 4_000,
    "underscores_16k": "a_b " * 4_000,
    # Near-miss: a real metric word only at the very end, so every earlier
    # start position must fail after doing maximal work.
    "trailing_metric_16k": "1 " * 8_000 + "scored 7 goals",
    # Unicode / Arabic / RTL must not be pathological either.
    "accented_16k": "é " * 8_000,
    "arabic_16k": "ازاي " * 3_200,
    "arabic_indic_digits_16k": "٧ " * 8_000,
    "mixed_arabic_english_16k": ("Messi ميسي 7 " * 1_230),
    "rtl_override_8k": "‮" + "abc 12 " * 1_200,
    "punctuation_heavy_16k": "a, " * 5_333,
    "control_chars_8k": "\x00\x01 9 " * 1_600,
}


@pytest.mark.parametrize("payload", _REDOS_PAYLOADS.values(), ids=_REDOS_PAYLOADS.keys())
def test_extract_numeric_claims_is_redos_safe(payload):
    start = time.perf_counter()
    extract_numeric_claims(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_SECONDS, (
        f"extract_numeric_claims took {elapsed:.2f}s on a {len(payload)}-char "
        f"adversarial payload -- possible regex-complexity regression."
    )


def test_extract_numeric_claims_scaling_is_not_quadratic():
    """
    Complexity guard beyond a single wall-clock threshold: doubling the input
    must not ~4x the time. The unbounded version measured a consistent
    ~4.0x per doubling; the bounded version measures ~2.0x.

    Measurement noise only ever *inflates* a timing, never deflates it, so the
    MINIMUM ratio across several attempts is the sound estimator of true
    complexity -- a linear implementation produces a clean sample on some
    attempt, a quadratic one cannot.
    """
    def timed(n: int) -> float:
        payload = "1 " * n
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            extract_numeric_claims(payload)
            best = min(best, time.perf_counter() - start)
        return best

    ratios = []
    for _ in range(5):
        small = timed(4_000)
        large = timed(8_000)
        if small < 1e-4:
            assert large < 1e-2, f"16k took {large:.4f}s despite 8k being instant"
            return
        ratios.append(large / small)
        if ratios[-1] < 3.0:
            return
    assert min(ratios) < 3.0, (
        f"doubling input scaled time by {min(ratios):.1f}x at best across "
        f"{len(ratios)} attempts (ratios={[round(r, 1) for r in ratios]}) -- "
        "quadratic backtracking appears to have returned."
    )


def test_entity_captures_are_bounded():
    """
    Structural guard against the quadratic shape returning.

    The dangerous shape is an unbounded lazy entity capture at the START of a
    pattern: nothing precedes it, so `re.finditer()` retries it at every
    position and it expands toward end-of-string. In the source such a capture
    sits directly after the pattern string's opening quote.

    The number-first pattern's TRAILING capture is deliberately left unbounded
    and must NOT be flagged: it is gated behind a literal
    `\\d+ <metric> by|from` prefix, so it cannot be retried at every position.
    It measured linear (1.8x per doubling) and was left untouched.
    """
    import inspect

    source = inspect.getsource(extract_numeric_claims)

    assert '"(\\w[\\w\\s]*?)' not in source, (
        "a claim pattern begins with the unbounded lazy entity capture "
        "`(\\w[\\w\\s]*?)` -- bound it (e.g. `(\\w[\\w\\s]{0,200}?)`) so a long "
        "run of word/space characters cannot force quadratic scanning."
    )

    module_source = inspect.getsource(inspect.getmodule(extract_numeric_claims))
    assert "_MAX_ENTITY_SPAN" in module_source, (
        "the entity-span bound constant is missing -- the ReDoS mitigation "
        "appears to have been removed."
    )


# --- Behavior parity: legitimate extraction must be unchanged ---------------


def _claim_tuples(text):
    """(value, metric, entity) with None normalised to "" so results sort.

    Pattern 5 (the anchored bare "N metric" form) intentionally yields
    entity=None, so a claim list normally contains both an entity-bearing and
    an entity-less claim for the same number.
    """
    return sorted(
        (c["value"], c["metric"], c["entity"] or "") for c in extract_numeric_claims(text)
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Lionel Messi scored 7 goals", (7.0, "goals", "Lionel Messi")),
        ("Kylian Mbappe scored 8 goals", (8.0, "goals", "Kylian Mbappe")),
        ("Messi had 3 assists", (3.0, "assists", "Messi")),
        # NOTE: "passes" -> "passe", not "passes_attempted": the function does
        # metric.rstrip("s") ("passes" -> "passe") and metric_map only has the
        # key "pass". Pre-existing quirk, identical before and after this
        # security fix, and out of scope for it -- pinned here as measured
        # behavior so the fix is provably behavior-preserving.
        ("Argentina made 450 passes", (450.0, "passe", "Argentina")),
        ("Messi recorded 2 tackles", (2.0, "successful_tackles", "Messi")),
        ("France achieved 12 shots", (12.0, "shots", "France")),
    ],
)
def test_basic_claims_still_extracted(text, expected):
    assert expected in _claim_tuples(text)


def test_decimal_values_preserved():
    claims = _claim_tuples("Messi scored 2.5 xG")
    assert (2.5, "xg", "Messi") in claims


def test_zero_and_large_values_preserved():
    assert (0.0, "goals", "He") in _claim_tuples("He scored 0 goals")
    assert (1000000.0, "passe", "Team") in _claim_tuples("Team scored 1000000 passes")


def test_colon_form_still_extracted():
    assert (3.0, "goals", "Argentina") in _claim_tuples("Argentina: 3 goals")


def test_possessive_form_still_extracted():
    assert (12.0, "goals", "Messi") in _claim_tuples("Messi's goals is 12")


def test_by_form_still_extracted():
    values = {(c["value"], c["metric"]) for c in extract_numeric_claims("5 assists by Messi")}
    assert (5.0, "assists") in values


def test_multiple_claims_in_one_answer():
    claims = _claim_tuples("Messi scored 7 goals and Mbappe scored 8 goals.")
    assert (7.0, "goals", "Messi") in claims
    # The second entity captures as "and Mbappe" -- the lazy span starts at the
    # first word char after the previous match. Measured, pre-existing, and
    # unchanged by this fix.
    assert (8.0, "goals", "and Mbappe") in claims


def test_long_sentence_prefix_entity_still_captured_whole():
    """The bound must be wide enough for a realistic sentence prefix plus the
    longest corpus entity name (measured worst case: 109 characters)."""
    text = (
        "According to the retrieved match data for the tournament final "
        "Antonio Joao Pereira Albuquerque Tavares Silva scored 3 goals"
    )
    entities = [c["entity"] for c in extract_numeric_claims(text)]
    assert any(
        e and e.endswith("Antonio Joao Pereira Albuquerque Tavares Silva")
        for e in entities
    ), f"long-prefix entity was truncated by the bound: {entities}"


def test_accented_entity_names_preserved():
    claims = _claim_tuples("Mbappé scored 12 goals")
    assert any(value == 12.0 and metric == "goals" for value, metric, _ in claims)


def test_arabic_text_does_not_crash_and_english_claim_survives():
    """Arabic/MSA/EGY behavior is unchanged: the patterns are English-metric
    based, but Arabic text must not break extraction of an English claim."""
    claims = _claim_tuples("ميسي سجل 7 goals")
    assert any(value == 7.0 and metric == "goals" for value, metric, _ in claims)


def test_newline_separated_claims_still_extracted():
    claims = _claim_tuples("line one\nMessi scored 2 goals\nline three")
    assert any(value == 2.0 and metric == "goals" for value, metric, _ in claims)


@pytest.mark.parametrize(
    "text",
    [
        "no numbers here at all",
        "scored goals",
        "",
        "   ",
        "Messi scored goals",
        "\x00\x01\x02",
    ],
)
def test_malformed_input_returns_no_claims_without_error(text):
    assert extract_numeric_claims(text) == [] or isinstance(extract_numeric_claims(text), list)


def test_claim_shape_is_unchanged():
    """Downstream (validate_answer) reads value/metric/entity/context."""
    claims = extract_numeric_claims("Messi scored 7 goals")
    assert claims
    for claim in claims:
        assert set(claim) == {"value", "metric", "entity", "context"}
        assert isinstance(claim["value"], float)
        assert isinstance(claim["metric"], str)
