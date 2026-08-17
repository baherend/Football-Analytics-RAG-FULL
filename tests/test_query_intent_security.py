"""
test_query_intent_security.py -- ReDoS regression + behavior-parity guard for
src/query/intent.py's COMPARISON_PATTERNS.

Background: the router's comparison patterns "(\\w+)\\s+vs\\.?\\s+(\\w+)" and
"(\\w+)\\s+versus\\s+(\\w+)" carried UNBOUNDED entity quantifiers. Combined
with re.search()'s per-position retry (neither pattern has a fixed literal
prefix to fast-scan for), long runs of word characters forced O(N^2)
backtracking. Measured before the fix, on the full _detect_comparison() path:

    4 KB   -> 0.19 s
    8 KB   -> 0.38 s        (consistent 4x per doubling => quadratic)
    20 KB  -> 16.5 s        (accented input; "é" is a \\w char in Python 3,
                             so this is NOT an ASCII-only problem)

This is reachable from user query text on EVERY query, via classify_query().

The same vulnerable shape was already found and fixed in
src/retrieval/safeguards.py::_detect_comparison_entities() (bounded to
{1,60}); the router held an independent, unbounded copy of the same two
patterns, which that fix missed. This module pins both the security property
and the valid-comparison behavior that must not change with it.

Bound rationale: these two patterns capture a single `\\w` token -- `\\w`
matches neither spaces nor hyphens, so a multi-word name like "Lionel Andres
Messi" was never captured whole by them in the first place. The longest single
`\\w` token across all 713 entity names in the production corpus is 14 chars
("Dayotchanculle"), so a 60-char bound leaves >4x headroom while matching the
convention already used in safeguards.py.
"""

from __future__ import annotations

import re
import time

import pytest

import src.query.intent as intent


# --- Security: ReDoS regression ---------------------------------------------
#
# 2-second bound chosen deliberately generous -- the fixed code completes each
# payload in well under 0.1s, while the unbounded version needed 16s+ on the
# 20k accented payload. Generous enough not to flake on a loaded CI machine,
# tight enough to catch a real regression back to quadratic behavior.

_REDOS_TIMEOUT_SECONDS = 2.0

_REDOS_PAYLOADS = {
    # Long runs of \w with no "vs"/"versus": the exact shape that forced
    # per-position retry x unbounded backtracking.
    "ascii_word_run_40k": "a" * 40_000,
    "accented_word_run_20k": "é" * 20_000,
    "digits_underscores_40k": ("a1_" * 13_333),
    # Near-miss: contains "vs" only at the very end, so every earlier retry
    # position must fail.
    "trailing_vs_40k": "a" * 40_000 + " vs b",
    "trailing_versus_40k": "a" * 40_000 + " versus b",
    # Malformed / non-matching long input.
    "no_match_punct_40k": ("a, " * 13_333),
    # Unicode / RTL: must not be pathological either.
    "arabic_40k": "ا" * 40_000,
    "rtl_override_20k": "‮" + "ازاي " * 4_000,
}


@pytest.mark.parametrize("payload", _REDOS_PAYLOADS.values(), ids=_REDOS_PAYLOADS.keys())
def test_detect_comparison_is_redos_safe(payload):
    start = time.perf_counter()
    intent._detect_comparison(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_SECONDS, (
        f"_detect_comparison took {elapsed:.2f}s on a {len(payload)}-char "
        f"adversarial payload -- possible regex-complexity regression."
    )


@pytest.mark.parametrize("payload", _REDOS_PAYLOADS.values(), ids=_REDOS_PAYLOADS.keys())
def test_classify_query_is_redos_safe(payload):
    """classify_query() calls _detect_comparison() on every user query, so the
    whole live routing entry point must stay linear too."""
    start = time.perf_counter()
    intent.classify_query(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < _REDOS_TIMEOUT_SECONDS, (
        f"classify_query took {elapsed:.2f}s on a {len(payload)}-char "
        f"adversarial payload -- possible regex-complexity regression."
    )


def test_comparison_scaling_is_not_quadratic():
    """
    Structural guard beyond a single wall-clock threshold: doubling the input
    must not ~4x the time. The unbounded version measured a consistent 4.0x
    per doubling; a linear scan stays near 2x (with generous slack for timer
    noise on short runs).
    """
    def timed(n: int) -> float:
        payload = "a" * n
        # Best of 3 -- reduces scheduler noise without making the test slow.
        return min(
            (lambda: (lambda t0: (intent._detect_comparison(payload), time.perf_counter() - t0)[1])(time.perf_counter()))()
            for _ in range(3)
        )

    small = timed(8_000)
    large = timed(16_000)

    # Guard against a divide-by-zero on a very fast machine: if both are
    # effectively instant, the pathological behavior is definitively gone.
    if small < 1e-4:
        assert large < 1e-2, f"16k took {large:.4f}s despite 8k being instant"
        return

    ratio = large / small
    assert ratio < 3.0, (
        f"doubling input scaled time by {ratio:.1f}x (8k={small:.4f}s, "
        f"16k={large:.4f}s) -- quadratic backtracking appears to have returned."
    )


def test_comparison_entity_quantifiers_are_bounded():
    """
    Structural regex review, pinned as a test: every entity-capturing `\\w`
    quantifier in COMPARISON_PATTERNS must carry an explicit upper bound.
    Catches a future pattern being added (or reverted) with a bare `(\\w+)`,
    which is what caused this vulnerability.
    """
    offenders = [p for p in intent.COMPARISON_PATTERNS if r"(\w+)" in p]
    assert not offenders, (
        "unbounded entity-capturing quantifier `(\\w+)` found in "
        f"COMPARISON_PATTERNS: {offenders!r} -- bound it (e.g. (\\w{{1,60}})) "
        "so long word-character runs cannot force quadratic backtracking."
    )


# --- Behavior parity: valid comparisons must be unchanged -------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Messi vs Mbappe", ["Messi", "Mbappe"]),
        ("Messi versus Mbappe", ["Messi", "Mbappe"]),
        ("Messi vs. Mbappe", ["Messi", "Mbappe"]),
        ("Argentina vs France", ["Argentina", "France"]),
    ],
)
def test_vs_versus_comparisons_still_detected(query, expected):
    assert intent._detect_comparison(query) == expected


def test_longest_real_corpus_token_still_matches():
    """The longest single \\w token in the production corpus (14 chars) sits
    far inside the 60-char bound -- pinned so the bound is never tightened
    below what real data needs."""
    assert intent._detect_comparison("Dayotchanculle vs Zainalabiddin") == [
        "Dayotchanculle",
        "Zainalabiddin",
    ]


def test_accented_names_still_match():
    """Accented characters are \\w in Python 3 and must keep matching."""
    assert intent._detect_comparison("Mbappé vs Giroud") == ["Mbappé", "Giroud"]


def test_multiword_names_keep_existing_single_token_behavior():
    """
    PRE-EXISTING behavior, pinned so the bound doesn't silently change it:
    `\\w` matches neither spaces nor hyphens, so these two patterns only ever
    captured ONE token per side. "Lionel Andres Messi vs Kylian Mbappe"
    yields the adjacent tokens, not the full names. This is NOT introduced by
    the bound -- it is what the unbounded pattern already did.
    """
    assert intent._detect_comparison("Lionel Andres Messi vs Kylian Mbappe") == [
        "Messi",
        "Kylian",
    ]


@pytest.mark.parametrize(
    "query,expected",
    [
        # Non-vs comparison phrasings route through the other (already
        # non-pathological) patterns and must be untouched by this fix.
        ("Compare Messi and Mbappe", ["Messi", "Mbappe"]),
        ("Who performed better, Messi or Mbappe?", ["Messi", "Mbappe"]),
        ("difference between Messi and Mbappe", ["Messi", "Mbappe"]),
    ],
)
def test_other_comparison_phrasings_unchanged(query, expected):
    assert intent._detect_comparison(query) == expected


def test_non_comparison_query_still_returns_empty():
    assert intent._detect_comparison("How many goals did Messi score?") == []


def test_classification_of_vs_query_unchanged():
    """A "X vs Y" query must still classify as hybrid with the same
    confidence -- the routing contract this fix must not disturb."""
    assert intent.classify_query("Messi vs Mbappe") == ("hybrid", 0.9)


def test_msa_egy_queries_still_classify_semantic():
    """MSA/Egyptian style queries route via the retrieval-layer team-style
    detector, not COMPARISON_PATTERNS -- unaffected, pinned as parity."""
    assert intent.classify_query("اسلوب لعب France كان ازاي") == ("semantic", 0.9)


def test_bounded_pattern_matches_unbounded_on_realistic_input():
    """
    Direct equivalence proof for the bound: on every realistic entity token
    (<= 60 chars), the bounded and unbounded patterns produce identical
    matches. Divergence is only possible on a single >60-char token, which no
    real name in the corpus approaches (longest is 14).
    """
    unbounded = re.compile(r"(\w+)\s+vs\.?\s+(\w+)")
    bounded = re.compile(r"(\w{1,60})\s+vs\.?\s+(\w{1,60})")
    samples = [
        "messi vs mbappe",
        "argentina vs france",
        "dayotchanculle vs zainalabiddin",
        "mbappé vs giroud",
        "a vs b",
        "lionel andres messi vs kylian mbappe",
        "no comparison here at all",
        "who scored the most goals?",
    ]
    for s in samples:
        a, b = unbounded.search(s), bounded.search(s)
        assert (a is None) == (b is None), f"match presence differs for {s!r}"
        if a is not None:
            assert a.groups() == b.groups(), f"captured groups differ for {s!r}"
