"""
src/team_style.py -- team playing-style / formation / tactics query detection.

Phase 5: moved verbatim out of `src/retrieval/safeguards.py` to close a real
reverse dependency. `src/query/intent.py` (UNDERSTAND) needs this classifier,
and importing it from `src/retrieval/` meant understanding depended on
retrieval -- backwards relative to the runtime flow
(UNDERSTAND -> PLAN -> RETRIEVE).

It could not simply move into `src/query/` either: `src/retrieval/safeguards.py`
uses the same detectors for its team-style document boost, so that direction
would have produced `retrieval -> query`, an equally reverse edge. A neutral
flat shared module -- the convention `src/stage_taxonomy.py`,
`src/embedding_config.py` and `src/artifacts.py` already follow -- lets both
layers depend downward on shared vocabulary instead of sideways on each other.

This module is **pure text classification**: no retrieval, no I/O, no chunk
store, no artifact paths.

NOTE on scope: `_normalize_arabic_for_matching()` is a general Arabic
matching helper, not a team-style concept. It lives here because it is shared
by exactly two callers -- this module's `_detect_team_style_entities()` and
`safeguards.py`'s `_detect_comparison_entities()` -- and duplicating it would
risk the two drifting apart (the same reasoning that keeps the BM25 tokenizer
duplication documented rather than silently forked). If more shared text
helpers accumulate, they belong in a dedicated text-matching module rather
than expanding this one.
"""

from __future__ import annotations

import re

__all__ = [
    "_normalize_arabic_for_matching",
    "_extract_latin_entity_spans",
    "_LATIN_ENTITY_SPAN",
    "_STYLE_KEYWORDS",
    "_STYLE_KEYWORDS_AR",
    "_detect_team_style_entities",
    "_detect_team_style_query",
]


# ---------------------------------------------------------------------------
# Arabic Safeguard-Trigger Normalization
# ---------------------------------------------------------------------------
#
# Light, deterministic normalization used ONLY for recognizing safeguard
# trigger phrases below -- never applied to the query actually sent to
# BM25/Dense retrieval. Unifies the most common Arabic spelling variants
# (alef-hamza forms, alef maksura) so that e.g. the benchmark's "ازاي" and
# the equally common "إزاي" both match the same trigger phrase, without
# needing to enumerate every spelling separately.


def _normalize_arabic_for_matching(text: str) -> str:
    for variant in ("أ", "إ", "آ"):
        text = text.replace(variant, "ا")
    text = text.replace("ى", "ي")
    # Collapse runs of whitespace (spaces/tabs/newlines) to a single space
    # so multi-word trigger phrases (e.g. "بتلعب ازاي") still match a
    # literal substring check when a query has irregular spacing.
    return re.sub(r"\s+", " ", text)


# A Latin-script span (canonical player/team name) inside an otherwise
# Arabic query -- used to recover the entity a safeguard should boost.
# Arabic-transliterated entity names (e.g. "ميسي") are a separate,
# deliberately deferred problem; this phase's queries keep entities in
# their canonical Latin form.
#
# Bounded to 60 characters (a generous margin over the longest real full
# names in the dataset, e.g. "Abdulelah Saad Hameed Al-Malki" = 31 chars):
# an earlier unbounded `[A-Za-z .'\-]*` here allowed the shared prefix
# between the quantified class and the trailing `[A-Za-z]` to create
# O(N) redundant backtracking splits per match attempt. Bounding it caps
# the worst-case backtracking search space to a constant, independent of
# input length -- verified with adversarial stress tests (see
# tests/test_arabic_safeguards.py's ReDoS regression tests).
_LATIN_ENTITY_SPAN = re.compile(r"[A-Za-z][A-Za-z .'\-]{0,58}[A-Za-z]|[A-Za-z]")


def _extract_latin_entity_spans(query: str, max_entities: int = 5) -> list[str]:
    """
    Every distinct contiguous Latin-script span in `query`, in order of
    first appearance, case-insensitively de-duplicated, bounded to
    `max_entities`.

    A single-entity query (e.g. gt-multi-03: "...Morocco...") returns
    exactly one span. A genuinely multi-entity query (e.g. gt-multi-04:
    "...Argentina وFrance...") returns all of them -- a caller must not
    silently act on only the first when more than one is present (see
    _detect_team_style_entities). `max_entities` bounds the amount of work
    a pathological input can force, consistent with this module's other
    bounded-regex hardening.
    """
    seen_lower: set[str] = set()
    spans: list[str] = []
    for match in _LATIN_ENTITY_SPAN.finditer(query):
        span = match.group(0).strip(" .'-")
        if len(span) <= 2:
            continue
        span_lower = span.lower()
        if span_lower in seen_lower:
            continue
        seen_lower.add(span_lower)
        spans.append(span)
        if len(spans) >= max_entities:
            break
    return spans



# ---------------------------------------------------------------------------
# Team Style Query Detection
# ---------------------------------------------------------------------------

_STYLE_KEYWORDS = {
    "style",
    "formation",
    "formations",
    "play pattern",
    "play patterns",
    "passing pattern",
    "passing patterns",
    "tactics",
    "approach",
    "playing style",
    "how they play",
    "how they played",
}

# Normalized (post alef-substitution) MSA/Egyptian team-style/formation/
# tactics trigger phrases. Arabic morphology means a stem like "تشكيل"
# already appears as a substring of its inflected/definite forms
# (التشكيل, تشكيلات, التشكيلات, ...), unlike English where "formation" and
# "formations" needed two separate entries.
_STYLE_KEYWORDS_AR = {
    "اسلوب",                 # style (اسلوب اللعب, اسلوبهم, باسلوب, ...)
    "تشكيل",                  # formation/lineup and its inflected forms
    "خطة",                   # tactical plan (الخطة, بخطة, ...)
    "نمط التمرير",            # passing pattern
    "انماط التمرير",          # passing patterns
    "لعبت ازاي",              # played how (EGY)
    "لعبوا ازاي",             # played how, plural (EGY)
    "بتلعب ازاي",             # is/was playing how (EGY)
    "يلعبوا ازاي",            # they play how (EGY)
    "كان عامل ازاي",          # was like how (EGY)
    "لعبهم كان عامل ازاي",     # their play was like how (EGY)
}


def _detect_team_style_entities(query: str) -> list[str]:
    """Return every team name a team-style/formation/tactics query names,
    or [] if it isn't a team-style query at all.

    English: at most one entity (unchanged, existing regex-based
    extraction -- English multi-entity extraction is out of scope for
    this fix; see _detect_team_style_query's docstring). MSA/Egyptian:
    every distinct Latin-script entity found (see
    _extract_latin_entity_spans) -- a query naming two teams (e.g. a
    style-comparison "multi" case such as gt-multi-04) gets both, instead
    of silently collapsing to whichever team is mentioned first."""
    query_lower = query.lower().strip()

    if any(keyword in query_lower for keyword in _STYLE_KEYWORDS):
        patterns = [
            (
                r"^(?:what\s+(?:was|were)\s+|describe\s+|how\s+did\s+)?"
                r"(.+?)(?:['?]s|s')\s+"
                r"(?:passing\s+patterns?|(?:most\s+common\s+)?formations?|"
                r"tactics|approach|(?:playing\s+)?style)\b"
            ),
            r"^(.+?)(?:['?]s|s')\s+(?:playing\s+)?style\b",
            r"^(?:what\s+(?:was|were)|how\s+did)\s+(.+?)\s+play\b",
            r"^(?:describe\s+)?(.+?)\s+(?:formations?|tactics|approach)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if not match:
                continue

            team_name = match.group(1).strip(" ,.?")
            team_name = re.sub(
                r"^(?:what\s+(?:was|were)|how\s+did|describe)\s+",
                "",
                team_name,
            ).strip()

            if len(team_name) > 2:
                return [team_name.title()]

        return []

    normalized = _normalize_arabic_for_matching(query_lower)
    if any(keyword in normalized for keyword in _STYLE_KEYWORDS_AR):
        return [entity.title() for entity in _extract_latin_entity_spans(query)]

    return []


def _detect_team_style_query(query: str) -> str | None:
    """Return the team name for a team-style/formation/tactics query when
    exactly one team is named, otherwise None -- including when the query
    names MULTIPLE teams (ambiguous which one a single-value caller should
    use; see _detect_team_style_entities for the multi-team case, used by
    _ensure_team_style_doc)."""
    entities = _detect_team_style_entities(query)
    return entities[0] if len(entities) == 1 else None

