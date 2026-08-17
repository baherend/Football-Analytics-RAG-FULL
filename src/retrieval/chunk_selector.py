"""Select candidate chunks that contain answer-bearing query evidence."""

from __future__ import annotations

import re
from typing import Any


# Latin letters/digits (unchanged) plus Arabic letters+diacritics
# (U+0621-U+065F -- HAMZA through the combining diacritic marks; excludes
# Arabic punctuation, which sits below U+0621: ، U+060C, ؛ U+061B, ؟
# U+061F) and Arabic-Indic digits (U+0660-U+0669), so Arabic punctuation
# acts as a separator exactly like English punctuation already does,
# instead of gluing onto the adjacent word token. No transliteration, no
# stemming/morphology beyond the existing English-only suffix rules in
# _normalize_token (which only ever match ASCII suffixes and are
# therefore inert -- a no-op -- on Arabic-range tokens).
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[ء-ٟ٠-٩]+", re.IGNORECASE)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "most", "of", "on", "or", "the", "their", "they", "to",
    "usually", "was", "were", "what", "when", "which", "who", "with",
}

_ENTITY_METADATA_FIELDS = {
    "team_name",
    "player_name",
    "home_team",
    "away_team",
}


def _normalize_token(token: str) -> str:
    """Apply small, deterministic English morphology normalization."""
    token = token.casefold()

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token


def _content_terms(text: str) -> set[str]:
    """Extract normalized non-stopword terms from text."""
    return {
        normalized
        for token in _TOKEN_PATTERN.findall(text or "")
        if (normalized := _normalize_token(token)) not in _STOP_WORDS
    }


def _chunk_entity_terms(chunk: dict[str, Any]) -> set[str]:
    """Extract entity terms from one candidate's metadata."""
    metadata = chunk.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    terms: set[str] = set()
    for field in _ENTITY_METADATA_FIELDS:
        value = metadata.get(field) or chunk.get(field)
        if value:
            terms.update(_content_terms(str(value)))

    return terms


def select_relevant_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Select chunks using entity grounding and marginal facet coverage."""
    if max_chunks <= 0 or not chunks:
        return []

    raw_query_terms = _content_terms(query)
    if not raw_query_terms:
        return []

    entity_terms_by_chunk = [
        _chunk_entity_terms(chunk)
        for chunk in chunks
    ]
    all_entity_terms = set().union(*entity_terms_by_chunk)
    mentioned_entity_terms = raw_query_terms & all_entity_terms

    query_evidence_terms = raw_query_terms - all_entity_terms
    if not query_evidence_terms:
        return chunks[:max_chunks]

    candidates: list[tuple[int, dict[str, Any], set[str]]] = []

    for position, chunk in enumerate(chunks):
        chunk_entity_terms = entity_terms_by_chunk[position]

        if (
            mentioned_entity_terms
            and chunk_entity_terms
            and not (chunk_entity_terms & mentioned_entity_terms)
        ):
            continue

        chunk_terms = _content_terms(str(chunk.get("text", "")))
        covered_terms = query_evidence_terms & chunk_terms

        if covered_terms:
            candidates.append((position, chunk, covered_terms))

    if not candidates:
        return chunks[:max_chunks]

    selected: list[dict[str, Any]] = []
    covered: set[str] = set()

    while candidates and len(selected) < max_chunks:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                len(candidates[index][2] - covered),
                len(candidates[index][2]),
                -candidates[index][0],
            ),
        )

        _, chunk, chunk_coverage = candidates.pop(best_index)
        new_coverage = chunk_coverage - covered

        if not new_coverage:
            break

        selected.append(chunk)
        covered.update(new_coverage)

    # Coverage selection can exhaust the available query facets before
    # reaching max_chunks. Backfill from the original ranking so Hybrid
    # preserves useful retrieval depth instead of returning an underfilled
    # result set. Keep entity grounding when the query names an entity.
    selected_objects = {id(chunk) for chunk in selected}
    for position, chunk in enumerate(chunks):
        if len(selected) >= max_chunks:
            break
        if id(chunk) in selected_objects:
            continue

        chunk_entity_terms = entity_terms_by_chunk[position]
        if (
            mentioned_entity_terms
            and chunk_entity_terms
            and not (chunk_entity_terms & mentioned_entity_terms)
        ):
            continue

        selected.append(chunk)
        selected_objects.add(id(chunk))

    return selected
