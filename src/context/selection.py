"""Select candidate chunks that contain answer-bearing query evidence."""

from __future__ import annotations

import re
from typing import Any
from collections import Counter


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

_MONTH_NUMBERS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _extract_query_match_dates(query: str) -> set[str]:
    """Extract explicit match dates from a query as YYYY-MM-DD values."""
    query_lower = query.casefold()
    dates: set[str] = set()

    for year, month, day in re.findall(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
        query_lower,
    ):
        try:
            dates.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
        except ValueError:
            pass

    month_names = "|".join(
        sorted((re.escape(name) for name in _MONTH_NUMBERS), key=len, reverse=True)
    )

    for day, month_name, year in re.findall(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\s+(\d{{4}})\b",
        query_lower,
    ):
        dates.add(
            f"{int(year):04d}-{_MONTH_NUMBERS[month_name]:02d}-{int(day):02d}"
        )

    for month_name, day, year in re.findall(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
        query_lower,
    ):
        dates.add(
            f"{int(year):04d}-{_MONTH_NUMBERS[month_name]:02d}-{int(day):02d}"
        )

    return dates


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

    query_match_dates = _extract_query_match_dates(query)

    exact_fixture_positions: set[int] = set()
    if query_match_dates:
        for position, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            home_team = metadata.get("home_team")
            away_team = metadata.get("away_team")
            match_date = str(metadata.get("match_date", ""))

            if not home_team or not away_team or match_date not in query_match_dates:
                continue

            home_terms = _content_terms(str(home_team))
            away_terms = _content_terms(str(away_team))

            if (
                home_terms
                and away_terms
                and home_terms <= raw_query_terms
                and away_terms <= raw_query_terms
            ):
                exact_fixture_positions.add(position)

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
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        has_fixture_metadata = bool(
            metadata.get("home_team")
            and metadata.get("away_team")
            and metadata.get("match_date")
        )
        if (
            exact_fixture_positions
            and has_fixture_metadata
            and position not in exact_fixture_positions
        ):
            continue

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
    selected_document_ids: set[str] = set()
    source_priority = {
        "match_pair_boost": 2,
        "match_fixture_expansion": 1,
    }
    document_counts = Counter((c.get("metadata", {}).get("document_id") or c.get("document_id")) for _, c, _ in candidates)

    while candidates and len(selected) < max_chunks:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                source_priority.get(candidates[index][1].get("source"), 0),
                len(candidates[index][2] - covered),
                (
                    document_counts.get(
                        candidates[index][1].get("metadata", {}).get("document_id")
                        or candidates[index][1].get("document_id"),
                        0,
                    )
                    if selected
                    else 0
                ),
                len(candidates[index][2]),
                -candidates[index][0],
            ),
        )

        _, chunk, chunk_coverage = candidates.pop(best_index)
        new_coverage = chunk_coverage - covered
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        document_id = metadata.get("document_id") or chunk.get("document_id")

        if not new_coverage:
            if chunk.get("source") == "match_fixture_expansion":
                if document_id not in selected_document_ids:
                    selected.append(chunk)
                    if document_id:
                        selected_document_ids.add(document_id)
                continue
            break

        selected.append(chunk)
        if document_id:
            selected_document_ids.add(document_id)
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
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        has_fixture_metadata = bool(
            metadata.get("home_team")
            and metadata.get("away_team")
            and metadata.get("match_date")
        )
        if (
            exact_fixture_positions
            and has_fixture_metadata
            and position not in exact_fixture_positions
        ):
            continue

        if (
            mentioned_entity_terms
            and chunk_entity_terms
            and not (chunk_entity_terms & mentioned_entity_terms)
        ):
            continue

        selected.append(chunk)
        selected_objects.add(id(chunk))

    return selected
