"""Assess whether selected chunks contain enough evidence to answer a query."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "the", "their", "they", "to", "use",
    "used", "was", "were", "what", "when", "which", "who", "with",
}

_ENTITY_METADATA_FIELDS = {
    "team_name",
    "player_name",
    "home_team",
    "away_team",
}


@dataclass(frozen=True)
class AnswerabilityAssessment:
    """Deterministic evidence-coverage result."""

    status: str
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]


def _normalize_token(token: str) -> str:
    token = token.casefold()

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ly"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if token.endswith("ss"):
        return token
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token


def _content_terms(text: str) -> set[str]:
    normalized_text = re.sub(
        r"\b([a-z0-9]+)[\'?]s\b",
        r"\1",
        text or "",
        flags=re.IGNORECASE,
    )
    return {
        normalized
        for token in _TOKEN_PATTERN.findall(normalized_text)
        if (normalized := _normalize_token(token)) not in _STOP_WORDS
    }


def _entity_terms(chunks: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        for field in _ENTITY_METADATA_FIELDS:
            value = metadata.get(field) or chunk.get(field)
            if value:
                terms.update(_content_terms(str(value)))

    return terms


def _entity_values(chunks: list[dict[str, Any]]) -> set[str]:
    """Return normalized entity names carried by candidate metadata."""
    values: set[str] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        for field in _ENTITY_METADATA_FIELDS:
            value = metadata.get(field) or chunk.get(field)
            if value:
                normalized = " ".join(
                    token.casefold()
                    for token in _TOKEN_PATTERN.findall(str(value))
                )
                if normalized:
                    values.add(normalized)

    return values


def _explicit_query_entity_phrases(query: str) -> set[str]:
    """Extract explicit title-cased entity-like phrases from the query."""
    phrases = re.findall(
        r"\b[A-Z][A-Za-z0-9'?\-]*(?:\s+[A-Z][A-Za-z0-9'?\-]*)*\b",
        query or "",
    )

    ignored = {"what", "when", "where", "which", "who", "how", "why"}

    normalized_phrases: set[str] = set()
    for phrase in phrases:
        normalized = " ".join(
            token.casefold()
            for token in _TOKEN_PATTERN.findall(phrase)
            if token.casefold() not in ignored
        )
        if normalized:
            normalized_phrases.add(normalized)

    return normalized_phrases


def assess_answerability(
    query: str,
    chunks: list[dict[str, Any]],
) -> AnswerabilityAssessment:
    """Classify evidence coverage as answerable, partial, or unanswerable."""
    if not chunks:
        return AnswerabilityAssessment(
            status="unanswerable",
            matched_terms=(),
            missing_terms=tuple(sorted(_content_terms(query))),
        )

    chunk_entity_terms = _entity_terms(chunks)
    chunk_entity_values = _entity_values(chunks)
    explicit_query_entities = _explicit_query_entity_phrases(query)

    grounded_entity_mentions = {
        phrase
        for phrase in explicit_query_entities
        if any(
            phrase == entity_value
            or phrase in entity_value
            or entity_value in phrase
            for entity_value in chunk_entity_values
        )
    }

    if (
        explicit_query_entities
        and chunk_entity_values
        and not grounded_entity_mentions
    ):
        return AnswerabilityAssessment(
            status="unanswerable",
            matched_terms=(),
            missing_terms=tuple(sorted(_content_terms(query))),
        )

    query_terms = _content_terms(query) - chunk_entity_terms
    if not query_terms:
        return AnswerabilityAssessment(
            status="unanswerable",
            matched_terms=(),
            missing_terms=(),
        )

    evidence_terms: set[str] = set()
    for chunk in chunks:
        evidence_terms.update(_content_terms(str(chunk.get("text", ""))))

    matched = query_terms & evidence_terms
    missing = query_terms - evidence_terms

    if not matched:
        status = "unanswerable"
    elif not missing:
        status = "answerable"
    else:
        status = "partially_answerable"

    return AnswerabilityAssessment(
        status=status,
        matched_terms=tuple(sorted(matched)),
        missing_terms=tuple(sorted(missing)),
    )
