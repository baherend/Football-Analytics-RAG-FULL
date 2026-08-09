"""
conversation_memory.py -- Session-scoped conversation memory search.

Conversation memory answers one question only: "what was the user talking
about?" It exists to help interpret follow-up questions (recently mentioned
players/teams, the previous comparison subject, an unresolved pronoun) --
never to supply football facts or statistics.

Football evidence continues to come exclusively from structured match
facts / BM25 / Chroma retrieval (06_retrieve_context.py). Nothing in this
module reads artifacts, match facts, or retrieval results, and nothing it
produces may be presented as verified or authoritative -- see
format_conversation_context().

Memory is strictly scoped per dataset. The scope key is whatever
`ArtifactPaths | None` the caller is currently using to select a runtime
dataset (see src/artifacts.py) -- the same value Batch 7 already treats as
"the source of truth for one selected dataset". Reusing it here means the
legacy flat WC2022 dataset (None), an explicitly namespaced WC2022 dataset
(ArtifactPaths(43, 106)), and any other namespaced competition/season all
get their own isolated memory automatically, with no separate keying
scheme to keep in sync.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Hashable, Sequence

from src.artifacts import ArtifactPaths

# Generic English words that are capitalized only because they start a
# sentence/question -- filtering them keeps entity extraction from treating
# "Compare", "Which", "Tell" etc. as candidate entities. This list is
# deliberately generic (no player/team/competition names).
_QUESTION_STOPWORDS = {
    "how", "what", "which", "who", "why", "when", "where",
    "tell", "describe", "explain", "compare",
    "is", "are", "was", "were", "did", "does", "do", "can", "could",
    "the", "a", "an", "this", "that", "these", "those",
    "me", "about", "and", "or", "in", "of", "to", "for", "with", "than",
    "i", "you", "we",
}

_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs", "it", "its",
}

_PRONOUN_RE = re.compile(r"\b(" + "|".join(_PRONOUNS) + r")\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_MAX_TURN_TEXT_LENGTH = 240


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant exchange, plus the candidate entities mentioned in it."""

    question: str
    answer: str
    entities: tuple[str, ...] = ()


def _extract_entities(text: str) -> tuple[str, ...]:
    """
    Extract candidate proper-noun entities as runs of capitalized words,
    skipping generic English/question-starter words.

    This is intentionally not a real NER model -- it is a bounded,
    deterministic heuristic for conversational continuity (recently
    mentioned players/teams), not a source of football knowledge.
    """
    entities: list[str] = []
    current: list[str] = []
    for word in _WORD_RE.findall(text):
        base = word.split("'")[0]
        if base and base[0].isupper() and base.lower() not in _QUESTION_STOPWORDS:
            current.append(base)
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))

    seen: set[str] = set()
    deduped: list[str] = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            deduped.append(entity)
    return tuple(deduped)


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _QUESTION_STOPWORDS}


def _score_turn(query_entities: set[str], query_tokens: set[str], turn: ConversationTurn) -> int:
    turn_entities = {e.lower() for e in turn.entities}
    turn_tokens = _tokenize(turn.question) | _tokenize(turn.answer)
    entity_overlap = len({e.lower() for e in query_entities} & turn_entities)
    token_overlap = len(query_tokens & turn_tokens)
    return entity_overlap * 3 + token_overlap


class ConversationMemory:
    """
    Session-scoped, dataset-isolated store of conversation turns with a
    small deterministic search over them.

    This class only stores and searches text -- it has no knowledge of
    football artifacts, match facts, or retrieval. Search results should be
    treated as conversational context for interpreting a follow-up
    question, never as football evidence (see format_conversation_context).
    """

    def __init__(self, max_turns_per_dataset: int = 20) -> None:
        self._turns: dict[Hashable, list[ConversationTurn]] = {}
        self._max_turns_per_dataset = max_turns_per_dataset

    def add_turn(self, dataset: ArtifactPaths | None, question: str, answer: str) -> None:
        """Record one turn under the given dataset scope."""
        # Reference-resolution entities come only from user-authored text.
        # Assistant-generated names must never become trusted entity targets.
        entities = _extract_entities(question)
        turn = ConversationTurn(question=question, answer=answer, entities=entities)
        turns = self._turns.setdefault(dataset, [])
        turns.append(turn)
        overflow = len(turns) - self._max_turns_per_dataset
        if overflow > 0:
            del turns[:overflow]

    def search(self, dataset: ArtifactPaths | None, query: str, k: int = 3) -> list[ConversationTurn]:
        """
        Return up to `k` previous turns (from this dataset scope only)
        relevant to `query`, most relevant/recent first.

        Turns are scored by shared entities and shared vocabulary with the
        query. If the query introduces no candidate entity of its own
        (e.g. a bare pronoun like "he", or a generic follow-up like "Which
        team had higher xG?"), it is treated as an implicit reference to
        the most recent turn(s) rather than excluded outright -- this is
        the bounded query-context mechanism this module provides, not a
        general coreference engine.
        """
        turns = self._turns.get(dataset, [])
        if not turns:
            return []

        query_entities = set(_extract_entities(query))
        query_tokens = _tokenize(query)

        scored = [(_score_turn(query_entities, query_tokens, turn), idx, turn)
                  for idx, turn in enumerate(turns)]
        relevant = [item for item in scored if item[0] > 0]

        if not relevant:
            if not query_entities and _PRONOUN_RE.search(query):
                # Pronoun-only follow-up: use only the most recent turn that
                # actually carries an entity, rather than treating the entire
                # conversation as relevant.
                relevant = next(
                    ([item] for item in reversed(scored) if item[2].entities),
                    [],
                )
            elif not query_entities and scored and len(scored[-1][2].entities) >= 2:
                # Generic entity-free comparison follow-up: only inherit the
                # immediately previous multi-entity context.
                relevant = [scored[-1]]
            elif query_entities and query.strip().lower().startswith("what about"):
                # Explicit topic continuation such as "What about X?"
                # inherits only the most recent entity-bearing turn.
                relevant = next(
                    ([item] for item in reversed(scored) if item[2].entities),
                    [],
                )

        relevant.sort(key=lambda item: (-item[0], -item[1]))
        return [turn for _, _, turn in relevant[:k]]

    def clear(self, dataset: ArtifactPaths | None) -> None:
        """Drop all turns for one dataset scope. Other scopes are untouched."""
        self._turns.pop(dataset, None)


def resolve_pronoun_references(query: str, relevant_turns: Sequence[ConversationTurn]) -> str:
    """
    Best-effort, bounded pronoun substitution for the text handed to
    retrieval/routing -- not for the question shown to the user.

    If `query` contains a generic pronoun (he/him/she/her/they/them/it/...)
    and the most relevant prior turn mentions a candidate entity, the
    pronoun is replaced with that entity so structured/semantic retrieval
    has something concrete to look up. Leaves `query` untouched otherwise.
    """
    if not relevant_turns:
        return query

    turn = next((turn for turn in relevant_turns if turn.entities), None)
    if turn is None:
        return query

    if _PRONOUN_RE.search(query):
        if len(turn.entities) >= 2 and re.search(
            r"\b(they|them|their|theirs)\b",
            query,
            flags=re.IGNORECASE,
        ):
            # Plural reference to a multi-entity prior turn: keep the user's
            # wording intact and append all candidate entities for retrieval.
            # Never collapse "they/them" to only the first entity.
            return f"{query} {' '.join(turn.entities)}"
        return _PRONOUN_RE.sub(turn.entities[0], query)

    query_entities = _extract_entities(query)

    if (
        query.strip().lower().startswith("what about")
        and len(query_entities) == 1
        and len(turn.entities) == 1
    ):
        previous_entity = turn.entities[0]
        new_entity = query_entities[0]
        return re.sub(
            re.escape(previous_entity),
            new_entity,
            turn.question,
            flags=re.IGNORECASE,
        )

    if not query_entities and len(turn.entities) >= 2:
        return f"{query} {' '.join(turn.entities)}"

    return query


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TURN_TEXT_LENGTH:
        return text
    return text[:_MAX_TURN_TEXT_LENGTH].rstrip() + "..."


def format_conversation_context(turns: Sequence[ConversationTurn]) -> str:
    """
    Render relevant turns as a clearly-labeled, non-authoritative
    conversation-context block for the generation prompt.

    This heading and wording are load-bearing: conversation memory must
    never be mistaken for verified football evidence, even if a prior
    assistant turn happened to contain a hallucination.
    """
    if not turns:
        return ""

    lines = [
        "## Conversation Context (reference only -- not verified football data)",
        "These are earlier turns in this conversation. They may help interpret "
        "pronouns or follow-up references (e.g. \"he\", \"they\", \"that team\"), "
        "but they are not a source of football facts or statistics -- they are "
        "not verified against match data.",
        "",
    ]
    for turn in turns:
        lines.append(f"Previously asked: {_truncate(turn.question)}")
        lines.append(f"Previously answered: {_truncate(turn.answer)}")
    return "\n".join(lines)
