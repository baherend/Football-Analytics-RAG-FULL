"""
src/context/evidence.py -- the Evidence Pack: the typed handoff between
Context Engineering and everything downstream of it.

Migration Step 4. An EvidencePack is what SELECT EVIDENCE produces and what
ANSWERABILITY / GENERATE consume. It is deliberately **minimal**: it carries
only fields that some current caller populates AND some current caller reads.
Fields from the target architecture with no consumer today (token budgets,
compression ratios, LLM-written summaries) are intentionally absent -- see
PROJECT_MEMORY.md's Architecture Decisions for what was deferred and why.

Two properties this type exists to guarantee:

1. **Provenance cannot be silently lost.** Every item keeps its `chunk_id`
   and `document_id`; `EvidencePack.from_chunks()` preserves the original
   candidate dict verbatim on `EvidenceItem.raw`, and `to_chunks()` returns
   those exact objects. Downstream citation code (07_prompting.py's
   `_semantic_citations`) therefore sees byte-identical input, and a future
   refactor cannot drop IDs without failing tests/test_evidence_pack.py.

2. **Evidence text is passive DATA, never instructions.** Nothing in this
   module interprets, executes, or promotes chunk text; items are inert
   containers. Rendering a pack for a prompt is a separate, explicit step
   (src/context/rendering.py) that labels each item as a numbered source.
   See AGENT_RULES.md §9 and docs/architecture/overview.md's trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Entity-bearing metadata fields, kept consistent with the selector and the
# answerability assessor so "entity coverage" means the same thing at every
# stage of the context pipeline.
_ENTITY_FIELDS = ("team_name", "player_name", "home_team", "away_team")


def _field(chunk: dict[str, Any], name: str) -> Any:
    """Read a field from a candidate's metadata, falling back to top level.

    Candidates reach Context Engineering from several retrieval paths (BM25,
    dense, and the boost/expansion safeguards), which populate entity fields
    at slightly different depths. This mirrors the same metadata-then-top-level
    lookup the selector and answerability already use, so no stage disagrees
    about what a candidate says about itself.
    """
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    value = metadata.get(name)
    if value is None:
        value = chunk.get(name)
    return value


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of retained evidence, with its provenance."""

    chunk_id: str | None
    document_id: str | None
    level: str | None
    text: str
    score: float | None
    source: str | None          # retrieval/selection provenance, e.g. "bm25",
                                # "dense", "team_style_boost", "sibling_expansion"
    team_name: str | None = None
    player_name: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    raw: dict[str, Any] | None = None   # the original candidate dict, verbatim

    @classmethod
    def from_chunk(cls, chunk: dict[str, Any]) -> "EvidenceItem":
        # "rrf_score" is the fused score when present; plain "score" is what
        # single-retriever and boost-injected candidates carry -- the same
        # precedence build_context() already uses when labelling a source.
        score = chunk.get("rrf_score", chunk.get("score"))
        return cls(
            chunk_id=chunk.get("chunk_id"),
            document_id=_field(chunk, "document_id"),
            level=_field(chunk, "level"),
            text=str(chunk.get("text", "")),
            score=score,
            source=chunk.get("source"),
            team_name=_field(chunk, "team_name"),
            player_name=_field(chunk, "player_name"),
            home_team=_field(chunk, "home_team"),
            away_team=_field(chunk, "away_team"),
            raw=chunk,
        )

    @property
    def entity_values(self) -> tuple[str, ...]:
        """Distinct entity names this item carries, in a stable order."""
        seen: list[str] = []
        for name in _ENTITY_FIELDS:
            value = getattr(self, name)
            if value and str(value) not in seen:
                seen.append(str(value))
        return tuple(seen)


@dataclass(frozen=True)
class EvidencePack:
    """The selected evidence for one query, plus its provenance."""

    query: str
    items: tuple[EvidenceItem, ...]
    candidates_considered: int = 0   # size of the pool selection chose from

    @classmethod
    def from_chunks(
        cls,
        query: str,
        chunks: list[dict[str, Any]] | None,
        candidates_considered: int | None = None,
    ) -> "EvidencePack":
        chunks = chunks or []
        return cls(
            query=query,
            items=tuple(EvidenceItem.from_chunk(chunk) for chunk in chunks),
            candidates_considered=(
                len(chunks) if candidates_considered is None else candidates_considered
            ),
        )

    def to_chunks(self) -> list[dict[str, Any]]:
        """The original candidate dicts, verbatim.

        Returns the very same objects `from_chunks()` received (not copies,
        not reconstructions) so existing consumers -- citation building,
        prompt formatting, answerability -- are byte-identical to the
        pre-migration path. This is what makes introducing the pack a
        representation change with provably zero behavior change.
        """
        return [
            item.raw if item.raw is not None else {}
            for item in self.items
        ]

    @property
    def chunk_ids(self) -> tuple[str | None, ...]:
        return tuple(item.chunk_id for item in self.items)

    @property
    def document_ids(self) -> tuple[str | None, ...]:
        return tuple(item.document_id for item in self.items)

    @property
    def entity_coverage(self) -> tuple[str, ...]:
        """Every distinct entity name present across the retained evidence."""
        seen: list[str] = []
        for item in self.items:
            for value in item.entity_values:
                if value not in seen:
                    seen.append(value)
        return tuple(seen)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)
