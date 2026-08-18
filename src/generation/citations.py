"""
src/generation/citations.py -- turning evidence provenance into user-facing
sources.

Migration Step 5: extracted verbatim from 07_prompting.py.

Finalization, not verification: this maps the chunk/document IDs carried by
the Step 4 Evidence Pack into the citation list shown to a user. It never
invents a source -- every semantic citation comes from a retrieved chunk that
was actually in the evidence, and every structured citation from a resolved
structured result.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
#
# Turns the same evidence already selected for generation -- retrieved
# semantic chunks, a usable StructuredResult/ComparisonResult -- into a
# small, deterministic, human-readable citation list. This never re-runs
# retrieval and never asks the LLM to invent or format sources: the LLM is
# not the source of citation truth (see internal [Source N] markers in
# format_context_for_prompt(), which remain unchanged and serve a
# different purpose -- grounding generation, not user display).

_SEMANTIC_LEVEL_LABELS = {
    "1": "Match summary",
    "2": "Match key events",
    "3": "Player match performance",
    "4": "Player summary",
    "team": "Team summary",
}


def _semantic_citation_label(chunk: dict) -> str:
    """Human-readable label derived only from metadata the chunk actually carries."""
    meta = chunk.get("metadata") or {}
    level = meta.get("level")
    base = _SEMANTIC_LEVEL_LABELS.get(level, f"Level {level}" if level else "Retrieved evidence")
    who = meta.get("player_name") or meta.get("team_name")
    return f"{base} — {who}" if who else base


def _semantic_citations(semantic_chunks: list[dict] | None) -> list[dict]:
    """
    One citation per retrieved chunk, deduplicated by chunk_id and in
    retrieval order (order can carry ranking meaning -- never re-sorted).
    """
    if not semantic_chunks:
        return []
    seen: set = set()
    citations = []
    for chunk in semantic_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id is not None:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
        meta = chunk.get("metadata") or {}
        citations.append({
            "type": "semantic",
            "label": _semantic_citation_label(chunk),
            "chunk_id": chunk_id,
            "document_id": meta.get("document_id"),
        })
    return citations


def _structured_citations(structured_result) -> list[dict]:
    """
    Citation(s) for a usable structured result. `structured_result` must
    already be confirmed usable by the caller (see is_usable_structured_result)
    -- this function does not re-derive that gate, mirroring how callers
    already compute `has_structured` once and reuse it.

    A ComparisonResult (duck-typed via `.values`, same convention as
    is_usable_structured_result) yields one citation per compared entity --
    both sides of a player or team comparison are represented automatically
    through this single path, with no separate comparison-citation logic.
    An ordinary StructuredResult yields exactly one citation.
    """
    if structured_result is None:
        return []

    values = getattr(structured_result, "values", None)
    if values is not None:
        metric = getattr(structured_result, "metric", None)
        citations = []
        for v in values:
            if v.value is None:
                continue
            label = f"{v.entity_name} — {metric}" if metric else v.entity_name
            citations.append({
                "type": "structured", "label": label,
                "chunk_id": None, "document_id": None,
            })
        return citations

    query = getattr(structured_result, "query", None)
    if query is None:
        return []
    entity = getattr(query, "entity", None)
    metric = getattr(query, "metric", None)
    name = getattr(query, "entity_name", None)
    if name is None:
        # Superlative/aggregate queries with no named entity: fall back to
        # the actual resolved record's own name field, never a guess.
        data = getattr(structured_result, "data", None) or []
        if data:
            name = data[0].get("player_name") or data[0].get("team_name")
    label = " — ".join(part for part in (entity, name, metric) if part) or "Structured statistics"
    return [{"type": "structured", "label": label, "chunk_id": None, "document_id": None}]


def build_user_citations(structured_result=None, semantic_chunks: list[dict] | None = None) -> list[dict]:
    """
    Deterministic citation list for user display, built only from evidence
    already used for this answer. Structured citations (if any) are listed
    first, mirroring generation's own precedence -- structured evidence is
    presented to the LLM as authoritative ahead of semantic supporting
    context (see answer_question() below).
    """
    return _structured_citations(structured_result) + _semantic_citations(semantic_chunks)


def group_citations(citations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (structured, semantic) for rendering -- selection stays shared; only display differs."""
    structured = [c for c in citations if c.get("type") == "structured"]
    semantic = [c for c in citations if c.get("type") == "semantic"]
    return structured, semantic


def render_citations_cli(citations: list[dict]) -> str:
    """Plain-text "Sources:" block for chat.py's terminal output. Empty string when there's nothing to show."""
    if not citations:
        return ""
    structured, semantic = group_citations(citations)
    lines = ["Sources:"]
    n = 1
    for c in structured:
        lines.append(f"[{n}] Structured data — {c['label']}")
        n += 1
    for c in semantic:
        suffix = f" — chunk {c['chunk_id']}" if c.get("chunk_id") else ""
        lines.append(f"[{n}] Semantic evidence — {c['label']}{suffix}")
        n += 1
    return "\n".join(lines)

