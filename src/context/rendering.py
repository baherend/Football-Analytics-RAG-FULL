"""
src/context/rendering.py -- turning an Evidence Pack into prompt-ready text.

Migration Step 4: `build_context()` moved here verbatim from
src/retrieval/search.py -- no logic changes. It is Context Engineering's last
stage (Order -> Evidence Pack -> rendered context), not retrieval's job.

**Trust boundary.** Rendering labels each piece of evidence as a numbered,
attributed source block. Chunk text is copied in as inert DATA and is never
interpreted, executed, or promoted to an instruction; any instruction-looking
sentence inside a retrieved chunk is just more source text. The
system/developer rules live in the prompt template (07_prompting.py's
build_prompt), never here. See AGENT_RULES.md §9.

Known divergence, recorded as debt rather than changed here: 07_prompting.py
has its own near-duplicate renderer (`format_context_for_prompt`) with a
slightly different source header, and which of the two reaches the LLM depends
on the route and the entry point (chat.py always uses the prompting one;
07_prompting.py's answer_question() uses this one only on the non-structured
branch). Unifying them changes generation semantics, so it belongs to
Migration Step 5 (Generation + verification split). See PROJECT_MEMORY.md.
"""

from __future__ import annotations

from typing import Any

from src.context.evidence import EvidencePack


def build_context(chunks: list[dict[str, Any]], max_length: int = 3000) -> str:
    """
    Build a context string from retrieved chunks.

    Formats chunks with metadata and truncates to max_length.
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        level = meta.get("level", "unknown")

        # Add a stable source label that matches the generation prompt's
        # citation contract and preserves exact chunk-level traceability.
        header = f"[Source {i+1}: Level {level}"
        if chunk.get("chunk_id"):
            header += f", chunk_id={chunk['chunk_id']}"
        if meta.get("player_name"):
            header += f", Player: {meta['player_name']}"
        if meta.get("team_name"):
            header += f", Team: {meta['team_name']}"
        if meta.get("match_id"):
            header += f", Match: {meta['match_id']}"

        # Show RRF score if available, otherwise show score
        score_key = "rrf_score" if "rrf_score" in chunk else "score"
        header += f", Score: {chunk.get(score_key, 0):.4f}]"

        text = chunk["text"]
        entry = f"{header}\n{text}\n"

        if current_length + len(entry) > max_length:
            break

        context_parts.append(entry)
        current_length += len(entry)

    return "\n".join(context_parts)


def render_pack(pack: EvidencePack, max_length: int = 3000) -> str:
    """
    Render an EvidencePack to prompt-ready context text.

    Pack-native entry point for the same rendering, so callers holding a pack
    don't have to unwrap it themselves. Delegates to build_context() over the
    pack's verbatim source chunks, so output is identical for identical
    evidence.
    """
    return build_context(pack.to_chunks(), max_length=max_length)
