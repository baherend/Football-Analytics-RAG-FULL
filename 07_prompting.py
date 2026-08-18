"""
07_prompting.py — Stage 7: Generation coordinator + compatibility boundary

Connects the retrieval/context pipeline to an LLM for grounded question
answering.

Input: user question
Output: grounded answer with cited sources

Migration Step 5 (Generation + Verification Split): this file used to hold
prompt templates, prompt construction, two provider adapters, numeric/
comparison/structured validation, and citation building all together. Those
implementations now live in focused packages:

    src/generation/prompt.py      -- policy templates, canonical evidence
                                     rendering, prompt + role-separated
                                     message construction
    src/generation/provider.py    -- model invocation (Groq / OpenRouter)
    src/generation/policy.py      -- pre-generation refusal gate
    src/generation/citations.py   -- evidence provenance -> user sources
    src/verification/validation.py -- numeric-claim grounding checks
    src/verification/comparison.py -- comparison/structured answer checks

What remains here is the **coordinator** (`answer_question()`) plus
compatibility re-exports. This module stays the seam on purpose, exactly like
`src/query/router.py` and `src/retrieval/search.py`: seven test modules load it
via `import_module("07_prompting")` and monkeypatch `route_and_execute`,
`ask_groq`, `generate_answer`, `build_prompt`, `validate_answer`, and
`GROQ_API_KEY` **on this module**, and `chat.py` reaches eleven attributes
through `prompting_mod.<name>`. `answer_question()` therefore calls those
names bare, so they resolve through this module's own globals and stay
patchable -- the same contract that shaped Migration Steps 2 and 3.

API Key:
    - Reads GROQ_API_KEY from environment or Streamlit secrets
    - NEVER hardcodes API keys
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.artifacts import ArtifactPaths
from src.conversation_memory import (
    ConversationMemory,
    format_conversation_context,
    resolve_pronoun_references,
)
from src.query.query_schema import ComparisonResult
from src.query.router import route_and_execute

# ---------------------------------------------------------------------------
# Compatibility re-exports -- see module docstring. chat.py, streamlit_app.py
# and tests/ reach these through this module.
# ---------------------------------------------------------------------------

from src.generation.citations import (
    build_user_citations,
    group_citations,
    render_citations_cli,
)
from src.generation.policy import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    is_unsupported_query,
    is_usable_structured_result,
)
from src.generation.prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_STRUCTURED,
    build_messages,
    build_prompt,
    format_context_for_prompt,
    render_evidence,
    select_system_prompt,
)
from src.generation.provider import (
    DEFAULT_MODEL,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    MODELS,
    OPENROUTER_API_URL,
    PROVIDER_KEYS,
    ask_groq,
    generate_answer,
    get_api_key,
)
from src.verification.comparison import (
    validate_comparison_answer,
    validate_structured_answer,
)
from src.verification.validation import (
    ValidationResult,
    extract_numeric_claims,
    validate_answer,
)

__all__ = [
    "answer_question",
    "build_prompt",
    "build_messages",
    "format_context_for_prompt",
    "generate_answer",
    "ask_groq",
    "get_api_key",
    "build_user_citations",
    "render_citations_cli",
    "validate_answer",
    "validate_comparison_answer",
    "validate_structured_answer",
    "is_usable_structured_result",
    "is_unsupported_query",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_WITH_STRUCTURED",
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "MODELS",
]


# ---------------------------------------------------------------------------
# End-to-end coordinator
# ---------------------------------------------------------------------------


def answer_question(question: str, api_key: str | None = None,
                    model: str | None = None,
                    artifact_paths: ArtifactPaths | None = None,
                    memory: ConversationMemory | None = None) -> tuple[str, list[dict]]:
    """
    End-to-end: route query, retrieve context, build prompt, generate answer.

    Uses structured routing when applicable (e.g. "Who scored the most goals?"
    → structured resolver), falls back to semantic retrieval otherwise.

    `memory`, if given, is searched under `artifact_paths`' dataset scope for
    conversation turns relevant to `question`. Relevant turns may resolve a
    pronoun (e.g. "he") in the *retrieval* query and are surfaced to the LLM
    as clearly-labeled, non-authoritative conversation context -- they never
    replace or get merged into the retrieved football evidence itself.

    Returns (answer_string, list_of_user_facing_citations) -- see
    build_user_citations() for the citation shape. The citation list is
    always [] for the deterministic refusal answer, even if evidence was
    retrieved but judged insufficient (see is_unsupported_query()) -- an
    empty citation list is a self-evident "no source list", not an ordinary
    "loading" gap; showing retrieved-but-insufficient evidence under a
    refusal would misleadingly imply it supports an answer it does not.
    """
    relevant_turns = memory.search(artifact_paths, question) if memory is not None else []
    retrieval_query = resolve_pronoun_references(question, relevant_turns)

    result = route_and_execute(retrieval_query, artifact_paths=artifact_paths)
    sr = getattr(result, "structured_result", None)
    has_structured = is_usable_structured_result(sr)

    if has_structured:
        # Mirror chat.py::process_query()'s context assembly: present the
        # structured fact as authoritative directly from `sr`, rather than
        # `result.context` -- for a hybrid route, execute_route() overwrites
        # `context` with semantic-only text after computing it, so the
        # structured explanation would otherwise be silently dropped.
        context = (
            "## Authoritative Data (Verified from Match Facts)\n\n"
            "The following numbers are VERIFIED and must be used EXACTLY:\n\n"
            + sr.explanation
        )
        if result.semantic_chunks:
            context += "\n\n" + format_context_for_prompt(result.semantic_chunks)
    else:
        context = result.context

    conversation_context = format_conversation_context(relevant_turns)
    full_context = f"{conversation_context}\n\n{context}" if conversation_context else context

    if is_unsupported_query(sr, getattr(result, "answerability", None)):
        answer = INSUFFICIENT_CONTEXT_MESSAGE
        citations: list[dict] = []
    else:
        # build_prompt() is still called (and still monkeypatched by tests) so
        # its exact legacy string stays observable. The provider receives the
        # role-separated form of the same content: developer policy in a
        # `system` message, retrieved evidence + question in a `user` message,
        # so evidence cannot occupy the system role. See
        # src/generation/prompt.py for why delimiters alone were insufficient.
        prompt = build_prompt(question, full_context, has_structured=has_structured)
        messages = build_messages(question, full_context, has_structured=has_structured)

        key = api_key or GROQ_API_KEY
        if not key:
            return "Missing GROQ_API_KEY. Please set it in Streamlit secrets or environment.", []

        answer = ask_groq(prompt, api_key=key, model=model, messages=messages)

        if has_structured:
            try:
                validation = validate_structured_answer(answer, sr)
                if not validation.is_valid:
                    answer = validation.corrected_answer or answer
            except Exception:
                # Validation is best-effort -- don't break the pipeline.
                pass

        citations = build_user_citations(sr if has_structured else None, result.semantic_chunks)

    if memory is not None:
        memory.add_turn(artifact_paths, question, answer)

    return answer, citations


if __name__ == "__main__":
    question = "Who scored the most goals?"
    answer, citations = answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()
    print(render_citations_cli(citations) or "Sources: none")
