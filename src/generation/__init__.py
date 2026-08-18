"""
src/generation/ -- grounded answer generation.

Migration Step 5. Receives query + policy + the Step 4 EvidencePack (already
approved by answerability) and produces a draft answer plus its citations.

    prompt.py     -- system/developer policy, canonical evidence rendering,
                     prompt + role-separated message construction
    provider.py   -- model invocation only (Groq / OpenRouter adapters)
    policy.py     -- pre-generation gate: generate, or refuse deterministically
    citations.py  -- evidence provenance -> user-facing sources

Generation does NOT retrieve, select evidence, or validate answers -- see
src/retrieval/, src/context/, and src/verification/ respectively.

**Trust boundary**: developer policy is sent in a `system` message and
retrieved evidence in a `user` message (prompt.build_messages()). Retrieved
text is untrusted DATA and cannot occupy the system role regardless of its
content. See AGENT_RULES.md §9.
"""

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
    MODELS,
    PROVIDER_KEYS,
    ask_groq,
    generate_answer,
    get_api_key,
)

__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_WITH_STRUCTURED",
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "MODELS",
    "PROVIDER_KEYS",
    "DEFAULT_MODEL",
    "select_system_prompt",
    "format_context_for_prompt",
    "render_evidence",
    "build_prompt",
    "build_messages",
    "get_api_key",
    "generate_answer",
    "ask_groq",
    "is_usable_structured_result",
    "is_unsupported_query",
    "build_user_citations",
    "group_citations",
    "render_citations_cli",
]
