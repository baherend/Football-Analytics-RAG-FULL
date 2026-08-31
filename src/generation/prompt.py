"""
src/generation/prompt.py -- system policy, evidence rendering, and prompt
construction for grounded generation.

Migration Step 5: extracted from 07_prompting.py -- templates and
`build_prompt()` moved verbatim, plus two additions described below. See
07_prompting.py for the coordinator that still owns `answer_question()` and
the compatibility re-exports existing callers use.

## Trust boundary (the reason `build_messages()` exists)

Before this phase BOTH provider adapters sent:

    messages=[{"role": "user", "content": prompt}]

where `prompt` was one string containing the system rules, the retrieved
evidence, AND the user's question. System/developer policy and untrusted
retrieved text therefore sat at the *same privilege level*, separated only by
markdown delimiters -- exactly the "delimiters are not a security boundary"
failure mode. A retrieved chunk saying "ignore previous instructions" was
structurally indistinguishable from the policy above it.

`build_messages()` fixes that at the level the model actually distinguishes:
policy goes in a `system` message, evidence + question go in a `user` message.
Retrieved text can no longer occupy the system role no matter what it
contains. This does not require deleting or sanitizing hostile text -- it
requires that such text stay inside the data boundary, which is now
structural rather than typographic.

`build_prompt()` is retained and returns the **byte-identical legacy string**
(system + context + question concatenated), because callers and tests depend
on it; `build_messages()` is what the providers now send. Parity between the
two is pinned in tests/test_generation_verification.py.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.context.evidence import EvidencePack
from src.context.rendering import build_context

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a football analytics assistant working with the selected competition and season data.
You answer questions based ONLY on the provided context from StatsBomb match data.

Rules:
1. Answer ONLY based on the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information, say: "I don't have enough data to answer this question."
3. NEVER guess or infer information not present in the context.
4. Cite specific matches, players, or statistics when possible.
5. Be precise with numbers (goals, xG, minutes, etc.).
6. Cite sources like [Source 1], [Source 2], etc."""


SYSTEM_PROMPT_WITH_STRUCTURED = """You are a football analytics assistant working with the selected competition and season data.
You answer questions based on structured data and retrieved context from StatsBomb match data.

CRITICAL RULE - STRUCTURED DATA IS AUTHORITATIVE:
When structured facts are provided as Authoritative Data, use those exact values.
Do not independently re-derive, estimate, or contradict verified structured values.

Rules:
1. Structured facts take precedence for numeric claims.
2. Answer ONLY from the supplied structured facts and retrieved context.
3. NEVER guess or infer information not present in the supplied evidence.
4. If the evidence is insufficient, say: "I don't have enough data to answer this question."
5. Cite specific matches, players, statistics, and [Source N] references when available."""


# Canonical refusal text -- matches the wording both system prompts already
# instruct the LLM to produce when evidence is insufficient (rule 2/4 above).
INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough data to answer this question."
ARABIC_INSUFFICIENT_CONTEXT_MESSAGE = "لا أملك بيانات كافية للإجابة عن هذا السؤال."

ARABIC_RESPONSE_INSTRUCTION = """
LANGUAGE RULE:
The user's question is in Arabic. Answer in Arabic.
Keep player names, team names, source labels, and exact numeric/statistical
values faithful to the supplied evidence. Do not translate or invent evidence.
""".strip()


def contains_arabic(text: str) -> bool:
    """Return True when text contains at least one Arabic-script letter."""
    return any(
        unicodedata.category(ch).startswith("L")
        and (
            "\u0600" <= ch <= "\u06ff"
            or "\u0750" <= ch <= "\u077f"
            or "\u08a0" <= ch <= "\u08ff"
            or "\ufb50" <= ch <= "\ufdff"
            or "\ufe70" <= ch <= "\ufeff"
        )
        for ch in text
    )


def response_language_for_question(question: str) -> str:
    """Derive the response language once from the original user question."""
    return "ar" if contains_arabic(question) else "en"


def localized_insufficient_context_message(question: str) -> str:
    """Return the deterministic refusal in the user's question language."""
    if response_language_for_question(question) == "ar":
        return ARABIC_INSUFFICIENT_CONTEXT_MESSAGE
    return INSUFFICIENT_CONTEXT_MESSAGE


def _system_prompt_for_question(question: str, has_structured: bool = False) -> str:
    """Add response-language policy without changing English legacy prompts."""
    system_prompt = select_system_prompt(has_structured)
    if response_language_for_question(question) == "ar":
        return f"{system_prompt}\n\n{ARABIC_RESPONSE_INSTRUCTION}"
    return system_prompt


def select_system_prompt(has_structured: bool = False) -> str:
    """The developer/system policy for this generation."""
    return SYSTEM_PROMPT_WITH_STRUCTURED if has_structured else SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Evidence Rendering
# ---------------------------------------------------------------------------


def format_context_for_prompt(chunks: list[dict], max_length: int = 3000) -> str:
    """Format retrieved semantic chunks into prompt-ready source blocks.

    Migration Step 5 note: this used to be a second, independent renderer that
    diverged from src/context/rendering.py::build_context() (different source
    header -- this one omitted the `Score:` field), with which one reached the
    LLM depending on route and entry point. It now delegates to the canonical
    Context Engineering renderer, so one evidence set renders one way
    everywhere. See PROJECT_MEMORY.md for the measured before/after.
    """
    return build_context(chunks, max_length=max_length)


def render_evidence(pack: EvidencePack, max_length: int = 3000) -> str:
    """Render an EvidencePack to prompt-ready evidence text.

    Pack-native entry point, so generation can consume the Step 4 Evidence
    Pack directly instead of unwrapping it into raw dicts at the call site.
    """
    return build_context(pack.to_chunks(), max_length=max_length)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def build_prompt(
    question: str,
    context: str,
    has_structured: bool = False,
) -> str:
    """Build a complete prompt for the LLM.

    Legacy single-string form, byte-identical to the pre-Step-5 output. Kept
    because chat.py and several tests call/patch it directly. Providers now
    send build_messages() instead -- see this module's docstring.
    """
    system_prompt = _system_prompt_for_question(question, has_structured)
    return f"""{system_prompt}

{_user_content(question, context)}"""


def _user_content(question: str, context: str) -> str:
    """The untrusted half of the prompt: retrieved evidence + the question.

    Everything in here is DATA. It is never treated as instructions, and
    build_messages() keeps it out of the system role.
    """
    return f"""## Retrieved Context

{context}

## Question

{question}

## Answer

Based on the retrieved context, here is my answer:"""


def build_messages(
    question: str,
    context: str,
    has_structured: bool = False,
) -> list[dict[str, Any]]:
    """Build a role-separated chat payload.

    Returns `[{"role": "system", ...}, {"role": "user", ...}]`: developer
    policy in the system role, retrieved evidence and the user's question in
    the user role. This is the structural trust boundary -- retrieved text
    cannot occupy the system role regardless of its content.

    Concatenating the two contents with a blank line reproduces
    build_prompt()'s legacy string exactly (pinned by test).
    """
    return [
        {"role": "system", "content": _system_prompt_for_question(question, has_structured)},
        {"role": "user", "content": _user_content(question, context)},
    ]
