"""
prompt_builder.py — Phase 6: Prompt Construction

Builds prompts for the LLM using retrieved context and user questions.
Follows the lab philosophy: the LLM receives only the final selected chunks.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a football analytics assistant specializing in FIFA World Cup 2022 data.
You answer questions based ONLY on the provided context from StatsBomb match data.

Rules:
1. Answer ONLY based on the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information to answer the question, say explicitly: "I don't have enough data in the retrieved context to answer this question."
3. NEVER guess or infer information not present in the context.
4. For subjective questions (e.g., "best player", "most aggressive"), refuse to answer unless the context provides explicit rankings or metrics that directly address the question.
5. For questions about events not in the data (e.g., other World Cups, future predictions), state that the data only covers the 2022 FIFA World Cup.
6. For questions requiring temporal analysis (e.g., "before/after a specific goal"), refuse unless the context explicitly contains timestamped data for that comparison.
7. Cite specific matches, players, or statistics when possible.
8. Be precise with numbers (goals, xG, minutes, etc.)."""

CONTEXT_TEMPLATE = """## Retrieved Context

The following documents were retrieved from the FIFA World Cup 2022 database:

{context}"""

QUESTION_TEMPLATE = """## Question

{question}"""

ANSWER_TEMPLATE = """## Answer

Based on the retrieved context, here is my answer:"""


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------


def build_prompt(
    question: str,
    context: str,
    include_system: bool = True,
    include_answer_prefix: bool = True,
) -> str:
    """
    Build a complete prompt for the LLM.

    Parameters:
        question: User's question
        context: Retrieved context from hybrid search
        include_system: Whether to include system instructions
        include_answer_prefix: Whether to include answer prefix

    Returns:
        Complete prompt string.
    """
    parts = []

    if include_system:
        parts.append(SYSTEM_PROMPT)
        parts.append("")

    parts.append(CONTEXT_TEMPLATE.format(context=context))
    parts.append("")
    parts.append(QUESTION_TEMPLATE.format(question=question))

    if include_answer_prefix:
        parts.append("")
        parts.append(ANSWER_TEMPLATE)

    return "\n".join(parts)


def build_messages(
    question: str,
    context: str,
) -> list[dict[str, str]]:
    """
    Build message list for chat-based LLM APIs.

    Returns list of {role, content} messages.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            CONTEXT_TEMPLATE.format(context=context)
            + "\n\n"
            + QUESTION_TEMPLATE.format(question=question)
        )},
    ]


# ---------------------------------------------------------------------------
# Context Formatting (for prompt)
# ---------------------------------------------------------------------------


def format_context_for_prompt(chunks: list[dict], max_length: int = 3000) -> str:
    """
    Format retrieved chunks into a readable context string for the prompt.

    Similar to build_context() but optimized for LLM consumption.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        level = meta.get("level", "unknown")

        # Build source line
        source = f"[Source {i+1}: Level {level}"
        if meta.get("player_name"):
            source += f", {meta['player_name']}"
        if meta.get("team_name"):
            source += f", {meta['team_name']}"
        if meta.get("match_id"):
            source += f", Match {meta['match_id']}"
        source += "]"

        text = chunk["text"]
        entry = f"{source}\n{text}\n"

        if current_length + len(entry) > max_length:
            break

        parts.append(entry)
        current_length += len(entry)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Honest Failure
# ---------------------------------------------------------------------------


def build_failure_prompt(question: str, reason: str) -> str:
    """
    Build a prompt for honest failure when context is insufficient.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Note\n\n"
        f"The retrieval system could not find sufficient context to answer this question. "
        f"Reason: {reason}\n\n"
        f"Please respond honestly that you don't have enough information to answer "
        f"this question based on the available FIFA World Cup 2022 data."
    )
