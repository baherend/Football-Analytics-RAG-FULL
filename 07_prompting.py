"""
07_prompting.py — Phase 6: Prompting, Generation, and Validation

Single file that takes context and produces a validated answer.

Combines:
- Prompt construction (system prompts, context formatting)
- LLM generation via Groq/OpenRouter APIs
- Answer validation against structured facts

Input: question + context (from 06_retrieve_context.py)
Output: validated answer string

Usage:
    from prompting_07 import build_prompt, generate_answer, validate_and_correct
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama-3.3-70b-versatile"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODELS = {
    "llama-3.3-70b-versatile": {"model": "llama-3.3-70b-versatile", "provider": "groq"},
    "llama-3.1-8b-instant": {"model": "llama-3.1-8b-instant", "provider": "groq"},
    "gemma2-9b-it": {"model": "gemma2-9b-it", "provider": "groq"},
    "mixtral-8x7b-32768": {"model": "mixtral-8x7b-32768", "provider": "groq"},
    "haiku": {"model": "anthropic/claude-3-haiku", "provider": "openrouter"},
    "sonnet": {"model": "anthropic/claude-3-sonnet", "provider": "openrouter"},
}

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


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


SYSTEM_PROMPT_WITH_STRUCTURED = """You are a football analytics assistant specializing in FIFA World Cup 2022 data.
You answer questions based on structured data and retrieved context from StatsBomb match data.

CRITICAL RULE — STRUCTURED DATA IS AUTHORITATIVE:
When structured facts are provided (marked as "Authoritative Data"), they are the
GROUND TRUTH. You MUST use these exact numbers. NEVER round, estimate, or
re-derive numbers independently. If the structured data says "Messi scored 7 goals,"
you must say exactly "7 goals" — not "about 7 goals" or "approximately 7 goals."

RULES:
1. Structured facts take precedence for ALL numeric claims. Retrieved text is only
   for narrative framing around those numbers.
2. If structured data provides a direct answer, present it confidently without hedging.
3. If the context doesn't contain enough information to answer the question, say
   explicitly: "I don't have enough data to answer this question."
4. NEVER guess or infer information not present in the context.
5. For subjective questions (e.g., "best player", "most aggressive"), refuse to
   answer unless the context provides explicit rankings or metrics.
6. Cite specific matches, players, or statistics when possible.
7. When comparing entities (e.g., "Messi vs Mbappé"), use the structured numbers
   for each entity and the retrieved text for narrative context."""

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
    has_structured: bool = False,
) -> str:
    """Build a complete prompt for the LLM."""
    parts = []

    if include_system:
        if has_structured:
            parts.append(SYSTEM_PROMPT_WITH_STRUCTURED)
        else:
            parts.append(SYSTEM_PROMPT)
        parts.append("")

    parts.append(CONTEXT_TEMPLATE.format(context=context))
    parts.append("")
    parts.append(QUESTION_TEMPLATE.format(question=question))

    if include_answer_prefix:
        parts.append("")
        parts.append(ANSWER_TEMPLATE)

    return "\n".join(parts)


def format_context_for_prompt(chunks: list[dict], max_length: int = 3000) -> str:
    """Format retrieved chunks into a readable context string for the prompt."""
    if not chunks:
        return "No relevant documents found."

    parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        level = meta.get("level", "unknown")

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


def format_structured_context(
    structured_explanation: str,
    additional_context: str = "",
) -> str:
    """Format structured facts for the prompt, marking them as authoritative."""
    parts = []

    if structured_explanation:
        parts.append("## Authoritative Data (Verified from Match Facts)")
        parts.append("")
        parts.append("The following numbers are VERIFIED and must be used EXACTLY:")
        parts.append("")
        parts.append(structured_explanation)
        parts.append("")

    if additional_context:
        parts.append("## Additional Context (Narrative)")
        parts.append("")
        parts.append(additional_context)

    return "\n".join(parts)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build message list for chat-based LLM APIs."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            CONTEXT_TEMPLATE.format(context=context)
            + "\n\n"
            + QUESTION_TEMPLATE.format(question=question)
        )},
    ]


def build_failure_prompt(question: str, reason: str) -> str:
    """Build a prompt for honest failure when context is insufficient."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Note\n\n"
        f"The retrieval system could not find sufficient context to answer this question. "
        f"Reason: {reason}\n\n"
        f"Please respond honestly that you don't have enough information to answer "
        f"this question based on the available FIFA World Cup 2022 data."
    )


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------


def get_api_key(provider: str = "groq") -> str:
    """Get API key from environment for the specified provider."""
    key_env = PROVIDER_KEYS.get(provider, "GROQ_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(
            f"{key_env} not set. "
            f"Set it with: $env:{key_env} = 'your-key-here'"
        )
    return key


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str:
    """Generate an answer using the LLM."""
    import httpx

    model_config = MODELS.get(model)
    if model_config:
        model_name = model_config["model"]
        provider = model_config["provider"]
    else:
        model_name = model
        provider = "groq"

    if not api_key:
        key_env = PROVIDER_KEYS.get(provider, "GROQ_API_KEY")
        api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(
            "API key not provided. Pass api_key parameter or set GROQ_API_KEY env var."
        )

    if provider == "groq":
        api_url = os.environ.get("GROQ_API_URL", GROQ_API_URL)
    else:
        api_url = OPENROUTER_API_URL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://football-analytics-rag.local"
        headers["X-Title"] = "Football Analytics RAG"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = httpx.post(api_url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def generate_answer_messages(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Generate an answer using message format (for chat APIs)."""
    import httpx

    if model in MODELS:
        model = MODELS[model]

    api_key = get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://football-analytics-rag.local",
        "X-Title": "Football Analytics RAG",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = httpx.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def answer_question(question: str, context: str, model: str = DEFAULT_MODEL) -> str:
    """End-to-end: build prompt and generate answer."""
    prompt = build_prompt(question, context)
    return generate_answer(prompt, model=model)


# ---------------------------------------------------------------------------
# Answer Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating an LLM answer against structured facts."""
    is_valid: bool
    contradictions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_answer: str | None = None

    def __str__(self):
        if self.is_valid:
            return "VALID"
        issues = [c["description"] for c in self.contradictions]
        return f"INVALID: {'; '.join(issues)}"


def extract_numeric_claims(text: str) -> list[dict]:
    """Extract numeric claims from text."""
    claims = []

    patterns = [
        r"(\w[\w\s]*?)\s+(?:scored|has|had|made|achieved|recorded)\s+(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        r"(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:by|from)\s+(\w[\w\s]*?)",
        r"(\w[\w\s]*?):\s*(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)",
        r"(\w[\w\s]*?)(?:'s|'s)\s+(goals?|assists?|shots?|passes?|minutes?|tackles?|interceptions?|xG)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?)",
        r"(?:^|\s)(\d+(?:\.\d+)?)\s+(goals?|assists?|shots?|passes?|minutes?|xG)(?:\s|$|[,.\])])",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) == 3:
                if groups[0].replace(".", "").isdigit():
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = groups[2].strip()
                elif groups[2].replace(".", "").isdigit():
                    entity = groups[0].strip()
                    metric = groups[1].lower()
                    value = float(groups[2])
                else:
                    entity = groups[0].strip()
                    value = float(groups[1])
                    metric = groups[2].lower()
            elif len(groups) == 2:
                if groups[0].replace(".", "").isdigit():
                    value = float(groups[0])
                    metric = groups[1].lower()
                    entity = None
                else:
                    continue
            else:
                continue

            metric = metric.rstrip("s")
            metric_map = {
                "goal": "goals", "assist": "assists", "shot": "shots",
                "pass": "passes_attempted", "minute": "minutes",
                "tackle": "successful_tackles", "interception": "successful_interceptions",
                "xg": "xg",
            }
            normalized_metric = metric_map.get(metric, metric)

            claims.append({
                "value": value,
                "metric": normalized_metric,
                "entity": entity,
                "context": text[max(0, match.start() - 20):match.end() + 20],
            })

    return claims


def validate_answer(
    llm_answer: str,
    structured_explanation: str,
    structured_value: float | int | None = None,
    structured_entity: str | None = None,
    structured_metric: str | None = None,
) -> ValidationResult:
    """Validate an LLM answer against structured facts."""
    result = ValidationResult(is_valid=True)

    llm_claims = extract_numeric_claims(llm_answer)

    expected_patterns = [
        r"(?:total|sum|count)?\s*(?:goals?|assists?|shots?|passes?|minutes?|xG)\s+(?:is|was|are|were)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:goals?|assists?|shots?|passes?|minutes?|xG)",
        r"(?:goals?|assists?|shots?|passes?|minutes?|xG)\s*:\s*(\d+(?:\.\d+)?)",
    ]

    expected_value = None
    for pattern in expected_patterns:
        expected_match = re.search(pattern, structured_explanation, re.IGNORECASE)
        if expected_match:
            expected_value = float(expected_match.group(1))
            break

    if expected_value is None and structured_value is not None:
        expected_value = float(structured_value)

    if expected_value is not None:
        for claim in llm_claims:
            if structured_metric and claim["metric"] != structured_metric:
                continue

            if structured_entity and claim["entity"]:
                entity_match = (
                    structured_entity.lower() in claim["entity"].lower() or
                    claim["entity"].lower() in structured_entity.lower()
                )
                if not entity_match:
                    continue

            if claim["value"] != expected_value:
                result.contradictions.append({
                    "llm_value": claim["value"],
                    "expected_value": expected_value,
                    "metric": claim["metric"],
                    "entity": claim["entity"],
                    "description": (
                        f"LLM claimed {claim['entity'] or 'entity'} has "
                        f"{claim['value']} {claim['metric']}, but structured "
                        f"data shows {expected_value}"
                    ),
                    "context": claim["context"],
                })

    if result.contradictions:
        result.is_valid = False
        result.corrected_answer = _generate_corrected_answer(
            llm_answer, structured_explanation, result.contradictions
        )

    return result


def _generate_corrected_answer(
    llm_answer: str,
    structured_explanation: str,
    contradictions: list[dict],
) -> str:
    """Generate a corrected answer when contradictions are detected."""
    corrected = llm_answer

    for contradiction in contradictions:
        old_value = contradiction["llm_value"]
        new_value = contradiction["expected_value"]
        metric = contradiction["metric"]

        old_pattern = f"{old_value:g}\\s+{metric}"
        new_text = f"{new_value:g} {metric}"
        corrected = re.sub(old_pattern, new_text, corrected, flags=re.IGNORECASE)

    if corrected == llm_answer:
        return (
            f"Based on the structured data:\n{structured_explanation}\n\n"
            f"(Note: The original answer contained incorrect numbers and was "
            f"replaced with verified data.)"
        )

    return (
        f"{corrected}\n\n"
        f"(Note: Numbers have been verified against structured data.)"
    )


def validate_and_correct(
    llm_answer: str,
    structured_result,
) -> tuple[str, ValidationResult]:
    """Validate an LLM answer and return corrected answer if needed."""
    if structured_result is None or not hasattr(structured_result, 'explanation'):
        return llm_answer, ValidationResult(is_valid=True)

    validation = validate_answer(
        llm_answer=llm_answer,
        structured_explanation=structured_result.explanation or "",
        structured_value=structured_result.aggregated_value,
        structured_entity=getattr(structured_result, 'entity_name', None),
        structured_metric=getattr(structured_result, 'metric', None),
    )

    if validation.is_valid:
        return llm_answer, validation
    else:
        answer = validation.corrected_answer or llm_answer
        return answer, validation
