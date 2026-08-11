"""
07_prompting.py — Stage 7: Prompting & LLM Generation

Builds prompts and generates answers using the Groq API (OpenAI-compatible).
Connects the retrieval pipeline to an LLM for grounded question answering.

Input: user question
Output: grounded answer with cited sources

API Key:
    - Reads GROQ_API_KEY from environment or Streamlit secrets
    - NEVER hardcodes API keys
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from importlib import import_module

from src.artifacts import ArtifactPaths
from src.conversation_memory import (
    ConversationMemory,
    format_conversation_context,
    resolve_pronoun_references,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "haiku"

MODELS = {
    "llama-3.3-70b-versatile": {
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
    },
    "llama-3.1-8b-instant": {
        "model": "llama-3.1-8b-instant",
        "provider": "groq",
    },
    "haiku": {
        "model": "anthropic/claude-3-haiku",
        "provider": "openrouter",
    },
    "sonnet": {
        "model": "anthropic/claude-3-sonnet",
        "provider": "openrouter",
    },
}

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


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


def is_unsupported_query(structured_result, answerability) -> bool:
    """
    True when there is no usable authoritative structured result AND the
    routed semantic evidence was explicitly assessed "unanswerable".

    A usable structured result (status "resolved"/"partial" with an
    explanation) always takes precedence -- semantic-only answerability
    must never veto a valid structured answer.
    """
    has_structured = bool(
        structured_result
        and getattr(structured_result, "status", None) in ("resolved", "partial")
        and getattr(structured_result, "explanation", None)
    )
    return (
        not has_structured
        and answerability is not None
        and getattr(answerability, "status", None) == "unanswerable"
    )


def build_prompt(
    question: str,
    context: str,
    has_structured: bool = False,
) -> str:
    """Build a complete prompt for the LLM."""
    system_prompt = SYSTEM_PROMPT_WITH_STRUCTURED if has_structured else SYSTEM_PROMPT
    return f"""{system_prompt}

## Retrieved Context

{context}

## Question

{question}

## Answer

Based on the retrieved context, here is my answer:"""


def format_context_for_prompt(chunks: list[dict], max_length: int = 3000) -> str:
    """Format retrieved semantic chunks into prompt-ready source blocks."""
    if not chunks:
        return "No relevant documents found."

    parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        level = meta.get("level", "unknown")

        source = f"[Source {i + 1}: Level {level}"
        if chunk.get("chunk_id"):
            source += f", chunk_id={chunk['chunk_id']}"
        if meta.get("player_name"):
            source += f", {meta['player_name']}"
        if meta.get("team_name"):
            source += f", {meta['team_name']}"
        if meta.get("match_id"):
            source += f", Match {meta['match_id']}"
        source += "]"

        entry = f"{source}\n{chunk['text']}\n"
        if current_length + len(entry) > max_length:
            break

        parts.append(entry)
        current_length += len(entry)

    return "\n".join(parts)


def get_api_key(provider: str | None = None) -> str:
    """Return the configured API key for a generation provider."""
    if provider is None:
        provider = MODELS[DEFAULT_MODEL]["provider"]

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
    """Generate an answer through the provider associated with ``model``."""
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
        key_env = PROVIDER_KEYS.get(provider, "GROQ_API_KEY")
        raise ValueError(
            f"{key_env} not set. Pass api_key or configure the environment variable."
        )

    api_url = (
        os.environ.get("GROQ_API_URL", GROQ_API_URL)
        if provider == "groq"
        else OPENROUTER_API_URL
    )

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
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# LLM Generation (Groq API)
# ---------------------------------------------------------------------------

def ask_groq(prompt: str, api_key: str | None = None,
             model: str | None = None) -> str:
    """Generate an answer using the Groq API."""
    import httpx

    key = api_key or GROQ_API_KEY
    if not key:
        return "Error: GROQ_API_KEY not set. Please configure your API key."

    mdl = model or GROQ_MODEL
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": mdl,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    try:
        response = httpx.post(GROQ_API_URL, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text[:200]}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"



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

    def __str__(self) -> str:
        if self.is_valid:
            return "VALID"
        issues = [c["description"] for c in self.contradictions]
        return f"INVALID: {'; '.join(issues)}"


def extract_numeric_claims(text: str) -> list[dict]:
    """Extract supported numeric football-stat claims from generated text."""
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
                "goal": "goals",
                "assist": "assists",
                "shot": "shots",
                "pass": "passes_attempted",
                "minute": "minutes",
                "tackle": "successful_tackles",
                "interception": "successful_interceptions",
                "xg": "xg",
            }

            claims.append(
                {
                    "value": value,
                    "metric": metric_map.get(metric, metric),
                    "entity": entity,
                    "context": text[max(0, match.start() - 20):match.end() + 20],
                }
            )

    return claims


def _generate_corrected_answer(
    llm_answer: str,
    structured_explanation: str,
    contradictions: list[dict],
) -> str:
    """Correct contradicted numeric claims using verified structured values."""
    corrected = llm_answer

    for contradiction in contradictions:
        old_value = contradiction["llm_value"]
        new_value = contradiction["expected_value"]
        metric = contradiction["metric"]

        old_pattern = rf"{old_value:g}\s+{re.escape(metric)}"
        new_text = f"{new_value:g} {metric}"
        corrected = re.sub(
            old_pattern,
            new_text,
            corrected,
            flags=re.IGNORECASE,
        )

    if corrected == llm_answer:
        return (
            f"Based on the structured data:\n{structured_explanation}\n\n"
            "(Note: The original answer contained incorrect numbers and was "
            "replaced with verified data.)"
        )

    return (
        f"{corrected}\n\n"
        "(Note: Numbers have been verified against structured data.)"
    )


def validate_answer(
    llm_answer: str,
    structured_explanation: str,
    structured_value: float | int | None = None,
    structured_entity: str | None = None,
    structured_metric: str | None = None,
) -> ValidationResult:
    """Validate generated numeric claims against authoritative structured facts."""
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
                    structured_entity.lower() in claim["entity"].lower()
                    or claim["entity"].lower() in structured_entity.lower()
                )
                if not entity_match:
                    continue

            if claim["value"] != expected_value:
                result.contradictions.append(
                    {
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
                    }
                )

    if result.contradictions:
        result.is_valid = False
        result.corrected_answer = _generate_corrected_answer(
            llm_answer,
            structured_explanation,
            result.contradictions,
        )

    return result


# ---------------------------------------------------------------------------
# End-to-End Answer
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

    Returns (answer_string, list_of_sources).
    """
    route_and_execute = import_module("06_retrieve_context").route_and_execute

    relevant_turns = memory.search(artifact_paths, question) if memory is not None else []
    retrieval_query = resolve_pronoun_references(question, relevant_turns)

    result = route_and_execute(retrieval_query, artifact_paths=artifact_paths)
    sr = getattr(result, "structured_result", None)
    has_structured = bool(
        sr and getattr(sr, "status", None) in ("resolved", "partial") and getattr(sr, "explanation", None)
    )
    sources = result.semantic_chunks or []

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
    else:
        prompt = build_prompt(question, full_context, has_structured=has_structured)

        key = api_key or GROQ_API_KEY
        if not key:
            return "Missing GROQ_API_KEY. Please set it in Streamlit secrets or environment.", sources

        answer = ask_groq(prompt, api_key=key, model=model)

        if has_structured:
            try:
                validation = validate_answer(
                    llm_answer=answer,
                    structured_explanation=sr.explanation,
                    structured_value=sr.aggregated_value,
                    structured_metric=getattr(sr.query, "metric", None),
                )
                if not validation.is_valid:
                    answer = validation.corrected_answer or answer
            except Exception:
                # Validation is best-effort -- don't break the pipeline.
                pass

    if memory is not None:
        memory.add_turn(artifact_paths, question, answer)

    return answer, sources


if __name__ == "__main__":
    question = "Who scored the most goals?"
    answer, sources = answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"\nSources ({len(sources)}):")
    for s in sources:
        print(f"  - {s.get('chunk_id', 'N/A')}: {s['text'][:80]}...")
