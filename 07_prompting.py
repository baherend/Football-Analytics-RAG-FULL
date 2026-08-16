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
from src.query.query_schema import ComparisonResult

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


def is_usable_structured_result(structured_result) -> bool:
    """
    True when `structured_result` carries evidence the generation layer
    may present as authoritative.

    Ordinary StructuredResult: unchanged from the existing contract --
    status "resolved"/"partial" with a non-empty explanation is usable (a
    "partial" StructuredResult may still carry a real aggregated_value
    with a scope caveat, e.g. a dropped filter -- see
    src/query/resolver.py).

    ComparisonResult (duck-typed via its `values` list, so this module
    doesn't need to import the query schema): a comparison is only usable
    when it is actually complete -- both compared entities produced a
    real numeric value. "resolved" already implies this (Step 2G). A
    "partial" comparison with a missing side (Step 2F leaves
    difference/outcome both None in that case) must NOT be treated as a
    fully verified two-sided comparison merely because its status string
    passes the ordinary resolved/partial check -- but a "partial"
    comparison where both values ARE present (the caveat came from an
    underlying entity's own StructuredResult, not from a missing side)
    remains usable, with its partial status left intact.
    """
    if not (
        structured_result
        and getattr(structured_result, "status", None) in ("resolved", "partial")
        and getattr(structured_result, "explanation", None)
    ):
        return False

    values = getattr(structured_result, "values", None)
    if values is None:
        return True  # Ordinary StructuredResult -- existing contract.

    # ComparisonResult: complete only when both entities resolved to a
    # real numeric value.
    return len(values) == 2 and values[0].value is not None and values[1].value is not None


def is_unsupported_query(structured_result, answerability) -> bool:
    """
    True when there is no usable authoritative structured result AND the
    routed semantic evidence was explicitly assessed "unanswerable".

    A usable structured result always takes precedence -- semantic-only
    answerability must never veto a valid structured answer.
    """
    return (
        not is_usable_structured_result(structured_result)
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
# Comparison Answer Validation
# ---------------------------------------------------------------------------

# Same metric-word vocabulary already used by extract_numeric_claims() above
# -- not a new vocabulary, just reused for entity-anchored matching.
_COMPARISON_METRIC_WORDS = {
    "goals": r"goals?",
    "assists": r"assists?",
    "shots": r"shots?",
    "passes_attempted": r"passes?",
    "minutes": r"minutes?",
    "successful_tackles": r"tackles?",
    "successful_interceptions": r"interceptions?",
    "xg": r"xG",
}


def _comparison_metric_word_pattern(metric: str) -> str:
    """Natural-language regex alternation for a canonical metric name."""
    return _COMPARISON_METRIC_WORDS.get(metric, re.escape(metric) + "s?")


def _claimed_value_near_entity(text: str, entity_name: str, metric_word: str) -> float | None:
    """
    Return the first numeric value `text` claims for `entity_name` near
    `metric_word`, or None if no such claim is made. Anchored on the
    exact (escaped) entity_name string rather than a generic \\w+ capture,
    so names with spaces, accents, apostrophes, or hyphens work safely.
    """
    escaped = re.escape(entity_name)
    patterns = [
        rf"{escaped}(?:'s)?\s*(?:scored|had|has|got|recorded|made)?\s*(\d+(?:\.\d+)?)\s+{metric_word}",
        rf"(\d+(?:\.\d+)?)\s+{metric_word}\s*(?:for|by|from)\s*{escaped}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


_NEGATION_PATTERN = re.compile(r"\b(?:not|n't|never|no\s+longer)\b", re.IGNORECASE)


def _claims_higher(text: str, name_x: str, name_y: str) -> bool:
    """
    True if `text` claims `name_x` scored/had more, or is higher, than
    `name_y` -- bounded to avoid matching across unrelated sentences.

    A negated claim ("X did not score more than Y") is deliberately NOT
    treated as an affirmative claim in either direction. Correctly
    interpreting a negated comparison's actual meaning would require real
    language understanding, which this deterministic validator
    intentionally does not attempt -- treating it as "no clear directional
    claim" (an omission, not a contradiction) is safer than guessing a
    polarity that may be wrong either way.
    """
    pattern = (
        rf"{re.escape(name_x)}\b([^.?!]{{0,60}}?)\b(?:more|higher)\b"
        rf"[^.?!]{{0,40}}?\bthan\b[^.?!]{{0,10}}?{re.escape(name_y)}"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return False
    if _NEGATION_PATTERN.search(match.group(1)):
        return False
    return True


def _claimed_difference(text: str, metric_word: str) -> float | None:
    """Return an explicitly stated numeric difference/margin, or None."""
    patterns = [
        rf"\bby\s+(\d+(?:\.\d+)?)\s+{metric_word}\b",
        rf"(?:difference|margin)\s+of\s+(\d+(?:\.\d+)?)\s*{metric_word}?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _generate_comparison_correction(comparison_result: ComparisonResult) -> str:
    """Deterministic corrected comparison sentence -- never another LLM call."""
    name_a, value_a = comparison_result.values[0].entity_name, comparison_result.values[0].value
    name_b, value_b = comparison_result.values[1].entity_name, comparison_result.values[1].value
    metric = comparison_result.metric

    if comparison_result.outcome == "tie":
        return f"{name_a} and {name_b} were tied at {value_a:g} {metric} each."

    if comparison_result.outcome == "entity_a_higher":
        higher_name, higher_value = name_a, value_a
        lower_name, lower_value = name_b, value_b
    else:
        higher_name, higher_value = name_b, value_b
        lower_name, lower_value = name_a, value_a

    return (
        f"{higher_name} had {higher_value:g} {metric} compared with {lower_name}'s {lower_value:g}, "
        f"so {higher_name} was higher by {comparison_result.difference:g}."
    )


def validate_comparison_answer(llm_answer: str, comparison_result: ComparisonResult) -> ValidationResult:
    """
    Validate a generated comparison answer against ComparisonResult's
    deterministic structured facts (both entity values, difference,
    outcome) -- reusing the same ValidationResult/corrected_answer
    contract as validate_answer(), for a two-entity shape instead of a
    single scalar value.

    Only meaningful for a complete comparison (both values present,
    outcome/difference already computed by ComparisonResult.__post_init__)
    -- Step 2H already keeps an incomplete comparison from reaching
    generation as authoritative evidence, so the guard below is a
    defensive no-op, not a new gate.

    Detects only explicit contradictions -- never requires the answer to
    mention every fact, and never flags an answer merely for omitting
    values, the difference, or for paraphrasing "higher" as "more":
      - a claimed value for a named entity that disagrees with its
        authoritative aggregated_value
      - a claimed "X is higher/more than Y" ordering that disagrees with
        the authoritative outcome (including a false claim over a tie)
      - an explicitly stated numeric difference that disagrees with the
        authoritative difference

    Never re-derives values from explanation text or calls an LLM to
    judge or correct the answer -- both detection and correction use
    only comparison_result's already-computed fields.
    """
    result = ValidationResult(is_valid=True)

    values = comparison_result.values
    if len(values) != 2 or values[0].value is None or values[1].value is None:
        return result  # incomplete comparison -- nothing to validate here

    name_a, value_a = values[0].entity_name, values[0].value
    name_b, value_b = values[1].entity_name, values[1].value
    metric_word = _comparison_metric_word_pattern(comparison_result.metric)

    contradictions = []

    for name, expected_value in ((name_a, value_a), (name_b, value_b)):
        claimed = _claimed_value_near_entity(llm_answer, name, metric_word)
        if claimed is not None and claimed != expected_value:
            contradictions.append({
                "description": (
                    f"LLM claimed {name} has {claimed:g} {comparison_result.metric}, "
                    f"but structured data shows {expected_value:g}"
                ),
            })

    claims_a_higher = _claims_higher(llm_answer, name_a, name_b)
    claims_b_higher = _claims_higher(llm_answer, name_b, name_a)
    if claims_a_higher and comparison_result.outcome != "entity_a_higher":
        contradictions.append({
            "description": f"LLM claimed {name_a} is higher than {name_b}, contradicting the authoritative outcome",
        })
    elif claims_b_higher and comparison_result.outcome != "entity_b_higher":
        contradictions.append({
            "description": f"LLM claimed {name_b} is higher than {name_a}, contradicting the authoritative outcome",
        })

    claimed_diff = _claimed_difference(llm_answer, metric_word)
    if (
        claimed_diff is not None
        and comparison_result.difference is not None
        and claimed_diff != comparison_result.difference
    ):
        contradictions.append({
            "description": (
                f"LLM stated a difference of {claimed_diff:g}, but the authoritative "
                f"difference is {comparison_result.difference:g}"
            ),
        })

    if contradictions:
        result.is_valid = False
        result.contradictions = contradictions
        result.corrected_answer = _generate_comparison_correction(comparison_result)

    return result


def validate_structured_answer(llm_answer: str, structured_result) -> ValidationResult:
    """
    Shared validation dispatch used by both chat.py::process_query() and
    answer_question(): validate_comparison_answer() for a ComparisonResult
    (explicit isinstance check -- no circular import, see
    src/query/query_schema.py), otherwise the existing scalar
    validate_answer(), unchanged. Both return the same ValidationResult,
    so callers apply identical is_valid/corrected_answer handling
    regardless of which validator ran.
    """
    if isinstance(structured_result, ComparisonResult):
        return validate_comparison_answer(llm_answer, structured_result)
    return validate_answer(
        llm_answer=llm_answer,
        structured_explanation=structured_result.explanation,
        structured_value=getattr(structured_result, "aggregated_value", None),
        structured_metric=getattr(getattr(structured_result, "query", None), "metric", None),
    )


# ---------------------------------------------------------------------------
# User-Facing Citations
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

    Returns (answer_string, list_of_user_facing_citations) -- see
    build_user_citations() for the citation shape. The citation list is
    always [] for the deterministic refusal answer, even if evidence was
    retrieved but judged insufficient (see is_unsupported_query()) -- an
    empty citation list is a self-evident "no source list", not an ordinary
    "loading" gap; showing retrieved-but-insufficient evidence under a
    refusal would misleadingly imply it supports an answer it does not.
    """
    route_and_execute = import_module("06_retrieve_context").route_and_execute

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
        prompt = build_prompt(question, full_context, has_structured=has_structured)

        key = api_key or GROQ_API_KEY
        if not key:
            return "Missing GROQ_API_KEY. Please set it in Streamlit secrets or environment.", []

        answer = ask_groq(prompt, api_key=key, model=model)

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
