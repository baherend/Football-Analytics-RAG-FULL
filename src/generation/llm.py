"""
llm.py — Phase 6: LLM Generation

Generates answers using an LLM via OpenRouter API.
Supports multiple models and handles errors gracefully.

Usage:
    from llm import generate_answer
    answer = generate_answer(prompt, model="anthropic/claude-3-haiku")
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default model (fast, cost-effective for RAG)
DEFAULT_MODEL = "anthropic/claude-3-haiku"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model options
MODELS = {
    "haiku": "anthropic/claude-3-haiku",
    "sonnet": "anthropic/claude-3-sonnet",
    "gpt4": "openai/gpt-4-turbo",
    "gpt3.5": "openai/gpt-3.5-turbo",
    "mistral": "mistralai/mistral-7b-instruct",
}


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Set it with: $env:OPENROUTER_API_KEY = 'your-key-here'"
        )
    return key


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Generate an answer using the LLM.

    Parameters:
        prompt: Complete prompt with context and question
        model: Model identifier (full name or shortcut)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Generated answer string.
    """
    import httpx

    # Resolve model shortcut
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = httpx.post(
            OPENROUTER_API_URL,
            json=payload,
            headers=headers,
            timeout=60.0,
        )
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
    """
    Generate an answer using message format (for chat APIs).

    Parameters:
        messages: List of {role, content} messages
        model: Model identifier
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Generated answer string.
    """
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
        response = httpx.post(
            OPENROUTER_API_URL,
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as e:
        return f"Error: API returned {e.response.status_code}: {e.response.text}"
    except httpx.TimeoutException:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    End-to-end: build prompt and generate answer.

    Parameters:
        question: User's question
        context: Retrieved context
        model: Model identifier

    Returns:
        Generated answer.
    """
    from prompt_builder import build_prompt

    prompt = build_prompt(question, context)
    return generate_answer(prompt, model=model)
