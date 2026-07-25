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
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# API endpoints
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model options (Groq models)
MODELS = {
    "llama-3.3-70b-versatile": {"model": "llama-3.3-70b-versatile", "provider": "groq"},
    "llama-3.1-8b-instant": {"model": "llama-3.1-8b-instant", "provider": "groq"},
    "gemma2-9b-it": {"model": "gemma2-9b-it", "provider": "groq"},
    "mixtral-8x7b-32768": {"model": "mixtral-8x7b-32768", "provider": "groq"},
    "haiku": {"model": "anthropic/claude-3-haiku", "provider": "openrouter"},
    "sonnet": {"model": "anthropic/claude-3-sonnet", "provider": "openrouter"},
}

# Provider API keys (read from environment)
PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# LLM Client
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
    """
    Generate an answer using the LLM.

    Parameters:
        prompt: Complete prompt with context and question
        model: Model identifier (full name or shortcut)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (lower = more deterministic)
        api_key: Optional API key (overrides environment variable)

    Returns:
        Generated answer string.
    """
    import httpx

    # Resolve model shortcut
    model_config = MODELS.get(model)
    if model_config:
        model_name = model_config["model"]
        provider = model_config["provider"]
    else:
        # Assume it's a full model name for Groq
        model_name = model
        provider = "groq"

    # Get provider API key (parameter takes priority)
    if not api_key:
        key_env = PROVIDER_KEYS.get(provider, "GROQ_API_KEY")
        api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(
            "API key not provided. Pass api_key parameter or set GROQ_API_KEY env var."
        )

    # Select API endpoint (allow override from environment)
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = httpx.post(
            api_url,
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
