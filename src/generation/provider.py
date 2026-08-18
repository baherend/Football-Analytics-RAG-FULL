"""
src/generation/provider.py -- LLM provider adapters (model invocation only).

Migration Step 5: extracted from 07_prompting.py -- model registry, API-key
resolution, and both HTTP call paths moved verbatim apart from the
role-separation change described below. See 07_prompting.py for the
coordinator and its compatibility re-exports.

This module invokes models. It does not build prompts (src/generation/
prompt.py), select evidence (src/context/), or validate answers
(src/verification/).

## Role separation (deliberate, security-motivated behavior change)

Both `ask_groq()` and `generate_answer()` previously sent
`messages=[{"role": "user", "content": prompt}]`, putting developer policy and
untrusted retrieved evidence in one user message. Both now accept an optional
`messages=` payload built by prompt.build_messages(), which places policy in a
`system` message and evidence + question in a `user` message.

The legacy `prompt=` string path is preserved unchanged for callers that still
pass one, so no existing signature breaks. Provider defaults, timeouts, retry
behavior (there is none), and error handling are untouched.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Configuration  (defaults deliberately unchanged -- see module docstring)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "haiku"

MODELS = {
    "haiku": {"provider": "openrouter", "model": "anthropic/claude-3.5-haiku"},
    "sonnet": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"},
    "gpt4o-mini": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
    "llama": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "llama-8b": {"provider": "groq", "model": "llama-3.1-8b-instant"},
}

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_api_key(provider: str | None = None) -> str:
    """Resolve the API key for a provider from the environment."""
    key_env = PROVIDER_KEYS.get(provider or "groq", "GROQ_API_KEY")
    return os.environ.get(key_env, "")


def _messages_for(prompt: str | None, messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Prefer a role-separated payload; fall back to the legacy single message.

    Keeping the legacy branch means callers still passing a plain string keep
    their exact previous behavior, while callers using build_messages() get
    the system/user privilege separation.
    """
    if messages is not None:
        return messages
    return [{"role": "user", "content": prompt or ""}]


def generate_answer(
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
    messages: list[dict[str, Any]] | None = None,
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
        "messages": _messages_for(prompt, messages),
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


def ask_groq(prompt: str | None = None, api_key: str | None = None,
             model: str | None = None,
             messages: list[dict[str, Any]] | None = None) -> str:
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
        "messages": _messages_for(prompt, messages),
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
