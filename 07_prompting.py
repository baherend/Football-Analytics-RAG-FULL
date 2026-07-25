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
from importlib import import_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a football analytics assistant specializing in FIFA World Cup 2022 data.
You answer questions based ONLY on the provided context from StatsBomb match data.

Rules:
1. Answer ONLY based on the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information, say: "I don't have enough data to answer this question."
3. NEVER guess or infer information not present in the context.
4. Cite specific matches, players, or statistics when possible.
5. Be precise with numbers (goals, xG, minutes, etc.).
6. Cite sources like [Source 1], [Source 2], etc."""


def build_prompt(question: str, context: str) -> str:
    """Build a complete prompt for the LLM."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Context

{context}

## Question

{question}

## Answer

Based on the retrieved context, here is my answer:"""


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
# End-to-End Answer
# ---------------------------------------------------------------------------

def answer_question(question: str, api_key: str | None = None,
                    model: str | None = None) -> tuple[str, list[dict]]:
    """
    End-to-end: retrieve context, build prompt, generate answer.

    Returns (answer_string, list_of_sources).
    """
    build_context = import_module("06_retrieve_context").build_context
    context, sources = build_context(question)
    prompt = build_prompt(question, context)

    key = api_key or GROQ_API_KEY
    if not key:
        return "Missing GROQ_API_KEY. Please set it in Streamlit secrets or environment.", sources

    answer = ask_groq(prompt, api_key=key, model=model)
    return answer, sources


if __name__ == "__main__":
    question = "How many goals did Messi score in the tournament?"
    answer, sources = answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"\nSources ({len(sources)}):")
    for s in sources:
        print(f"  - {s.get('chunk_id', 'N/A')}: {s['text'][:80]}...")
