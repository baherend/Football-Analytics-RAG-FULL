"""
streamlit_app.py — Streamlit UI for FIFA World Cup 2022 RAG System

Usage:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Football Analytics RAG",
    page_icon="⚽",
    layout="wide",
)

# Load custom CSS
css_path = Path("styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Secrets Fallback
# ---------------------------------------------------------------------------
# Professor's required pattern: read from st.secrets first, fall back to env.
# Uses OPENROUTER_API_KEY in secrets.toml (mapped to GROQ_API_KEY in code).

try:
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets.get("OPENROUTER_API_KEY", "")
    if not os.environ.get("GROQ_API_URL"):
        os.environ["GROQ_API_URL"] = st.secrets.get("OPENROUTER_API_URL", "")
except Exception:
    pass

# Also set the model from secrets if available
try:
    _default_model = st.secrets.get("OPENROUTER_MODEL", "llama-3.3-70b-versatile")
except Exception:
    _default_model = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Module Loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_modules():
    """Load pipeline modules once."""
    import importlib.util

    def _import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    retrieve = _import("retrieve", "06_retrieve_context.py")
    prompting = _import("prompting", "07_prompting.py")

    return retrieve, prompting


@st.cache_resource
def load_data():
    """Load match_facts.json once."""
    import json
    path = Path("output/match_facts.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Pipeline Status
# ---------------------------------------------------------------------------


def check_pipeline():
    """Check which pipeline artifacts exist."""
    artifacts = {
        "match_facts.json": Path("output/match_facts.json").exists(),
        "documents.json": Path("output/documents.json").exists(),
        "chunks.json": Path("output/chunks.json").exists(),
        "BM25 index": Path("output/indices/bm25.pkl").exists(),
        "ChromaDB": Path("output/chroma_db/chroma.sqlite3").exists(),
    }
    return artifacts


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


with st.sidebar:
    st.title("⚽ Football Analytics")
    st.markdown("---")

    # Pipeline status
    st.subheader("Pipeline Status")
    artifacts = check_pipeline()
    for name, exists in artifacts.items():
        icon = "✅" if exists else "❌"
        st.markdown(f"{icon} {name}")

    st.markdown("---")

    # Settings
    st.subheader("Settings")
    model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0,
    )
    mode = st.selectbox(
        "Query Mode",
        ["hybrid", "structured", "semantic"],
        index=0,
    )
    k = st.slider("Retrieved chunks (k)", 1, 10, 5)

    st.markdown("---")

    # API configuration (manual override for local dev)
    st.subheader("API Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Override for local dev. Leave blank to use st.secrets.",
        value="",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")

    # Stats
    data = load_data()
    if data:
        st.subheader("Data Stats")
        st.metric("Player Records", len(data.get("player_match_facts", [])))
        st.metric("Match Records", len(data.get("match_facts", [])))
        st.metric("Team Records", len(data.get("team_match_facts", [])))


# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------


st.title("⚽ FIFA World Cup 2022 — Football Analytics RAG")
st.markdown("Ask questions about the 2022 FIFA World Cup. The system retrieves relevant data and generates answers using structured facts and semantic search.")

# Load modules
try:
    retrieve, prompting = load_modules()
except Exception as e:
    st.error(f"Failed to load pipeline modules: {e}")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route"):
            with st.expander("Routing Details"):
                st.json(msg["route"])
        if msg.get("context"):
            with st.expander("Retrieved Context"):
                st.text(msg["context"][:2000])

# Chat input
if query := st.chat_input("Ask about FIFA World Cup 2022..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            t0 = time.time()

            # Route + retrieve
            route = retrieve.route_query(query)
            result = retrieve.execute_route(route, semantic_k=k)

            # Build context
            parts = []
            has_structured = False
            sr = result.structured_result
            if sr and sr.status in ("resolved", "partial") and sr.explanation:
                parts.append(
                    "## Authoritative Data (Verified from Match Facts)\n\n"
                    f"{sr.explanation}"
                )
                has_structured = True
            elif sr and sr.status == "empty":
                parts.append(
                    "NOTE: The structured data did not contain a direct answer. "
                    "Only answer if the context clearly addresses the question."
                )
            if result.semantic_chunks:
                parts.append(
                    prompting.format_context_for_prompt(result.semantic_chunks)
                )

            context = "\n\n".join(parts) if parts else "No relevant data found."

            # Build prompt
            prompt = prompting.build_prompt(
                query, context, has_structured=has_structured
            )

            # Generate answer
            try:
                answer = prompting.generate_answer(prompt, model=model)
            except Exception as e:
                answer = f"**Error:** {e}\n\n**Retrieved Context:**\n{context[:1000]}"

            elapsed = time.time() - t0

            # Display answer
            st.markdown(answer)

            # Display metadata
            col1, col2, col3 = st.columns(3)
            col1.metric("Route", route.path)
            col2.metric("Confidence", f"{route.confidence:.2f}")
            col3.metric("Time", f"{elapsed:.1f}s")

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "route": {
                    "path": route.path,
                    "confidence": route.confidence,
                    "reason": route.reason,
                },
                "context": context,
            })

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

# Example questions
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("**Example Questions:**")
    examples = [
        "How many goals did Messi score?",
        "Who scored the most goals in the tournament?",
        "Compare Messi and Mbappé's performance",
        "How did France play in the final?",
        "Who had the highest xG?",
        "Argentina vs France in the Final",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()
