"""
app_simple.py — Simplified Streamlit UI
"""
import sys
import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Football RAG", page_icon="⚽", layout="wide")

# Load CSS
css_path = Path("styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚽ Football Analytics")
    st.markdown("---")

    # API Config
    st.subheader("API Configuration")
    api_key = st.text_input("Groq API Key", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    model = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ])

    st.markdown("---")
    st.markdown("**Quick Test Queries:**")
    st.code("""
How many goals did Messi score?
Who scored the most goals?
Compare Messi and Mbappé
How did France play in the final?
Who was the most aggressive player?
    """)

# Main
st.title("⚽ FIFA World Cup 2022 — RAG Chat")
st.markdown("Ask questions about the 2022 FIFA World Cup.")

# Check API key
if not api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar to enable answers.")
    st.stop()

# Set env var for other modules
os.environ["GROQ_API_KEY"] = api_key

# Load modules
@st.cache_resource
def load_pipeline():
    import importlib.util
    def _import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    return {
        "router": _import("router", "08_router.py"),
        "prompt": _import("prompt_builder", "src/generation/prompt_builder.py"),
        "llm": _import("llm", "src/generation/llm.py"),
        "retrieve": _import("retrieve", "07_retrieve_context.py"),
    }

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask about FIFA World Cup 2022..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Process
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()

            try:
                # Route
                route = pipeline["router"].route_query(query)
                result = pipeline["router"].execute_route(route, semantic_k=5)

                # Build context
                parts = []
                has_structured = False
                sr = result.structured_result
                if sr and sr.status in ("resolved", "partial") and sr.explanation:
                    parts.append(f"## Authoritative Data\n\n{sr.explanation}")
                    has_structured = True
                elif sr and sr.status == "empty":
                    parts.append("NOTE: The structured data did not contain a direct answer. Only answer if the context clearly addresses the question.")
                if result.semantic_chunks:
                    parts.append(pipeline["prompt"].format_context_for_prompt(result.semantic_chunks))

                context = "\n\n".join(parts) if parts else "No relevant data found."

                # Build prompt
                prompt = pipeline["prompt"].build_prompt(query, context, has_structured=has_structured)

                # Generate
                answer = pipeline["llm"].generate_answer(prompt, model=model, api_key=api_key)

                elapsed = time.time() - t0

                # Display
                st.markdown(answer)
                st.caption(f"Route: {route.path} | Time: {elapsed:.1f}s")

                # Store
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.messages.append({"role": "user", "content": query})

            except Exception as e:
                st.error(f"Error: {e}")
