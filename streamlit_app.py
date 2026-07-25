"""
streamlit_app.py — Streamlit UI for FIFA World Cup 2022 RAG Assistant

Connects to the RAG pipeline via 07_prompting.py.
Reads API key from Streamlit secrets for deployment.

Usage:
    streamlit run streamlit_app.py
"""

from importlib import import_module

import streamlit as st

# ---------------------------------------------------------------------------
# Load RAG module
# ---------------------------------------------------------------------------

rag = import_module("07_prompting")

# Configure API key from Streamlit secrets
try:
    if not rag.GROQ_API_KEY:
        rag.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    rag.GROQ_MODEL = st.secrets.get("GROQ_MODEL", rag.GROQ_MODEL)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FIFA World Cup 2022 — RAG Assistant",
    page_icon="⚽",
    layout="wide",
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0B1220;
    }
    [data-testid="stSidebar"] {
        background-color: #0F1A2E;
        border-right: 1px solid #1E293B;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    p, li, span, label {
        color: #C9D1D9;
    }
    [data-testid="stMetric"] {
        background-color: #1A2332;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #00C853 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8B95A5 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1A2332;
        border: 1px solid #2D3748;
        border-radius: 16px;
        padding: 16px;
        margin: 8px 0;
    }
    [data-testid="stChatInput"] {
        background-color: #1A2332;
        border: 2px solid #2D3748;
        border-radius: 12px;
        color: #FFFFFF;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #00C853;
    }
    .stButton > button {
        background-color: #1A2332;
        color: #FFFFFF;
        border: 1px solid #2D3748;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #2D3748;
        border-color: #00C853;
    }
    [data-testid="stExpander"] {
        background-color: #1A2332;
        border: 1px solid #2D3748;
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary {
        color: #00C853;
        font-weight: 600;
    }
    code {
        background-color: #1A2332;
        color: #00C853;
        border-radius: 4px;
        padding: 2px 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚽ Football Analytics")
    st.markdown("---")

    st.subheader("Settings")
    model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0,
    )
    k = st.slider("Sources to retrieve", 1, 5, 3)

    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This RAG assistant answers questions about the **FIFA World Cup 2022** using:
    - **2,835+ documents** from StatsBomb open data
    - **Hybrid retrieval** (BM25 + dense embeddings + RRF)
    - **Groq LLM** for answer generation
    """)

    st.markdown("---")
    st.caption("Built with Streamlit · StatsBomb Open Data")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("⚽ FIFA World Cup 2022 — RAG Assistant")
st.markdown("Ask questions about matches, players, and teams from the 2022 FIFA World Cup.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    meta = src.get("metadata", {})
                    st.markdown(f"**[{meta.get('level', '?')}]** {src['text'][:300]}...")
                    st.caption(f"Score: {src.get('rrf_score', src.get('score', 0)):.4f}")

# Chat input
if question := st.chat_input("Ask about the FIFA World Cup 2022..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            answer, sources = rag.answer_question(
                question,
                api_key=rag.GROQ_API_KEY,
                model=model,
            )
            st.markdown(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        meta = src.get("metadata", {})
                        level = meta.get("level", "?")
                        player = meta.get("player_name", "")
                        team = meta.get("team_name", "")
                        label = f"Level {level}"
                        if player:
                            label += f" · {player}"
                        if team:
                            label += f" · {team}"
                        st.markdown(f"**[{label}]**")
                        st.markdown(f"> {src['text'][:400]}")
                        score_key = "rrf_score" if "rrf_score" in src else "score"
                        st.caption(f"Relevance score: {src.get(score_key, 0):.4f}")

    # Store messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# Example questions (shown when chat is empty)
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "How many goals did Messi score in the tournament?",
        "Who scored the most goals?",
        "Compare Messi and Mbappé's performance",
        "How did France play in the final?",
        "What was Argentina's playing style?",
        "Who had the highest xG?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
