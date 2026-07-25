"""
evaluation_benchmark.py — Benchmark Dataset for RAG Evaluation

30 diverse queries covering structured, semantic, hybrid, and edge cases.
Each query has expected outcomes for automated evaluation.

Usage:
    from tests.evaluation_benchmark import BENCHMARK, get_by_category
    numeric_queries = get_by_category("numeric")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Benchmark Schema
# ---------------------------------------------------------------------------
#
# Each entry is a dict with:
#   id:                       unique identifier
#   query:                    natural language query
#   expected_route:           "structured" | "semantic" | "hybrid"
#   expected_answer:          expected aggregated value (for structured queries)
#   expected_answer_contains: substring the explanation/answer must contain
#   expected_chunks_level:    expected document level in retrieved chunks
#   expected_chunks_keywords: keywords that should appear in retrieved chunks
#   category:                 "numeric" | "superlative" | "slice" | "semantic" | "hybrid" | "edge_case"
#   notes:                    optional human-readable notes


BENCHMARK: list[dict] = [
    # -----------------------------------------------------------------------
    # Structured Numeric (6)
    # -----------------------------------------------------------------------
    {
        "id": "num-01",
        "query": "How many goals did Messi score?",
        "expected_route": "structured",
        "expected_answer": 7,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Direct numeric query for a specific player",
    },
    {
        "id": "num-02",
        "query": "How many goals did Mbappé score?",
        "expected_route": "structured",
        "expected_answer": 8,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Direct numeric query with accented name",
    },
    {
        "id": "num-03",
        "query": "What is Messi's xG?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": "6.0",
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Float metric — check approximate value",
    },
    {
        "id": "num-04",
        "query": "How many minutes did Messi play?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": "733",
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Minutes metric — check approximate value",
    },
    {
        "id": "num-05",
        "query": "How many assists did Mbappé have?",
        "expected_route": "structured",
        "expected_answer": 2,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Assists metric",
    },
    {
        "id": "num-06",
        "query": "How many goals did Argentina score in the tournament?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "numeric",
        "notes": "Team-level aggregation via player-level data",
    },

    # -----------------------------------------------------------------------
    # Structured Superlative (4)
    # -----------------------------------------------------------------------
    {
        "id": "sup-01",
        "query": "Who scored the most goals?",
        "expected_route": "structured",
        "expected_answer": 8,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "superlative",
        "notes": "Top scorer — Mbappé with 8 goals",
    },
    {
        "id": "sup-02",
        "query": "Who had the highest xG in the tournament?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "superlative",
        "notes": "Top xG player",
    },
    {
        "id": "sup-03",
        "query": "Which player had the most assists?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "superlative",
        "notes": "Top assist provider",
    },
    {
        "id": "sup-04",
        "query": "Who scored the most goals in the tournament?",
        "expected_route": "structured",
        "expected_answer": 8,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "superlative",
        "notes": "Alternative phrasing for top scorer",
    },

    # -----------------------------------------------------------------------
    # Structured Slice (4)
    # -----------------------------------------------------------------------
    {
        "id": "sli-01",
        "query": "How many goals did Messi score in knockout matches?",
        "expected_route": "structured",
        "expected_answer": 5,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "slice",
        "notes": "Filtered numeric — knockout only",
    },
    {
        "id": "sli-02",
        "query": "How many goals did Messi score in the group stage?",
        "expected_route": "structured",
        "expected_answer": 2,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "slice",
        "notes": "Filtered numeric — group stage only",
    },
    {
        "id": "sli-03",
        "query": "How many goals did Messi score against France?",
        "expected_route": "structured",
        "expected_answer": 2,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "slice",
        "notes": "Filtered numeric — opponent filter",
    },
    {
        "id": "sli-04",
        "query": "How many goals did Mbappé score in the Final?",
        "expected_route": "structured",
        "expected_answer": 3,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "slice",
        "notes": "Filtered numeric — stage filter + specific player. Mbappé scored a hat-trick (3 goals) in the Final.",
    },

    # -----------------------------------------------------------------------
    # Semantic (8)
    # -----------------------------------------------------------------------
    {
        "id": "sem-01",
        "query": "How did France play in the final?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["France", "Final"],
        "category": "semantic",
        "notes": "Match-specific descriptive query",
    },
    {
        "id": "sem-02",
        "query": "Describe Messi's tournament performance",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "4",
        "expected_chunks_keywords": ["Messi"],
        "category": "semantic",
        "notes": "Player tournament summary",
    },
    {
        "id": "sem-03",
        "query": "What was Argentina's playing style?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "team",
        "expected_chunks_keywords": ["Argentina"],
        "category": "semantic",
        "notes": "Team-level style analysis",
    },
    {
        "id": "sem-04",
        "query": "Tell me about the Semi-final between Argentina and Croatia",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["Argentina", "Croatia", "Semi"],
        "category": "semantic",
        "notes": "Match-specific with stage",
    },
    {
        "id": "sem-05",
        "query": "How did Mbappé perform in the group stage?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": ["Mbappé", "group"],
        "category": "semantic",
        "notes": "Player performance in specific stage",
    },
    {
        "id": "sem-06",
        "query": "What happened in the opening match?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["Qatar", "Ecuador"],
        "category": "semantic",
        "notes": "Opening match — Qatar vs Ecuador",
    },
    {
        "id": "sem-07",
        "query": "Describe the Final between Argentina and France",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["Argentina", "France", "Final"],
        "category": "semantic",
        "notes": "Final match description",
    },
    {
        "id": "sem-08",
        "query": "How did Argentina's defense perform?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": ["Argentina", "tackle", "interception"],
        "category": "semantic",
        "notes": "Defensive performance — may match L3/L4 chunks",
    },

    # -----------------------------------------------------------------------
    # Hybrid (4)
    # -----------------------------------------------------------------------
    {
        "id": "hyb-01",
        "query": "Compare Messi and Mbappé's tournament performance",
        "expected_route": "hybrid",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "4",
        "expected_chunks_keywords": ["Messi", "Mbappé"],
        "category": "hybrid",
        "notes": "Comparison query — needs both L4 docs",
    },
    {
        "id": "hyb-02",
        "query": "Who performed better, Messi or Mbappé?",
        "expected_route": "hybrid",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "4",
        "expected_chunks_keywords": ["Messi", "Mbappé"],
        "category": "hybrid",
        "notes": "Comparison query — alternative phrasing",
    },
    {
        "id": "hyb-03",
        "query": "Argentina vs France in the Final",
        "expected_route": "hybrid",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["Argentina", "France"],
        "category": "hybrid",
        "notes": "Match comparison with structured context",
    },
    {
        "id": "hyb-04",
        "query": "Messi's goals and how he scored them",
        "expected_route": "hybrid",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": ["Messi", "goal"],
        "category": "hybrid",
        "notes": "Structured numeric + descriptive context",
    },

    # -----------------------------------------------------------------------
    # Edge Cases (4)
    # -----------------------------------------------------------------------
    {
        "id": "edge-01",
        "query": "How many goals did Zidane score?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "edge_case",
        "notes": "Player not in WC 2022 — should return empty",
    },
    {
        "id": "edge-02",
        "query": "What is the meaning of football?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "edge_case",
        "notes": "Out-of-scope question — no relevant chunks expected",
    },
    {
        "id": "edge-03",
        "query": "Who won the World Cup?",
        "expected_route": "semantic",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": "1",
        "expected_chunks_keywords": ["Argentina", "Final"],
        "category": "edge_case",
        "notes": "Answerable from L1 Final match doc",
    },
    {
        "id": "edge-04",
        "query": "How many total goals were scored in the tournament?",
        "expected_route": "structured",
        "expected_answer": None,
        "expected_answer_contains": None,
        "expected_chunks_level": None,
        "expected_chunks_keywords": [],
        "category": "edge_case",
        "notes": "Aggregation query — may not resolve cleanly",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_by_category(category: str) -> list[dict]:
    """Return benchmark entries for a specific category."""
    return [b for b in BENCHMARK if b["category"] == category]


def get_by_id(entry_id: str) -> dict | None:
    """Return a benchmark entry by its ID."""
    for b in BENCHMARK:
        if b["id"] == entry_id:
            return b
    return None


def get_categories() -> list[str]:
    """Return all unique categories in the benchmark."""
    return sorted(set(b["category"] for b in BENCHMARK))


def summary() -> str:
    """Return a human-readable summary of the benchmark."""
    lines = [f"Benchmark: {len(BENCHMARK)} queries"]
    for cat in get_categories():
        entries = get_by_category(cat)
        lines.append(f"  {cat}: {len(entries)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
