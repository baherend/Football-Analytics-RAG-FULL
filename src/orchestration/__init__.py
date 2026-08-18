"""
src/orchestration/ -- the answer policy shared by all runtime interfaces.

Phase B1. Narrow by design: it holds only what `chat.py::process_query()` and
`07_prompting.py::answer_question()` implemented identically -- context
assembly, the refusal gate, and post-generation verification + citations.

    interface adapter (chat.py / 07_prompting.py)
        routing, prompt construction, generation, interface-specific state
            |
            v
    src/orchestration/policy.py
        assemble_context -> should_refuse -> finalize_answer

Routing, prompt building and generation are deliberately NOT here: they carry
monkeypatch contracts on the interface modules, and forcing them in would mean
injecting them back out again. See policy.py's docstring for the evidence.

This is not a god orchestrator and must not become one. Add a responsibility
here only when more than one interface genuinely shares it.
"""

from src.orchestration.policy import (
    AssembledContext,
    FinalizedAnswer,
    assemble_context,
    finalize_answer,
    should_refuse,
)

__all__ = [
    "AssembledContext",
    "FinalizedAnswer",
    "assemble_context",
    "should_refuse",
    "finalize_answer",
]
