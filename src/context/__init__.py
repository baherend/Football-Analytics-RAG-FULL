"""
src/context/ -- Context Engineering: choosing and presenting the evidence
subset shown downstream.

Migration Step 4. The layer boundary this package exists to draw:

    RETRIEVAL           find/rank candidates
      (src/retrieval/)  BM25 + dense + fusion + retrieval safeguards
            |
            v
      Candidate Pool
            |
            v
    CONTEXT ENGINEERING choose the evidence subset presented downstream
      (this package)    coverage selection -> ordering -> Evidence Pack
                                           -> rendered context
            |
            v
      ANSWERABILITY / GENERATION

Modules:

    evidence.py       -- EvidenceItem / EvidencePack (the typed handoff)
    selection.py      -- marginal-coverage evidence selection
    rendering.py      -- Evidence Pack -> prompt-ready context text
    answerability.py  -- is the retained evidence enough to answer?

Deliberately NOT implemented (no current need or consumer -- see
PROJECT_MEMORY.md's Architecture Decisions):

    Deduplicate  -- measured redundant: retrieval's safeguards already dedupe
                    by chunk_id (candidate-pool duplicate count was 0 across
                    every baseline case), so a dedup stage here would be a
                    no-op box added only to match a diagram.
    Token budget -- a budget already exists implicitly as max_chunks (count)
                    and max_length (characters); no token-aware need proven.
    Compress     -- DEFERRED, NOT YET JUSTIFIED: no demonstrated budget
                    overflow, no consumer, and no way to evaluate information
                    loss today.

Guiding principle: the smallest high-signal context that still covers the
evidence requirements -- not the largest possible Top-K.
"""

from src.context.answerability import AnswerabilityAssessment, assess_answerability
from src.context.evidence import EvidenceItem, EvidencePack
from src.context.rendering import build_context, render_pack
from src.context.selection import select_relevant_chunks

__all__ = [
    "AnswerabilityAssessment",
    "assess_answerability",
    "EvidenceItem",
    "EvidencePack",
    "build_context",
    "render_pack",
    "select_relevant_chunks",
]
