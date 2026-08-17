"""Retrieval-stage utilities: find and rank candidates.

Migration Step 4: evidence selection (`select_relevant_chunks`) and
answerability moved to `src/context/` -- they choose/assess the evidence
subset shown downstream, which is Context Engineering's responsibility, not
retrieval's. Import them from `src.context` instead. Nothing outside this
package imported them through here, so no compatibility shim was needed.
"""

__all__: list[str] = []
