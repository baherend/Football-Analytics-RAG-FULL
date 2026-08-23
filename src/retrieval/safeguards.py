"""
src/retrieval/safeguards.py -- Retrieval-side safeguards: Arabic-aware
comparison, team-style, and match-level entity/document boosting, plus
query-entity sibling expansion.

Migration Step 2 (Retrieval Split): mechanically extracted from
src/retrieval/search.py's Arabic Safeguard-Trigger Normalization, Comparison
Entity Detection, Team Style Query Detection, and Match-Level Query
Detection sections -- no logic changes. See src/retrieval/search.py for the
compatibility re-exports existing callers keep using.

Does NOT do BM25, dense retrieval, fusion, or orchestration -- see
src/retrieval/bm25.py, dense.py, fusion.py, service.py for those.
"""

from __future__ import annotations

import re

from src.artifacts import ArtifactPaths
# Phase 5: team-style classification moved to the neutral src/team_style.py so
# that src/query/intent.py can classify a query without importing retrieval.
# Imported back here for the two safeguards that genuinely need it:
# _detect_comparison_entities() shares the Arabic matching normalizer, and
# _ensure_team_style_doc() below uses the entity detector.
from src.team_style import (
    _detect_team_style_entities,
    _normalize_arabic_for_matching,
)


def _get_chunks(artifact_paths: ArtifactPaths | None = None) -> list[dict]:
    """
    Load chunks via src.retrieval.search's cached loader.

    Deliberately a lazy (call-time, not import-time) import: _load_chunks()
    and its cache dict (_chunks_cache) must stay defined in search.py --
    src/evaluation/retrieval_evaluator.py::reset_retrieval_caches() resets that cache
    by reassigning the `_chunks_cache` attribute directly on the
    `src.retrieval.search` module object between benchmark cases. Re-exporting
    the cache dict into this module by name would silently desync it from
    search.py's copy (attribute reassignment doesn't propagate back into the
    module that actually owns it), breaking that reset contract. A top-level
    (import-time) import here would also create a circular import with
    search.py, which imports these detection/ensure functions back for
    re-export.
    """
    from src.retrieval import search as _search

    return _search._load_chunks(artifact_paths.chunks if artifact_paths is not None else None)


# ---------------------------------------------------------------------------
# Comparison Entity Detection
# ---------------------------------------------------------------------------

# Normalized (post alef-substitution) MSA/Egyptian comparison markers.
# Deliberately no bare "X و Y" fallback (unlike English's broad "X or Y"):
# "و" is a far more common, ambiguous conjunction in Arabic than "or" is in
# English, so an equivalent broad fallback would create excessive false
# positives -- see test_comparison_no_false_positive_on_and_without_comparison_words.
_AR_BETTER_WORD = r"(?:احسن|افضل)"
# Bounded to 60 characters -- see _LATIN_ENTITY_SPAN's comment in
# src/team_style.py for
# why an unbounded quantifier here is a regex-complexity (ReDoS) risk,
# confirmed via adversarial stress testing: the un-anchored patterns below
# (no fixed literal prefix for re.search to fast-scan for) combine
# re.search()'s per-position retry with this quantifier's backtracking,
# producing O(N^2)-or-worse behavior on long non-matching input without
# a bound.
#
# Greedy (not non-greedy): a trailing entity group anchored only by a
# generic "end of phrase" marker (whitespace/؟/?/end-of-string) needs to
# consume the FULL multi-word name greedily -- a non-greedy version stops
# at the first word (e.g. "Lionel" instead of "Lionel Andres Messi"),
# since bare whitespace after the first word already satisfies a
# non-greedy trailing anchor. Greedy backtracks (bounded, so still fast)
# to give back exactly the separator whitespace the rest of the pattern
# needs -- confirmed correct for both single- and multi-word names.
_AR_LATIN_ENTITY = r"[A-Za-z][A-Za-z .'\-]{0,58}"


def _detect_comparison_entities(query: str) -> list[str]:
    """
    Detect if a query is comparing two entities (players/teams).

    Returns list of entity names if comparison detected, empty list otherwise.
    Recognizes English ("compare X and Y", "X vs Y", "who was better, X or
    Y") and MSA/Egyptian ("قارن بين X و Y", "مين أحسن X ولا Y", "X ولا Y مين
    كان أفضل", "مين الأفضل بين X و Y") comparison phrasing. Arabic patterns
    match entities as bounded Latin-script spans directly (_AR_LATIN_ENTITY).
    """
    import re

    query_lower = query.lower().strip()

    # Bounded (to 80/60/40 chars respectively -- generous for any real
    # entity name/filler clause) for the same regex-complexity reason as
    # _AR_LATIN_ENTITY above: these patterns have no fixed literal prefix
    # (or, for "compare"/"who...better", the ambiguous part still follows
    # unbounded through re.search()'s per-position retries on long
    # non-matching input), confirmed vulnerable via adversarial stress
    # testing for the "X vs Y" and "X or Y" patterns specifically -- the
    # other two were bounded defensively even though not proven exploitable.

    # Pattern: "Compare X and Y"
    match = re.search(r"compare\s+(.{1,80}?)\s+and\s+(.{1,80}?)(?:\s|'s|$|\?)", query_lower)
    if match:
        return [match.group(1).strip().rstrip("'s"), match.group(2).strip().rstrip("'s")]

    # Pattern: "X vs Y"
    match = re.search(r"(\w{1,60})\s+vs\.?\s+(\w{1,60})", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Pattern: "Who performed better ... X or Y"
    match = re.search(r"who\s+(?:performed|played|did)\s+better.{0,40}?(\w{1,60})\s+or\s+(\w{1,60})", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Pattern: "X or Y" (simple)
    match = re.search(r"(\w{1,60})\s+or\s+(\w{1,60})(?:\s|$|\?)", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # --- MSA / Egyptian comparison patterns ---
    # `[,،]?\s+` (rather than plain `\s+`) tolerates an optional comma
    # (Latin or Arabic) directly after an entity, e.g. "قارن بين Messi, و
    # Mbappe" -- a natural pause a person might type before the connector.
    normalized = _normalize_arabic_for_matching(query_lower)
    arabic_patterns = [
        # "قارن بين X و Y"
        rf"قارن\s+بين\s+({_AR_LATIN_ENTITY})[,،]?\s+و\s+({_AR_LATIN_ENTITY})(?:\s|؟|\?|$)",
        # "قارن X مع Y"
        rf"قارن\s+({_AR_LATIN_ENTITY})[,،]?\s+مع\s+({_AR_LATIN_ENTITY})(?:\s|؟|\?|$)",
        # "مين أحسن X ولا Y" / "مين كان أحسن X ولا Y"
        rf"مين\s+(?:كان\s+)?{_AR_BETTER_WORD}[,،]?\s+({_AR_LATIN_ENTITY})[,،]?\s+ولا\s+({_AR_LATIN_ENTITY})(?:\s|؟|\?|$)",
        # "X ولا Y مين كان أفضل"
        rf"({_AR_LATIN_ENTITY})[,،]?\s+ولا\s+({_AR_LATIN_ENTITY})\s+مين\s+(?:كان\s+)?{_AR_BETTER_WORD}",
        # "مين الأفضل بين X و Y"
        rf"مين\s+(?:هو\s+)?ال{_AR_BETTER_WORD}\s+بين\s+({_AR_LATIN_ENTITY})[,،]?\s+و\s+({_AR_LATIN_ENTITY})(?:\s|؟|\?|$)",
    ]
    for pattern in arabic_patterns:
        match = re.search(pattern, normalized)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]

    return []


def _find_l4_document(player_name: str, chunks: list[dict]) -> dict | None:
    """Find a player's L4 tournament summary document in chunks."""
    player_lower = player_name.lower()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("level") == "4":
            chunk_player = chunk.get("player_name", "").lower()
            if player_lower in chunk_player or player_lower in chunk.get("text", "").lower()[:100]:
                return chunk

    return None


def _ensure_comparison_entities(
    query: str,
    results: list[dict],
    k: int,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Ensure that when a query compares two entities, both entities' L4
    tournament summary documents are included in the final top-k results.

    Strategy: check only the top-k (not all candidates), and if an entity's
    L4 doc is missing from top-k, find it in the chunk store and prepend it.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
    """
    entities = _detect_comparison_entities(query)
    if len(entities) < 2:
        return results

    # Load chunks for L4 lookup
    chunks = _get_chunks(artifact_paths)

    # Check current top-k for existing L4 docs
    top_k = results[:k]
    additions = []

    for entity in entities:
        entity_lower = entity.lower()

        # Check if entity's L4 doc is already in top-k
        has_l4_in_topk = any(
            r.get("metadata", {}).get("level") == "4" and
            entity_lower in r.get("metadata", {}).get("player_name", "").lower()
            for r in top_k
        )

        if not has_l4_in_topk:
            # Find entity's L4 chunk-0 (the main summary) in all chunks
            for chunk in chunks:
                if chunk.get("level") == "4" and chunk["chunk_id"].endswith("-chunk-0"):
                    player_name = (chunk.get("player_name") or
                                   chunk.get("metadata", {}).get("player_name", "")).lower()
                    if entity_lower in player_name:
                        additions.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "metadata": {
                                "document_id": chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"),
                                "level": "4",
                                "player_name": chunk.get("player_name") or chunk.get("metadata", {}).get("player_name"),
                                "team_name": chunk.get("team_name") or chunk.get("metadata", {}).get("team_name"),
                            },
                            "score": 0.01,
                            "rrf_score": 0.01,
                            "source": "comparison_boost",
                        })
                        break

    # Prepend additions so they appear in top-k
    if additions:
        results = additions + results

    # Prepend additions and deduplicate
    if additions:
        seen_ids = set()
        deduped = []
        for r in additions + results:
            if r["chunk_id"] not in seen_ids:
                deduped.append(r)
                seen_ids.add(r["chunk_id"])
        return deduped

    return results

def _ensure_team_style_doc(
    query: str,
    results: list[dict],
    k: int,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    When a query asks about one or more teams' playing style, ensure each
    named team's team-level analysis document is included in the top-k
    results. A query naming multiple teams (e.g. a style-comparison
    "multi" case) gets each team's document, not just the first-mentioned
    one -- see _detect_team_style_entities.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
    """
    team_names = _detect_team_style_entities(query)
    if not team_names:
        return results

    top_k = results[:k]
    chunks = None
    additions = []

    for team_name in team_names:
        team_lower = team_name.lower()

        # Check if this team's doc is already in top-k
        has_team_in_topk = any(
            r.get("metadata", {}).get("level") == "team" and
            team_lower in (r.get("metadata", {}).get("team_name") or r.get("team_name") or "").lower()
            for r in top_k
        )
        if has_team_in_topk:
            continue

        if chunks is None:
            chunks = _get_chunks(artifact_paths)

        for chunk in chunks:
            if chunk.get("level") == "team":
                chunk_team = (chunk.get("team_name") or
                              chunk.get("metadata", {}).get("team_name", "")).lower()
                if team_lower in chunk_team and chunk["chunk_id"].endswith("-chunk-0"):
                    additions.append({
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "metadata": {
                            "document_id": chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"),
                            "level": "team",
                            "team_name": chunk.get("team_name") or chunk.get("metadata", {}).get("team_name"),
                        },
                        "score": 0.01,
                        "rrf_score": 0.01,
                        "source": "team_style_boost",
                    })
                    break

    if not additions:
        return results

    # Prepend additions and deduplicate (mirrors _ensure_comparison_entities).
    seen_ids = set()
    deduped = []
    for r in additions + results:
        if r["chunk_id"] not in seen_ids:
            deduped.append(r)
            seen_ids.add(r["chunk_id"])
    return deduped


# ---------------------------------------------------------------------------
# Match-Level Query Detection
# ---------------------------------------------------------------------------

_MATCH_QUERY_PATTERNS = [
    r"how\s+did\s+(.+?)\s+(?:perform|play|do)\b",
    r"what\s+happened\s+in\s+(?:the\s+)?(?:match\s+)?(?:between\s+)?(.+?)\s+(?:and|vs)",
    r"how\s+did\s+(.+?)\s+fare",
    r"describe\s+(.+?)(?:'s|s')\s+(?:match|game|performance)",
    r"(.+?)(?:'s|s')\s+(?:match|game)\s+(?:against|vs)",
]

_STAGE_KEYWORDS = {
    "semi": "Semi-finals",
    "semi-final": "Semi-finals",
    "quarter": "Quarter-finals",
    "quarter-final": "Quarter-finals",
    "final": "Final",
    "round of 16": "Round of 16",
    "group": "Group Stage",
    "group stage": "Group Stage",
    "3rd place": "3rd Place Final",
}


def _detect_match_query(query: str) -> tuple[str | None, str | None]:
    """Return the detected team and stage for a specific-match query."""
    query_lower = query.lower().strip()

    stage = None
    for keyword, stage_name in _STAGE_KEYWORDS.items():
        if keyword in query_lower:
            stage = stage_name
            break

    head_to_head_patterns = [
        (
            r"\b(?:in|between)\s+(?:the\s+)?"
            r"([a-z?-?][a-z?-? .'-]+?)\s+"
            r"(?:vs\.?|versus|and)\s+"
            r"([a-z?-?][a-z?-? .'-]+?)"
            r"(?=\s+(?:final|semi-finals?|semi-final|quarter-finals?|"
            r"quarter-final|round of 16|group stage|3rd place final)\b|\?|$)"
        ),
        (
            r"^([a-z?-?][a-z?-? .'-]+?)\s+"
            r"(?:vs\.?|versus)\s+"
            r"([a-z?-?][a-z?-? .'-]+?)"
            r"(?=\s+(?:final|semi-finals?|semi-final|quarter-finals?|"
            r"quarter-final|round of 16|group stage|3rd place final)\b|\?|$)"
        ),
    ]

    for pattern in head_to_head_patterns:
        match = re.search(pattern, query_lower)
        if match:
            team_name = match.group(1).strip(" ,.?")
            if len(team_name) > 2:
                return team_name.title(), stage

    for pattern in _MATCH_QUERY_PATTERNS:
        match = re.search(pattern, query_lower)
        if not match:
            continue

        team_name = match.group(1).strip()
        for word in ["the", "a", "an", "in", "during", "at", "did"]:
            team_name = team_name.replace(f" {word} ", " ").strip()

        if len(team_name) > 2:
            return team_name.title(), stage

    return None, stage



def _detect_match_teams(query: str) -> tuple[str | None, str | None]:
    """Detect two teams mentioned in a head-to-head match query."""
    query_lower = query.lower().strip()
    query_lower = re.sub(r"^in\s+", "", query_lower)

    patterns = [
        # Possessive result phrasing:
        # "Chelsea's 2-2 draw with Tottenham"
        # "Manchester City's 6-1 win over Newcastle"
        # "Everton's 2-3 loss to West Ham United"
        r"(?:how\s+did\s+)?([a-z][a-z .'-]+?)(?:'s|s')\s+"
        r"\d+\s*[-\u2013]\s*\d+\s+"
        r"(?:draw\s+with|win\s+over|loss\s+to|lost\s+to|defeat\s+to)\s+"
        r"([a-z][a-z .'-]+?)"
        r"(?=\s+(?:on|in|during|unfold|match|game|\d)|\s*,|\?|$)",

        # Verb-first result phrasing:
        # "Manchester United beat Arsenal 3-2"
        r"(?:how\s+did\s+)?([a-z][a-z .'-]+?)\s+beat\s+"
        r"([a-z][a-z .'-]+?)\s+\d+\s*[-\u2013]\s*\d+"
        r"(?=\s*(?:,|\?|$)|\s+(?:on|in|during|unfold))",

        # Explicit "match between X and Y" phrasing.
        r"(?:the\s+)?match\s+between\s+"
        r"([a-z][a-z .'-]+?)\s+and\s+"
        r"([a-z][a-z .'-]+?)"
        r"(?=\s+(?:on|in|during)|[?.!,]|$)",

        # Generic head-to-head phrasing retained for compatibility.
        r"([a-z][a-z .'-]+?)\s+(?:vs\.?|versus|and|with)\s+"
        r"([a-z][a-z .'-]+?)"
        r"(?=\s+(?:on|in|during|match|game|draw|win|lost|\d)|\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            first = match.group(1).strip(" ,.?").title()
            second = match.group(2).strip(" ,.?").title()

            for suffix in (" Draw", " Win", " Lost", " Loss"):
                if first.endswith(suffix):
                    first = first[:-len(suffix)].strip()

            if first.casefold() in {"draw", "win", "lost", "loss"}:
                continue

            if len(first) > 2 and len(second) > 2:
                return first, second

    return None, None



def _boost_match_pair_candidates(
    query: str,
    results: list[dict],
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """Promote the exact L1 fixture identified by teams and score."""

    first_team, second_team = _detect_match_teams(query)

    if not first_team or not second_team:
        return results

    # Reuse the structured-query team's canonical name resolver so aliases /
    # partial names such as "Tottenham" -> "Tottenham Hotspur" are resolved
    # against the active competition dataset, never a different namespace.
    from src.query.resolver import DATA_PATH, _load_data, _resolve_team_name

    data_path = artifact_paths.match_facts if artifact_paths is not None else DATA_PATH
    try:
        data = _load_data(data_path)
        first_team = _resolve_team_name(first_team, data) or first_team
        second_team = _resolve_team_name(second_team, data) or second_team
    except (FileNotFoundError, KeyError, OSError):
        # Retrieval must retain its previous best-effort behaviour if the
        # structured artifact is unavailable.
        pass

    first_cf = first_team.casefold()
    second_cf = second_team.casefold()

    score_match = re.search(r"\b(\d+)\s*[-\u2013]\s*(\d+)\b", query)
    query_score = (
        (int(score_match.group(1)), int(score_match.group(2)))
        if score_match
        else None
    )

    chunks = _get_chunks(artifact_paths)
    matching_fixtures: dict[object, dict] = {}

    for chunk in chunks:
        if chunk.get("level") != "1":
            continue

        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        home = str(metadata.get("home_team", "")).casefold()
        away = str(metadata.get("away_team", "")).casefold()

        same_orientation = home == first_cf and away == second_cf
        reverse_orientation = home == second_cf and away == first_cf

        if not (same_orientation or reverse_orientation):
            continue

        if query_score is not None:
            first_score, second_score = query_score
            home_score = metadata.get("home_score")
            away_score = metadata.get("away_score")

            if same_orientation:
                score_matches = (
                    home_score == first_score
                    and away_score == second_score
                )
            else:
                score_matches = (
                    home_score == second_score
                    and away_score == first_score
                )

            if not score_matches:
                continue

        fixture_key = (
            chunk.get("match_id")
            or metadata.get("match_id")
            or chunk.get("document_id")
            or metadata.get("document_id")
        )
        if fixture_key is not None:
            matching_fixtures.setdefault(fixture_key, chunk)

    # Multiple chunks from one L1 document are one fixture, not ambiguity.
    # Only refuse the boost when genuinely different matches still satisfy
    # the available team/score evidence.
    if len(matching_fixtures) != 1:
        return results

    chunk = next(iter(matching_fixtures.values()))

    results = [
        item
        for item in results
        if item.get("chunk_id") != chunk.get("chunk_id")
    ]

    results.insert(
        0,
        {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": chunk.get("document_id"),
            "text": chunk.get("text", ""),
            "metadata": {
                **chunk.get("metadata", {}),
                "document_id": chunk.get("document_id")
                or chunk.get("metadata", {}).get("document_id"),
                "level": chunk.get("level")
                or chunk.get("metadata", {}).get("level"),
                "match_id": chunk.get("match_id")
                or chunk.get("metadata", {}).get("match_id"),
            },
            "score": 0.0,
            "rrf_score": 0.0,
            "source": "match_pair_boost",
        },
    )

    return results

def _ensure_match_summary(
    query: str,
    results: list[dict],
    k: int,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Include an L1 match summary only for a genuine match query, without
    displacing stronger retrieved results.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
    """
    team_name, stage = _detect_match_query(query)
    if not team_name and not stage:
        return results

    query_lower = query.lower().strip()
    explicit_match_intent = any(
        marker in query_lower
        for marker in (
            " match",
            " game",
            "between ",
            " vs ",
            " versus ",
            "what happened",
            "key events",
        )
    )

    if team_name is None and not explicit_match_intent:
        return results

    chunks = _get_chunks(artifact_paths)
    team_lower = (team_name or "").lower()

    if team_name:
        known_teams = {
            (
                chunk.get("team_name")
                or chunk.get("metadata", {}).get("team_name")
                or ""
            ).strip().lower()
            for chunk in chunks
            if chunk.get("level") == "team"
        }
        known_teams.discard("")

        is_known_team = any(
            team_lower == known
            or team_lower in known
            or known in team_lower
            for known in known_teams
        )
        if not is_known_team:
            return results

    def _matches_stage(text: str, requested_stage: str) -> bool:
        text_lower = (text or "").lower()[:300]
        requested = requested_stage.lower()

        if requested == "final":
            exclusions = (
                "semi-final",
                "semi final",
                "quarter-final",
                "quarter final",
                "3rd place final",
                "third place final",
            )
            if any(value in text_lower for value in exclusions):
                return False
            return re.search(r"\bfinal\b", text_lower) is not None

        return requested in text_lower

    top_k = results[:k]
    already_present = any(
        item.get("metadata", {}).get("level") == "1"
        and (
            team_lower in (item.get("text", "") or "").lower()[:300]
            if team_lower
            else True
        )
        and (
            _matches_stage(item.get("text", ""), stage)
            if stage
            else True
        )
        for item in top_k
    )
    if already_present:
        return results

    for chunk in chunks:
        if chunk.get("level") != "1":
            continue

        text = chunk.get("text", "")
        team_match = team_lower in text.lower()[:300] if team_lower else True
        stage_match = _matches_stage(text, stage) if stage else True

        if not (team_match and stage_match):
            continue

        if k <= 1:
            return results

        addition = {
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "metadata": {
                "document_id": (
                    chunk.get("document_id")
                    or chunk.get("metadata", {}).get("document_id")
                ),
                "level": "1",
                "match_id": (
                    chunk.get("match_id")
                    or chunk.get("metadata", {}).get("match_id")
                ),
                "home_team": chunk.get("metadata", {}).get("home_team"),
                "away_team": chunk.get("metadata", {}).get("away_team"),
            },
            "score": 0.01,
            "rrf_score": 0.01,
            "source": "match_summary_boost",
        }

        existing = [
            item
            for item in results
            if item.get("chunk_id") != addition["chunk_id"]
        ]

        insert_at = min(k - 1, len(existing))
        return existing[:insert_at] + [addition] + existing[insert_at:]

    return results



def _expand_exact_fixture_candidates(
    query: str,
    results: list[dict],
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """Expand a trusted exact-fixture boost across answer-bearing levels.

    Only a unique ``match_pair_boost`` is trusted as the fixture identity.
    Level 1/2 chunks from that match may be added directly. Level 3 is added
    only for the player explicitly requested by a ``how did X perform``
    clause, resolved against players from that same match.
    """
    boosted_match_ids: set[str] = set()

    for result in results:
        if result.get("source") != "match_pair_boost":
            continue

        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        match_id = result.get("match_id") or metadata.get("match_id")
        if match_id is not None:
            boosted_match_ids.add(str(match_id))

    if len(boosted_match_ids) != 1:
        return results

    target_match_id = next(iter(boosted_match_ids))
    raw_chunks = _get_chunks(artifact_paths)

    same_match_chunks = []
    for chunk in raw_chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        match_id = chunk.get("match_id") or metadata.get("match_id")
        if match_id is not None and str(match_id) == target_match_id:
            same_match_chunks.append(chunk)

    if not same_match_chunks:
        return results

    requested_player = None
    player_match = re.search(
        r"\bhow\s+did\s+((?:(?!\bhow\s+did\b).)+?)\s+perform\b",
        query,
        flags=re.IGNORECASE,
    )
    if player_match:
        requested_player = player_match.group(1).strip(" ,.?")

    resolved_player = None
    if requested_player:
        player_names = []
        seen_player_names = set()

        for chunk in same_match_chunks:
            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            level = str(chunk.get("level") or metadata.get("level") or "")
            if level != "3":
                continue

            player_name = (
                chunk.get("player_name")
                or metadata.get("player_name")
            )
            if player_name and player_name not in seen_player_names:
                seen_player_names.add(player_name)
                player_names.append(player_name)

        if player_names:
            from src.query.resolver import _resolve_player_name

            local_data = {
                "player_match_facts": [
                    {"player_name": name}
                    for name in player_names
                ]
            }
            resolved_player = _resolve_player_name(
                requested_player,
                local_data,
            )

    eligible_chunk_ids: set[str] = set()

    for raw_chunk in same_match_chunks:
        raw_metadata = raw_chunk.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        level = str(
            raw_chunk.get("level")
            or raw_metadata.get("level")
            or ""
        )

        if level not in {"1", "2", "3"}:
            continue

        if level == "3":
            player_name = (
                raw_chunk.get("player_name")
                or raw_metadata.get("player_name")
            )
            if not resolved_player or player_name != resolved_player:
                continue

        chunk_id = raw_chunk.get("chunk_id")
        if chunk_id:
            eligible_chunk_ids.add(chunk_id)

    expanded = []
    for item in results:
        promoted = dict(item)
        if (
            promoted.get("chunk_id") in eligible_chunk_ids
            and promoted.get("source") != "match_pair_boost"
        ):
            promoted["source"] = "match_fixture_expansion"
        expanded.append(promoted)

    seen_chunk_ids = {item.get("chunk_id") for item in expanded}

    for raw_chunk in same_match_chunks:
        raw_metadata = raw_chunk.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        level = str(
            raw_chunk.get("level")
            or raw_metadata.get("level")
            or ""
        )

        if level not in {"1", "2", "3"}:
            continue

        if level == "3":
            player_name = (
                raw_chunk.get("player_name")
                or raw_metadata.get("player_name")
            )
            if not resolved_player or player_name != resolved_player:
                continue

        chunk_id = raw_chunk.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue

        document_id = (
            raw_chunk.get("document_id")
            or raw_metadata.get("document_id")
        )

        metadata = dict(raw_metadata)
        metadata.setdefault("document_id", document_id)

        for field in (
            "level",
            "match_id",
            "player_name",
            "team_name",
            "home_team",
            "away_team",
            "match_date",
        ):
            if metadata.get(field) is None and raw_chunk.get(field) is not None:
                metadata[field] = raw_chunk.get(field)

        expanded.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": raw_chunk.get("text", ""),
            "metadata": metadata,
            "score": 0.0,
            "rrf_score": 0.0,
            "source": "match_fixture_expansion",
        })
        seen_chunk_ids.add(chunk_id)

    return expanded

def _expand_query_entity_siblings(
    query: str, results: list[dict], artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Add sibling chunks for candidate documents whose entity appears in query.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py. This safeguard must never
    silently reload the legacy WC2022 chunks for another dataset's query.
    """
    query_lower = query.casefold()
    entity_fields = ("team_name", "player_name", "home_team", "away_team")
    target_document_ids: set[str] = set()

    raw_chunks = _get_chunks(artifact_paths)
    raw_chunks_by_id = {
        chunk.get("chunk_id"): chunk
        for chunk in raw_chunks
        if chunk.get("chunk_id")
    }

    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        raw_chunk = raw_chunks_by_id.get(result.get("chunk_id"), {})
        raw_metadata = raw_chunk.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        entity_values = [
            metadata.get(field)
            or result.get(field)
            or raw_metadata.get(field)
            or raw_chunk.get(field)
            for field in entity_fields
        ]
        entity_matches = any(
            str(value).strip().casefold() in query_lower
            for value in entity_values
            if value
        )

        if entity_matches:
            document_id = (
                result.get("document_id")
                or metadata.get("document_id")
                or raw_chunk.get("document_id")
                or raw_metadata.get("document_id")
            )
            if document_id:
                target_document_ids.add(str(document_id))

    if not target_document_ids:
        return results

    expanded = list(results)
    seen_chunk_ids = {item.get("chunk_id") for item in expanded}

    for raw_chunk in raw_chunks:
        raw_metadata = raw_chunk.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        document_id = (
            raw_chunk.get("document_id")
            or raw_metadata.get("document_id")
        )
        chunk_id = raw_chunk.get("chunk_id")

        if document_id not in target_document_ids or chunk_id in seen_chunk_ids:
            continue

        metadata = dict(raw_metadata)
        metadata.setdefault("document_id", document_id)
        for field in ("level", "match_id", "player_name", "team_name", "home_team", "away_team"):
            if metadata.get(field) is None and raw_chunk.get(field) is not None:
                metadata[field] = raw_chunk.get(field)

        expanded.append({
            "chunk_id": chunk_id,
            "text": raw_chunk.get("text", ""),
            "metadata": metadata,
            "score": 0.0,
            "rrf_score": 0.0,
            "source": "sibling_expansion",
        })
        seen_chunk_ids.add(chunk_id)

    return expanded
