"""Representative Arabic query-understanding coverage matrix."""

from __future__ import annotations

import pytest

from src.query.parsing import parse_structured_query
from src.query.router import route_query


@pytest.mark.parametrize(
    ("question", "entity"),
    [
        ("من هو هداف كأس العالم 2022 وكم هدفًا سجل؟", "player"),
        ("من سجل أكبر عدد من الأهداف؟", "player"),
        ("ما الفريق الأكثر تسجيلًا للأهداف؟", "team"),
    ],
)
def test_arabic_goal_superlatives_are_structured(question: str, entity: str):
    route = route_query(question)
    parsed = parse_structured_query(question)

    assert route.path == "structured"
    assert parsed is not None
    assert parsed.entity == entity
    assert parsed.metric == "goals"
    assert parsed.aggregation == "max"


@pytest.mark.parametrize(
    "question",
    [
        "كيف لعبت Argentina في المباراة النهائية؟",
        "كيف لعب Leicester City خلال الموسم؟",
    ],
)
def test_arabic_qualitative_questions_remain_semantic(question: str):
    route = route_query(question)

    assert route.path == "semantic"
    assert parse_structured_query(question) is None


@pytest.mark.xfail(
    strict=True,
    reason="Arabic named-player numeric parsing is not implemented",
)
@pytest.mark.parametrize(
    "question",
    [
        "كم هدفًا سجل Lionel Messi؟",
        "كم هدفًا سجل Jamie Vardy؟",
    ],
)
def test_arabic_named_player_numeric_questions_are_structured(question: str):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.intent == "numeric"
    assert route.structured_query.entity == "player"
    assert route.structured_query.metric == "goals"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Arabic non-goal superlatives lack complete metric/aggregation "
        "normalization and parser patterns"
    ),
)
@pytest.mark.parametrize(
    ("question", "entity", "metric"),
    [
        ("من لديه أكبر عدد من التمريرات الحاسمة؟", "player", "assists"),
        ("ما الفريق صاحب أعلى xG؟", "team", "xg"),
    ],
)
def test_arabic_non_goal_superlatives_are_structured(
    question: str,
    entity: str,
    metric: str,
):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.intent == "superlative"
    assert route.structured_query.entity == entity
    assert route.structured_query.metric == metric
    assert route.structured_query.aggregation == "max"


@pytest.mark.xfail(
    strict=True,
    reason="Arabic compositional dependency parsing is not implemented",
)
def test_arabic_compositional_dependency_routes_hybrid():
    route = route_query("كيف لعب الفريق الأكثر تسجيلًا للأهداف؟")

    assert route.path == "hybrid"
    assert route.dependency_query is not None
    assert route.dependency_query.entity == "team"
    assert route.dependency_query.metric == "goals"


@pytest.mark.xfail(
    strict=True,
    reason="Arabic comparison routing is not implemented",
)
@pytest.mark.parametrize(
    "question",
    [
        "قارن بين Lionel Messi و Kylian Mbappe في الأهداف.",
        "قارن بين Harry Kane و Jamie Vardy في الأهداف.",
    ],
)
def test_arabic_comparisons_route_hybrid(question: str):
    assert route_query(question).path == "hybrid"
