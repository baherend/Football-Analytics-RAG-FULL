from pathlib import Path

from src import cache


def test_data_hash_changes_when_same_file_changes(tmp_path):
    data_path = tmp_path / "match_facts.json"
    data_path.write_text('{"value": 1}', encoding="utf-8")

    cache.clear_all_caches()
    first_hash = cache._compute_data_hash(data_path)

    data_path.write_text('{"value": 999999}', encoding="utf-8")
    second_hash = cache._compute_data_hash(data_path)

    assert second_hash != first_hash


def test_data_hash_is_path_aware_for_different_files(tmp_path):
    path_a = tmp_path / "a" / "match_facts.json"
    path_b = tmp_path / "b" / "match_facts.json"
    path_a.parent.mkdir()
    path_b.parent.mkdir()

    content = '{"value": 1}'
    path_a.write_text(content, encoding="utf-8")
    path_b.write_text(content, encoding="utf-8")

    hash_a = cache._compute_data_hash(path_a)
    hash_b = cache._compute_data_hash(path_b)

    assert hash_a != hash_b


def test_missing_data_hash_is_path_aware(tmp_path):
    missing_a = tmp_path / "a" / "match_facts.json"
    missing_b = tmp_path / "b" / "match_facts.json"

    hash_a = cache._compute_data_hash(missing_a)
    hash_b = cache._compute_data_hash(missing_b)

    assert hash_a != hash_b


def test_load_data_cache_is_path_aware(tmp_path):
    import os
    import src.query.resolver as resolver

    path_a = tmp_path / "a" / "match_facts.json"
    path_b = tmp_path / "b" / "match_facts.json"
    path_a.parent.mkdir()
    path_b.parent.mkdir()

    path_a.write_text('{"dataset": "A"}', encoding="utf-8")
    path_b.write_text('{"dataset": "B"}', encoding="utf-8")

    same_time = 1_700_000_000
    os.utime(path_a, (same_time, same_time))
    os.utime(path_b, (same_time, same_time))

    resolver._data_cache_by_path.clear()

    data_a = resolver._load_data(path_a)
    data_b = resolver._load_data(path_b)

    assert data_a["dataset"] == "A"
    assert data_b["dataset"] == "B"


def test_explicit_in_memory_datasets_do_not_share_structured_cache():
    from src.query.query_schema import StructuredQuery
    from src.query.resolver import resolve

    cache.clear_all_caches()

    query = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Test Player",
    )

    def dataset(goals):
        return {
            "player_match_facts": [{
                "player_name": "Test Player",
                "goals": goals,
            }],
            "match_facts": [],
            "team_match_facts": [],
        }

    result_a = resolve(query, data=dataset(1))
    result_b = resolve(query, data=dataset(9))

    assert result_a.aggregated_value == 1
    assert result_b.aggregated_value == 9


def test_load_data_reloads_when_same_path_changes(tmp_path):
    import src.query.resolver as resolver

    data_path = tmp_path / "match_facts.json"
    data_path.write_text('{"dataset": "A"}', encoding="utf-8")

    resolver._data_cache_by_path.clear()
    first = resolver._load_data(data_path)

    data_path.write_text('{"dataset": "BBBB"}', encoding="utf-8")
    second = resolver._load_data(data_path)

    assert first["dataset"] == "A"
    assert second["dataset"] == "BBBB"


def test_file_backed_resolve_still_uses_structured_cache(monkeypatch):
    import src.query.resolver as resolver
    from src.query.query_schema import StructuredQuery

    cache.clear_all_caches()

    data = {
        "player_match_facts": [{"player_name": "Test Player", "goals": 4}],
        "match_facts": [],
        "team_match_facts": [],
    }
    load_calls = {"count": 0}

    def fake_load_data(path=None):
        load_calls["count"] += 1
        return data

    monkeypatch.setattr(resolver, "_load_data", fake_load_data)
    monkeypatch.setattr(cache, "_compute_data_hash", lambda data_path=Path("output/match_facts.json"): "stable-test-hash")

    query = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Test Player",
    )

    first = resolver.resolve(query)
    second = resolver.resolve(query)

    assert first.aggregated_value == 4
    assert second.aggregated_value == 4
    assert load_calls["count"] == 1


def test_structured_cache_key_includes_full_stage_taxonomy_semantics(monkeypatch):
    import src.query.resolver as resolver
    from src.query.query_schema import StructuredQuery
    from src.stage_taxonomy import StageTaxonomy

    cache.clear_all_caches()

    data = {
        "player_match_facts": [{"player_name": "Test Player", "goals": 1}],
        "match_facts": [],
        "team_match_facts": [],
    }
    seen_keys = []

    monkeypatch.setattr(resolver, "_load_data", lambda path=None: data)
    monkeypatch.setattr(
        cache,
        "get_cached_structured_result",
        lambda query_key, data_path=None: seen_keys.append(query_key) or None,
    )
    monkeypatch.setattr(
        cache, "set_cached_structured_result",
        lambda query_key, result, data_path=None: None,
    )

    query = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Test Player",
    )

    taxonomy_a = StageTaxonomy(
        stages=frozenset({"League Stage", "Final"}),
        knockout_stages=frozenset({"Final"}),
        group_stages=frozenset({"League Stage"}),
        stage_order=("League Stage", "Final"),
    )
    taxonomy_b = StageTaxonomy(
        stages=frozenset({"League Stage", "Final"}),
        knockout_stages=frozenset(),
        group_stages=frozenset(),
        stage_order=("Final", "League Stage"),
    )

    resolver.resolve(query, stage_taxonomy=taxonomy_a)
    resolver.resolve(query, stage_taxonomy=taxonomy_b)

    assert len(seen_keys) == 2
    assert seen_keys[0] != seen_keys[1]


def test_embedding_model_cache_distinguishes_different_model_names(monkeypatch):
    """Cache behavior: MiniLM and MPNet model instances must be
    distinguished correctly -- requesting two different model names must
    never return the same cached instance, and requesting the same name
    twice must return the exact same instance (the existing cache-hit
    contract, unaffected by adding a second model)."""
    from src.embedding_config import MINILM, MPNET_MULTILINGUAL

    class _FakeModel:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda name: _FakeModel(name),
    )
    cache.clear_all_caches()

    minilm_instance = cache.get_embedding_model(MINILM.hf_name)
    mpnet_instance = cache.get_embedding_model(MPNET_MULTILINGUAL.hf_name)
    minilm_instance_again = cache.get_embedding_model(MINILM.hf_name)

    assert minilm_instance is not mpnet_instance
    assert minilm_instance.name == MINILM.hf_name
    assert mpnet_instance.name == MPNET_MULTILINGUAL.hf_name
    # Cache hit: re-requesting the same model name returns the same instance.
    assert minilm_instance_again is minilm_instance


def test_embedding_model_cache_default_resolves_minilm(monkeypatch):
    """Backward compatibility: get_embedding_model()'s zero-argument
    default must still resolve MiniLM after the hardcoded literal was
    replaced with a reference to the central registry."""
    from src.embedding_config import MINILM

    class _FakeModel:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda name: _FakeModel(name),
    )
    cache.clear_all_caches()

    instance = cache.get_embedding_model()

    assert instance.name == MINILM.hf_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_resolve_from_text_preserves_file_backed_structured_cache(monkeypatch):
    import src.query.resolver as resolver

    cache.clear_all_caches()

    data = {
        "player_match_facts": [{"player_name": "Test Player", "goals": 4}],
        "match_facts": [],
        "team_match_facts": [],
    }
    cache_calls = {"get": 0, "set": 0}

    monkeypatch.setattr(resolver, "_load_data", lambda path=None: data)

    def fake_get(query_key, data_path=None):
        cache_calls["get"] += 1
        return None

    def fake_set(query_key, result, data_path=None):
        cache_calls["set"] += 1

    monkeypatch.setattr(cache, "get_cached_structured_result", fake_get)
    monkeypatch.setattr(cache, "set_cached_structured_result", fake_set)

    result = resolver.resolve_from_text("How many goals did Test Player score?")

    assert result.aggregated_value == 4
    assert cache_calls["get"] == 1
    assert cache_calls["set"] == 1
