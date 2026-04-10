"""Tests for VerdictCache and CachedJudgeGrader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.cache import CachedJudgeGrader, VerdictCache


@pytest.fixture
def cache(tmp_path: Path) -> VerdictCache:
    return VerdictCache(root=tmp_path / "cache", enabled=True)


def _conv() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_disabled_cache_get_always_returns_none(tmp_path: Path):
    c = VerdictCache(root=tmp_path / "x", enabled=False)
    key = c.make_key("m", "sha", _conv(), "rt", 1)
    assert c.get(key) is None


def test_disabled_cache_put_writes_nothing(tmp_path: Path):
    root = tmp_path / "x"
    c = VerdictCache(root=root, enabled=False)
    key = c.make_key("m", "sha", _conv(), "rt", 1)
    c.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert not root.exists() or not any(root.iterdir())


def test_make_key_is_deterministic_across_instances(tmp_path: Path):
    c1 = VerdictCache(root=tmp_path / "a")
    c2 = VerdictCache(root=tmp_path / "b")
    k1 = c1.make_key("m", "sha", _conv(), "rt", 1)
    k2 = c2.make_key("m", "sha", _conv(), "rt", 1)
    assert k1 == k2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"judge_model": "m2"},
        {"judge_prompt_sha": "sha2"},
        {"conversation": [{"role": "user", "content": "different"}]},
        {"rubric_text": "different"},
        {"k_index": 2},
    ],
)
def test_make_key_changes_when_any_input_changes(cache: VerdictCache, kwargs):
    base = dict(
        judge_model="m",
        judge_prompt_sha="sha",
        conversation=_conv(),
        rubric_text="rt",
        k_index=1,
    )
    base_key = cache.make_key(**base)
    other_key = cache.make_key(**{**base, **kwargs})
    assert base_key != other_key


def test_put_then_get_round_trips(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    verdict = CriterionVerdict(criterion="rt", criteria_met=True, explanation="ok")
    cache.put(key, verdict)
    assert cache.get(key) == verdict


def test_get_missing_key_returns_none(cache: VerdictCache):
    assert cache.get("nonexistent") is None


def test_corrupt_cache_file_returns_none_and_logs(cache: VerdictCache, caplog):
    """A corrupt JSON entry must be treated as a miss and logged as a warning."""
    import logging

    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    path = cache._path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {")
    with caplog.at_level(logging.WARNING, logger="healthbench_agent.llm_eval.cache.store"):
        assert cache.get(key) is None
    assert any("Corrupt cache entry" in record.message for record in caplog.records)


def test_files_are_sharded_by_first_two_hex_chars(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    shard = key[:2]
    assert (cache.root / shard).is_dir()


def test_clear_removes_cache_root(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    cache.clear()
    assert not cache.root.exists() or not any(cache.root.iterdir())


def test_stats_reports_hits_misses_and_size(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    assert cache.get(key) is None  # miss
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert cache.get(key) is not None  # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size_bytes"] > 0


def test_two_instances_share_cache_files(tmp_path: Path):
    root = tmp_path / "shared"
    a = VerdictCache(root=root)
    b = VerdictCache(root=root)
    key = a.make_key("m", "sha", _conv(), "rt", 1)
    a.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert b.get(key) is not None


def _make_inner(verdicts: list[CriterionVerdict]) -> MagicMock:
    inner = MagicMock(spec=["grade"])
    inner.grade.return_value = verdicts
    return inner


def test_cached_proxy_first_call_delegates(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    proxy = CachedJudgeGrader(inner, cache, "openai/gpt-4.1@1.0", "sha1", k_index=1)
    out = proxy.grade(_conv(), [RubricItem(criterion="r1", points=1.0)])
    assert out == [verdict]
    assert inner.grade.call_count == 1


def test_cached_proxy_second_call_hits_cache(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    proxy = CachedJudgeGrader(inner, cache, "openai/gpt-4.1@1.0", "sha1", k_index=1)
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    proxy.grade(_conv(), rubrics)
    proxy.grade(_conv(), rubrics)
    assert inner.grade.call_count == 1


def test_cached_proxy_distinct_k_index_yields_separate_entries(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    CachedJudgeGrader(inner, cache, "m", "s", k_index=1).grade(_conv(), rubrics)
    CachedJudgeGrader(inner, cache, "m", "s", k_index=2).grade(_conv(), rubrics)
    assert inner.grade.call_count == 2  # both missed


def test_cached_proxy_disabled_cache_always_delegates(tmp_path: Path):
    disabled = VerdictCache(root=tmp_path / "off", enabled=False)
    inner = _make_inner([CriterionVerdict(criterion="r1", criteria_met=True, explanation="")])
    proxy = CachedJudgeGrader(inner, disabled, "m", "s", k_index=1)
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    proxy.grade(_conv(), rubrics)
    proxy.grade(_conv(), rubrics)
    assert inner.grade.call_count == 2


def test_cached_proxy_batches_misses_into_one_inner_call(cache: VerdictCache):
    v1 = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    v2 = CriterionVerdict(criterion="r2", criteria_met=False, explanation="")
    inner = _make_inner([v1, v2])
    proxy = CachedJudgeGrader(inner, cache, "m", "s", k_index=1)
    out = proxy.grade(
        _conv(),
        [RubricItem(criterion="r1", points=1.0), RubricItem(criterion="r2", points=1.0)],
    )
    assert out == [v1, v2]
    assert inner.grade.call_count == 1
    args, _ = inner.grade.call_args
    assert len(args[1]) == 2  # both rubrics passed in one delegated call


def test_cached_proxy_preserves_input_rubric_order(cache: VerdictCache):
    v1 = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    v2 = CriterionVerdict(criterion="r2", criteria_met=False, explanation="")
    # Pre-populate r1, force a miss on r2. Rubric key format must match
    # the composite used by CachedJudgeGrader.grade().
    cached_key = cache.make_key("m", "s", _conv(), "r1|points=1.0|tags=()", 1)
    cache.put(cached_key, v1)
    inner = _make_inner([v2])  # only the missed one
    proxy = CachedJudgeGrader(inner, cache, "m", "s", k_index=1)
    out = proxy.grade(
        _conv(),
        [RubricItem(criterion="r1", points=1.0), RubricItem(criterion="r2", points=1.0)],
    )
    assert out[0] == v1
    assert out[1] == v2


def test_cache_distinguishes_rubrics_with_identical_criterion_text_but_different_points(
    cache: VerdictCache,
):
    """Two rubrics sharing criterion text but differing on points must not collide.

    Regression test for a cache-key collision where the key was built
    from ``item.criterion`` alone. Now the proxy falls back to
    ``criterion|points=...|tags=...`` when ``criterion_id`` is missing,
    so identical text with different points keeps distinct cache entries.
    """
    v_positive = CriterionVerdict(criterion="foo", criteria_met=True, explanation="")
    v_penalty = CriterionVerdict(criterion="foo", criteria_met=True, explanation="")
    inner = _make_inner([v_positive, v_penalty])
    proxy = CachedJudgeGrader(inner, cache, "m", "s", k_index=1)
    rubrics = [
        RubricItem(criterion="foo", points=1.0),
        RubricItem(criterion="foo", points=-1.0),
    ]
    proxy.grade(_conv(), rubrics)
    # Both rubrics missed the cache on the first pass; had the old
    # collision-prone key been in use the second rubric would have
    # re-used the first entry and reported only one miss.
    stats = cache.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0
