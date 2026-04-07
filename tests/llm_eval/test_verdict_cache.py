"""Tests for VerdictCache and CachedJudgeGrader."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.llm_eval.verdict_cache import VerdictCache


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
