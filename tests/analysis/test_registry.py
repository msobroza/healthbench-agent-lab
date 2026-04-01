"""Tests for healthbench_agent.analysis.registry.

Covers register_analysis, run_one, run_category, run_all, and AnalysisEntry
using the ZOMBIES heuristic. Each test patches _analysis_registry to avoid polluting
the module-level state used by exploration.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from healthbench_agent.analysis.registry import (
    AnalysisEntry,
    _analysis_registry,
    register_analysis,
    run_all,
    run_category,
    run_one,
)
from healthbench_agent.domain import HealthBenchDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(monkeypatch):
    """Replace _analysis_registry with an empty dict for the duration of the test."""
    fresh: dict = {}
    monkeypatch.setattr("healthbench_agent.analysis.registry._analysis_registry", fresh)
    return fresh


@pytest.fixture
def main_dataset():
    return HealthBenchDataset(subset="main", samples=[], source_path="/tmp/main.jsonl")


@pytest.fixture
def hard_dataset():
    return HealthBenchDataset(subset="hard", samples=[], source_path="/tmp/hard.jsonl")


def _stub_fn(datasets, output_dir, save=False) -> dict[str, Any]:
    return {d.subset: {"called": True} for d in datasets}


# ---------------------------------------------------------------------------
# AnalysisEntry
# ---------------------------------------------------------------------------


def test_analysis_entry_repr_contains_name_and_category():
    entry = AnalysisEntry(
        name="my_analysis",
        category="exploration",
        datasets=["main"],
        function=_stub_fn,
    )
    r = repr(entry)
    assert "my_analysis" in r
    assert "exploration" in r


def test_analysis_entry_stores_all_fields():
    entry = AnalysisEntry(
        name="test",
        category="insights",
        datasets=["main", "hard"],
        function=_stub_fn,
    )
    assert entry.name == "test"
    assert entry.category == "insights"
    assert entry.datasets == ["main", "hard"]
    assert entry.function is _stub_fn


# ---------------------------------------------------------------------------
# register_analysis
# ---------------------------------------------------------------------------


def test_register_analysis_stores_entry_in_registry(isolated_registry):
    @register_analysis(name="count_samples", category="exploration", datasets=["main"])
    def count_samples(datasets, output_dir, save=False):
        return {}

    assert "count_samples" in isolated_registry


def test_register_analysis_returns_function_unchanged(isolated_registry):
    def my_fn(datasets, output_dir, save=False):
        return {}

    result = register_analysis(name="my_fn", category="exploration", datasets=["main"])(my_fn)
    assert result is my_fn


def test_register_analysis_overwrites_existing_entry(isolated_registry):
    def fn_v1(datasets, output_dir, save=False):
        return {"version": 1}

    def fn_v2(datasets, output_dir, save=False):
        return {"version": 2}

    register_analysis(name="dup", category="exploration", datasets=["main"])(fn_v1)
    register_analysis(name="dup", category="exploration", datasets=["main"])(fn_v2)

    assert isolated_registry["dup"].function is fn_v2


def test_register_analysis_stores_datasets_list(isolated_registry):
    @register_analysis(name="multi_subset", category="exploration", datasets=["main", "hard"])
    def multi_fn(datasets, output_dir, save=False):
        return {}

    assert isolated_registry["multi_subset"].datasets == ["main", "hard"]


# ---------------------------------------------------------------------------
# run_one
# ---------------------------------------------------------------------------


def test_run_one_raises_value_error_for_unknown_name(isolated_registry, tmp_path):
    with pytest.raises(ValueError, match="not registered"):
        run_one("nonexistent", [], tmp_path)


def test_run_one_returns_dict_keyed_by_name(isolated_registry, tmp_path, main_dataset):
    @register_analysis(name="simple", category="exploration", datasets=["main"])
    def simple(datasets, output_dir, save=False):
        return {"main": {"ok": True}}

    result = run_one("simple", [main_dataset], tmp_path)
    assert "simple" in result


def test_run_one_filters_datasets_to_declared_subsets(isolated_registry, tmp_path, main_dataset, hard_dataset):
    received: list = []

    @register_analysis(name="main_only", category="exploration", datasets=["main"])
    def main_only(datasets, output_dir, save=False):
        received.extend(datasets)
        return {}

    run_one("main_only", [main_dataset, hard_dataset], tmp_path)
    assert all(d.subset == "main" for d in received)


def test_run_one_passes_save_flag_to_function(isolated_registry, tmp_path, main_dataset):
    received_save: list = []

    @register_analysis(name="save_check", category="exploration", datasets=["main"])
    def save_check(datasets, output_dir, save=False):
        received_save.append(save)
        return {}

    run_one("save_check", [main_dataset], tmp_path, save=True)
    assert received_save == [True]


# ---------------------------------------------------------------------------
# run_category
# ---------------------------------------------------------------------------


def test_run_category_returns_only_matching_category(isolated_registry, tmp_path, main_dataset):
    @register_analysis(name="exp_fn", category="exploration", datasets=["main"])
    def exp_fn(datasets, output_dir, save=False):
        return {}

    @register_analysis(name="viz_fn", category="visualization", datasets=["main"])
    def viz_fn(datasets, output_dir, save=False):
        return {}

    result = run_category("exploration", [main_dataset], tmp_path)
    assert "exp_fn" in result
    assert "viz_fn" not in result


def test_run_category_empty_registry_returns_empty_dict(isolated_registry, tmp_path):
    result = run_category("exploration", [], tmp_path)
    assert result == {}


def test_run_category_unknown_category_returns_empty_dict(isolated_registry, tmp_path):
    @register_analysis(name="some_fn", category="exploration", datasets=["main"])
    def some_fn(datasets, output_dir, save=False):
        return {}

    result = run_category("visualization", [], tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def test_run_all_returns_all_registered_analyses(isolated_registry, tmp_path, main_dataset):
    @register_analysis(name="fn_a", category="exploration", datasets=["main"])
    def fn_a(datasets, output_dir, save=False):
        return {}

    @register_analysis(name="fn_b", category="insights", datasets=["main"])
    def fn_b(datasets, output_dir, save=False):
        return {}

    result = run_all([main_dataset], tmp_path)
    assert "fn_a" in result
    assert "fn_b" in result


def test_run_all_empty_registry_returns_empty_dict(isolated_registry, tmp_path):
    result = run_all([], tmp_path)
    assert result == {}


def test_run_all_filters_subsets_per_entry(isolated_registry, tmp_path, main_dataset, hard_dataset):
    received: dict[str, list] = {}

    @register_analysis(name="main_fn", category="exploration", datasets=["main"])
    def main_fn(datasets, output_dir, save=False):
        received["main_fn"] = [d.subset for d in datasets]
        return {}

    @register_analysis(name="hard_fn", category="exploration", datasets=["hard"])
    def hard_fn(datasets, output_dir, save=False):
        received["hard_fn"] = [d.subset for d in datasets]
        return {}

    run_all([main_dataset, hard_dataset], tmp_path)
    assert received["main_fn"] == ["main"]
    assert received["hard_fn"] == ["hard"]
