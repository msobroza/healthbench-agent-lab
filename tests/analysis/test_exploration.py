"""Tests for healthbench_agent.analysis.exploration.

Verifies the analysis convention: save=False returns correct dict keys per
subset without writing files; save=True writes CSV artifacts. Uses a minimal
in-memory HealthBenchDataset to avoid network and disk I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from healthbench_agent.analysis.exploration import (
    compute_prompt_structure,
    compute_rubric_size,
    compute_rubric_tag_frequency,
    compute_sample_counts,
    compute_score_range_stats,
)
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    prompt_id: str = "p001",
    rubric_points: list[float] | None = None,
    tags: list[str] | None = None,
    ideal_completions_data: dict | None = None,
) -> HealthBenchSample:
    if rubric_points is None:
        rubric_points = [1.0, -0.5]
    if tags is None:
        tags = ["theme: cardiology", "axis: accuracy"]
    rubrics = [
        {"criterion": f"criterion_{i}", "points": p, "tags": tags}
        for i, p in enumerate(rubric_points)
    ]
    return HealthBenchSample.from_dict(
        {
            "prompt_id": prompt_id,
            "prompt": [{"role": "user", "content": "I have chest pain."}],
            "rubrics": rubrics,
            "example_tags": tags,
            "ideal_completions_data": ideal_completions_data,
        }
    )


@pytest.fixture
def main_dataset():
    samples = [_make_sample("p001"), _make_sample("p002")]
    return HealthBenchDataset(subset="main", samples=samples, source_path="/tmp/main.jsonl")


@pytest.fixture
def hard_dataset():
    samples = [_make_sample("p003", rubric_points=[0.5])]
    return HealthBenchDataset(subset="hard", samples=samples, source_path="/tmp/hard.jsonl")


@pytest.fixture
def empty_dataset():
    return HealthBenchDataset(subset="consensus", samples=[], source_path="/tmp/empty.jsonl")


# ---------------------------------------------------------------------------
# compute_sample_counts
# ---------------------------------------------------------------------------


def test_sample_counts_save_false_returns_dict_keyed_by_subset(main_dataset, tmp_path):
    result = compute_sample_counts([main_dataset], tmp_path, save=False)
    assert "main" in result


def test_sample_counts_save_false_writes_no_files(main_dataset, tmp_path):
    compute_sample_counts([main_dataset], tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_sample_counts_total_samples_matches_dataset_length(main_dataset, tmp_path):
    result = compute_sample_counts([main_dataset], tmp_path, save=False)
    assert result["main"]["total_samples"] == len(main_dataset.samples)


def test_sample_counts_empty_dataset_returns_empty_subset_dict(empty_dataset, tmp_path):
    result = compute_sample_counts([empty_dataset], tmp_path, save=False)
    assert result["consensus"] == {}


def test_sample_counts_save_true_writes_csv(main_dataset, tmp_path):
    compute_sample_counts([main_dataset], tmp_path, save=True)
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) >= 1


def test_sample_counts_multiple_datasets_keyed_separately(main_dataset, hard_dataset, tmp_path):
    result = compute_sample_counts([main_dataset, hard_dataset], tmp_path, save=False)
    assert "main" in result
    assert "hard" in result


# ---------------------------------------------------------------------------
# compute_rubric_size
# ---------------------------------------------------------------------------


def test_rubric_size_save_false_returns_dict_keyed_by_subset(main_dataset, tmp_path):
    result = compute_rubric_size([main_dataset], tmp_path, save=False)
    assert "main" in result


def test_rubric_size_save_false_writes_no_files(main_dataset, tmp_path):
    compute_rubric_size([main_dataset], tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_rubric_size_contains_expected_keys(main_dataset, tmp_path):
    result = compute_rubric_size([main_dataset], tmp_path, save=False)
    stats = result["main"]
    assert "rubric_size_stats" in stats


def test_rubric_size_empty_dataset_returns_empty_subset_dict(empty_dataset, tmp_path):
    result = compute_rubric_size([empty_dataset], tmp_path, save=False)
    assert result["consensus"] == {}


# ---------------------------------------------------------------------------
# compute_rubric_tag_frequency
# ---------------------------------------------------------------------------


def test_rubric_tag_frequency_save_false_returns_dict_keyed_by_subset(main_dataset, tmp_path):
    result = compute_rubric_tag_frequency([main_dataset], tmp_path, save=False)
    assert "main" in result


def test_rubric_tag_frequency_save_false_writes_no_files(main_dataset, tmp_path):
    compute_rubric_tag_frequency([main_dataset], tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_rubric_tag_frequency_empty_dataset_returns_empty_subset_dict(empty_dataset, tmp_path):
    result = compute_rubric_tag_frequency([empty_dataset], tmp_path, save=False)
    assert result["consensus"] == {}


# ---------------------------------------------------------------------------
# compute_prompt_structure
# ---------------------------------------------------------------------------


def test_prompt_structure_save_false_returns_dict_keyed_by_subset(main_dataset, tmp_path):
    result = compute_prompt_structure([main_dataset], tmp_path, save=False)
    assert "main" in result


def test_prompt_structure_save_false_writes_no_files(main_dataset, tmp_path):
    compute_prompt_structure([main_dataset], tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_prompt_structure_empty_dataset_returns_empty_subset_dict(empty_dataset, tmp_path):
    result = compute_prompt_structure([empty_dataset], tmp_path, save=False)
    assert result["consensus"] == {}


# ---------------------------------------------------------------------------
# compute_score_range_stats
# ---------------------------------------------------------------------------


def test_score_range_stats_save_false_returns_dict_keyed_by_subset(main_dataset, tmp_path):
    result = compute_score_range_stats([main_dataset], tmp_path, save=False)
    assert "main" in result


def test_score_range_stats_save_false_writes_no_files(main_dataset, tmp_path):
    compute_score_range_stats([main_dataset], tmp_path, save=False)
    assert list(tmp_path.iterdir()) == []


def test_score_range_stats_empty_dataset_returns_empty_subset_dict(empty_dataset, tmp_path):
    result = compute_score_range_stats([empty_dataset], tmp_path, save=False)
    assert result["consensus"] == {}


# ---------------------------------------------------------------------------
# Convention: all exploration functions return empty dict for empty input list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        compute_sample_counts,
        compute_rubric_size,
        compute_rubric_tag_frequency,
        compute_prompt_structure,
        compute_score_range_stats,
    ],
)
def test_exploration_fn_with_no_datasets_returns_empty_dict(fn, tmp_path):
    result = fn([], tmp_path, save=False)
    assert result == {}
