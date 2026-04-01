"""Tests for healthbench_agent.analysis.exploration.

Verifies the analysis convention: save=False returns correct dict keys per
subset without writing files; save=True writes CSV artifacts. Uses a minimal
in-memory HealthBenchDataset to avoid network and disk I/O.
"""

from __future__ import annotations

import pytest

from healthbench_agent.analysis.exploration import (
    compute_prompt_structure,
    compute_rubric_size,
    compute_rubric_tag_frequency,
    compute_sample_counts,
    compute_score_range_stats,
)
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample

# consensus_dataset, multi_turn_sample injected from tests/conftest.py


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
        tags = ["theme:cardiology", "axis:accuracy"]
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


# ---------------------------------------------------------------------------
# compute_sample_counts — ideal_completions_data detection (conftest fixtures)
#
# consensus_dataset : 1 sample with ideal_completions_data populated
# main_dataset      : 2 samples without ideal_completions_data (local fixture)
# ---------------------------------------------------------------------------


def test_sample_counts_detects_ideal_completions_present(consensus_dataset, tmp_path):
    result = compute_sample_counts([consensus_dataset], tmp_path, save=False)
    assert result["consensus"]["with_ideal_completions_count"] == 1
    assert result["consensus"]["with_ideal_completions_fraction"] == pytest.approx(1.0)


def test_sample_counts_detects_ideal_completions_absent(main_dataset, tmp_path):
    result = compute_sample_counts([main_dataset], tmp_path, save=False)
    n = len(main_dataset.samples)
    assert result["main"]["without_ideal_completions_count"] == n
    assert result["main"]["without_ideal_completions_fraction"] == pytest.approx(1.0)


def test_sample_counts_physician_group_distribution_populated(consensus_dataset, tmp_path):
    result = compute_sample_counts([consensus_dataset], tmp_path, save=False)
    distribution = result["consensus"]["physician_group_distribution"]
    assert len(distribution) > 0


def test_sample_counts_unique_prompt_ids_equals_sample_count(consensus_dataset, tmp_path):
    result = compute_sample_counts([consensus_dataset], tmp_path, save=False)
    total = result["consensus"]["total_samples"]
    unique = result["consensus"]["unique_prompt_ids"]
    assert unique == total


# ---------------------------------------------------------------------------
# compute_prompt_structure — multi-turn detection (conftest fixtures)
#
# multi_turn_sample: 3-turn prompt (user → assistant → user)
# ---------------------------------------------------------------------------


def test_prompt_structure_multi_turn_sample_counted_as_multi_turn(
    multi_turn_sample, tmp_path
):
    dataset = HealthBenchDataset(
        subset="main",
        samples=[multi_turn_sample],
        source_path="/tmp/test.jsonl",
    )
    result = compute_prompt_structure([dataset], tmp_path, save=False)
    assert result["main"]["multi_turn_count"] == 1
    assert result["main"]["single_turn_count"] == 0


def test_prompt_structure_mixed_dataset_distinguishes_turn_types(
    main_dataset, multi_turn_sample, tmp_path
):
    # local main_dataset has 2 single-turn samples; adding multi_turn_sample gives 3 total
    mixed = HealthBenchDataset(
        subset="main",
        samples=main_dataset.samples + [multi_turn_sample],
        source_path="/tmp/mixed.jsonl",
    )
    result = compute_prompt_structure([mixed], tmp_path, save=False)
    assert result["main"]["single_turn_count"] == 2
    assert result["main"]["multi_turn_count"] == 1


def test_prompt_structure_multi_turn_num_turns_stats_min_is_three(
    multi_turn_sample, tmp_path
):
    dataset = HealthBenchDataset(
        subset="hard",
        samples=[multi_turn_sample],
        source_path="/tmp/hard.jsonl",
    )
    result = compute_prompt_structure([dataset], tmp_path, save=False)
    assert result["hard"]["num_turns_stats"]["min"] == 3


# ---------------------------------------------------------------------------
# compute_rubric_size — realistic rubric counts (conftest fixtures)
#
# consensus_dataset : sample_with_ideal has 3 rubric items
# ---------------------------------------------------------------------------


def test_rubric_size_reflects_actual_item_count_from_consensus(
    consensus_dataset, tmp_path
):
    result = compute_rubric_size([consensus_dataset], tmp_path, save=False)
    stats = result["consensus"]["rubric_size_stats"]
    assert stats["min"] == 3
    assert stats["max"] == 3
    assert stats["mean"] == pytest.approx(3.0)
