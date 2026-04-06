"""Extended tests for healthbench_agent.analysis.exploration.

Covers functions and save=True paths not exercised in test_exploration.py:
compute_rubric_points_distribution, compute_positive_vs_penalty_stats,
compute_rubric_tags_per_sample, compute_example_tag_frequency,
compute_tag_prefix_distribution, compute_data_quality, compute_subset_overlap,
and save=True branches of functions already partially tested.
"""

from __future__ import annotations

import pytest

from healthbench_agent.analysis.exploration import (
    compute_data_quality,
    compute_example_tag_frequency,
    compute_positive_vs_penalty_stats,
    compute_prompt_structure,
    compute_rubric_points_distribution,
    compute_rubric_size,
    compute_rubric_tag_frequency,
    compute_rubric_tags_per_sample,
    compute_score_range_stats,
    compute_subset_overlap,
    compute_tag_prefix_distribution,
)
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample, RubricItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    prompt_id: str = "p001",
    rubric_points: list[float] | None = None,
    rubric_tags: list[list[str]] | None = None,
    example_tags: list[str] | None = None,
    prompt_content: str = "What should I do?",
) -> HealthBenchSample:
    if rubric_points is None:
        rubric_points = [1.0, -0.5]
    if rubric_tags is None:
        rubric_tags = [["level:example", "axis:accuracy"]] * len(rubric_points)
    if example_tags is None:
        example_tags = ["theme:general_medicine", "axis:accuracy"]
    rubrics = [
        RubricItem(criterion=f"criterion_{i}", points=p, tags=tags)
        for i, (p, tags) in enumerate(zip(rubric_points, rubric_tags))
    ]
    return HealthBenchSample(
        prompt_id=prompt_id,
        prompt=[{"role": "user", "content": prompt_content}],
        rubrics=rubrics,
        example_tags=example_tags,
        ideal_completions_data=None,
        canary=None,
    )


@pytest.fixture
def single_sample_dataset() -> HealthBenchDataset:
    s = _make_sample("p1", rubric_points=[2.0, -1.0, 0.5])
    return HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/s.jsonl")


@pytest.fixture
def three_sample_dataset() -> HealthBenchDataset:
    s1 = _make_sample("p1", rubric_points=[1.0, -0.5])
    s2 = _make_sample("p2", rubric_points=[2.0, 3.0])
    s3 = _make_sample("p3", rubric_points=[-1.0])
    return HealthBenchDataset(subset="main", samples=[s1, s2, s3], source_path="/tmp/t.jsonl")


@pytest.fixture
def empty_dataset() -> HealthBenchDataset:
    return HealthBenchDataset(subset="consensus", samples=[], source_path="/tmp/e.jsonl")


# ---------------------------------------------------------------------------
# compute_rubric_size — save=True branch
# ---------------------------------------------------------------------------


def test_rubric_size_save_true_writes_csv(main_dataset, tmp_path):
    compute_rubric_size([main_dataset], tmp_path, save=True)
    assert any(tmp_path.glob("compute_rubric_size_*.csv"))


def test_rubric_size_value_counts_maps_size_to_count(single_sample_dataset, tmp_path):
    result = compute_rubric_size([single_sample_dataset], tmp_path, save=False)
    counts = result["main"]["rubric_size_value_counts"]
    # single sample has 3 rubric items → size 3 maps to count 1
    assert counts[3] == 1


# ---------------------------------------------------------------------------
# compute_prompt_structure — save=True branch
# ---------------------------------------------------------------------------


def test_prompt_structure_save_true_writes_csv(main_dataset, tmp_path):
    compute_prompt_structure([main_dataset], tmp_path, save=True)
    assert any(tmp_path.glob("compute_prompt_structure_*.csv"))


# ---------------------------------------------------------------------------
# compute_score_range_stats — save=True branch
# ---------------------------------------------------------------------------


def test_score_range_stats_save_true_writes_csv(main_dataset, tmp_path):
    compute_score_range_stats([main_dataset], tmp_path, save=True)
    assert any(tmp_path.glob("compute_score_range_stats_*.csv"))


def test_score_range_stats_max_possible_non_negative(single_sample_dataset, tmp_path):
    result = compute_score_range_stats([single_sample_dataset], tmp_path, save=False)
    assert result["main"]["max_possible_score_stats"]["min"] >= 0.0


def test_score_range_stats_min_possible_non_positive(single_sample_dataset, tmp_path):
    result = compute_score_range_stats([single_sample_dataset], tmp_path, save=False)
    # min_possible is sum of negative points → ≤ 0
    assert result["main"]["min_possible_score_stats"]["max"] <= 0.0


def test_score_range_width_equals_max_minus_min(tmp_path):
    s = _make_sample("p1", rubric_points=[3.0, -1.0])
    dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
    result = compute_score_range_stats([dataset], tmp_path, save=False)
    width = result["main"]["score_range_width_stats"]["mean"]
    assert width == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# compute_rubric_tag_frequency — save=True branch
# ---------------------------------------------------------------------------


def test_rubric_tag_frequency_save_true_writes_csv(main_dataset, tmp_path):
    compute_rubric_tag_frequency([main_dataset], tmp_path, save=True)
    assert any(tmp_path.glob("compute_rubric_tag_frequency_*.csv"))


def test_rubric_tag_frequency_total_rubric_items_correct(single_sample_dataset, tmp_path):
    result = compute_rubric_tag_frequency([single_sample_dataset], tmp_path, save=False)
    assert result["main"]["total_rubric_items"] == 3


def test_rubric_tag_frequency_fraction_with_tag_is_one_when_all_tagged(tmp_path):
    s = _make_sample("p1", rubric_tags=[["axis:accuracy"], ["axis:completeness"]])
    dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
    result = compute_rubric_tag_frequency([dataset], tmp_path, save=False)
    assert result["main"]["fraction_with_tag"] == pytest.approx(1.0)


def test_rubric_tag_frequency_fraction_without_tag_is_one_when_none_tagged(tmp_path):
    s = _make_sample("p1", rubric_tags=[[], []])
    dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
    result = compute_rubric_tag_frequency([dataset], tmp_path, save=False)
    assert result["main"]["fraction_without_tag"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_rubric_points_distribution
# ---------------------------------------------------------------------------


class TestComputeRubricPointsDistribution:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_rubric_points_distribution([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_rubric_points_distribution([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_rubric_points_distribution([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_rubric_points_distribution([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in (
            "points_value_counts",
            "points_stats",
            "fraction_positive",
            "fraction_negative",
            "fraction_zero",
            "total_rubric_items",
        ):
            assert k in keys

    def test_fractions_sum_to_one(self, single_sample_dataset, tmp_path):
        result = compute_rubric_points_distribution([single_sample_dataset], tmp_path, save=False)
        r = result["main"]
        total = r["fraction_positive"] + r["fraction_negative"] + r["fraction_zero"]
        assert total == pytest.approx(1.0)

    def test_total_rubric_items_matches_actual_count(self, single_sample_dataset, tmp_path):
        result = compute_rubric_points_distribution([single_sample_dataset], tmp_path, save=False)
        # single_sample_dataset has one sample with 3 rubric items
        assert result["main"]["total_rubric_items"] == 3

    def test_count_positive_correct(self, tmp_path):
        s = _make_sample("p1", rubric_points=[1.0, 2.0, -1.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_rubric_points_distribution([dataset], tmp_path, save=False)
        assert result["main"]["count_positive"] == 2
        assert result["main"]["count_negative"] == 1
        assert result["main"]["count_zero"] == 0

    def test_fraction_zero_one_when_all_zero_points(self, tmp_path):
        s = _make_sample("p1", rubric_points=[0.0, 0.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_rubric_points_distribution([dataset], tmp_path, save=False)
        assert result["main"]["fraction_zero"] == pytest.approx(1.0)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_rubric_points_distribution([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_rubric_points_distribution_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_rubric_points_distribution([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_positive_vs_penalty_stats
# ---------------------------------------------------------------------------


class TestComputePositiveVsPenaltyStats:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_positive_vs_penalty_stats([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_positive_vs_penalty_stats([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_positive_vs_penalty_stats([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_positive_vs_penalty_stats([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in (
            "total_possible_points_stats",
            "total_penalty_points_stats",
            "penalty_mass_ratio_stats",
            "samples_with_zero_possible_points",
        ):
            assert k in keys

    def test_samples_with_zero_possible_points_correct(self, tmp_path):
        s_pos = _make_sample("p1", rubric_points=[1.0])  # has positive
        s_neg = _make_sample("p2", rubric_points=[-1.0])  # only penalty → zero possible
        dataset = HealthBenchDataset(
            subset="main", samples=[s_pos, s_neg], source_path="/tmp/x.jsonl"
        )
        result = compute_positive_vs_penalty_stats([dataset], tmp_path, save=False)
        assert result["main"]["samples_with_zero_possible_points"] == 1

    def test_total_possible_points_mean_correct(self, tmp_path):
        s1 = _make_sample("p1", rubric_points=[2.0, -1.0])  # possible = 2.0
        s2 = _make_sample("p2", rubric_points=[4.0])  # possible = 4.0
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_positive_vs_penalty_stats([dataset], tmp_path, save=False)
        assert result["main"]["total_possible_points_stats"]["mean"] == pytest.approx(3.0)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_positive_vs_penalty_stats([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_positive_vs_penalty_stats_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_positive_vs_penalty_stats([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_rubric_tags_per_sample
# ---------------------------------------------------------------------------


class TestComputeRubricTagsPerSample:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_rubric_tags_per_sample([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_rubric_tags_per_sample([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_rubric_tags_per_sample([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_rubric_tags_per_sample([main_dataset], tmp_path, save=False)
        assert "unique_tags_per_sample_stats" in result["main"]
        assert "samples_with_zero_tags" in result["main"]

    def test_samples_with_zero_tags_counted_correctly(self, tmp_path):
        s_tagged = _make_sample("p1", rubric_tags=[["axis:accuracy"]])
        s_untagged = _make_sample("p2", rubric_tags=[[]])
        dataset = HealthBenchDataset(
            subset="main", samples=[s_tagged, s_untagged], source_path="/tmp/x.jsonl"
        )
        result = compute_rubric_tags_per_sample([dataset], tmp_path, save=False)
        assert result["main"]["samples_with_zero_tags"] == 1

    def test_unique_tags_per_sample_stats_min_at_least_zero(self, main_dataset, tmp_path):
        result = compute_rubric_tags_per_sample([main_dataset], tmp_path, save=False)
        assert result["main"]["unique_tags_per_sample_stats"]["min"] >= 0

    def test_duplicate_tags_across_rubric_items_deduplicated(self, tmp_path):
        # same tag "axis:accuracy" on two rubric items → unique count = 1
        s = _make_sample("p1", rubric_tags=[["axis:accuracy"], ["axis:accuracy"]])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_rubric_tags_per_sample([dataset], tmp_path, save=False)
        assert result["main"]["unique_tags_per_sample_stats"]["min"] == 1

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_rubric_tags_per_sample([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_rubric_tags_per_sample_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_rubric_tags_per_sample([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_example_tag_frequency
# ---------------------------------------------------------------------------


class TestComputeExampleTagFrequency:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_example_tag_frequency([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_example_tag_frequency([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_example_tag_frequency([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_example_tag_frequency([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in (
            "tag_value_counts",
            "total_unique_example_tags",
            "mean_example_tags_per_sample",
            "samples_with_zero_example_tags",
        ):
            assert k in keys

    def test_samples_with_zero_example_tags_correct(self, tmp_path):
        s_tagged = _make_sample("p1", example_tags=["theme:cardiology"])
        s_empty = HealthBenchSample(
            prompt_id="p2",
            prompt=[{"role": "user", "content": "hi"}],
            rubrics=[RubricItem(criterion="c", points=1.0, tags=[])],
            example_tags=[],
            ideal_completions_data=None,
            canary=None,
        )
        dataset = HealthBenchDataset(
            subset="main", samples=[s_tagged, s_empty], source_path="/tmp/x.jsonl"
        )
        result = compute_example_tag_frequency([dataset], tmp_path, save=False)
        assert result["main"]["samples_with_zero_example_tags"] == 1

    def test_unique_tags_count_deduplicates_across_samples(self, tmp_path):
        # Both samples have the same tag → still only 1 unique tag
        s1 = _make_sample("p1", example_tags=["theme:cardiology"])
        s2 = _make_sample("p2", example_tags=["theme:cardiology"])
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_example_tag_frequency([dataset], tmp_path, save=False)
        assert result["main"]["total_unique_example_tags"] == 1

    def test_mean_example_tags_per_sample_correct(self, tmp_path):
        s1 = _make_sample("p1", example_tags=["t1", "t2"])  # 2 tags
        s2 = _make_sample("p2", example_tags=["t3"])  # 1 tag
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_example_tag_frequency([dataset], tmp_path, save=False)
        assert result["main"]["mean_example_tags_per_sample"] == pytest.approx(1.5)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_example_tag_frequency([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_example_tag_frequency_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_example_tag_frequency([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_tag_prefix_distribution
# ---------------------------------------------------------------------------


class TestComputeTagPrefixDistribution:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_tag_prefix_distribution([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_tag_prefix_distribution([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_tag_prefix_distribution([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_tag_prefix_distribution([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in (
            "example_tag_prefix_distribution",
            "rubric_tag_prefix_distribution",
            "example_tag_unique_prefixes",
            "rubric_tag_unique_prefixes",
        ):
            assert k in keys

    def test_theme_prefix_found_when_tags_have_theme_prefix(self, tmp_path):
        s = _make_sample("p1", example_tags=["theme:cardiology", "axis:accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_tag_prefix_distribution([dataset], tmp_path, save=False)
        assert "theme" in result["main"]["example_tag_prefix_distribution"]

    def test_prefixes_list_is_sorted(self, tmp_path):
        s = _make_sample("p1", example_tags=["theme:cardiology", "axis:accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_tag_prefix_distribution([dataset], tmp_path, save=False)
        prefixes = result["main"]["example_tag_unique_prefixes"]
        assert prefixes == sorted(prefixes)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_tag_prefix_distribution([main_dataset], tmp_path, save=True)
        csv_files = list(tmp_path.glob("compute_tag_prefix_distribution_*.csv"))
        assert len(csv_files) >= 1

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_tag_prefix_distribution([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_data_quality
# ---------------------------------------------------------------------------


class TestComputeDataQuality:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_data_quality([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_data_quality([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_data_quality([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_data_quality([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in (
            "total_samples",
            "total_rubric_items",
            "empty_rubrics_count",
            "empty_rubrics_fraction",
            "zero_points_rubric_items_count",
            "zero_points_rubric_items_fraction",
            "duplicate_prompt_id_count",
            "duplicate_prompt_id_fraction",
            "zero_possible_points_samples_count",
            "zero_possible_points_samples_fraction",
            "empty_rubric_tags_count",
            "empty_rubric_tags_fraction",
            "empty_example_tags_count",
            "empty_example_tags_fraction",
        ):
            assert k in keys

    def test_total_samples_matches_dataset_length(self, main_dataset, tmp_path):
        result = compute_data_quality([main_dataset], tmp_path, save=False)
        assert result["main"]["total_samples"] == len(main_dataset.samples)

    def test_duplicate_prompt_id_detected(self, tmp_path):
        s1 = _make_sample("same-id")
        s2 = _make_sample("same-id")
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_data_quality([dataset], tmp_path, save=False)
        assert result["main"]["duplicate_prompt_id_count"] == 1

    def test_empty_rubrics_counted_correctly(self, tmp_path):
        s_empty = HealthBenchSample(
            prompt_id="p1",
            prompt=[{"role": "user", "content": "hi"}],
            rubrics=[],
            example_tags=[],
            ideal_completions_data=None,
            canary=None,
        )
        s_normal = _make_sample("p2")
        dataset = HealthBenchDataset(
            subset="main", samples=[s_empty, s_normal], source_path="/tmp/x.jsonl"
        )
        result = compute_data_quality([dataset], tmp_path, save=False)
        assert result["main"]["empty_rubrics_count"] == 1
        assert result["main"]["empty_rubrics_fraction"] == pytest.approx(0.5)

    def test_zero_points_rubric_items_counted(self, tmp_path):
        s = _make_sample("p1", rubric_points=[1.0, 0.0, -0.5])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_data_quality([dataset], tmp_path, save=False)
        assert result["main"]["zero_points_rubric_items_count"] == 1

    def test_zero_possible_points_detected_when_only_penalties(self, tmp_path):
        s = _make_sample("p1", rubric_points=[-1.0, -0.5])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_data_quality([dataset], tmp_path, save=False)
        assert result["main"]["zero_possible_points_samples_count"] == 1

    def test_fractions_in_zero_to_one_range(self, main_dataset, tmp_path):
        result = compute_data_quality([main_dataset], tmp_path, save=False)
        r = result["main"]
        for key in (k for k in r if k.endswith("_fraction")):
            assert 0.0 <= r[key] <= 1.0

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_data_quality([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_data_quality_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_data_quality([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_subset_overlap
# ---------------------------------------------------------------------------


class TestComputeSubsetOverlap:
    def test_returns_empty_dict_when_only_hard_present(self, hard_dataset, tmp_path):
        result = compute_subset_overlap([hard_dataset], tmp_path, save=False)
        assert result == {}

    def test_returns_empty_dict_when_only_consensus_present(self, consensus_dataset, tmp_path):
        result = compute_subset_overlap([consensus_dataset], tmp_path, save=False)
        assert result == {}

    def test_returns_empty_dict_with_no_datasets(self, tmp_path):
        assert compute_subset_overlap([], tmp_path, save=False) == {}

    def test_result_contains_expected_keys_when_both_present(
        self, hard_dataset, consensus_dataset, tmp_path
    ):
        result = compute_subset_overlap([hard_dataset, consensus_dataset], tmp_path, save=False)
        for k in (
            "hard_only_count",
            "consensus_only_count",
            "intersection_count",
            "hard_total",
            "consensus_total",
            "jaccard_similarity",
        ):
            assert k in result

    def test_save_false_writes_no_files(self, hard_dataset, consensus_dataset, tmp_path):
        compute_subset_overlap([hard_dataset, consensus_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_jaccard_zero_when_no_shared_ids(self, tmp_path):
        hard = HealthBenchDataset(
            subset="hard",
            samples=[_make_sample("hard-only-id")],
            source_path="/tmp/h.jsonl",
        )
        consensus = HealthBenchDataset(
            subset="consensus",
            samples=[_make_sample("consensus-only-id")],
            source_path="/tmp/c.jsonl",
        )
        result = compute_subset_overlap([hard, consensus], tmp_path, save=False)
        assert result["jaccard_similarity"] == pytest.approx(0.0)

    def test_jaccard_one_when_all_ids_shared(self, tmp_path):
        shared_sample = _make_sample("shared-id")
        hard = HealthBenchDataset(
            subset="hard", samples=[shared_sample], source_path="/tmp/h.jsonl"
        )
        consensus = HealthBenchDataset(
            subset="consensus", samples=[shared_sample], source_path="/tmp/c.jsonl"
        )
        result = compute_subset_overlap([hard, consensus], tmp_path, save=False)
        assert result["jaccard_similarity"] == pytest.approx(1.0)
        assert result["intersection_count"] == 1
        assert result["hard_only_count"] == 0
        assert result["consensus_only_count"] == 0

    def test_hard_only_count_correct(self, tmp_path):
        hard = HealthBenchDataset(
            subset="hard",
            samples=[_make_sample("h1"), _make_sample("shared")],
            source_path="/tmp/h.jsonl",
        )
        consensus = HealthBenchDataset(
            subset="consensus",
            samples=[_make_sample("shared"), _make_sample("c1")],
            source_path="/tmp/c.jsonl",
        )
        result = compute_subset_overlap([hard, consensus], tmp_path, save=False)
        assert result["hard_only_count"] == 1
        assert result["consensus_only_count"] == 1
        assert result["intersection_count"] == 1

    def test_save_true_writes_csv(self, hard_dataset, consensus_dataset, tmp_path):
        compute_subset_overlap([hard_dataset, consensus_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("compute_subset_overlap_*.csv"))
