"""Tests for healthbench_agent.analysis.visualization.

Verifies the analysis convention: save=False returns correct dict keys per
subset without writing any files; save=True writes PNG artifacts. Uses fixtures
from conftest.py (main_dataset, hard_dataset, consensus_dataset, empty_dataset).
"""

from __future__ import annotations

import pytest

from healthbench_agent.analysis.visualization import (
    plot_criteria_weight_histogram,
    plot_multiturn_complexity_comparison,
    plot_penalty_heatmap,
    plot_penalty_ratio_cdf,
    plot_positive_share_by_axis,
    plot_positive_share_heatmap,
    plot_score_ceiling_distribution,
    plot_theme_axis_heatmap,
)
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample, RubricItem

# ---------------------------------------------------------------------------
# Local fixture helpers
# ---------------------------------------------------------------------------


def _make_sample(
    prompt_id: str,
    rubric_points: list[float],
    example_tags: list[str] | None = None,
) -> HealthBenchSample:
    if example_tags is None:
        example_tags = ["theme:cardiology", "axis:accuracy"]
    rubrics = [
        RubricItem(criterion=f"c{i}", points=p, tags=["axis:accuracy"])
        for i, p in enumerate(rubric_points)
    ]
    return HealthBenchSample(
        prompt_id=prompt_id,
        prompt=[{"role": "user", "content": "question"}],
        rubrics=rubrics,
        example_tags=example_tags,
        ideal_completions_data=None,
        canary=None,
    )


@pytest.fixture
def two_theme_dataset() -> HealthBenchDataset:
    s1 = _make_sample("p1", [1.0, -0.5], ["theme:cardiology", "axis:accuracy"])
    s2 = _make_sample("p2", [2.0], ["theme:oncology", "axis:completeness"])
    return HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/t.jsonl")


@pytest.fixture
def empty_dataset() -> HealthBenchDataset:
    return HealthBenchDataset(subset="consensus", samples=[], source_path="/tmp/e.jsonl")


# ---------------------------------------------------------------------------
# plot_score_ceiling_distribution
# ---------------------------------------------------------------------------


class TestPlotScoreCeilingDistribution:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_score_ceiling_distribution([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_score_ceiling_distribution([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_score_ceiling_distribution([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_stats_key(self, main_dataset, tmp_path):
        result = plot_score_ceiling_distribution([main_dataset], tmp_path, save=False)
        assert "stats" in result["main"]

    def test_stats_contains_mean(self, main_dataset, tmp_path):
        result = plot_score_ceiling_distribution([main_dataset], tmp_path, save=False)
        assert "mean" in result["main"]["stats"]

    def test_mean_is_non_negative_when_all_rubrics_positive(self, tmp_path):
        s = _make_sample("p1", [2.0, 3.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = plot_score_ceiling_distribution([dataset], tmp_path, save=False)
        assert result["main"]["stats"]["mean"] == pytest.approx(5.0)

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_score_ceiling_distribution([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("score_ceiling_distribution_*.png"))

    def test_save_true_result_contains_paths_key(self, main_dataset, tmp_path):
        result = plot_score_ceiling_distribution([main_dataset], tmp_path, save=True)
        assert "paths" in result["main"]
        assert len(result["main"]["paths"]) == 1

    def test_multiple_subsets_all_keyed(self, main_dataset, consensus_dataset, tmp_path):
        result = plot_score_ceiling_distribution(
            [main_dataset, consensus_dataset], tmp_path, save=False
        )
        assert "main" in result
        assert "consensus" in result

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_score_ceiling_distribution([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_theme_axis_heatmap
# ---------------------------------------------------------------------------


class TestPlotThemeAxisHeatmap:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_theme_axis_heatmap([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_theme_axis_heatmap([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_theme_axis_heatmap([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = plot_theme_axis_heatmap([main_dataset], tmp_path, save=False)
        for k in ("cross_table", "themes", "axes"):
            assert k in result["main"]

    def test_themes_is_sorted_list(self, two_theme_dataset, tmp_path):
        result = plot_theme_axis_heatmap([two_theme_dataset], tmp_path, save=False)
        themes = result["main"]["themes"]
        assert themes == sorted(themes)

    def test_axes_is_sorted_list(self, two_theme_dataset, tmp_path):
        result = plot_theme_axis_heatmap([two_theme_dataset], tmp_path, save=False)
        axes = result["main"]["axes"]
        assert axes == sorted(axes)

    def test_two_themes_yield_two_entries_in_cross_table(self, two_theme_dataset, tmp_path):
        result = plot_theme_axis_heatmap([two_theme_dataset], tmp_path, save=False)
        assert len(result["main"]["themes"]) == 2

    def test_samples_without_tags_classified_as_unknown(self, tmp_path):
        s = _make_sample("p1", [1.0], example_tags=[])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = plot_theme_axis_heatmap([dataset], tmp_path, save=False)
        assert "_unknown" in result["main"]["themes"]

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_theme_axis_heatmap([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("theme_axis_heatmap_*.png"))

    def test_save_true_result_contains_paths_key(self, main_dataset, tmp_path):
        result = plot_theme_axis_heatmap([main_dataset], tmp_path, save=True)
        assert "paths" in result["main"]

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_theme_axis_heatmap([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_criteria_weight_histogram
# ---------------------------------------------------------------------------


class TestPlotCriteriaWeightHistogram:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_criteria_weight_histogram([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_criteria_weight_histogram([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_criteria_weight_histogram([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = plot_criteria_weight_histogram([main_dataset], tmp_path, save=False)
        for k in ("points_value_counts", "stats"):
            assert k in result["main"]

    def test_points_value_counts_contains_actual_point_values(self, tmp_path):
        s = _make_sample("p1", [1.0, -1.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = plot_criteria_weight_histogram([dataset], tmp_path, save=False)
        counts = result["main"]["points_value_counts"]
        assert 1.0 in counts
        assert -1.0 in counts

    def test_count_in_stats_matches_total_rubric_items(self, tmp_path):
        s1 = _make_sample("p1", [1.0, -0.5, 2.0])
        s2 = _make_sample("p2", [0.5])
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = plot_criteria_weight_histogram([dataset], tmp_path, save=False)
        assert int(result["main"]["stats"]["count"]) == 4

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_criteria_weight_histogram([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("criteria_weight_histogram_*.png"))

    def test_save_true_result_contains_paths_key(self, main_dataset, tmp_path):
        result = plot_criteria_weight_histogram([main_dataset], tmp_path, save=True)
        assert "paths" in result["main"]

    def test_multiple_datasets_keyed_separately(self, main_dataset, hard_dataset, tmp_path):
        result = plot_criteria_weight_histogram(
            [main_dataset, hard_dataset], tmp_path, save=False
        )
        assert "main" in result
        assert "hard" in result

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_criteria_weight_histogram([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_penalty_heatmap
# ---------------------------------------------------------------------------


class TestPlotPenaltyHeatmap:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_penalty_heatmap([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_penalty_heatmap([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_penalty_heatmap([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_dataset_with_penalties_returns_themes_and_axes(self, tmp_path):
        s = _make_sample("p1", [1.0, -2.0], ["theme:cardiology", "axis:accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = plot_penalty_heatmap([dataset], tmp_path, save=False)
        assert "themes" in result["main"]
        assert "axes" in result["main"]

    def test_no_penalties_returns_empty(self, tmp_path):
        s = _make_sample("p1", [1.0, 2.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = plot_penalty_heatmap([dataset], tmp_path, save=False)
        assert result["main"] == {}

    def test_save_true_writes_png(self, tmp_path):
        s = _make_sample("p1", [1.0, -2.0], ["theme:cardiology", "axis:accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        plot_penalty_heatmap([dataset], tmp_path, save=True)
        assert any(tmp_path.glob("penalty_heatmap_*.png"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_penalty_heatmap([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_multiturn_complexity_comparison
# ---------------------------------------------------------------------------


class TestPlotMultiturnComplexityComparison:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_multiturn_complexity_comparison([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_multiturn_complexity_comparison([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_multiturn_complexity_comparison([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_turn_type_keys(self, main_dataset, tmp_path):
        result = plot_multiturn_complexity_comparison([main_dataset], tmp_path, save=False)
        subset_data = result["main"]
        assert "single_turn_count" in subset_data
        assert "multi_turn_count" in subset_data

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_multiturn_complexity_comparison([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("multiturn_complexity_*.png"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_multiturn_complexity_comparison([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_penalty_ratio_cdf
# ---------------------------------------------------------------------------


class TestPlotPenaltyRatioCdf:
    def test_returns_overlay_key(self, main_dataset, tmp_path):
        result = plot_penalty_ratio_cdf([main_dataset], tmp_path, save=False)
        assert "overlay" in result

    def test_overlay_contains_per_subset(self, main_dataset, tmp_path):
        result = plot_penalty_ratio_cdf([main_dataset], tmp_path, save=False)
        assert "per_subset" in result["overlay"]
        assert "main" in result["overlay"]["per_subset"]

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_penalty_ratio_cdf([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_dict(self, empty_dataset, tmp_path):
        result = plot_penalty_ratio_cdf([empty_dataset], tmp_path, save=False)
        assert result == {}

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_penalty_ratio_cdf([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("penalty_ratio_cdf_*.png"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_penalty_ratio_cdf([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_positive_share_by_axis
# ---------------------------------------------------------------------------


class TestPlotPositiveShareByAxis:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_positive_share_by_axis([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_result_contains_share_keys(self, main_dataset, tmp_path):
        result = plot_positive_share_by_axis([main_dataset], tmp_path, save=False)
        assert "positive_share" in result["main"]
        assert "penalty_share" in result["main"]

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_positive_share_by_axis([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_positive_share_by_axis([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_positive_share_by_axis([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("positive_share_by_axis_*.png"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_positive_share_by_axis([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# plot_positive_share_heatmap
# ---------------------------------------------------------------------------


class TestPlotPositiveShareHeatmap:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = plot_positive_share_heatmap([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = plot_positive_share_heatmap([main_dataset], tmp_path, save=False)
        for k in ("share_table", "themes", "axes"):
            assert k in result["main"]

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        plot_positive_share_heatmap([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = plot_positive_share_heatmap([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_save_true_writes_png(self, main_dataset, tmp_path):
        plot_positive_share_heatmap([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("positive_share_heatmap_*.png"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert plot_positive_share_heatmap([], tmp_path, save=False) == {}
