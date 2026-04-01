"""Tests for healthbench_agent.analysis.insights.

Verifies the analysis convention: save=False returns correct dict keys per
subset without writing files; save=True writes CSV artifacts. Uses fixtures
from conftest.py (main_dataset, hard_dataset, consensus_dataset, empty_dataset).
"""

from __future__ import annotations

import pytest

from healthbench_agent.analysis.insights import (
    compute_criterion_text_length_stats,
    compute_emergency_axis_profile,
    compute_hard_subset_signature,
    compute_penalty_concentration,
    compute_prompt_length_vs_rubric_complexity,
    compute_rubric_axis_difficulty,
    compute_score_ceiling_by_theme,
    compute_theme_axis_breakdown,
)
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample, RubricItem

# ---------------------------------------------------------------------------
# Local fixtures — extend conftest.py fixtures with insights-specific shapes
# ---------------------------------------------------------------------------


def _make_sample(
    prompt_id: str = "p001",
    rubric_points: list[float] | None = None,
    rubric_axes: list[str] | None = None,
    example_tags: list[str] | None = None,
    prompt_content: str = "What should I do about chest pain?",
    criterion_text: str = "Advise the patient to seek emergency care.",
) -> HealthBenchSample:
    if rubric_points is None:
        rubric_points = [1.0, -0.5]
    if rubric_axes is None:
        rubric_axes = ["accuracy", "accuracy"]
    if example_tags is None:
        example_tags = ["theme:cardiology", "axis:accuracy"]
    rubrics = [
        RubricItem(
            criterion=criterion_text,
            points=p,
            tags=[f"axis:{ax}"],
        )
        for p, ax in zip(rubric_points, rubric_axes)
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
def two_theme_dataset() -> HealthBenchDataset:
    """Dataset with two different themes so cross-tables are non-trivial."""
    s1 = _make_sample("p1", example_tags=["theme:cardiology", "axis:accuracy"])
    s2 = _make_sample("p2", example_tags=["theme:oncology", "axis:completeness"])
    return HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/t.jsonl")


@pytest.fixture
def penalty_heavy_dataset() -> HealthBenchDataset:
    """Dataset with a large penalty item to test penalty concentration metrics."""
    s1 = _make_sample("p1", rubric_points=[2.0, -3.0], example_tags=["theme:emergency"])
    s2 = _make_sample("p2", rubric_points=[1.0], example_tags=["theme:cardiology"])
    return HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/p.jsonl")


@pytest.fixture
def main_dataset_for_signature() -> HealthBenchDataset:
    s1 = _make_sample("p1", rubric_points=[1.0, 2.0, -0.5], example_tags=["theme:cardiology"])
    s2 = _make_sample("p2", rubric_points=[3.0], example_tags=["theme:oncology"])
    return HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/m.jsonl")


@pytest.fixture
def hard_dataset_for_signature() -> HealthBenchDataset:
    s1 = _make_sample("p3", rubric_points=[1.0, 2.0, 3.0, 4.0], example_tags=["theme:cardiology"])
    return HealthBenchDataset(subset="hard", samples=[s1], source_path="/tmp/h.jsonl")


@pytest.fixture
def empty_dataset() -> HealthBenchDataset:
    return HealthBenchDataset(subset="consensus", samples=[], source_path="/tmp/e.jsonl")


# ---------------------------------------------------------------------------
# compute_theme_axis_breakdown
# ---------------------------------------------------------------------------


class TestComputeThemeAxisBreakdown:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_theme_axis_breakdown([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_theme_axis_breakdown([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_theme_axis_breakdown([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_theme_axis_breakdown([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        assert "cross_table" in keys
        assert "theme_counts" in keys
        assert "axis_counts" in keys
        assert "total_samples" in keys
        assert "missing_theme_count" in keys
        assert "missing_axis_count" in keys

    def test_total_samples_matches_dataset_length(self, main_dataset, tmp_path):
        result = compute_theme_axis_breakdown([main_dataset], tmp_path, save=False)
        assert result["main"]["total_samples"] == len(main_dataset.samples)

    def test_samples_without_theme_tag_counted_as_unknown(self, tmp_path):
        s = _make_sample("p1", example_tags=[])  # no theme tag
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_theme_axis_breakdown([dataset], tmp_path, save=False)
        assert result["main"]["missing_theme_count"] == 1
        # axis comes from rubric tags, not example_tags — rubrics have axis:accuracy
        assert result["main"]["missing_axis_count"] == 0

    def test_two_themes_yield_two_theme_count_entries(self, two_theme_dataset, tmp_path):
        result = compute_theme_axis_breakdown([two_theme_dataset], tmp_path, save=False)
        assert len(result["main"]["theme_counts"]) == 2

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_theme_axis_breakdown([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("theme_axis_breakdown_*.csv"))

    def test_multiple_datasets_keyed_separately(self, main_dataset, consensus_dataset, tmp_path):
        result = compute_theme_axis_breakdown([main_dataset, consensus_dataset], tmp_path, save=False)
        assert "main" in result
        assert "consensus" in result

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_theme_axis_breakdown([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_rubric_axis_difficulty
# ---------------------------------------------------------------------------


class TestComputeRubricAxisDifficulty:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_rubric_axis_difficulty([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_rubric_axis_difficulty([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_rubric_axis_difficulty([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_rubric_axis_difficulty([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        assert "axis_stats" in keys
        assert "total_rubric_items" in keys
        assert "axes_found" in keys

    def test_total_rubric_items_matches_actual_count(self, tmp_path):
        s1 = _make_sample("p1", rubric_points=[1.0, -0.5])
        s2 = _make_sample("p2", rubric_points=[2.0])
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_rubric_axis_difficulty([dataset], tmp_path, save=False)
        assert result["main"]["total_rubric_items"] == 3

    def test_penalty_fraction_correct(self, tmp_path):
        s = _make_sample("p1", rubric_points=[1.0, -1.0], rubric_axes=["accuracy", "accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_rubric_axis_difficulty([dataset], tmp_path, save=False)
        axis_stats = result["main"]["axis_stats"]["accuracy"]
        assert axis_stats["penalty_fraction"] == pytest.approx(0.5)
        assert axis_stats["positive_fraction"] == pytest.approx(0.5)

    def test_axes_found_is_sorted_list(self, tmp_path):
        s = _make_sample("p1", rubric_points=[1.0, 2.0], rubric_axes=["completeness", "accuracy"])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_rubric_axis_difficulty([dataset], tmp_path, save=False)
        axes = result["main"]["axes_found"]
        assert axes == sorted(axes)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_rubric_axis_difficulty([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("rubric_axis_difficulty_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_rubric_axis_difficulty([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_penalty_concentration
# ---------------------------------------------------------------------------


class TestComputePenaltyConcentration:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_penalty_concentration([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_penalty_concentration([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_penalty_concentration([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_penalty_concentration([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in ("theme_total_penalty", "theme_mean_penalty", "total_penalty_points",
                  "total_positive_points", "penalty_to_positive_ratio"):
            assert k in keys

    def test_penalty_to_positive_ratio_none_when_no_positive_points(self, tmp_path):
        s = _make_sample("p1", rubric_points=[-1.0])  # only penalty, no positive
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_penalty_concentration([dataset], tmp_path, save=False)
        assert result["main"]["penalty_to_positive_ratio"] is None

    def test_total_penalty_computed_correctly(self, penalty_heavy_dataset, tmp_path):
        result = compute_penalty_concentration([penalty_heavy_dataset], tmp_path, save=False)
        # s1 penalty = 3.0, s2 penalty = 0.0 → total = 3.0
        assert result["main"]["total_penalty_points"] == pytest.approx(3.0, abs=1e-3)

    def test_total_positive_computed_correctly(self, penalty_heavy_dataset, tmp_path):
        result = compute_penalty_concentration([penalty_heavy_dataset], tmp_path, save=False)
        # s1 positive = 2.0, s2 positive = 1.0 → total = 3.0
        assert result["main"]["total_positive_points"] == pytest.approx(3.0, abs=1e-3)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_penalty_concentration([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("penalty_concentration_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_penalty_concentration([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_criterion_text_length_stats
# ---------------------------------------------------------------------------


class TestComputeCriterionTextLengthStats:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_criterion_text_length_stats([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_criterion_text_length_stats([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_criterion_text_length_stats([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_criterion_text_length_stats([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        assert "overall_length_stats" in keys
        assert "length_stats_by_axis" in keys
        assert "total_rubric_items" in keys

    def test_total_rubric_items_is_sum_across_all_samples(self, tmp_path):
        s1 = _make_sample("p1", rubric_points=[1.0, -0.5])
        s2 = _make_sample("p2", rubric_points=[1.0])
        dataset = HealthBenchDataset(subset="main", samples=[s1, s2], source_path="/tmp/x.jsonl")
        result = compute_criterion_text_length_stats([dataset], tmp_path, save=False)
        assert result["main"]["total_rubric_items"] == 3

    def test_overall_length_stats_mean_is_positive(self, main_dataset, tmp_path):
        result = compute_criterion_text_length_stats([main_dataset], tmp_path, save=False)
        assert result["main"]["overall_length_stats"]["mean"] > 0

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_criterion_text_length_stats([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("criterion_text_length_stats_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_criterion_text_length_stats([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_score_ceiling_by_theme
# ---------------------------------------------------------------------------


class TestComputeScoreCeilingByTheme:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_score_ceiling_by_theme([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_score_ceiling_by_theme([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_score_ceiling_by_theme([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_score_ceiling_by_theme([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in ("theme_mean_possible_points", "theme_mean_penalty_ratio",
                  "theme_sample_counts", "overall_mean_possible_points",
                  "overall_mean_penalty_ratio"):
            assert k in keys

    def test_overall_mean_possible_points_is_positive_when_samples_have_positive_rubrics(
        self, tmp_path
    ):
        s = _make_sample("p1", rubric_points=[3.0, 2.0])
        dataset = HealthBenchDataset(subset="main", samples=[s], source_path="/tmp/x.jsonl")
        result = compute_score_ceiling_by_theme([dataset], tmp_path, save=False)
        assert result["main"]["overall_mean_possible_points"] == pytest.approx(5.0)

    def test_theme_sample_counts_sum_equals_total_samples(self, two_theme_dataset, tmp_path):
        result = compute_score_ceiling_by_theme([two_theme_dataset], tmp_path, save=False)
        total = sum(result["main"]["theme_sample_counts"].values())
        assert total == len(two_theme_dataset.samples)

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_score_ceiling_by_theme([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("score_ceiling_by_theme_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_score_ceiling_by_theme([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_emergency_axis_profile
# ---------------------------------------------------------------------------


class TestComputeEmergencyAxisProfile:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_emergency_axis_profile([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_emergency_axis_profile([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_emergency_axis_profile([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_emergency_axis_profile([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        for k in ("emergency_rubric_item_count", "emergency_rubric_item_fraction",
                  "samples_with_emergency_item_count", "samples_with_emergency_item_fraction",
                  "emergency_penalty_points_total", "theme_counts"):
            assert k in keys

    def test_no_emergency_items_yields_zero_count(self, tmp_path):
        s = RubricItem(criterion="routine check", points=1.0, tags=["axis:completeness"])
        sample = HealthBenchSample(
            prompt_id="p1", prompt=[{"role": "user", "content": "hi"}],
            rubrics=[s], example_tags=[], ideal_completions_data=None, canary=None,
        )
        dataset = HealthBenchDataset(subset="main", samples=[sample], source_path="/tmp/x.jsonl")
        result = compute_emergency_axis_profile([dataset], tmp_path, save=False)
        assert result["main"]["emergency_rubric_item_count"] == 0
        assert result["main"]["samples_with_emergency_item_count"] == 0

    def test_safety_axis_item_counted_as_emergency(self, tmp_path):
        s = RubricItem(criterion="routine check", points=-1.0, tags=["axis:safety"])
        sample = HealthBenchSample(
            prompt_id="p1", prompt=[{"role": "user", "content": "hi"}],
            rubrics=[s], example_tags=["theme:general"], ideal_completions_data=None, canary=None,
        )
        dataset = HealthBenchDataset(subset="main", samples=[sample], source_path="/tmp/x.jsonl")
        result = compute_emergency_axis_profile([dataset], tmp_path, save=False)
        assert result["main"]["emergency_rubric_item_count"] == 1
        assert result["main"]["emergency_penalty_points_total"] == pytest.approx(1.0)

    def test_emergency_keyword_in_criterion_text_counted_as_emergency(self, tmp_path):
        s = RubricItem(criterion="Call emergency services immediately", points=2.0,
                       tags=["axis:accuracy"])
        sample = HealthBenchSample(
            prompt_id="p1", prompt=[{"role": "user", "content": "hi"}],
            rubrics=[s], example_tags=["theme:general"], ideal_completions_data=None, canary=None,
        )
        dataset = HealthBenchDataset(subset="main", samples=[sample], source_path="/tmp/x.jsonl")
        result = compute_emergency_axis_profile([dataset], tmp_path, save=False)
        assert result["main"]["emergency_rubric_item_count"] == 1
        assert result["main"]["samples_with_emergency_item_count"] == 1

    def test_fractions_in_zero_to_one_range(self, main_dataset, tmp_path):
        result = compute_emergency_axis_profile([main_dataset], tmp_path, save=False)
        r = result["main"]
        assert 0.0 <= r["emergency_rubric_item_fraction"] <= 1.0
        assert 0.0 <= r["samples_with_emergency_item_fraction"] <= 1.0

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_emergency_axis_profile([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("emergency_axis_profile_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_emergency_axis_profile([], tmp_path, save=False) == {}


# ---------------------------------------------------------------------------
# compute_hard_subset_signature
# ---------------------------------------------------------------------------


class TestComputeHardSubsetSignature:
    def test_returns_empty_dict_when_only_main_present(
        self, main_dataset_for_signature, tmp_path
    ):
        result = compute_hard_subset_signature([main_dataset_for_signature], tmp_path, save=False)
        assert result == {}

    def test_returns_empty_dict_when_only_hard_present(
        self, hard_dataset_for_signature, tmp_path
    ):
        result = compute_hard_subset_signature([hard_dataset_for_signature], tmp_path, save=False)
        assert result == {}

    def test_returns_empty_dict_with_no_datasets(self, tmp_path):
        assert compute_hard_subset_signature([], tmp_path, save=False) == {}

    def test_result_contains_expected_keys_when_both_subsets_present(
        self, main_dataset_for_signature, hard_dataset_for_signature, tmp_path
    ):
        result = compute_hard_subset_signature(
            [main_dataset_for_signature, hard_dataset_for_signature], tmp_path, save=False
        )
        for k in ("main_stats", "hard_stats", "delta",
                  "theme_distribution_main", "theme_distribution_hard"):
            assert k in result

    def test_delta_keys_include_all_metric_names(
        self, main_dataset_for_signature, hard_dataset_for_signature, tmp_path
    ):
        result = compute_hard_subset_signature(
            [main_dataset_for_signature, hard_dataset_for_signature], tmp_path, save=False
        )
        for metric in ("rubric_size", "total_possible_points", "penalty_mass_ratio",
                       "prompt_char_length"):
            assert metric in result["delta"]

    def test_save_false_writes_no_files(
        self, main_dataset_for_signature, hard_dataset_for_signature, tmp_path
    ):
        compute_hard_subset_signature(
            [main_dataset_for_signature, hard_dataset_for_signature], tmp_path, save=False
        )
        assert list(tmp_path.iterdir()) == []

    def test_save_true_writes_csv(
        self, main_dataset_for_signature, hard_dataset_for_signature, tmp_path
    ):
        compute_hard_subset_signature(
            [main_dataset_for_signature, hard_dataset_for_signature], tmp_path, save=True
        )
        assert any(tmp_path.glob("hard_subset_signature*.csv"))

    def test_hard_rubric_size_larger_than_main(
        self, main_dataset_for_signature, hard_dataset_for_signature, tmp_path
    ):
        result = compute_hard_subset_signature(
            [main_dataset_for_signature, hard_dataset_for_signature], tmp_path, save=False
        )
        # hard has 4 rubric items per sample, main has 3 and 1 → mean ~2; hard mean = 4
        assert result["hard_stats"]["rubric_size"] > result["main_stats"]["rubric_size"]


# ---------------------------------------------------------------------------
# compute_prompt_length_vs_rubric_complexity
# ---------------------------------------------------------------------------


class TestComputePromptLengthVsRubricComplexity:
    def test_save_false_returns_dict_keyed_by_subset(self, main_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        assert "main" in result

    def test_save_false_writes_no_files(self, main_dataset, tmp_path):
        compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        assert list(tmp_path.iterdir()) == []

    def test_empty_dataset_returns_empty_subset_dict(self, empty_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([empty_dataset], tmp_path, save=False)
        assert result["consensus"] == {}

    def test_result_contains_expected_keys(self, main_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        keys = result["main"].keys()
        assert "length_vs_rubric_size" in keys
        assert "length_vs_possible_points" in keys
        assert "sample_count" in keys

    def test_correlation_keys_present(self, main_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        corr = result["main"]["length_vs_rubric_size"]
        for k in ("pearson_r", "pearson_p", "spearman_r", "spearman_p"):
            assert k in corr

    def test_sample_count_matches_dataset_length(self, main_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        assert result["main"]["sample_count"] == len(main_dataset.samples)

    def test_pearson_r_in_valid_range(self, main_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=False)
        r = result["main"]["length_vs_rubric_size"]["pearson_r"]
        assert -1.0 <= r <= 1.0

    def test_save_true_writes_csv(self, main_dataset, tmp_path):
        compute_prompt_length_vs_rubric_complexity([main_dataset], tmp_path, save=True)
        assert any(tmp_path.glob("prompt_length_vs_rubric_complexity_*.csv"))

    def test_no_datasets_returns_empty_dict(self, tmp_path):
        assert compute_prompt_length_vs_rubric_complexity([], tmp_path, save=False) == {}

    def test_multiple_datasets_all_keyed(self, main_dataset, consensus_dataset, tmp_path):
        result = compute_prompt_length_vs_rubric_complexity(
            [main_dataset, consensus_dataset], tmp_path, save=False
        )
        assert "main" in result
        assert "consensus" in result
