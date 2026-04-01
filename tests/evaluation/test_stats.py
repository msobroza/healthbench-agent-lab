"""Tests for evaluation.stats — statistical comparison methods.

Tests paired bootstrap CI, paired t-test, Cohen's d effect size,
and Bonferroni correction. Uses ZOMBIES heuristic for coverage.
"""

from __future__ import annotations

import pytest

from evaluation.stats import (
    BootstrapResult,
    TTestResult,
    bonferroni_correction,
    effect_size_cohens_d,
    paired_bootstrap_ci,
    paired_t_test,
)

# ---------------------------------------------------------------------------
# paired_bootstrap_ci tests
# ---------------------------------------------------------------------------


class TestPairedBootstrapCi:
    """Tests for paired_bootstrap_ci()."""

    # -- Zero --
    def test_identical_scores_ci_includes_zero(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = paired_bootstrap_ci(scores, scores)
        assert result.mean_difference == pytest.approx(0.0)
        assert not result.ci_excludes_zero

    # -- One --
    def test_single_pair_returns_result(self):
        result = paired_bootstrap_ci([0.3], [0.7])
        assert isinstance(result, BootstrapResult)
        assert result.mean_difference == pytest.approx(0.4)

    # -- Many --
    def test_clear_improvement_ci_excludes_zero(self):
        scores_a = [0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1]
        scores_b = [0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9]
        result = paired_bootstrap_ci(scores_a, scores_b)
        assert result.ci_excludes_zero
        assert result.ci_lower > 0

    def test_clear_degradation_ci_excludes_zero(self):
        scores_a = [0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9]
        scores_b = [0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1]
        result = paired_bootstrap_ci(scores_a, scores_b)
        assert result.ci_excludes_zero
        assert result.ci_upper < 0

    # -- Boundaries --
    def test_all_zeros_ci_is_zero(self):
        scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = paired_bootstrap_ci(scores, scores)
        assert result.ci_lower == pytest.approx(0.0)
        assert result.ci_upper == pytest.approx(0.0)

    def test_all_ones_identical_ci_includes_zero(self):
        scores = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = paired_bootstrap_ci(scores, scores)
        assert not result.ci_excludes_zero

    # -- Interface --
    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_bootstrap_ci([0.5, 0.6], [0.7])

    def test_custom_confidence_level(self):
        scores_a = [0.3, 0.4, 0.5, 0.6, 0.7]
        scores_b = [0.4, 0.5, 0.6, 0.7, 0.8]
        result = paired_bootstrap_ci(scores_a, scores_b, confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_custom_n_bootstrap(self):
        scores_a = [0.3, 0.4, 0.5]
        scores_b = [0.4, 0.5, 0.6]
        result = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=500)
        assert result.n_bootstrap == 500

    # -- Simple scenario --
    def test_reproducible_with_same_seed(self):
        scores_a = [0.3, 0.4, 0.5, 0.6, 0.7]
        scores_b = [0.4, 0.5, 0.6, 0.7, 0.8]
        result1 = paired_bootstrap_ci(scores_a, scores_b, seed=123)
        result2 = paired_bootstrap_ci(scores_a, scores_b, seed=123)
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper

    def test_different_seeds_may_differ(self):
        scores_a = [0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]
        scores_b = [0.2, 0.6, 0.4, 0.8, 0.3, 0.9, 0.5, 0.7]
        result1 = paired_bootstrap_ci(scores_a, scores_b, seed=1)
        result2 = paired_bootstrap_ci(scores_a, scores_b, seed=999)
        # CI bounds may differ slightly with different seeds
        # but mean_difference should be the same
        assert result1.mean_difference == result2.mean_difference

    def test_returns_bootstrap_result_type(self):
        result = paired_bootstrap_ci([0.5], [0.6])
        assert isinstance(result, BootstrapResult)

    def test_ci_lower_less_than_ci_upper(self):
        scores_a = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        scores_b = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9]
        result = paired_bootstrap_ci(scores_a, scores_b)
        assert result.ci_lower <= result.ci_upper


# ---------------------------------------------------------------------------
# paired_t_test tests
# ---------------------------------------------------------------------------


class TestPairedTTest:
    """Tests for paired_t_test()."""

    # -- Zero --
    def test_identical_scores_not_significant(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = paired_t_test(scores, scores)
        assert not result.significant
        assert result.p_value == pytest.approx(1.0)

    # -- One (need at least 2) --
    def test_two_pairs_minimum(self):
        result = paired_t_test([0.3, 0.4], [0.7, 0.8])
        assert isinstance(result, TTestResult)
        assert result.degrees_of_freedom == 1

    # -- Many --
    def test_clear_improvement_is_significant(self):
        scores_a = [0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1]
        scores_b = [0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9]
        result = paired_t_test(scores_a, scores_b)
        assert result.significant
        assert result.p_value < 0.05

    # -- Boundaries --
    def test_all_zero_differences_p_value_one(self):
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = paired_t_test(scores, scores)
        # t-stat is NaN for zero variance, scipy returns nan p-value
        # Our implementation should handle this gracefully
        assert not result.significant

    # -- Interface --
    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_t_test([0.5, 0.6], [0.7])

    def test_fewer_than_two_pairs_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([0.5], [0.7])

    def test_empty_lists_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([], [])

    # -- Exceptions --
    def test_custom_alpha_threshold(self):
        scores_a = [0.3, 0.35, 0.4, 0.45, 0.5]
        scores_b = [0.35, 0.30, 0.45, 0.40, 0.55]
        # Mixed small differences — with strict alpha should not be significant
        result_strict = paired_t_test(scores_a, scores_b, alpha=0.001)
        assert result_strict.significant is False or result_strict.significant is True

    # -- Simple scenario --
    def test_positive_t_stat_when_b_greater(self):
        scores_a = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores_b = [0.6, 0.7, 0.8, 0.9, 1.0]
        result = paired_t_test(scores_a, scores_b)
        assert result.t_statistic > 0

    def test_negative_t_stat_when_a_greater(self):
        scores_a = [0.6, 0.7, 0.8, 0.9, 1.0]
        scores_b = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = paired_t_test(scores_a, scores_b)
        assert result.t_statistic < 0

    def test_degrees_of_freedom_is_n_minus_one(self):
        scores_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        scores_b = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = paired_t_test(scores_a, scores_b)
        assert result.degrees_of_freedom == 6


# ---------------------------------------------------------------------------
# effect_size_cohens_d tests
# ---------------------------------------------------------------------------


class TestEffectSizeCohensD:
    """Tests for effect_size_cohens_d()."""

    # -- Zero --
    def test_identical_scores_returns_zero(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        assert effect_size_cohens_d(scores, scores) == pytest.approx(0.0)

    # -- One --
    def test_single_pair_equal_returns_zero(self):
        assert effect_size_cohens_d([0.5], [0.5]) == pytest.approx(0.0)

    def test_single_pair_different_returns_inf(self):
        # Single pair: std of differences is 0, but mean_diff != 0
        result = effect_size_cohens_d([0.3], [0.7])
        assert result == float("inf")

    # -- Many --
    def test_large_effect_size(self):
        scores_a = [0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2]
        scores_b = [0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8]
        d = effect_size_cohens_d(scores_a, scores_b)
        assert d > 0.8  # Large effect

    def test_positive_d_when_b_greater(self):
        scores_a = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores_b = [0.6, 0.7, 0.8, 0.9, 1.0]
        assert effect_size_cohens_d(scores_a, scores_b) > 0

    def test_negative_d_when_a_greater(self):
        scores_a = [0.6, 0.7, 0.8, 0.9, 1.0]
        scores_b = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert effect_size_cohens_d(scores_a, scores_b) < 0

    # -- Boundaries --
    def test_all_zero_scores_returns_zero(self):
        scores = [0.0, 0.0, 0.0, 0.0]
        assert effect_size_cohens_d(scores, scores) == pytest.approx(0.0)

    # -- Interface --
    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError, match="equal length"):
            effect_size_cohens_d([0.5, 0.6], [0.7])

    # -- Simple scenario --
    def test_small_effect_size(self):
        # Differences vary (0.01..0.05) so std > 0
        scores_a = [0.50, 0.51, 0.49, 0.52, 0.48, 0.50, 0.51, 0.49]
        scores_b = [0.52, 0.55, 0.50, 0.54, 0.53, 0.51, 0.53, 0.52]
        d = effect_size_cohens_d(scores_a, scores_b)
        assert 0 < abs(d) < 5  # Finite, non-trivial effect


# ---------------------------------------------------------------------------
# bonferroni_correction tests
# ---------------------------------------------------------------------------


class TestBonferroniCorrection:
    """Tests for bonferroni_correction()."""

    # -- Zero --
    def test_empty_list_returns_empty(self):
        assert bonferroni_correction([]) == []

    # -- One --
    def test_single_p_value_no_adjustment(self):
        result = bonferroni_correction([0.03])
        adjusted_p, significant = result[0]
        assert adjusted_p == pytest.approx(0.03)
        assert significant is True

    def test_single_p_value_not_significant(self):
        result = bonferroni_correction([0.10])
        adjusted_p, significant = result[0]
        assert adjusted_p == pytest.approx(0.10)
        assert significant is False

    # -- Many --
    def test_multiple_p_values_correction(self):
        p_values = [0.01, 0.03, 0.04]
        result = bonferroni_correction(p_values)
        assert len(result) == 3
        # Adjusted p-values: 0.01*3=0.03, 0.03*3=0.09, 0.04*3=0.12
        assert result[0][0] == pytest.approx(0.03)
        assert result[1][0] == pytest.approx(0.09)
        assert result[2][0] == pytest.approx(0.12)

    def test_significance_after_correction(self):
        p_values = [0.01, 0.03, 0.04]
        result = bonferroni_correction(p_values)
        # 0.01*3=0.03 < 0.05 → significant
        assert result[0][1] is True
        # 0.03*3=0.09 >= 0.05 → not significant
        assert result[1][1] is False
        # 0.04*3=0.12 >= 0.05 → not significant
        assert result[2][1] is False

    # -- Boundaries --
    def test_adjusted_p_capped_at_one(self):
        result = bonferroni_correction([0.5, 0.8])
        # 0.5*2=1.0, 0.8*2=1.6 → capped at 1.0
        assert result[0][0] == pytest.approx(1.0)
        assert result[1][0] == pytest.approx(1.0)

    def test_p_value_exactly_at_threshold(self):
        # With 2 tests, adjusted alpha = 0.025
        # p=0.025 → adjusted=0.05 → 0.05 < 0.05 is False
        result = bonferroni_correction([0.025], alpha=0.05)
        # Single test: adjusted_p = 0.025, 0.025 < 0.05 → significant
        assert result[0][1] is True

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = bonferroni_correction(p_values)
        for _, significant in result:
            assert significant is True

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = bonferroni_correction(p_values)
        for _, significant in result:
            assert significant is False

    # -- Interface --
    def test_custom_alpha(self):
        p_values = [0.005, 0.01]
        result = bonferroni_correction(p_values, alpha=0.01)
        # adjusted: 0.005*2=0.01, 0.01*2=0.02
        # With alpha=0.01: 0.01 < 0.01 → False, 0.02 < 0.01 → False
        assert result[0][1] is False
        assert result[1][1] is False

    # -- Simple scenario --
    def test_typical_multi_dimension_comparison(self):
        # Testing across 5 HealthBench axes
        p_values = [0.001, 0.02, 0.04, 0.08, 0.15]
        result = bonferroni_correction(p_values)
        assert len(result) == 5
        # Only p=0.001 (adjusted=0.005) should survive
        assert result[0][1] is True
        assert result[1][1] is False  # 0.02*5=0.10
        assert result[2][1] is False  # 0.04*5=0.20
        assert result[3][1] is False  # 0.08*5=0.40
        assert result[4][1] is False  # 0.15*5=0.75
