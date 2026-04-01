"""Statistical comparison methods for paired agent evaluation.

Implements paired bootstrap CI, paired t-test, Cohen's d effect size,
and Bonferroni correction as specified in SPEC §5.5.

Decision rule: An improvement is reported only when the 95% bootstrap CI
for the paired difference excludes zero AND p < 0.05 after Bonferroni
correction (when testing multiple dimensions).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class BootstrapResult:
    """Result of a paired bootstrap confidence interval test.

    Attributes:
        mean_difference: Mean of the paired differences.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        ci_excludes_zero: Whether the CI excludes zero (significant).
        confidence_level: Confidence level used (e.g. 0.95).
        n_bootstrap: Number of bootstrap samples drawn.
    """

    mean_difference: float
    ci_lower: float
    ci_upper: float
    ci_excludes_zero: bool
    confidence_level: float
    n_bootstrap: int


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute paired bootstrap confidence interval for score differences.

    Primary comparison method — determines if the improvement CI excludes
    zero. Uses paired differences (same conversations, different agents)
    to reduce variance.

    Args:
        scores_a: Per-conversation scores for agent A.
        scores_b: Per-conversation scores for agent B.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CI bounds and significance.

    Raises:
        ValueError: If score lists have different lengths.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have equal length: {len(scores_a)} vs {len(scores_b)}"
        )

    differences = np.array(scores_b) - np.array(scores_a)
    mean_diff = float(np.mean(differences))

    rng = random.Random(seed)
    n = len(differences)
    bootstrap_means: list[float] = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample = differences[indices]
        bootstrap_means.append(float(np.mean(sample)))

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    ci_excludes_zero = ci_lower > 0 or ci_upper < 0

    return BootstrapResult(
        mean_difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_excludes_zero=ci_excludes_zero,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


@dataclass
class TTestResult:
    """Result of a paired t-test.

    Attributes:
        t_statistic: The t-statistic value.
        p_value: Two-tailed p-value.
        significant: Whether p < 0.05.
        degrees_of_freedom: Degrees of freedom (n - 1).
    """

    t_statistic: float
    p_value: float
    significant: bool
    degrees_of_freedom: int


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> TTestResult:
    """Run a paired t-test on per-conversation score differences.

    Quick significance check. Assumes roughly normal score differences.

    Args:
        scores_a: Per-conversation scores for agent A.
        scores_b: Per-conversation scores for agent B.
        alpha: Significance threshold (default 0.05).

    Returns:
        TTestResult with t-statistic, p-value, and significance.

    Raises:
        ValueError: If score lists have different lengths or fewer than 2 pairs.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have equal length: {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 paired observations for t-test")

    t_stat, p_value = scipy_stats.ttest_rel(scores_b, scores_a)
    t_stat = float(t_stat)
    p_value = float(p_value)

    # scipy returns NaN for zero-variance differences; treat as not significant
    if np.isnan(p_value):
        p_value = 1.0
        t_stat = 0.0

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        significant=bool(p_value < alpha),
        degrees_of_freedom=len(scores_a) - 1,
    )


def effect_size_cohens_d(
    scores_a: list[float],
    scores_b: list[float],
) -> float:
    """Compute Cohen's d effect size for paired differences.

    Measures practical significance — is the improvement meaningful?
    Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,
    0.5-0.8 = medium, > 0.8 = large.

    Args:
        scores_a: Per-conversation scores for agent A.
        scores_b: Per-conversation scores for agent B.

    Returns:
        Cohen's d value. Positive means B > A.

    Raises:
        ValueError: If score lists have different lengths.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have equal length: {len(scores_a)} vs {len(scores_b)}"
        )

    differences = np.array(scores_b) - np.array(scores_a)
    mean_diff = float(np.mean(differences))

    if len(differences) < 2:
        return 0.0 if mean_diff == 0 else float("inf")

    std_diff = float(np.std(differences, ddof=1))

    if std_diff == 0:
        return 0.0 if mean_diff == 0 else float("inf")

    return mean_diff / std_diff


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply Bonferroni correction for multiple hypothesis testing.

    When testing across multiple themes/axes simultaneously, adjusts
    the significance threshold to control the family-wise error rate.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Desired family-wise significance level (default 0.05).

    Returns:
        List of (adjusted_p_value, is_significant) tuples.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []

    adjusted_alpha = alpha / n_tests
    return [
        (min(p * n_tests, 1.0), p * n_tests < alpha)
        for p in p_values
    ]
