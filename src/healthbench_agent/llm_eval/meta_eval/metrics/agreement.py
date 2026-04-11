"""Agreement meta-evaluation metrics.

Metrics in this category measure how closely a judge's verdicts agree
with the expected (gold) labels: overall HealthBench gold score, Cohen's
kappa inter-rater agreement, and binary Krippendorff's alpha. Each
function is pure and takes a filtered verdict DataFrame.
"""

from __future__ import annotations

from statistics import fmean
from typing import TYPE_CHECKING, cast

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.scoring import calculate_score, clip_score

from .registry import MetricLevel, register_meta_metric

if TYPE_CHECKING:
    import pandas as pd


@register_meta_metric(
    "gold_score",
    level=MetricLevel.SAMPLE,
    description="Mean clipped HealthBench score on gold responses (target = 1.0)",
)
def gold_score(verdicts: pd.DataFrame) -> float:
    """Mean clipped HealthBench score the judge gives to gold responses.

    Rebuilds (rubrics, verdicts) lists per (prompt_id, sample_k) group
    and delegates to ``calculate_score`` + ``clip_score`` so meta-eval
    cannot drift from production scoring.

    Args:
        verdicts: DataFrame of verdict rows filtered to
            ``gold_source == "ideal_completion"``. Must contain
            ``prompt_id``, ``sample_k``, ``criterion``, ``points`` and
            ``observed_met`` columns.

    Returns:
        Mean clipped score in ``(-inf, 1.0]`` across every
        (prompt_id, sample_k) group. ``0.0`` when ``verdicts`` is empty
        or every group had ``calculate_score`` return ``None``.
    """
    if len(verdicts) == 0:
        return 0.0

    per_sample_scores: list[float] = []
    for _, group in verdicts.groupby(["prompt_id", "sample_k"], sort=False):
        rubric_items = [
            RubricItem(
                criterion=str(row.criterion),
                points=float(cast(float, row.points)),
                tags=[],
            )
            for row in group.itertuples(index=False)
        ]
        criterion_verdicts = [
            CriterionVerdict(
                criterion=str(row.criterion),
                criteria_met=bool(row.observed_met),
                explanation="",
            )
            for row in group.itertuples(index=False)
        ]
        raw = calculate_score(rubric_items, criterion_verdicts)
        if raw is None:
            continue
        per_sample_scores.append(clip_score(raw))

    return fmean(per_sample_scores) if per_sample_scores else 0.0


def _majority_vote_columns(df: pd.DataFrame) -> tuple[list[bool], list[bool]]:
    """Collapse k passes per (prompt_id, rubric_key, gold_source) to majority vote."""
    if len(df) == 0:
        return [], []
    grouped = df.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
    observed: list[bool] = []
    expected: list[bool] = []
    for _, group in grouped:
        observed.append(bool(group["observed_met"].mean() > 0.5))
        expected.append(bool(group["expected_met"].iloc[0]))
    return observed, expected


@register_meta_metric(
    "cohens_kappa",
    level=MetricLevel.ANY,
    description="Inter-rater agreement vs expected verdicts",
)
def cohens_kappa(verdicts: pd.DataFrame) -> float:
    """Cohen's kappa between judge majority vote and expected verdicts.

    Args:
        verdicts: DataFrame of verdict rows. Collapsed to per-(prompt_id,
            rubric_key, gold_source) majority votes before scoring.

    Returns:
        Cohen's kappa in ``[-1.0, 1.0]``. Returns ``0.0`` when
        ``verdicts`` is empty.
    """
    from sklearn.metrics import cohen_kappa_score

    observed, expected = _majority_vote_columns(verdicts)
    if not observed:
        return 0.0
    return float(cohen_kappa_score(expected, observed))


@register_meta_metric(
    "krippendorff_alpha",
    level=MetricLevel.ANY,
    description="Binary two-coder Krippendorff alpha",
)
def krippendorff_alpha(verdicts: pd.DataFrame) -> float:
    """Closed-form Krippendorff's alpha for binary, two-coder data.

    α = 1 - D_o / D_e where D_o is the observed disagreement (Hamming
    distance) and D_e is the expected disagreement under chance.

    Args:
        verdicts: DataFrame of verdict rows. Collapsed to per-(prompt_id,
            rubric_key, gold_source) majority votes before scoring.

    Returns:
        Alpha in ``(-inf, 1.0]``. Returns ``0.0`` when ``verdicts`` is
        empty; returns ``1.0`` when observed and expected agree perfectly
        even if the expected-disagreement denominator is zero.
    """
    observed, expected = _majority_vote_columns(verdicts)
    n = len(observed)
    if n == 0:
        return 0.0
    disagreements = sum(1 for o, e in zip(observed, expected, strict=True) if o != e)
    do_metric = disagreements / n
    p1 = (sum(observed) + sum(expected)) / (2 * n)
    de_metric = 2 * p1 * (1 - p1)
    if de_metric == 0:
        return 1.0 if do_metric == 0 else 0.0
    return float(1.0 - do_metric / de_metric)
