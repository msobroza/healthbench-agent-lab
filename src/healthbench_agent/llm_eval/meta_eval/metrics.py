"""Built-in meta-evaluation metrics.

Each function is registered with :func:`register_meta_metric` so that
importing this module at package load time populates the registry.
Metric functions are pure — they take a filtered DataFrame and return a
scalar (or a nested dict for dict-valued metrics).
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


@register_meta_metric(
    "calibration_curve",
    level=MetricLevel.ANY,
    description="Bootstrap SE of agreement at k = 1, 3, 5, 7",
)
def calibration_curve(verdicts: pd.DataFrame) -> dict[int, float]:
    """Bootstrap SE of per-(prompt_id, rubric_key) agreement at k in {1,3,5,7}.

    Args:
        verdicts: DataFrame of verdict rows. Filtered to ``sample_k <= k``
            for each ``k`` in ``{1, 3, 5, 7}`` and collapsed to per-rubric
            majority votes before computing agreement.

    Returns:
        Mapping of ``k`` to the standard error of per-rubric agreement
        at that k. Empty when ``verdicts`` is empty or every k had fewer
        than two agreement observations.
    """
    import math

    if len(verdicts) == 0:
        return {}

    curve: dict[int, float] = {}
    for k in (1, 3, 5, 7):
        subset = verdicts[verdicts["sample_k"] <= k]
        if len(subset) == 0:
            continue
        grouped = subset.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
        agreements: list[float] = []
        for _, group in grouped:
            majority = bool(group["observed_met"].mean() > 0.5)
            expected = bool(group["expected_met"].iloc[0])
            agreements.append(1.0 if majority == expected else 0.0)
        n = len(agreements)
        if n < 2:
            continue
        mean = sum(agreements) / n
        variance = sum((a - mean) ** 2 for a in agreements) / (n - 1)
        curve[k] = math.sqrt(variance / n)
    return curve


@register_meta_metric(
    "per_dimension_confusion",
    level=MetricLevel.ANY,
    description="tp/fp/tn/fn per dimension (e.g. axis name)",
)
def per_dimension_confusion(verdicts: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Group by ``dimension`` column and return tp/fp/tn/fn per dimension.

    Args:
        verdicts: DataFrame of verdict rows. Must contain ``dimension``,
            ``observed_met`` and ``expected_met`` columns. Null
            ``dimension`` values are bucketed under ``"unspecified"``.

    Returns:
        Mapping of dimension name to a ``{"tp", "fp", "tn", "fn"}``
        confusion-count dict. Empty when ``verdicts`` is empty.
    """
    result: dict[str, dict[str, int]] = {}
    if len(verdicts) == 0:
        return result
    df = verdicts.copy()
    df["dimension"] = df["dimension"].fillna("unspecified")
    for dim, group in df.groupby("dimension", sort=False):
        tp = int(((group["observed_met"]) & (group["expected_met"])).sum())
        fp = int(((group["observed_met"]) & (~group["expected_met"])).sum())
        tn = int(((~group["observed_met"]) & (~group["expected_met"])).sum())
        fn = int(((~group["observed_met"]) & (group["expected_met"])).sum())
        result[str(dim)] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return result


@register_meta_metric(
    "adversarial_accuracy",
    level=MetricLevel.RUBRIC,
    description="Accuracy on example_meets / example_fails pairs",
)
def adversarial_accuracy(verdicts: pd.DataFrame) -> float:
    """Plain accuracy: fraction of rows where observed_met == expected_met.

    Args:
        verdicts: DataFrame of verdict rows filtered to adversarial
            ``example_meets`` / ``example_fails`` gold sources. Must
            contain ``observed_met`` and ``expected_met`` columns.

    Returns:
        Accuracy in ``[0.0, 1.0]``. Returns ``0.0`` when ``verdicts`` is
        empty.
    """
    if len(verdicts) == 0:
        return 0.0
    matches = (verdicts["observed_met"] == verdicts["expected_met"]).sum()
    return float(matches / len(verdicts))


@register_meta_metric(
    "adversarial_prf1",
    level=MetricLevel.RUBRIC,
    description="Precision / recall / F1 on adversarial pairs",
)
def adversarial_prf1(verdicts: pd.DataFrame) -> dict[str, float]:
    """Precision / recall / F1 / support via sklearn.

    Args:
        verdicts: DataFrame of verdict rows filtered to adversarial
            ``example_meets`` / ``example_fails`` gold sources. Must
            contain ``observed_met`` and ``expected_met`` columns.

    Returns:
        Dict with ``precision``, ``recall``, ``f1`` and ``support`` float
        entries. All zero when ``verdicts`` is empty; ``support`` falls
        back to ``len(verdicts)`` when sklearn returns ``None``.
    """
    from sklearn.metrics import precision_recall_fscore_support

    if len(verdicts) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    precision, recall, f1, support = precision_recall_fscore_support(
        verdicts["expected_met"].astype(bool),
        verdicts["observed_met"].astype(bool),
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": float(support if support is not None else len(verdicts)),
    }


@register_meta_metric(
    "per_criterion_metrics",
    level=MetricLevel.RUBRIC,
    description="Per-criterion accuracy / precision / recall / F1",
)
def per_criterion_metrics(verdicts: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Group by rubric_key and return accuracy/precision/recall/f1 per criterion.

    Args:
        verdicts: DataFrame of verdict rows. Must contain ``rubric_key``,
            ``observed_met`` and ``expected_met`` columns.

    Returns:
        Mapping of ``rubric_key`` to a dict with ``accuracy``,
        ``precision``, ``recall`` and ``f1`` float entries. Empty when
        ``verdicts`` is empty.
    """
    from sklearn.metrics import precision_recall_fscore_support

    result: dict[str, dict[str, float]] = {}
    if len(verdicts) == 0:
        return result
    for key, group in verdicts.groupby("rubric_key", sort=False):
        observed = group["observed_met"].astype(bool)
        expected = group["expected_met"].astype(bool)
        accuracy = float((observed == expected).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, observed, average="binary", zero_division=0
        )
        result[str(key)] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result
