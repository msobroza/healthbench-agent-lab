"""Stratified meta-evaluation metrics.

Metrics in this category break down judge performance along a stratum
(dimension, rubric, or number of judge passes) rather than reporting a
single scalar: per-dimension confusion counts, per-criterion precision
/ recall / F1, and the bootstrap-SE calibration curve over k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import MetricLevel, register_meta_metric

if TYPE_CHECKING:
    import pandas as pd


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
