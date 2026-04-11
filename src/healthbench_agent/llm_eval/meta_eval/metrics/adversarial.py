"""Adversarial meta-evaluation metrics.

Metrics in this category operate on verdict rows sourced from a
rubric's ``example_meets`` / ``example_fails`` adversarial pairs — the
known-good and known-bad responses embedded in the rubric itself. They
measure how well the judge separates the two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import MetricLevel, register_meta_metric

if TYPE_CHECKING:
    import pandas as pd


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
