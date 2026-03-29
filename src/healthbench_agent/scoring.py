"""HealthBench scoring formula.

Implements the per-example and aggregate scoring defined in SPEC.md §3.3.
All functions are pure (no I/O, no side effects) and depend only on types
from `.types`. Higher-level modules (evaluation/, agents/) may call these
but must not be imported here.
"""

from __future__ import annotations

from .models import CriterionVerdict, EvalResult, RubricCriterion


def criterion_score(
    criteria: list[RubricCriterion],
    verdicts: list[CriterionVerdict],
) -> float:
    """Compute the HealthBench score for a single conversation.

    Applies the formula: sum(met weights) / sum(max(0, weight)) per SPEC.md §3.3.
    Criteria with no matching verdict are treated as not met.

    Args:
        criteria: All rubric criteria for the conversation.
        verdicts: LLM-judge verdicts, one per criterion.

    Returns:
        Score in (-inf, 1.0]. Returns 0.0 when max_points is zero (no positive
        criteria exist). Negative when penalty criteria dominate.
    """
    verdict_map = {v.criterion_id: v.met for v in verdicts}
    met_points = sum(c.weight for c in criteria if verdict_map.get(c.criterion_id, False))
    max_points = sum(max(0.0, c.weight) for c in criteria)
    if max_points == 0.0:
        return 0.0
    return met_points / max_points


def clip_score(score: float) -> float:
    """Clip a per-example score to [0, 1] before aggregation.

    Args:
        score: Raw per-example score, may be negative or greater than 1.

    Returns:
        Score clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, score))


def aggregate_scores(results: list[EvalResult]) -> float:
    """Compute the overall benchmark score as the clipped mean across conversations.

    Each per-example score is clipped to [0, 1] before averaging, following
    the HealthBench aggregation convention.

    Args:
        results: Evaluated conversations, one result per conversation.

    Returns:
        Mean clipped score in [0.0, 1.0]. Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    return sum(clip_score(r.score) for r in results) / len(results)


def stratified_scores(
    results: list[EvalResult],
    dimension: str,
) -> dict[str, float]:
    """Aggregate scores grouped by a stratification dimension.

    Args:
        results: Evaluated conversations.
        dimension: Attribute name on `EvalResult` holding the per-label score
            mapping. Typically 'theme_scores' or 'axis_scores'.

    Returns:
        Mapping of category label to mean clipped score across all conversations
        that have a score for that label. Labels absent from all results are
        omitted from the output.
    """
    buckets: dict[str, list[float]] = {}
    for result in results:
        scores = getattr(result, dimension, {})
        for label, score in scores.items():
            buckets.setdefault(label, []).append(clip_score(score))
    return {label: sum(values) / len(values) for label, values in buckets.items()}
