"""HealthBench scoring formula.

Implements the per-example and aggregate scoring defined in SPEC.md §3.3,
aligned with simple-evals calculate_score() in healthbench_eval.py.
All functions are pure (no I/O, no side effects) and depend only on types
from `.models`. Higher-level modules (evaluation/, agents/) may call these
but must not be imported here.
"""

from __future__ import annotations

from .data_models import CriterionVerdict, RubricItem, SingleEvalResult


def calculate_score(
    rubric_items: list[RubricItem],
    verdicts: list[CriterionVerdict],
) -> float | None:
    """Compute the HealthBench score for a single conversation.

    Mirrors simple-evals calculate_score(). Applies the formula:
    sum(met points) / sum(max(0, points)) per SPEC.md §3.3.

    Args:
        rubric_items: All rubric items for the conversation.
        verdicts: LLM-judge verdicts, one per rubric item in the same order.

    Returns:
        Score in (-inf, 1.0], or None when total_possible_points is zero
        (no positive criteria exist, e.g. a tag-level subset with only
        penalty criteria). Negative when penalty criteria dominate.
    """
    total_possible_points = sum(item.points for item in rubric_items if item.points > 0)
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        item.points
        for item, verdict in zip(rubric_items, verdicts, strict=True)
        if verdict.criteria_met
    )
    return achieved_points / total_possible_points


def clip_score(score: float) -> float:
    """Clip a per-example score to [0, 1] before aggregation.

    Args:
        score: Raw per-example score, may be negative or greater than 1.

    Returns:
        Score clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, score))


def aggregate_scores(results: list[SingleEvalResult]) -> float:
    """Compute the overall benchmark score as the clipped mean across conversations.

    Each per-example score is clipped to [0, 1] before averaging, following
    the HealthBench aggregation convention. Samples with a None score are skipped.

    Args:
        results: Per-conversation SingleEvalResult instances.

    Returns:
        Mean clipped score in [0.0, 1.0]. Returns 0.0 for an empty list or
        when all scores are None.
    """
    scored = [r.score for r in results if r.score is not None]
    if not scored:
        return 0.0
    return sum(clip_score(s) for s in scored) / len(scored)


def stratified_scores(results: list[SingleEvalResult]) -> dict[str, float]:
    """Aggregate scores grouped by theme and axis labels stored in metrics.

    Each SingleEvalResult.metrics dict maps label → per-sample score. This
    function collects and averages those scores across all results.

    Args:
        results: Per-conversation SingleEvalResult instances.

    Returns:
        Mapping of label to mean clipped score across all conversations that
        carry a score for that label. Labels absent from all results are omitted.
    """
    buckets: dict[str, list[float]] = {}
    for result in results:
        for label, score in result.metrics.items():
            buckets.setdefault(label, []).append(clip_score(score))
    return {label: sum(values) / len(values) for label, values in buckets.items()}
