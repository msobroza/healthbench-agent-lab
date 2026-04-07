"""Meta-evaluation registry, runner, and built-in metrics for LLM-as-judge.

The module is dataset-agnostic: it operates on lists of LabelledSample.
HealthBench-specific glue (subset loading, ideal completion extraction)
lives in ``cli_meta_eval.py``.

Adding a new metric is one decorated function — name, level, description,
and the pure DataFrame transform. Zero changes anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import TYPE_CHECKING, Any, cast

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.scoring import calculate_score, clip_score

if TYPE_CHECKING:
    import pandas as pd


class MetricLevel(StrEnum):
    """Which row subset a metric operates on."""

    SAMPLE = "sample"
    RUBRIC = "rubric"
    ANY = "any"


Metric = Callable[["pd.DataFrame"], Any]


@dataclass(frozen=True)
class MetricSpec:
    """Registered metric metadata.

    Attributes:
        name: Unique identifier (used by --metrics CLI flag and metrics.json).
        fn: The pure metric function. Takes a level-filtered DataFrame.
        level: Which gold_source rows the runner passes to ``fn``.
        description: One-line human-readable summary for ``list-metrics``.
    """

    name: str
    fn: Metric
    level: MetricLevel
    description: str


_METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_meta_metric(
    name: str,
    *,
    level: MetricLevel,
    description: str,
) -> Callable[[Metric], Metric]:
    """Decorator that registers a meta-evaluation metric by name + level."""

    def decorator(fn: Metric) -> Metric:
        _METRIC_REGISTRY[name] = MetricSpec(name=name, fn=fn, level=level, description=description)
        return fn

    return decorator


def get_meta_metric(name: str) -> MetricSpec:
    """Look up a registered metric by name.

    Raises:
        KeyError: When ``name`` is not registered.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric {name!r} is not registered. Available: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def registered_meta_metrics() -> dict[str, MetricSpec]:
    """Return the live registry mapping (mutating it affects the registry)."""
    return _METRIC_REGISTRY


AXIS_TAG_PREFIX = "axis: "
"""Prefix used by HealthBench's stratified-sample helper for axis tags.

The trailing space matches the form written by
``healthbench_agent.dataset.split_utils._extract_stratum``. Shared with
the CLI's ``axis_extractor`` so the two helpers cannot drift.
"""


class EmptyFilterError(ValueError):
    """Raised when a sample/rubric filter combination eliminates all rows.

    Carries the names (or repr) of the active filters so the CLI can show
    the user exactly which flags caused the empty result.
    """

    def __init__(self, sample_filter: Any, rubric_filter: Any) -> None:
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter
        super().__init__(
            f"Empty filter result. sample_filter={sample_filter!r}, rubric_filter={rubric_filter!r}"
        )


def axis_filter(*axes: str) -> Callable[[RubricItem], bool]:
    """Keep rubrics whose ``category`` or ``axis: <name>`` tag matches any of *axes*."""
    wanted = set(axes)

    def predicate(item: RubricItem) -> bool:
        if item.category in wanted:
            return True
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX) and tag[len(AXIS_TAG_PREFIX) :].strip() in wanted:
                return True
        return False

    return predicate


def metadata_filter(**conditions: Any) -> Callable[[LabelledSample], bool]:
    """Keep samples where every metadata key equals the given value.

    Top-level LabelledSample attributes (``language``, ``specialty``,
    ``user_persona``) are checked against the attribute; other keys are
    looked up in ``sample.metadata``.
    """
    top_level = {"language", "specialty", "user_persona"}

    def predicate(sample: LabelledSample) -> bool:
        for key, value in conditions.items():
            if key in top_level:
                if getattr(sample, key) != value:
                    return False
            else:
                if sample.metadata.get(key) != value:
                    return False
        return True

    return predicate


def specialty_filter(*specialties: str) -> Callable[[LabelledSample], bool]:
    """Keep samples whose ``specialty`` field is in *specialties*."""
    wanted = set(specialties)

    def predicate(sample: LabelledSample) -> bool:
        return sample.specialty in wanted

    return predicate


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
