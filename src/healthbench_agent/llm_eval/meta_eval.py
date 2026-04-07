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
    """Cohen's kappa between judge majority vote and expected verdicts."""
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
    """Bootstrap SE of per-(prompt_id, rubric_key) agreement at k in {1,3,5,7}."""
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
    """Group by ``dimension`` column and return tp/fp/tn/fn per dimension."""
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
    """Plain accuracy: fraction of rows where observed_met == expected_met."""
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
    """Precision / recall / F1 / support via sklearn."""
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
    """Group by rubric_key and return accuracy/precision/recall/f1 per criterion."""
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
