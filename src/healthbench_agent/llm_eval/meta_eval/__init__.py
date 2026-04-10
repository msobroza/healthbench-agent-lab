"""Meta-evaluation registry, runner, and built-in metrics for LLM-as-judge.

Re-exports the public surface of the sub-modules so that existing call
sites such as ``from healthbench_agent.llm_eval.meta_eval import X``
keep working after the split. Importing :mod:`.metrics` at package load
time is required to populate the metric registry via its
``@register_meta_metric`` decorators.
"""

from __future__ import annotations

# Importing metrics for its side-effects (@register_meta_metric decorators).
from . import metrics as _metrics  # noqa: F401
from .api import _build_judge_for_meta_eval, _load_subset_for_meta_eval, meta_evaluate
from .filters import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    axis_filter,
    metadata_filter,
    specialty_filter,
)
from .fixtures import FakeJudge, demo_labelled_set
from .metrics import (
    adversarial_accuracy,
    adversarial_prf1,
    calibration_curve,
    cohens_kappa,
    gold_score,
    krippendorff_alpha,
    per_criterion_metrics,
    per_dimension_confusion,
)
from .registry import (
    MetricLevel,
    MetricSpec,
    get_meta_metric,
    register_meta_metric,
    registered_meta_metrics,
)
from .results_view import MetricResultsView
from .runner import run_meta_eval

__all__ = [
    "AXIS_TAG_PREFIX",
    "EmptyFilterError",
    "FakeJudge",
    "MetricLevel",
    "MetricResultsView",
    "MetricSpec",
    "_build_judge_for_meta_eval",
    "_load_subset_for_meta_eval",
    "adversarial_accuracy",
    "adversarial_prf1",
    "axis_filter",
    "calibration_curve",
    "cohens_kappa",
    "demo_labelled_set",
    "get_meta_metric",
    "gold_score",
    "krippendorff_alpha",
    "meta_evaluate",
    "metadata_filter",
    "per_criterion_metrics",
    "per_dimension_confusion",
    "register_meta_metric",
    "registered_meta_metrics",
    "run_meta_eval",
    "specialty_filter",
]
