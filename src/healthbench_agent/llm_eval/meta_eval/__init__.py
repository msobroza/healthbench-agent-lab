"""Meta-evaluation registry, runner, metrics, and result helpers.

Re-exports the public surface of every sub-module so that existing
call sites such as ``from healthbench_agent.llm_eval.meta_eval import X``
keep working after the nested reorganisation. Importing the
:mod:`.metrics` subpackage at load time is what populates the metric
registry via its ``@register_meta_metric`` decorators.
"""

from __future__ import annotations

# Importing metrics for its side-effects (@register_meta_metric decorators).
from . import metrics as _metrics  # noqa: F401
from .api import _build_judge_for_meta_eval, _load_subset_for_meta_eval, meta_evaluate
from .demo_data import demo_labelled_set
from .filters import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    axis_filter,
    metadata_filter,
    specialty_filter,
)
from .metrics import (
    MetricLevel,
    MetricSpec,
    adversarial_accuracy,
    adversarial_prf1,
    calibration_curve,
    cohens_kappa,
    get_meta_metric,
    gold_score,
    krippendorff_alpha,
    per_criterion_metrics,
    per_dimension_confusion,
    register_meta_metric,
    registered_meta_metrics,
)
from .oracle_judge import OracleJudge
from .results import (
    MetricResultsView,
    load_results,
    plot_calibration_curve,
    plot_dimension_confusion,
    save_results,
)
from .runner import run_meta_eval

__all__ = [
    "AXIS_TAG_PREFIX",
    "EmptyFilterError",
    "MetricLevel",
    "MetricResultsView",
    "MetricSpec",
    "OracleJudge",
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
    "load_results",
    "meta_evaluate",
    "metadata_filter",
    "per_criterion_metrics",
    "per_dimension_confusion",
    "plot_calibration_curve",
    "plot_dimension_confusion",
    "register_meta_metric",
    "registered_meta_metrics",
    "run_meta_eval",
    "save_results",
    "specialty_filter",
]
