"""Built-in meta-evaluation metrics, grouped by category.

This package re-exports the metric registry (``MetricLevel``,
``MetricSpec``, ``register_meta_metric``, ``get_meta_metric``,
``registered_meta_metrics``) and every registered metric function. The
category submodules (:mod:`.agreement`, :mod:`.stratified`,
:mod:`.adversarial`) are imported at package load time for their
``@register_meta_metric`` decorator side-effects.
"""

from __future__ import annotations

# Importing the category modules for their @register_meta_metric side-effects
# is what populates the metric registry at package load time.
from . import adversarial as _adversarial  # noqa: F401
from . import agreement as _agreement  # noqa: F401
from . import stratified as _stratified  # noqa: F401
from .adversarial import adversarial_accuracy, adversarial_prf1
from .agreement import cohens_kappa, gold_score, krippendorff_alpha
from .registry import (
    MetricLevel,
    MetricSpec,
    get_meta_metric,
    register_meta_metric,
    registered_meta_metrics,
)
from .stratified import calibration_curve, per_criterion_metrics, per_dimension_confusion

__all__ = [
    "MetricLevel",
    "MetricSpec",
    "adversarial_accuracy",
    "adversarial_prf1",
    "calibration_curve",
    "cohens_kappa",
    "get_meta_metric",
    "gold_score",
    "krippendorff_alpha",
    "per_criterion_metrics",
    "per_dimension_confusion",
    "register_meta_metric",
    "registered_meta_metrics",
]
