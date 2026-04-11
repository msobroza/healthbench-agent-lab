"""Meta-evaluation result wrappers, IO helpers, and plot helpers.

Re-exports :class:`MetricResultsView`, :func:`save_results`,
:func:`load_results`, :func:`plot_calibration_curve`, and
:func:`plot_dimension_confusion` from the sibling :mod:`.view`,
:mod:`.io`, and :mod:`.plots` submodules so callers get a single
flat import path.
"""

from __future__ import annotations

from .io import load_results, save_results
from .plots import plot_calibration_curve, plot_dimension_confusion
from .view import MetricResultsView

__all__ = [
    "MetricResultsView",
    "load_results",
    "plot_calibration_curve",
    "plot_dimension_confusion",
    "save_results",
]
