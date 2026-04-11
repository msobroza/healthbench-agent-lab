"""Matplotlib plot helpers for :class:`MetricResults` scores.

Kept as free functions so callers can render plots from a raw
:class:`MetricResults` without constructing a :class:`MetricResultsView`
first, and so the view layer does not hard-depend on matplotlib.
"""

from __future__ import annotations

from typing import Any

from healthbench_agent.domain.meta_evaluation import MetricResults


def plot_calibration_curve(results: MetricResults, ax: Any = None) -> Any:
    """Plot the bootstrap SE calibration curve over k.

    Args:
        results: Pure-domain metric results containing the
            ``calibration_curve`` score.
        ax: Existing matplotlib Axes to draw into. When None, a new
            figure and axes are created via ``plt.subplots()``.

    Note:
        When ``ax`` is None, this creates a new matplotlib Figure and
        registers it with pyplot's global state. The Figure is
        accessible via ``ax.figure``; long-running callers should call
        ``plt.close(ax.figure)`` (or ``plt.close('all')``) to release it.
        To avoid this entirely, pass an existing ``ax`` from your own
        ``plt.subplots()`` call.

    Returns:
        The matplotlib Axes used for plotting.

    Raises:
        KeyError: When ``calibration_curve`` is not in ``results.scores``.
        ImportError: When matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "Plotting requires matplotlib. Install with: uv sync --extra viz"
        ) from exc

    if "calibration_curve" not in results.scores:
        raise KeyError("calibration_curve is not in results.scores")
    curve = results.scores["calibration_curve"]
    if ax is None:
        _, ax = plt.subplots()
    ks_raw = list(curve.keys())
    try:
        ks = sorted(ks_raw, key=lambda k: int(k))
    except (TypeError, ValueError):
        ks = sorted(ks_raw)
    ax.plot(ks, [curve[k] for k in ks], marker="o")
    ax.set_xlabel("k (number of judge passes)")
    ax.set_ylabel("Bootstrap SE of agreement")
    ax.set_title("Calibration curve")
    return ax


def plot_dimension_confusion(results: MetricResults, ax: Any = None) -> Any:
    """Stacked bar chart of per-dimension confusion counts.

    Args:
        results: Pure-domain metric results containing the
            ``per_dimension_confusion`` score.
        ax: Existing matplotlib Axes to draw into. When None, a new
            figure and axes are created via ``plt.subplots()``.

    Note:
        When ``ax`` is None, this creates a new matplotlib Figure and
        registers it with pyplot's global state. The Figure is
        accessible via ``ax.figure``; long-running callers should call
        ``plt.close(ax.figure)`` (or ``plt.close('all')``) to release it.
        To avoid this entirely, pass an existing ``ax`` from your own
        ``plt.subplots()`` call.

    Returns:
        The matplotlib Axes used for plotting.

    Raises:
        KeyError: When ``per_dimension_confusion`` is not in
            ``results.scores``.
        ImportError: When matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with: uv sync --extra viz"
        ) from exc

    if "per_dimension_confusion" not in results.scores:
        raise KeyError("per_dimension_confusion is not in results.scores")
    confusion = results.scores["per_dimension_confusion"]
    if ax is None:
        _, ax = plt.subplots()
    dims = list(confusion.keys())
    tp = [confusion[d]["tp"] for d in dims]
    fp = [confusion[d]["fp"] for d in dims]
    tn = [confusion[d]["tn"] for d in dims]
    fn = [confusion[d]["fn"] for d in dims]
    ax.bar(dims, tp, label="tp")
    ax.bar(dims, fp, bottom=tp, label="fp")
    ax.bar(dims, tn, bottom=[a + b for a, b in zip(tp, fp, strict=True)], label="tn")
    ax.bar(
        dims,
        fn,
        bottom=[a + b + c for a, b, c in zip(tp, fp, tn, strict=True)],
        label="fn",
    )
    ax.legend()
    ax.set_ylabel("count")
    ax.set_title("Per-dimension confusion")
    return ax
