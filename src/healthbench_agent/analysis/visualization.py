"""Visualization analyses for HealthBench datasets.

Implements figure-producing analyses registered under the 'visualization'
category. Each function builds a flat DataFrame early via build_rubric_dataframe
or build_sample_dataframe, then uses pandas pivot/groupby for aggregation and
matplotlib/seaborn for rendering.

Uses matplotlib's OOP interface (matplotlib.figure.Figure) to avoid mutating
global pyplot state inside a library module.

Import chain: .registry → .utils → ..domain.dataset.
Does not import from agents/ or evaluation/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns

from ..domain.dataset import HealthBenchDataset
from .registry import register_analysis
from .utils import (
    DEFAULT_PERCENTILES,
    build_rubric_dataframe,
    build_sample_dataframe,
    series_stats,
)

logger = logging.getLogger(__name__)


def _save_figure(fig: matplotlib.figure.Figure, path: Path) -> None:
    """Save *fig* to *path*, creating parent directories as needed.

    Args:
        fig: Matplotlib Figure to save.
        path: Full output file path (PNG).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info("Saved figure to %s", path)


# ---------------------------------------------------------------------------
# 1. plot_score_ceiling_distribution
# ---------------------------------------------------------------------------


@register_analysis(
    name="score_ceiling_distribution",
    category="visualization",
    datasets=["main", "hard", "consensus"],
)
def plot_score_ceiling_distribution(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot the distribution of total_possible_points per sample per subset.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one histogram PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with stats and optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)
        values = df["total_possible_points"]

        subset_result: dict[str, Any] = {
            "stats": series_stats(values, percentiles=DEFAULT_PERCENTILES),
        }

        if save:
            fig = matplotlib.figure.Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.hist(values, bins=30, edgecolor="white", linewidth=0.4)
            ax.set_title(f"Score Ceiling Distribution — {subset}")
            ax.set_xlabel("Total Possible Points")
            ax.set_ylabel("Number of Samples")
            path = output_dir / f"score_ceiling_distribution_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: score ceiling distribution — mean=%.2f",
            subset,
            subset_result["stats"]["mean"],
        )

    return results


# ---------------------------------------------------------------------------
# 2. plot_theme_axis_heatmap
# ---------------------------------------------------------------------------


@register_analysis(
    name="theme_axis_heatmap",
    category="visualization",
    datasets=["main", "hard", "consensus"],
)
def plot_theme_axis_heatmap(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot a heatmap of rubric item counts by theme × axis per subset.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one heatmap PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with cross_table, themes, axes, and
        optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)
        pivot = df.groupby(["theme", "axis"]).size().unstack(fill_value=0)

        subset_result: dict[str, Any] = {
            "cross_table": pivot.to_dict(orient="index"),
            "themes": sorted(pivot.index.tolist()),
            "axes": sorted(pivot.columns.tolist()),
        }

        if save:
            fig = matplotlib.figure.Figure(
                figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.6))
            )
            ax = fig.add_subplot(111)
            sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
            ax.set_title(f"Theme × Axis Rubric Item Counts — {subset}")
            ax.set_xlabel("Axis")
            ax.set_ylabel("Theme")
            fig.tight_layout()
            path = output_dir / f"theme_axis_heatmap_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: theme × axis heatmap — %s themes × %s axes",
            subset,
            len(subset_result["themes"]),
            len(subset_result["axes"]),
        )

    return results


# ---------------------------------------------------------------------------
# 3. plot_criteria_weight_histogram
# ---------------------------------------------------------------------------


@register_analysis(
    name="criteria_weight_histogram",
    category="visualization",
    datasets=["main", "hard", "consensus"],
)
def plot_criteria_weight_histogram(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot the distribution of rubric item point values per subset.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one histogram PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with points_value_counts, stats, and
        optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)
        points = df["points"]

        subset_result: dict[str, Any] = {
            "points_value_counts": {
                float(cast(float, k)): int(v)
                for k, v in points.round(1).value_counts().sort_index().items()
            },
            "stats": series_stats(points, percentiles=DEFAULT_PERCENTILES),
        }

        if save:
            fig = matplotlib.figure.Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.hist(points, bins=40, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"Criteria Weight Distribution — {subset}")
            ax.set_xlabel("Points")
            ax.set_ylabel("Number of Rubric Items")
            path = output_dir / f"criteria_weight_histogram_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: criteria weight histogram — %s rubric items",
            subset,
            int(subset_result["stats"]["count"]),
        )

    return results


# ---------------------------------------------------------------------------
# 4. plot_penalty_heatmap
# ---------------------------------------------------------------------------


@register_analysis(
    name="penalty_heatmap",
    category="visualization",
    datasets=["main", "hard"],
)
def plot_penalty_heatmap(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot a heatmap of mean penalty points per rubric item by theme × axis.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one heatmap PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with penalty_table, themes, axes, and
        optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)
        neg = df[df["points"] < 0].copy()
        neg["penalty"] = neg["points"].abs()

        if neg.empty:
            results[subset] = {}
            continue

        pivot = (
            neg.pivot_table(
                values="penalty",
                index="theme",
                columns="axis",
                aggfunc="mean",
            )
            .fillna(0)
            .round(2)
        )

        subset_result: dict[str, Any] = {
            "penalty_table": pivot.to_dict(orient="index"),
            "themes": sorted(pivot.index.tolist()),
            "axes": sorted(pivot.columns.tolist()),
        }

        if save:
            fig = matplotlib.figure.Figure(
                figsize=(max(6, len(pivot.columns) * 1.4), max(4, len(pivot) * 0.7))
            )
            ax = fig.add_subplot(111)
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", linewidths=0.5, ax=ax)
            ax.set_title(f"Mean Penalty per Rubric Item (theme × axis) — {subset}")
            ax.set_xlabel("Axis")
            ax.set_ylabel("Theme")
            fig.tight_layout()
            path = output_dir / f"penalty_heatmap_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: penalty heatmap — %s themes × %s axes",
            subset,
            len(subset_result["themes"]),
            len(subset_result["axes"]),
        )

    return results


# ---------------------------------------------------------------------------
# 5. plot_multiturn_complexity_comparison
# ---------------------------------------------------------------------------


@register_analysis(
    name="multiturn_complexity_comparison",
    category="visualization",
    datasets=["main", "hard"],
)
def plot_multiturn_complexity_comparison(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compare rubric complexity between single-turn and multi-turn prompts.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one box plot PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with single_turn/multi_turn stats and
        optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)
        df["turn_type"] = df["num_user_turns"].apply(
            lambda x: "single_turn" if x <= 1 else "multi_turn"
        )
        df["penalty_mass_ratio"] = df["penalty_mass_ratio"].fillna(0.0)

        metrics = ["rubric_size", "total_possible_points", "penalty_mass_ratio"]

        group_stats: dict[str, dict[str, Any]] = {}
        for turn_type, group in df.groupby("turn_type"):
            group_stats[str(turn_type)] = {
                m: series_stats(group[m], percentiles=DEFAULT_PERCENTILES) for m in metrics
            }

        counts = df["turn_type"].value_counts().to_dict()
        subset_result: dict[str, Any] = {
            "single_turn": group_stats.get("single_turn", {}),
            "multi_turn": group_stats.get("multi_turn", {}),
            "single_turn_count": counts.get("single_turn", 0),
            "multi_turn_count": counts.get("multi_turn", 0),
        }

        if save:
            fig = matplotlib.figure.Figure(figsize=(10, 4))
            for idx, metric in enumerate(metrics):
                ax = fig.add_subplot(1, 3, idx + 1)
                single = df.loc[df["turn_type"] == "single_turn", metric]
                multi = df.loc[df["turn_type"] == "multi_turn", metric]
                ax.boxplot([single, multi], tick_labels=["Single", "Multi"], widths=0.5)
                ax.set_title(metric.replace("_", " ").title())
                ax.set_ylabel(metric)
            fig.suptitle(f"Single-Turn vs Multi-Turn Complexity — {subset}", y=1.02)
            fig.tight_layout()
            path = output_dir / f"multiturn_complexity_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: multiturn comparison — single=%s, multi=%s",
            subset,
            subset_result["single_turn_count"],
            subset_result["multi_turn_count"],
        )

    return results


# ---------------------------------------------------------------------------
# 6. plot_penalty_ratio_cdf
# ---------------------------------------------------------------------------


@register_analysis(
    name="penalty_ratio_cdf",
    category="visualization",
    datasets=["main", "hard"],
)
def plot_penalty_ratio_cdf(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot overlaid CDFs of penalty_mass_ratio for main vs hard subsets.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where the overlay PNG is saved when save=True.
        save: When True, write one overlay CDF PNG to output_dir.

    Returns:
        Dict with key 'overlay' containing per_subset stats and optional paths.
    """
    per_subset: dict[str, dict[str, Any]] = {}
    ratio_series: dict[str, pd.Series] = {}

    for dataset in datasets:
        if not dataset.samples:
            continue

        df = build_sample_dataframe(dataset)
        ratios = df["penalty_mass_ratio"].dropna()

        if ratios.empty:
            continue

        ratio_series[dataset.subset] = ratios
        per_subset[dataset.subset] = series_stats(ratios, percentiles=DEFAULT_PERCENTILES)

    if not per_subset:
        return {}

    result: dict[str, Any] = {"overlay": {"per_subset": per_subset}}

    if save:
        fig = matplotlib.figure.Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for subset, series in sorted(ratio_series.items()):
            sorted_vals = np.sort(np.asarray(series.values))
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, label=subset, linewidth=1.5)
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", label="ratio = 1.0")
        ax.set_title("Penalty Mass Ratio CDF — Main vs Hard")
        ax.set_xlabel("Penalty Mass Ratio (abs penalty / positive points)")
        ax.set_ylabel("Cumulative Fraction of Samples")
        ax.legend()
        ax.set_xlim(left=0)
        fig.tight_layout()
        path = output_dir / "penalty_ratio_cdf_overlay.png"
        _save_figure(fig, path)
        result["overlay"]["paths"] = [str(path)]

    logger.info("Penalty ratio CDF — subsets: %s", list(per_subset.keys()))

    return result


# ---------------------------------------------------------------------------
# 7. plot_positive_share_by_axis
# ---------------------------------------------------------------------------


@register_analysis(
    name="positive_share_by_axis",
    category="visualization",
    datasets=["main", "hard"],
)
def plot_positive_share_by_axis(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot a diverging bar chart of positive vs penalty share per axis.

    positive_share = sum(positive points) / sum(abs(all points)) for each axis.
    Bars extend right for reward share and left for penalty share.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one diverging bar chart per subset to output_dir.

    Returns:
        Dict keyed by subset label with positive_share and penalty_share
        per axis, and optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)

        pos_by_axis = df[df["points"] > 0].groupby("axis")["points"].sum()
        total_by_axis = df["points"].abs().groupby(df["axis"]).sum()
        positive_share = (pos_by_axis / total_by_axis).sort_values().round(4)
        penalty_share = (1 - positive_share).round(4)

        subset_result: dict[str, Any] = {
            "positive_share": positive_share.to_dict(),
            "penalty_share": penalty_share.to_dict(),
        }

        if save:
            fig = matplotlib.figure.Figure(figsize=(8, max(3, len(positive_share) * 0.6)))
            ax = fig.add_subplot(111)
            ax.barh(positive_share.index, positive_share, color="steelblue", label="Positive share")
            ax.barh(positive_share.index, -penalty_share, color="salmon", label="Penalty share")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("\u2190 Penalty share | Positive share \u2192")
            ax.set_title(f"Reward vs Penalty Balance by Axis — {subset}")
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"positive_share_by_axis_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: positive share by axis — %s axes",
            subset,
            len(positive_share),
        )

    return results


# ---------------------------------------------------------------------------
# 8. plot_positive_share_heatmap
# ---------------------------------------------------------------------------


@register_analysis(
    name="positive_share_heatmap",
    category="visualization",
    datasets=["main", "hard"],
)
def plot_positive_share_heatmap(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot a heatmap of positive share by theme × axis.

    Each cell = sum(positive points) / sum(abs(all points)) for that
    (theme, axis) combination. Green = reward-heavy, red = penalty-heavy.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where PNG figures are saved when save=True.
        save: When True, write one heatmap PNG per subset to output_dir.

    Returns:
        Dict keyed by subset label with share_table, themes, axes, and
        optional paths.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)

        pivot = df.pivot_table(
            values="points",
            index="theme",
            columns="axis",
            aggfunc=lambda x: x[x > 0].sum() / x.abs().sum() if x.abs().sum() > 0 else 0,
        ).round(3)

        subset_result: dict[str, Any] = {
            "share_table": pivot.to_dict(orient="index"),
            "themes": sorted(pivot.index.tolist()),
            "axes": sorted(pivot.columns.tolist()),
        }

        if save:
            fig = matplotlib.figure.Figure(
                figsize=(max(6, len(pivot.columns) * 1.4), max(4, len(pivot) * 0.7))
            )
            ax = fig.add_subplot(111)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0.5,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title(f"Positive Share (theme × axis) — {subset}")
            ax.set_xlabel("Axis")
            ax.set_ylabel("Theme")
            fig.tight_layout()
            path = output_dir / f"positive_share_heatmap_{subset}.png"
            _save_figure(fig, path)
            subset_result["paths"] = [str(path)]

        results[subset] = subset_result
        logger.info(
            "Subset %s: positive share heatmap — %s themes × %s axes",
            subset,
            len(subset_result["themes"]),
            len(subset_result["axes"]),
        )

    return results
