"""Cross-dimensional insights analyses for HealthBench datasets.

Implements cross-dimensional breakdowns (theme × axis, rubric axis difficulty,
penalty concentration by theme, criterion text length, score ceiling by theme,
emergency axis profile, hard subset signature, prompt-length vs rubric
complexity) registered under the 'insights' category.

Each function builds a flat DataFrame early via build_rubric_dataframe or
build_sample_dataframe, then applies pandas groupby/pivot/agg operations.

Import chain: .registry → .utils → ..domain.dataset.
Does not import from agents/ or evaluation/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

from ..domain.dataset import HealthBenchDataset
from .registry import register_analysis
from .utils import (
    DEFAULT_PERCENTILES,
    build_rubric_dataframe,
    build_sample_dataframe,
    save_csv,
    series_stats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. compute_theme_axis_breakdown
# ---------------------------------------------------------------------------


@register_analysis(
    name="theme_axis_breakdown",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_theme_axis_breakdown(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Cross-tabulate samples by theme and axis example tags per subset.

    Reads the 'theme' and 'axis' prefixes from each sample's example_tags.
    Samples missing one or both dimensions are counted under '_unknown'.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset cross-tabulation CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        cross_table (theme → {axis → count}), theme_counts, axis_counts,
        total_samples, missing_theme_count, missing_axis_count.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)
        rdf = build_rubric_dataframe(dataset)
        cross_pivot = rdf.groupby(["theme", "axis"]).size().unstack(fill_value=0)
        cross_table = cross_pivot.to_dict(orient="index")

        subset_result: dict[str, Any] = {
            "cross_table": cross_table,
            "theme_counts": df["theme"].value_counts().to_dict(),
            "axis_counts": rdf["axis"].value_counts().to_dict(),
            "total_samples": len(df),
            "missing_theme_count": int((df["theme"] == "_unknown").sum()),
            "missing_axis_count": int((rdf["axis"] == "_unknown").sum()),
        }

        if save:
            flat = [
                {"theme": theme, "axis": axis, "count": count}
                for theme, axis_map in cross_table.items()
                for axis, count in axis_map.items()
            ]
            save_csv(
                pd.DataFrame(flat),
                output_dir / f"theme_axis_breakdown_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: theme × axis — %s themes, %s axes",
            subset,
            len(subset_result["theme_counts"]),
            len(subset_result["axis_counts"]),
        )

    return results


# ---------------------------------------------------------------------------
# 2. compute_rubric_axis_difficulty
# ---------------------------------------------------------------------------


@register_analysis(
    name="rubric_axis_difficulty",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_rubric_axis_difficulty(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse rubric point distributions grouped by rubric-item axis tag.

    For each axis value found in rubric-item tags, reports count, points
    descriptive stats, and penalty/positive fractions.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset axis difficulty CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        axis_stats ({axis → {count, points_stats, penalty_fraction,
        positive_fraction}}), total_rubric_items, axes_found.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)

        axis_stats: dict[str, Any] = {}
        for axis_label, group in df.groupby("axis"):
            pts = group["points"]
            n = len(pts)
            axis_stats[str(axis_label)] = {
                "count": n,
                "points_stats": series_stats(pts, percentiles=DEFAULT_PERCENTILES),
                "penalty_fraction": float((pts < 0).sum()) / n,
                "positive_fraction": float((pts > 0).sum()) / n,
            }

        subset_result: dict[str, Any] = {
            "axis_stats": axis_stats,
            "total_rubric_items": len(df),
            "axes_found": sorted(axis_stats.keys()),
        }

        if save:
            flat = [
                {
                    "axis": axis,
                    "count": s["count"],
                    "penalty_fraction": s["penalty_fraction"],
                    "positive_fraction": s["positive_fraction"],
                }
                for axis, s in axis_stats.items()
            ]
            save_csv(
                pd.DataFrame(flat),
                output_dir / f"rubric_axis_difficulty_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: axis difficulty — %s axes across %s rubric items",
            subset,
            len(axis_stats),
            len(df),
        )

    return results


# ---------------------------------------------------------------------------
# 3. compute_penalty_concentration
# ---------------------------------------------------------------------------


@register_analysis(
    name="penalty_concentration",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_penalty_concentration(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Identify which themes concentrate the highest penalty risk per subset.

    For each theme computes total and mean penalty mass per sample.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset penalty concentration CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        theme_total_penalty, theme_mean_penalty,
        total_penalty_points, total_positive_points,
        penalty_to_positive_ratio.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)

        total_penalty = float(df["total_penalty_points"].sum())
        total_positive = float(df["total_possible_points"].sum())

        subset_result: dict[str, Any] = {
            "theme_total_penalty": df.groupby("theme")["total_penalty_points"]
            .sum()
            .round(4)
            .to_dict(),
            "theme_mean_penalty": df.groupby("theme")["total_penalty_points"]
            .mean()
            .round(4)
            .to_dict(),
            "total_penalty_points": round(total_penalty, 4),
            "total_positive_points": round(total_positive, 4),
            "penalty_to_positive_ratio": (
                round(total_penalty / total_positive, 4) if total_positive else None
            ),
        }

        if save:
            save_csv(
                df.groupby("theme")[["total_penalty_points", "total_possible_points"]]
                .agg(["sum", "mean"])
                .round(4)
                .reset_index(),
                output_dir / f"penalty_concentration_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: penalty concentration — total_penalty=%.1f, ratio=%s",
            subset,
            total_penalty,
            subset_result["penalty_to_positive_ratio"],
        )

    return results


# ---------------------------------------------------------------------------
# 4. compute_criterion_text_length_stats
# ---------------------------------------------------------------------------


@register_analysis(
    name="criterion_text_length_stats",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_criterion_text_length_stats(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse criterion text length distribution grouped by rubric-item axis.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset criterion length CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        overall_length_stats, length_stats_by_axis, total_rubric_items.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_rubric_dataframe(dataset)

        length_stats_by_axis = {
            str(axis): series_stats(group["criterion_length"], percentiles=DEFAULT_PERCENTILES)
            for axis, group in df.groupby("axis")
        }

        subset_result: dict[str, Any] = {
            "overall_length_stats": series_stats(
                df["criterion_length"], percentiles=DEFAULT_PERCENTILES
            ),
            "length_stats_by_axis": length_stats_by_axis,
            "total_rubric_items": len(df),
        }

        if save:
            save_csv(
                df[["axis", "criterion_length"]],
                output_dir / f"criterion_text_length_stats_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: criterion text length — %s rubric items, %s axes",
            subset,
            len(df),
            len(length_stats_by_axis),
        )

    return results


# ---------------------------------------------------------------------------
# 5. compute_score_ceiling_by_theme
# ---------------------------------------------------------------------------


@register_analysis(
    name="score_ceiling_by_theme",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_score_ceiling_by_theme(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compute mean score ceiling and penalty ratio per theme per subset.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset theme score ceiling CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        theme_mean_possible_points, theme_mean_penalty_ratio,
        theme_sample_counts, overall_mean_possible_points,
        overall_mean_penalty_ratio.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)

        subset_result: dict[str, Any] = {
            "theme_mean_possible_points": df.groupby("theme")["total_possible_points"]
            .mean()
            .round(4)
            .to_dict(),
            "theme_mean_penalty_ratio": df.dropna(subset=["penalty_mass_ratio"])
            .groupby("theme")["penalty_mass_ratio"]
            .mean()
            .round(4)
            .to_dict(),
            "theme_sample_counts": df["theme"].value_counts().to_dict(),
            "overall_mean_possible_points": round(float(df["total_possible_points"].mean()), 4),
            "overall_mean_penalty_ratio": round(float(df["penalty_mass_ratio"].dropna().mean()), 4),
        }

        if save:
            save_csv(
                df.groupby("theme")
                .agg(
                    mean_possible_points=("total_possible_points", "mean"),
                    mean_penalty_ratio=("penalty_mass_ratio", "mean"),
                    sample_count=("theme", "count"),
                )
                .round(4)
                .reset_index(),
                output_dir / f"score_ceiling_by_theme_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: score ceiling by theme — %s themes",
            subset,
            len(subset_result["theme_sample_counts"]),
        )

    return results


# ---------------------------------------------------------------------------
# 6. compute_emergency_axis_profile
# ---------------------------------------------------------------------------


@register_analysis(
    name="emergency_axis_profile",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_emergency_axis_profile(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Profile samples and rubric items flagged as safety- or emergency-critical.

    A rubric item is emergency-critical when its axis tag is 'safety' or its
    criterion text contains 'emergency' (case-insensitive).

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset emergency profile CSV.

    Returns:
        Dict keyed by subset label. Each value has keys:
        emergency_rubric_item_count, emergency_rubric_item_fraction,
        samples_with_emergency_item_count, samples_with_emergency_item_fraction,
        emergency_penalty_points_total, theme_counts.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        rdf = build_rubric_dataframe(dataset)
        sdf = build_sample_dataframe(dataset)

        rdf["is_emergency"] = (rdf["axis"] == "safety") | rdf["criterion"].str.lower().str.contains(
            "emergency", na=False
        )

        emergency_items = rdf[rdf["is_emergency"]]
        emergency_item_count = len(emergency_items)
        emergency_penalty_total = float(
            emergency_items.loc[emergency_items["points"] < 0, "points"].abs().sum()
        )

        emergency_prompt_ids = set(emergency_items["prompt_id"])
        samples_with_emergency = len(emergency_prompt_ids)
        total_samples = len(sdf)

        # Theme counts for samples with emergency items
        theme_counts = (
            sdf[sdf["prompt_id"].isin(emergency_prompt_ids)]["theme"].value_counts().to_dict()
        )

        subset_result: dict[str, Any] = {
            "emergency_rubric_item_count": emergency_item_count,
            "emergency_rubric_item_fraction": (
                emergency_item_count / len(rdf) if len(rdf) else 0.0
            ),
            "samples_with_emergency_item_count": samples_with_emergency,
            "samples_with_emergency_item_fraction": (
                samples_with_emergency / total_samples if total_samples else 0.0
            ),
            "emergency_penalty_points_total": round(emergency_penalty_total, 4),
            "theme_counts": theme_counts,
        }

        if save:
            save_csv(
                pd.DataFrame(
                    [
                        {"metric": k, "value": v}
                        for k, v in subset_result.items()
                        if k != "theme_counts"
                    ]
                ),
                output_dir / f"emergency_axis_profile_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: emergency profile — %s/%s samples, %s rubric items",
            subset,
            samples_with_emergency,
            total_samples,
            emergency_item_count,
        )

    return results


# ---------------------------------------------------------------------------
# 7. compute_hard_subset_signature
# ---------------------------------------------------------------------------


@register_analysis(
    name="hard_subset_signature",
    category="insights",
    datasets=["main", "hard"],
)
def compute_hard_subset_signature(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compare structural metrics between 'hard' and 'main' subsets.

    Args:
        datasets: Loaded HealthBench datasets filtered to 'main' and 'hard'.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a comparison CSV to output_dir.

    Returns:
        Dict with keys: main_stats, hard_stats, delta,
        theme_distribution_main, theme_distribution_hard.
        Returns empty dict if both subsets are not present.
    """
    by_subset = {d.subset: d for d in datasets}
    if "main" not in by_subset or "hard" not in by_subset:
        logger.info(
            "hard_subset_signature requires both 'main' and 'hard'; found: %s",
            list(by_subset.keys()),
        )
        return {}

    metrics = ["rubric_size", "total_possible_points", "penalty_mass_ratio", "prompt_char_length"]

    def _subset_stats(ds: HealthBenchDataset) -> dict[str, float]:
        df = build_sample_dataframe(ds)
        return {
            "rubric_size": round(float(df["rubric_size"].mean()), 4),
            "total_possible_points": round(float(df["total_possible_points"].mean()), 4),
            "penalty_mass_ratio": round(float(df["penalty_mass_ratio"].dropna().mean()), 4),
            "prompt_char_length": round(float(df["prompt_char_length"].mean()), 4),
            "sample_count": len(df),
        }

    main_stats = _subset_stats(by_subset["main"])
    hard_stats = _subset_stats(by_subset["hard"])
    delta = {m: round(hard_stats[m] - main_stats[m], 4) for m in metrics}

    main_df = build_sample_dataframe(by_subset["main"])
    hard_df = build_sample_dataframe(by_subset["hard"])

    result: dict[str, Any] = {
        "main_stats": main_stats,
        "hard_stats": hard_stats,
        "delta": delta,
        "theme_distribution_main": main_df["theme"].value_counts().to_dict(),
        "theme_distribution_hard": hard_df["theme"].value_counts().to_dict(),
    }

    if save:
        comparison_rows = [
            {"metric": m, "main": main_stats[m], "hard": hard_stats[m], "delta": delta[m]}
            for m in metrics
        ]
        save_csv(pd.DataFrame(comparison_rows), output_dir / "hard_subset_signature.csv")

    logger.info(
        "hard_subset_signature: rubric_size delta=%.2f, possible_points delta=%.2f",
        delta["rubric_size"],
        delta["total_possible_points"],
    )

    return result


# ---------------------------------------------------------------------------
# 8. compute_prompt_length_vs_rubric_complexity
# ---------------------------------------------------------------------------


@register_analysis(
    name="prompt_length_vs_rubric_complexity",
    category="insights",
    datasets=["main", "hard", "consensus"],
)
def compute_prompt_length_vs_rubric_complexity(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compute Pearson and Spearman correlations between prompt length and rubric complexity.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset sample-level CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        length_vs_rubric_size, length_vs_possible_points, sample_count.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            results[subset] = {}
            continue

        df = build_sample_dataframe(dataset)

        def _corr(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
            if len(x) < 2:
                return {
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_r": None,
                    "spearman_p": None,
                }
            pearson_result = scipy_stats.pearsonr(x, y)
            spearman_result = scipy_stats.spearmanr(x, y)
            return {
                "pearson_r": round(float(pearson_result.statistic), 4),
                "pearson_p": round(float(pearson_result.pvalue), 4),
                "spearman_r": round(float(spearman_result.statistic), 4),
                "spearman_p": round(float(spearman_result.pvalue), 4),
            }

        x = df["prompt_char_length"]
        subset_result: dict[str, Any] = {
            "length_vs_rubric_size": _corr(x, df["rubric_size"]),
            "length_vs_possible_points": _corr(x, df["total_possible_points"]),
            "sample_count": len(df),
        }

        if save:
            save_csv(
                df[["prompt_char_length", "rubric_size", "total_possible_points"]],
                output_dir / f"prompt_length_vs_rubric_complexity_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: prompt_length correlations — "
            "vs_rubric_size pearson_r=%s, vs_possible_points pearson_r=%s",
            subset,
            subset_result["length_vs_rubric_size"]["pearson_r"],
            subset_result["length_vs_possible_points"]["pearson_r"],
        )

    return results
