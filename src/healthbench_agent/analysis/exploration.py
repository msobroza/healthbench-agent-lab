"""Descriptive statistics analyses for HealthBench datasets.

Implements the full suite of exploration analyses registered under the
'exploration' category. Each function accepts a list of HealthBenchDataset
objects, computes per-subset statistics using pandas, and optionally saves
CSV artifacts to disk.

Import chain: .registry → ..data_models.
Does not import from agents/ or evaluation/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .registry import register_analysis
from .utils import DEFAULT_PERCENTILES, parse_tag_prefixes, save_csv, series_stats
from ..domain.dataset import HealthBenchDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. compute_sample_counts
# ---------------------------------------------------------------------------


@register_analysis(
    name="sample_counts",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_sample_counts(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Count samples, unique prompt IDs, and ideal-completion coverage per subset.

    For each dataset subset:
    - Total number of samples.
    - Number of unique prompt_ids.
    - Count and fraction of samples that have ideal_completions_data.
    - Count and fraction of samples that do not have ideal_completions_data.
    - Value counts of the ideal_completions_group field found inside
      ideal_completions_data (when present).

    Args:
        datasets: Loaded HealthBench datasets to analyse. Filtered to
            'main', 'hard', and 'consensus' subsets.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a summary CSV for each subset to output_dir.

    Returns:
        Dict keyed by subset label. Each value is a nested dict with keys:
        total_samples, unique_prompt_ids, with_ideal_completions_count,
        with_ideal_completions_fraction, without_ideal_completions_count,
        without_ideal_completions_fraction, physician_group_distribution.
        Returns an empty dict for any subset with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        rows = []
        for sample in dataset.samples:
            has_ideal = sample.ideal_completions_data is not None
            group = None
            if has_ideal and isinstance(sample.ideal_completions_data, dict):
                group = sample.ideal_completions_data.get("ideal_completions_group")
            rows.append(
                {
                    "prompt_id": sample.prompt_id,
                    "has_ideal_completions": has_ideal,
                    "ideal_completions_group": group,
                }
            )

        dataframe = pd.DataFrame(rows)
        total = len(dataframe)
        with_ideal_count = int(dataframe["has_ideal_completions"].sum())
        without_ideal_count = total - with_ideal_count

        group_distribution = (
            dataframe["ideal_completions_group"]
            .dropna()
            .value_counts()
            .to_dict()
        )

        subset_result: dict[str, Any] = {
            "total_samples": total,
            "unique_prompt_ids": int(dataframe["prompt_id"].nunique()),
            "with_ideal_completions_count": with_ideal_count,
            "with_ideal_completions_fraction": with_ideal_count / total,
            "without_ideal_completions_count": without_ideal_count,
            "without_ideal_completions_fraction": without_ideal_count / total,
            "physician_group_distribution": group_distribution,
        }

        if save:
            summary_df = pd.DataFrame(
                [
                    {
                        "metric": key,
                        "value": str(value),
                    }
                    for key, value in subset_result.items()
                    if key != "physician_group_distribution"
                ]
            )
            save_csv(
                summary_df,
                output_dir / f"compute_sample_counts_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info("Subset %s: %s total samples", subset, total)

    return results


# ---------------------------------------------------------------------------
# 2. compute_prompt_structure
# ---------------------------------------------------------------------------


@register_analysis(
    name="prompt_structure",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_prompt_structure(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse the turn structure and character lengths of prompts per subset.

    For each sample computes:
    - Number of turns in the prompt MessageList.
    - Total character length of all turns combined (content as string).
    - Character length of the last user turn.

    Returns aggregate stats (min, max, mean, median, p95) for each metric
    and counts of single-turn vs multi-turn samples.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset stats CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        num_turns_stats, total_char_length_stats,
        last_user_turn_char_length_stats, single_turn_count,
        multi_turn_count. Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        rows = []
        for sample in dataset.samples:
            num_turns = len(sample.prompt)
            total_chars = sum(
                len(str(turn.get("content", ""))) for turn in sample.prompt
            )
            last_user_chars = 0
            for turn in reversed(sample.prompt):
                if turn.get("role") == "user":
                    last_user_chars = len(str(turn.get("content", "")))
                    break
            rows.append(
                {
                    "num_turns": num_turns,
                    "total_char_length": total_chars,
                    "last_user_turn_char_length": last_user_chars,
                }
            )

        dataframe = pd.DataFrame(rows)

        subset_result: dict[str, Any] = {
            "num_turns_stats": series_stats(
                dataframe["num_turns"], percentiles=DEFAULT_PERCENTILES
            ),
            "total_char_length_stats": series_stats(
                dataframe["total_char_length"], percentiles=DEFAULT_PERCENTILES
            ),
            "last_user_turn_char_length_stats": series_stats(
                dataframe["last_user_turn_char_length"], percentiles=DEFAULT_PERCENTILES
            ),
            "single_turn_count": int((dataframe["num_turns"] == 1).sum()),
            "multi_turn_count": int((dataframe["num_turns"] > 1).sum()),
        }

        if save:
            save_csv(dataframe, output_dir / f"compute_prompt_structure_{subset}.csv")

        results[subset] = subset_result
        logger.info(
            "Subset %s: %s samples analysed for prompt structure",
            subset,
            len(dataframe),
        )

    return results


# ---------------------------------------------------------------------------
# 3. compute_rubric_size
# ---------------------------------------------------------------------------


@register_analysis(
    name="rubric_size",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_rubric_size(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse the distribution of rubric items per sample for each subset.

    Computes descriptive statistics (min, max, mean, median, std, p25, p75,
    p95) for the count of rubric items per sample, plus a value_counts of
    exact rubric sizes.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset counts CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        rubric_size_stats (dict of stats), rubric_size_value_counts (dict
        mapping exact rubric size to sample count). Returns empty dict for
        subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        rubric_sizes = pd.Series(
            [len(sample.rubrics) for sample in dataset.samples],
            name="rubric_size",
        )

        subset_result: dict[str, Any] = {
            "rubric_size_stats": series_stats(
                rubric_sizes,
                percentiles=DEFAULT_PERCENTILES,
            ),
            "rubric_size_value_counts": {
                int(k): int(v)
                for k, v in rubric_sizes.value_counts().sort_index().items()
            },
        }

        if save:
            size_df = rubric_sizes.to_frame()
            save_csv(
                size_df,
                output_dir / f"compute_rubric_size_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: rubric sizes computed for %s samples",
            subset,
            len(rubric_sizes),
        )

    return results


# ---------------------------------------------------------------------------
# 4. compute_rubric_points_distribution
# ---------------------------------------------------------------------------


@register_analysis(
    name="rubric_points_distribution",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_rubric_points_distribution(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse the distribution of points values across all rubric items per subset.

    Flattens all rubric items across all samples and reports:
    - Value counts of points values rounded to 1 decimal place.
    - Descriptive stats (min, max, mean, median, std) of the points values.
    - Fraction of rubric items with points > 0, < 0, == 0.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset points CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        points_value_counts, points_stats, fraction_positive,
        fraction_negative, fraction_zero. Returns empty dict for subsets
        with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        all_points = pd.Series(
            [
                round(rubric_item.points, 1)
                for sample in dataset.samples
                for rubric_item in sample.rubrics
            ],
            name="points",
            dtype=float,
        )

        total_items = len(all_points)
        positive_count = int((all_points > 0).sum())
        negative_count = int((all_points < 0).sum())
        zero_count = int((all_points == 0).sum())

        subset_result: dict[str, Any] = {
            "points_value_counts": {
                float(k): int(v)
                for k, v in all_points.value_counts().sort_index().items()
            },
            "points_stats": series_stats(
                all_points,
                percentiles=DEFAULT_PERCENTILES,
            ),
            "fraction_positive": positive_count / total_items if total_items else 0.0,
            "fraction_negative": negative_count / total_items if total_items else 0.0,
            "fraction_zero": zero_count / total_items if total_items else 0.0,
            "count_positive": positive_count,
            "count_negative": negative_count,
            "count_zero": zero_count,
            "total_rubric_items": total_items,
        }

        if save:
            points_df = all_points.to_frame()
            save_csv(
                points_df,
                output_dir / f"compute_rubric_points_distribution_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: %s total rubric items analysed for points distribution",
            subset,
            total_items,
        )

    return results


# ---------------------------------------------------------------------------
# 5. compute_positive_vs_penalty_stats
# ---------------------------------------------------------------------------


@register_analysis(
    name="positive_vs_penalty_stats",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_positive_vs_penalty_stats(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse the balance between positive and penalty points per sample per subset.

    For each sample computes:
    - total_possible_points: sum of all positive rubric points.
    - total_penalty_points: absolute value of the sum of all negative points.
    - penalty_mass_ratio: abs(penalty) / total_possible_points when
      total_possible_points > 0, else None.

    Returns aggregate stats (mean, median, p95) for each metric, plus a
    count of samples where total_possible_points == 0.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset per-sample CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        total_possible_points_stats, total_penalty_points_stats,
        penalty_mass_ratio_stats, samples_with_zero_possible_points.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        rows = []
        for sample in dataset.samples:
            total_possible = sum(
                rubric_item.points
                for rubric_item in sample.rubrics
                if rubric_item.points > 0
            )
            total_penalty = abs(
                sum(
                    rubric_item.points
                    for rubric_item in sample.rubrics
                    if rubric_item.points < 0
                )
            )
            penalty_mass_ratio = (
                total_penalty / total_possible if total_possible > 0 else None
            )
            rows.append(
                {
                    "total_possible_points": total_possible,
                    "total_penalty_points": total_penalty,
                    "penalty_mass_ratio": penalty_mass_ratio,
                }
            )

        dataframe = pd.DataFrame(rows)

        subset_result: dict[str, Any] = {
            "total_possible_points_stats": series_stats(
                dataframe["total_possible_points"],
                percentiles=DEFAULT_PERCENTILES,
            ),
            "total_penalty_points_stats": series_stats(
                dataframe["total_penalty_points"],
                percentiles=DEFAULT_PERCENTILES,
            ),
            "penalty_mass_ratio_stats": series_stats(
                dataframe["penalty_mass_ratio"].dropna(),
                percentiles=DEFAULT_PERCENTILES,
            ),
            "samples_with_zero_possible_points": int(
                (dataframe["total_possible_points"] == 0).sum()
            ),
        }

        if save:
            save_csv(
                dataframe,
                output_dir / f"compute_positive_vs_penalty_stats_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: positive vs penalty stats computed for %s samples",
            subset,
            len(dataframe),
        )

    return results


# ---------------------------------------------------------------------------
# 6. compute_score_range_stats
# ---------------------------------------------------------------------------


@register_analysis(
    name="score_range_stats",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_score_range_stats(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compute the achievable score range for each sample per subset.

    For each sample:
    - max_possible_score: sum of max(0, points) over all rubric items
      (best-case score, all positive criteria met, no penalties).
    - min_possible_score: sum of negative points (worst-case, no positive
      criteria met, all penalties applied). The value is negative or zero.
    - score_range_width: max_possible_score - min_possible_score.

    Returns aggregate stats (min, max, mean, median, std, p5, p95) for all
    three metrics.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset per-sample CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        max_possible_score_stats, min_possible_score_stats,
        score_range_width_stats. Returns empty dict for subsets with no
        samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        rows = []
        for sample in dataset.samples:
            points = [rubric_item.points for rubric_item in sample.rubrics]
            max_possible = sum(max(0.0, p) for p in points)
            min_possible = sum(p for p in points if p < 0)
            score_range_width = max_possible - min_possible
            rows.append(
                {
                    "max_possible_score": max_possible,
                    "min_possible_score": min_possible,
                    "score_range_width": score_range_width,
                }
            )

        dataframe = pd.DataFrame(rows)

        subset_result: dict[str, Any] = {
            "max_possible_score_stats": series_stats(
                dataframe["max_possible_score"],
                percentiles=DEFAULT_PERCENTILES,
            ),
            "min_possible_score_stats": series_stats(
                dataframe["min_possible_score"],
                percentiles=DEFAULT_PERCENTILES,
            ),
            "score_range_width_stats": series_stats(
                dataframe["score_range_width"],
                percentiles=DEFAULT_PERCENTILES,
            ),
        }

        if save:
            save_csv(
                dataframe,
                output_dir / f"compute_score_range_stats_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: score range stats computed for %s samples",
            subset,
            len(dataframe),
        )

    return results


# ---------------------------------------------------------------------------
# 7. compute_rubric_tag_frequency
# ---------------------------------------------------------------------------


@register_analysis(
    name="rubric_tag_frequency",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_rubric_tag_frequency(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Count how often each rubric-item tag appears across a subset.

    Flattens all tags from all rubric items across all samples and computes:
    - value_counts mapping tag → occurrence count.
    - Total unique tags.
    - Total tag occurrences.
    - Mean tags per rubric item.
    - Fraction of rubric items with at least one tag.
    - Fraction of rubric items with zero tags.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset tag frequency CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        tag_value_counts, total_unique_tags, total_tag_occurrences,
        mean_tags_per_rubric_item, fraction_with_tag,
        fraction_without_tag. Returns empty dict for subsets with no
        samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        all_tags: list[str] = []
        total_rubric_items = 0
        items_with_tag = 0

        for sample in dataset.samples:
            for rubric_item in sample.rubrics:
                total_rubric_items += 1
                if rubric_item.tags:
                    items_with_tag += 1
                all_tags.extend(rubric_item.tags)

        tag_series = pd.Series(all_tags, name="tag")
        tag_counts = tag_series.value_counts()

        subset_result: dict[str, Any] = {
            "tag_value_counts": tag_counts.to_dict(),
            "total_unique_tags": int(tag_series.nunique()),
            "total_tag_occurrences": len(all_tags),
            "mean_tags_per_rubric_item": (
                len(all_tags) / total_rubric_items if total_rubric_items else 0.0
            ),
            "fraction_with_tag": (
                items_with_tag / total_rubric_items if total_rubric_items else 0.0
            ),
            "fraction_without_tag": (
                (total_rubric_items - items_with_tag) / total_rubric_items
                if total_rubric_items
                else 0.0
            ),
            "total_rubric_items": total_rubric_items,
        }

        if save:
            tag_df = tag_counts.reset_index()
            tag_df.columns = pd.Index(["tag", "count"])
            save_csv(
                tag_df,
                output_dir / f"compute_rubric_tag_frequency_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: %s unique rubric tags across %s rubric items",
            subset,
            subset_result["total_unique_tags"],
            total_rubric_items,
        )

    return results


# ---------------------------------------------------------------------------
# 8. compute_rubric_tags_per_sample
# ---------------------------------------------------------------------------


@register_analysis(
    name="rubric_tags_per_sample",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_rubric_tags_per_sample(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse the number of unique rubric-item tags per sample per subset.

    For each sample counts the total number of unique tags across all of its
    rubric items. Returns aggregate stats (min, max, mean, median, p95) and
    a count of samples that have zero tags across all rubric items.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset per-sample tag count CSV to
            output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        unique_tags_per_sample_stats, samples_with_zero_tags.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        unique_tag_counts = pd.Series(
            [
                len({tag for rubric_item in sample.rubrics for tag in rubric_item.tags})
                for sample in dataset.samples
            ],
            name="unique_tags_per_sample",
        )

        subset_result: dict[str, Any] = {
            "unique_tags_per_sample_stats": series_stats(
                unique_tag_counts,
                percentiles=DEFAULT_PERCENTILES,
            ),
            "samples_with_zero_tags": int((unique_tag_counts == 0).sum()),
        }

        if save:
            save_csv(
                unique_tag_counts.to_frame(),
                output_dir / f"compute_rubric_tags_per_sample_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: unique rubric tags per sample computed for %s samples",
            subset,
            len(unique_tag_counts),
        )

    return results


# ---------------------------------------------------------------------------
# 9. compute_example_tag_frequency
# ---------------------------------------------------------------------------


@register_analysis(
    name="example_tag_frequency",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_example_tag_frequency(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Count how often each example-level tag appears across samples in a subset.

    Flattens all example_tags across all samples and reports:
    - value_counts mapping tag → sample count (not occurrence count; each
      sample contributes at most once per tag due to de-duplication per
      sample before aggregation).
    - Total unique example tags.
    - Mean, median, and max example_tags per sample.
    - Count of samples with zero example_tags.

    Note: value_counts here counts total occurrences (a tag may appear in
    multiple samples); "per-sample" occurrence semantics are conveyed by
    the mean/median/max metrics.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset tag frequency CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        tag_value_counts, total_unique_example_tags,
        mean_example_tags_per_sample, median_example_tags_per_sample,
        max_example_tags_per_sample, samples_with_zero_example_tags.
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        all_tags: list[str] = []
        per_sample_counts: list[int] = []
        samples_with_zero = 0

        for sample in dataset.samples:
            tag_count = len(sample.example_tags)
            per_sample_counts.append(tag_count)
            if tag_count == 0:
                samples_with_zero += 1
            all_tags.extend(sample.example_tags)

        tag_series = pd.Series(all_tags, name="example_tag")
        count_series = pd.Series(per_sample_counts, name="example_tags_per_sample")

        subset_result: dict[str, Any] = {
            "tag_value_counts": tag_series.value_counts().to_dict(),
            "total_unique_example_tags": int(tag_series.nunique()),
            "mean_example_tags_per_sample": float(count_series.mean()),
            "median_example_tags_per_sample": float(count_series.median()),
            "max_example_tags_per_sample": int(count_series.max()),
            "samples_with_zero_example_tags": samples_with_zero,
        }

        if save:
            tag_df = tag_series.value_counts().reset_index()
            tag_df.columns = pd.Index(["tag", "count"])
            save_csv(
                tag_df,
                output_dir / f"compute_example_tag_frequency_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: %s unique example tags found",
            subset,
            subset_result["total_unique_example_tags"],
        )

    return results


# ---------------------------------------------------------------------------
# 10. compute_tag_prefix_distribution
# ---------------------------------------------------------------------------



@register_analysis(
    name="tag_prefix_distribution",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_tag_prefix_distribution(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Analyse 'key: value' prefix structure for both example and rubric tags.

    Many tags follow the pattern "key: value" (e.g. "theme: emergency
    medicine", "axis: accuracy"). This function splits each tag on the
    first ': ' occurrence, groups values by prefix, and returns value_counts
    per prefix for both example-level tags and rubric-item tags.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write per-subset CSVs for example and rubric tag
            prefix distributions to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys:
        example_tag_prefix_distribution (dict mapping prefix →
        {value: count}), rubric_tag_prefix_distribution (same structure),
        example_tag_unique_prefixes (list), rubric_tag_unique_prefixes (list).
        Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        all_example_tags: list[str] = []
        all_rubric_tags: list[str] = []

        for sample in dataset.samples:
            all_example_tags.extend(sample.example_tags)
            for rubric_item in sample.rubrics:
                all_rubric_tags.extend(rubric_item.tags)

        example_prefix_distribution = parse_tag_prefixes(all_example_tags)
        rubric_prefix_distribution = parse_tag_prefixes(all_rubric_tags)

        subset_result: dict[str, Any] = {
            "example_tag_prefix_distribution": example_prefix_distribution,
            "rubric_tag_prefix_distribution": rubric_prefix_distribution,
            "example_tag_unique_prefixes": sorted(example_prefix_distribution.keys()),
            "rubric_tag_unique_prefixes": sorted(rubric_prefix_distribution.keys()),
        }

        if save:
            example_rows = [
                {"prefix": prefix, "value": value, "count": count}
                for prefix, value_counts in example_prefix_distribution.items()
                for value, count in value_counts.items()
            ]
            rubric_rows = [
                {"prefix": prefix, "value": value, "count": count}
                for prefix, value_counts in rubric_prefix_distribution.items()
                for value, count in value_counts.items()
            ]
            if example_rows:
                save_csv(
                    pd.DataFrame(example_rows),
                    output_dir / f"compute_tag_prefix_distribution_example_{subset}.csv",
                )
            if rubric_rows:
                save_csv(
                    pd.DataFrame(rubric_rows),
                    output_dir / f"compute_tag_prefix_distribution_rubric_{subset}.csv",
                )

        results[subset] = subset_result
        logger.info(
            "Subset %s: %s example tag prefixes, %s rubric tag prefixes",
            subset,
            len(example_prefix_distribution),
            len(rubric_prefix_distribution),
        )

    return results


# ---------------------------------------------------------------------------
# 11. compute_data_quality
# ---------------------------------------------------------------------------


@register_analysis(
    name="data_quality",
    category="exploration",
    datasets=["main", "hard", "consensus"],
)
def compute_data_quality(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Audit data quality indicators for each dataset subset.

    Checks the following quality metrics:
    - Samples with empty rubrics list.
    - Rubric items with points == 0.
    - Duplicate prompt_ids (count of prompt_ids appearing more than once).
    - Samples where total_possible_points == 0 (calculate_score returns None).
    - Rubric items with empty tags list.
    - Samples with empty example_tags list.

    All counts are accompanied by their fraction of the relevant total.

    Args:
        datasets: Loaded HealthBench datasets to analyse.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write a per-subset quality summary CSV to output_dir.

    Returns:
        Dict keyed by subset label. Each value has keys ending in _count and
        _fraction for each quality metric, plus total_samples and
        total_rubric_items. Returns empty dict for subsets with no samples.
    """
    results: dict[str, Any] = {}

    for dataset in datasets:
        subset = dataset.subset

        if not dataset.samples:
            logger.info("Subset %s has no samples; skipping", subset)
            results[subset] = {}
            continue

        total_samples = len(dataset.samples)
        prompt_ids = [sample.prompt_id for sample in dataset.samples]
        duplicate_prompt_id_count = int(
            pd.Series(prompt_ids).duplicated().sum()
        )

        empty_rubrics_count = 0
        zero_points_rubric_items_count = 0
        zero_possible_points_samples_count = 0
        empty_rubric_tags_count = 0
        empty_example_tags_count = 0
        total_rubric_items = 0

        for sample in dataset.samples:
            if not sample.rubrics:
                empty_rubrics_count += 1

            if not sample.example_tags:
                empty_example_tags_count += 1

            total_possible = 0.0
            for rubric_item in sample.rubrics:
                total_rubric_items += 1
                if rubric_item.points == 0:
                    zero_points_rubric_items_count += 1
                if not rubric_item.tags:
                    empty_rubric_tags_count += 1
                if rubric_item.points > 0:
                    total_possible += rubric_item.points

            if total_possible == 0.0:
                zero_possible_points_samples_count += 1

        def _fraction(count: int, total: int) -> float:
            return count / total if total else 0.0

        subset_result: dict[str, Any] = {
            "total_samples": total_samples,
            "total_rubric_items": total_rubric_items,
            "empty_rubrics_count": empty_rubrics_count,
            "empty_rubrics_fraction": _fraction(empty_rubrics_count, total_samples),
            "zero_points_rubric_items_count": zero_points_rubric_items_count,
            "zero_points_rubric_items_fraction": _fraction(
                zero_points_rubric_items_count, total_rubric_items
            ),
            "duplicate_prompt_id_count": duplicate_prompt_id_count,
            "duplicate_prompt_id_fraction": _fraction(
                duplicate_prompt_id_count, total_samples
            ),
            "zero_possible_points_samples_count": zero_possible_points_samples_count,
            "zero_possible_points_samples_fraction": _fraction(
                zero_possible_points_samples_count, total_samples
            ),
            "empty_rubric_tags_count": empty_rubric_tags_count,
            "empty_rubric_tags_fraction": _fraction(
                empty_rubric_tags_count, total_rubric_items
            ),
            "empty_example_tags_count": empty_example_tags_count,
            "empty_example_tags_fraction": _fraction(
                empty_example_tags_count, total_samples
            ),
        }

        if save:
            quality_df = pd.DataFrame(
                [{"metric": k, "value": v} for k, v in subset_result.items()]
            )
            save_csv(
                quality_df,
                output_dir / f"compute_data_quality_{subset}.csv",
            )

        results[subset] = subset_result
        logger.info(
            "Subset %s: data quality audited (%s samples, %s rubric items)",
            subset,
            total_samples,
            total_rubric_items,
        )

    return results


# ---------------------------------------------------------------------------
# 12. compute_subset_overlap
# ---------------------------------------------------------------------------


@register_analysis(
    name="subset_overlap",
    category="exploration",
    datasets=["hard", "consensus"],
)
def compute_subset_overlap(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Compare prompt_id overlap between the 'hard' and 'consensus' subsets.

    Returns counts of prompt_ids exclusive to each subset, shared across
    both, and the Jaccard similarity coefficient. Only meaningful when both
    'hard' and 'consensus' datasets are present; returns an empty dict
    otherwise.

    Args:
        datasets: Loaded HealthBench datasets filtered to 'hard' and
            'consensus' subsets.
        output_dir: Directory where CSV artifacts are saved when save=True.
        save: When True, write an overlap summary CSV to output_dir.

    Returns:
        Dict with keys: hard_only_count, consensus_only_count,
        intersection_count, hard_total, consensus_total, jaccard_similarity.
        Returns empty dict if both 'hard' and 'consensus' datasets are not
        simultaneously present.
    """
    dataset_by_subset = {dataset.subset: dataset for dataset in datasets}

    if "hard" not in dataset_by_subset or "consensus" not in dataset_by_subset:
        logger.info(
            "subset_overlap requires both 'hard' and 'consensus' datasets; "
            "found only: %s",
            list(dataset_by_subset.keys()),
        )
        return {}

    hard_ids = {sample.prompt_id for sample in dataset_by_subset["hard"].samples}
    consensus_ids = {
        sample.prompt_id for sample in dataset_by_subset["consensus"].samples
    }

    hard_only = hard_ids - consensus_ids
    consensus_only = consensus_ids - hard_ids
    intersection = hard_ids & consensus_ids
    union = hard_ids | consensus_ids

    jaccard_similarity = len(intersection) / len(union) if union else 0.0

    result: dict[str, Any] = {
        "hard_only_count": len(hard_only),
        "consensus_only_count": len(consensus_only),
        "intersection_count": len(intersection),
        "hard_total": len(hard_ids),
        "consensus_total": len(consensus_ids),
        "jaccard_similarity": jaccard_similarity,
    }

    if save:
        summary_df = pd.DataFrame(
            [{"metric": k, "value": v} for k, v in result.items()]
        )
        save_csv(
            summary_df,
            output_dir / "compute_subset_overlap_hard_consensus.csv",
        )

    logger.info(
        "Subset overlap: hard=%s, consensus=%s, intersection=%s, jaccard=%.4f",
        len(hard_ids),
        len(consensus_ids),
        len(intersection),
        jaccard_similarity,
    )

    return result
