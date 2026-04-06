"""Shared utilities for all analysis modules.

Provides stateless helper functions, shared constants, and DataFrame builders
used across healthbench_agent.analysis.exploration, insights, and visualization.
Does not import from agents/ or evaluation/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..domain.dataset import HealthBenchDataset, HealthBenchSample

logger = logging.getLogger(__name__)

# Percentile levels used consistently across all descriptive stats analyses.
DEFAULT_PERCENTILES: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95]


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to CSV and log the path at INFO level.

    Args:
        dataframe: The DataFrame to write.
        path: Full file path (including filename) for the output CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=True)
    logger.info("Saved CSV to %s", path)


def series_stats(series: pd.Series, percentiles: list[float] | None = None) -> dict[str, Any]:
    """Compute descriptive stats for a numeric Series and return as a plain dict.

    Args:
        series: Numeric pandas Series to describe.
        percentiles: Quantile levels to include alongside the standard
            describe() output. Defaults to DEFAULT_PERCENTILES.

    Returns:
        Dict with keys derived from pandas describe() output (count, mean,
        std, min, percentile labels, max). All values are plain Python floats.
    """
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    description = series.describe(percentiles=percentiles)
    result: dict[str, Any] = {}
    for key, value in description.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            result[str(key)] = value
    return result


def tag_value(tags: list[str], prefix: str) -> str | None:
    """Return the value portion of the first tag matching 'prefix:value', or None.

    Args:
        tags: Flat list of raw tag strings (e.g. 'theme:emergency_referrals').
        prefix: Prefix to match, e.g. 'theme' or 'axis'.

    Returns:
        Value after ':', or None if no matching tag is found.
    """
    target = prefix + ":"
    for tag in tags:
        if tag.startswith(target):
            return tag[len(target) :]
    return None


def total_positive_points(sample: HealthBenchSample) -> float:
    """Sum of positive rubric points for a sample.

    Args:
        sample: A HealthBench sample with rubrics.

    Returns:
        Sum of points for rubric items where points > 0.
    """
    return sum(r.points for r in sample.rubrics if r.points > 0)


def total_penalty_points(sample: HealthBenchSample) -> float:
    """Absolute sum of negative rubric points for a sample.

    Args:
        sample: A HealthBench sample with rubrics.

    Returns:
        Absolute value of the sum of negative points.
    """
    return abs(sum(r.points for r in sample.rubrics if r.points < 0))


def total_prompt_chars(sample: HealthBenchSample) -> int:
    """Total character length across all prompt turns.

    Args:
        sample: A HealthBench sample with a prompt MessageList.

    Returns:
        Sum of len(content) across all turns.
    """
    return sum(len(str(t.get("content", ""))) for t in sample.prompt)


def build_rubric_dataframe(dataset: HealthBenchDataset) -> pd.DataFrame:
    """Build a flat DataFrame with one row per rubric item, enriched with sample-level tags.

    Columns: prompt_id, theme, axis, points, criterion, criterion_length.

    Args:
        dataset: A single loaded HealthBench dataset.

    Returns:
        DataFrame with one row per rubric item. Empty DataFrame with the
        expected columns when the dataset has no samples.
    """
    columns = ["prompt_id", "theme", "axis", "points", "criterion", "criterion_length"]
    if not dataset.samples:
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "prompt_id": sample.prompt_id,
            "theme": tag_value(sample.example_tags, "theme") or "_unknown",
            "axis": tag_value(rubric_item.tags, "axis") or "_unknown",
            "points": rubric_item.points,
            "criterion": rubric_item.criterion,
            "criterion_length": len(rubric_item.criterion),
        }
        for sample in dataset.samples
        for rubric_item in sample.rubrics
    ]
    return pd.DataFrame(rows)


def build_sample_dataframe(dataset: HealthBenchDataset) -> pd.DataFrame:
    """Build a flat DataFrame with one row per sample, enriched with computed metrics.

    Columns: prompt_id, theme, rubric_size, total_possible_points,
    total_penalty_points, penalty_mass_ratio, prompt_char_length,
    num_turns, num_user_turns.

    Args:
        dataset: A single loaded HealthBench dataset.

    Returns:
        DataFrame with one row per sample. Empty DataFrame with the
        expected columns when the dataset has no samples.
    """
    columns = [
        "prompt_id",
        "theme",
        "rubric_size",
        "total_possible_points",
        "total_penalty_points",
        "penalty_mass_ratio",
        "prompt_char_length",
        "num_turns",
        "num_user_turns",
    ]
    if not dataset.samples:
        return pd.DataFrame(columns=columns)

    rows = []
    for s in dataset.samples:
        pos = total_positive_points(s)
        pen = total_penalty_points(s)
        rows.append(
            {
                "prompt_id": s.prompt_id,
                "theme": tag_value(s.example_tags, "theme") or "_unknown",
                "rubric_size": len(s.rubrics),
                "total_possible_points": pos,
                "total_penalty_points": pen,
                "penalty_mass_ratio": pen / pos if pos > 0 else None,
                "prompt_char_length": total_prompt_chars(s),
                "num_turns": len(s.prompt),
                "num_user_turns": sum(1 for t in s.prompt if t.get("role") == "user"),
            }
        )
    return pd.DataFrame(rows)


def parse_tag_prefixes(tags: list[str]) -> dict[str, dict[str, int]]:
    """Parse 'key: value' tags and aggregate value_counts per prefix.

    Tags that do not contain ':' are collected under the special key
    '_no_prefix'.

    Args:
        tags: Flat list of raw tag strings.

    Returns:
        Dict mapping prefix string to a nested dict of value → count.
    """
    prefix_map: dict[str, dict[str, int]] = {}
    for tag in tags:
        if ":" in tag:
            prefix, value = tag.split(":", maxsplit=1)
            prefix = prefix.strip()
            value = value.strip()
        else:
            prefix = "_no_prefix"
            value = tag.strip()
        prefix_map.setdefault(prefix, {})
        prefix_map[prefix][value] = prefix_map[prefix].get(value, 0) + 1
    return prefix_map
