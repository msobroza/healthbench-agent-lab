"""Shared utilities for all analysis modules.

Provides stateless helper functions and shared constants used across
healthbench_agent.analysis.exploration, insights, and visualization.
Does not import from agents/, evaluation/, or dataset/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

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


def parse_tag_prefixes(tags: list[str]) -> dict[str, dict[str, int]]:
    """Parse 'key: value' tags and aggregate value_counts per prefix.

    Tags that do not contain ': ' are collected under the special key
    '_no_prefix'.

    Args:
        tags: Flat list of raw tag strings.

    Returns:
        Dict mapping prefix string to a nested dict of value → count.
    """
    prefix_map: dict[str, dict[str, int]] = {}
    for tag in tags:
        if ": " in tag:
            prefix, value = tag.split(": ", maxsplit=1)
            prefix = prefix.strip()
            value = value.strip()
        else:
            prefix = "_no_prefix"
            value = tag.strip()
        prefix_map.setdefault(prefix, {})
        prefix_map[prefix][value] = prefix_map[prefix].get(value, 0) + 1
    return prefix_map
