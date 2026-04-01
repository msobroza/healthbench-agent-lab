"""Analysis function registry for the HealthBench agent project.

Provides a decorator-based registry that maps analysis function names to
their metadata and callables. Higher-level runners use this module to
discover and execute registered analyses by name or category.

Dependency position: imported by analysis sub-modules (exploration.py,
visualization.py, etc.) and by top-level experiment scripts. Does not
import from agents/ or evaluation/.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain.dataset import DatasetSubset, HealthBenchDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry entry type
# ---------------------------------------------------------------------------

AnalysisFunction = Callable[
    [list[HealthBenchDataset], Path, bool],
    dict[str, Any],
]

CategoryOption = str  # "exploration" | "insights" | "visualization"


class AnalysisEntry:
    """Metadata and callable for a single registered analysis.

    Attributes:
        name: Unique identifier used to look up and invoke the analysis.
        category: Grouping label — one of 'exploration', 'insights',
            or 'visualization'.
        datasets: Subset labels this analysis is declared to support.
        function: The analysis callable, accepting datasets, output_dir,
            and save flag.
    """

    def __init__(
        self,
        name: str,
        category: CategoryOption,
        datasets: list[DatasetSubset],
        function: AnalysisFunction,
    ) -> None:
        self.name = name
        self.category = category
        self.datasets = datasets
        self.function = function

    def __repr__(self) -> str:
        return (
            f"AnalysisEntry(name={self.name!r}, category={self.category!r}, "
            f"datasets={self.datasets!r})"
        )


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_analysis_registry: dict[str, AnalysisEntry] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_analysis(
    name: str,
    category: CategoryOption,
    datasets: list[DatasetSubset],
) -> Callable[[AnalysisFunction], AnalysisFunction]:
    """Decorator that registers an analysis function in the module-level registry.

    The decorated function is stored under *name* in ``_analysis_registry``.
    Calling the decorator a second time with the same name overwrites the
    previous entry and logs a warning.

    Args:
        name: Unique identifier for the analysis.  Used as the key in
            ``_analysis_registry`` and passed to ``run_one``.
        category: Logical grouping — one of 'exploration', 'insights', or
            'visualization'.
        datasets: List of dataset subset labels the analysis supports.
            Values must be valid ``DatasetSubset`` literals: 'main', 'hard',
            or 'consensus'.

    Returns:
        A decorator that registers the wrapped function and returns it
        unchanged so the function remains callable under its original name.

    Example:
        @register_analysis(name="sample_counts", category="exploration",
                           datasets=["main", "hard", "consensus"])
        def compute_sample_counts(
            datasets: list[HealthBenchDataset],
            output_dir: Path,
            save: bool = False,
        ) -> dict[str, Any]:
            ...
    """

    def decorator(function: AnalysisFunction) -> AnalysisFunction:
        if name in _analysis_registry:
            logger.warning(
                "Analysis %r is already registered; overwriting with %s",
                name,
                function.__qualname__,
            )
        _analysis_registry[name] = AnalysisEntry(
            name=name,
            category=category,
            datasets=list(datasets),
            function=function,
        )
        return function

    return decorator


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_one(
    name: str,
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Run a single registered analysis by name.

    Filters *datasets* to only those whose ``subset`` attribute appears in
    the entry's declared dataset list before passing them to the analysis
    function.

    Args:
        name: Registered analysis name to execute.
        datasets: Available datasets; will be filtered to the subset labels
            declared in the registry entry.
        output_dir: Directory passed to the analysis function for saving
            artifacts.
        save: When True the analysis function is instructed to persist
            output files.

    Returns:
        Dict mapping *name* to the result returned by the analysis function.

    Raises:
        ValueError: If *name* is not present in ``_analysis_registry``.
    """
    if name not in _analysis_registry:
        raise ValueError(
            f"Analysis {name!r} is not registered. "
            f"Available analyses: {sorted(_analysis_registry)}"
        )

    entry = _analysis_registry[name]
    filtered_datasets = [d for d in datasets if d.subset in entry.datasets]

    logger.info(
        "Running analysis %r (category=%s, subsets=%s)",
        name,
        entry.category,
        [d.subset for d in filtered_datasets],
    )

    result = entry.function(filtered_datasets, output_dir, save)
    return {name: result}


def run_category(
    category: CategoryOption,
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Run all registered analyses that belong to a given category.

    Analyses are executed in insertion order. Each analysis receives only
    the datasets whose subset labels match the entry's declared list.

    Args:
        category: Category label used to filter registry entries — one of
            'exploration', 'insights', or 'visualization'.
        datasets: Available datasets passed (after subset filtering) to each
            matching analysis.
        output_dir: Directory passed to each analysis for saving artifacts.
        save: When True each analysis is instructed to persist output files.

    Returns:
        Dict mapping each matching analysis name to the result returned by
        that analysis function.
    """
    matching_entries = [
        entry for entry in _analysis_registry.values() if entry.category == category
    ]

    logger.info(
        "Running %s analyses in category %r",
        len(matching_entries),
        category,
    )

    results: dict[str, Any] = {}
    for entry in matching_entries:
        filtered_datasets = [d for d in datasets if d.subset in entry.datasets]
        logger.info(
            "Running analysis %r (subsets=%s)",
            entry.name,
            [d.subset for d in filtered_datasets],
        )
        results[entry.name] = entry.function(filtered_datasets, output_dir, save)

    return results


def run_all(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Run every registered analysis in insertion order.

    Each analysis receives only the datasets whose subset labels match the
    entry's declared list.

    Args:
        datasets: Available datasets passed (after subset filtering) to each
            analysis.
        output_dir: Directory passed to each analysis for saving artifacts.
        save: When True each analysis is instructed to persist output files.

    Returns:
        Dict mapping every registered analysis name to the result returned
        by that analysis function.
    """
    logger.info("Running all %s registered analyses", len(_analysis_registry))

    results: dict[str, Any] = {}
    for entry in _analysis_registry.values():
        filtered_datasets = [d for d in datasets if d.subset in entry.datasets]
        logger.info(
            "Running analysis %r (category=%s, subsets=%s)",
            entry.name,
            entry.category,
            [d.subset for d in filtered_datasets],
        )
        results[entry.name] = entry.function(filtered_datasets, output_dir, save)

    return results
