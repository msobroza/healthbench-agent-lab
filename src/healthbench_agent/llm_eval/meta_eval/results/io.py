"""Save/load helpers for :class:`MetricResults` artefacts.

Kept as free functions (no class methods) so callers can serialise a
raw :class:`MetricResults` without first constructing a
:class:`MetricResultsView`. The view layer is strictly for ergonomics.
"""

from __future__ import annotations

import json
from pathlib import Path

from healthbench_agent.domain.meta_evaluation import MetricResults

from .view import MetricResultsView


def save_results(results: MetricResults, run_dir: Path | str) -> None:
    """Write the results as indented ``metrics.json`` in ``run_dir``.

    Args:
        results: Pure-domain metric results to persist.
        run_dir: Destination directory; created if it does not exist.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(results.to_dict(), indent=2))


def load_results(run_dir: Path | str) -> MetricResultsView:
    """Load a previously-saved view from ``run_dir``.

    Reads ``metrics.json`` and reattaches ``verdicts.parquet`` to the
    result if it exists alongside it.

    Args:
        run_dir: Directory previously written by :func:`save_results`.

    Returns:
        A new view wrapping the deserialised :class:`MetricResults`.

    Raises:
        FileNotFoundError: If ``metrics.json`` is missing.
        ValueError: If the persisted schema_version is newer than this
            build understands.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    data = json.loads(metrics_path.read_text())
    results = MetricResults.from_dict(data)
    verdicts_path = run_dir / "verdicts.parquet"
    if verdicts_path.exists():
        results.verdicts_path = verdicts_path
    return MetricResultsView(results=results)
