"""User-facing wrapper around the pure-domain MetricResults dataclass.

Adds REPL/Jupyter ergonomics and a lazy ``verdicts`` parquet loader
without dragging matplotlib or pyarrow into the domain layer. Returned
by ``meta_evaluate`` and ``run_meta_eval`` so users get rich UX by
default.

File-IO helpers live in :mod:`.results_io` and plot helpers live in
:mod:`.results_plots`; both take the pure-domain
:class:`MetricResults` as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from healthbench_agent.domain.meta_evaluation import MetricResults

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MetricResultsView:
    """Sklearn-style ergonomics around a MetricResults dataclass.

    Wraps a pure-domain :class:`MetricResults` and adds rich-text REPL
    output, a pandas conversion, and a lazy verdicts-parquet loader.
    Pandas and pyarrow are imported lazily inside the methods that need
    them so the rest of the package stays usable when those optional
    deps are missing.

    Attributes:
        results: The wrapped pure-domain dataclass.
        _verdicts_cache: Memoised verdicts DataFrame, populated on the
            first call to :meth:`verdicts`. ``pd`` is only imported
            under ``TYPE_CHECKING`` so the annotation stays as a lazy
            string thanks to ``from __future__ import annotations``.
    """

    results: MetricResults
    _verdicts_cache: pd.DataFrame | None = None

    # ---- pretty printing -------------------------------------------------

    def __repr__(self) -> str:
        """One-line REPL summary with judge identity and sample counts."""
        meta = self.results.judge_metadata
        judge = meta.get("judge_model", "unknown")
        k = meta.get("k", meta.get("n_samples", "?"))
        return f"MetricResultsView(judge={judge}, k={k}, n_samples={self.results.n_samples_graded})"

    def summary(self) -> str:
        """Render a multi-line text table of all metric scores.

        Returns:
            A string with a header line, separator, column titles, and
            one row per metric showing its name, level, and value.
        """
        from healthbench_agent.llm_eval.meta_eval.registry import get_meta_metric

        meta = self.results.judge_metadata
        header_line = (
            f"MetricResults(judge={meta.get('judge_model', '?')}, "
            f"k={meta.get('k', meta.get('n_samples', '?'))}, "
            f"n={self.results.n_samples_graded})"
        )
        rule = "─" * 60
        rows = [header_line, rule, f"{'METRIC':<26} {'LEVEL':<8} VALUE"]
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value.upper()
            except KeyError:
                level = "?"
            rows.append(f"{name:<26} {level:<8} {value}")
        return "\n".join(rows)

    # ---- conversions -----------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """Convert scores into a tidy DataFrame with metric/level/value columns.

        Dict-valued metrics are flattened to ``"<metric>.<sub_key>"`` rows
        so each output row holds a single scalar value.

        Returns:
            A pandas DataFrame with columns ``metric``, ``level``,
            ``value``; one row per scalar metric value.
        """
        import pandas as pd

        from healthbench_agent.llm_eval.meta_eval.registry import get_meta_metric

        rows: list[dict[str, Any]] = []
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value
            except KeyError:
                level = "?"
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    rows.append({"metric": f"{name}.{sub_key}", "level": level, "value": sub_val})
            else:
                rows.append({"metric": name, "level": level, "value": value})
        return pd.DataFrame(rows)

    # ---- verdicts loader -------------------------------------------------

    def verdicts(self) -> pd.DataFrame:
        """Load and memoise the verdicts parquet referenced by the results.

        Returns:
            A pandas DataFrame parsed from ``self.results.verdicts_path``.

        Raises:
            FileNotFoundError: If ``verdicts_path`` is None (the view was
                constructed in-memory without an output directory).
        """
        import pandas as pd

        if self._verdicts_cache is not None:
            return self._verdicts_cache
        if self.results.verdicts_path is None:
            raise FileNotFoundError(
                "verdicts_path is None — view was constructed in-memory without an output_dir"
            )
        self._verdicts_cache = pd.read_parquet(self.results.verdicts_path)
        return self._verdicts_cache

    # ---- comparison ------------------------------------------------------

    def compare(self, other: MetricResultsView) -> pd.DataFrame:
        """Diff two MetricResultsView instances into a row-per-metric DataFrame.

        Scalar metrics get a numeric ``delta = other - self``; dict-valued
        metrics (e.g. ``per_dimension_confusion``, ``calibration_curve``)
        get the sentinel string ``"see details"`` so the column stays
        present without mixing shapes.

        Args:
            other: The other view to compare against.

        Returns:
            A DataFrame with columns ``metric``, ``self``, ``other``,
            ``delta``, one row per distinct metric name across both views,
            sorted alphabetically.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        all_keys = sorted(set(self.results.scores) | set(other.results.scores))
        for name in all_keys:
            self_val = self.results.scores.get(name)
            other_val = other.results.scores.get(name)
            if isinstance(self_val, int | float) and isinstance(other_val, int | float):
                delta: float | str = float(other_val) - float(self_val)
            else:
                delta = "see details"
            rows.append({"metric": name, "self": self_val, "other": other_val, "delta": delta})
        return pd.DataFrame(rows)
