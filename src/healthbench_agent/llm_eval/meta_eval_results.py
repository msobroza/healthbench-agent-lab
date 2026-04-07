"""User-facing wrapper around the pure-domain MetricResults dataclass.

Adds REPL/Jupyter/IO/plot helpers without dragging matplotlib, pandas,
or pyarrow into the domain layer. Returned by ``meta_evaluate`` and
``run_meta_eval`` so users get rich UX by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from healthbench_agent.domain.meta_evaluation import MetricResults

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MetricResultsView:
    """Sklearn-style ergonomics around a MetricResults dataclass.

    Wraps a pure-domain :class:`MetricResults` and adds rich-text REPL
    output, an HTML representation for Jupyter, pandas/markdown
    conversions, and disk persistence helpers. Pandas and pyarrow are
    imported lazily inside the methods that need them so the rest of
    the package stays usable when those optional deps are missing.

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
        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

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

    def _repr_html_(self) -> str:
        """Render an HTML table of all metric scores for Jupyter."""
        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

        rows_html = []
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value.upper()
            except KeyError:
                level = "?"
            rows_html.append(f"<tr><td>{name}</td><td>{level}</td><td>{value}</td></tr>")
        return (
            "<table>"
            "<thead><tr><th>metric</th><th>level</th><th>value</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
        )

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

        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

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

    def to_markdown(self) -> str:
        """Render the scores as a Markdown table via :meth:`to_pandas`."""
        return self.to_pandas().to_markdown(index=False)

    # ---- IO --------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path | str) -> MetricResultsView:
        """Load a previously-saved view from ``run_dir``.

        Reads ``metrics.json`` and reattaches ``verdicts.parquet`` to the
        result if it exists alongside it.

        Args:
            run_dir: Directory previously written by :meth:`save`.

        Returns:
            A new view wrapping the deserialised :class:`MetricResults`.

        Raises:
            FileNotFoundError: If ``metrics.json`` is missing.
            ValueError: If the persisted schema_version is newer than
                this build understands.
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
        return cls(results=results)

    def save(self, run_dir: Path | str) -> None:
        """Write the wrapped results as indented ``metrics.json`` in ``run_dir``.

        Args:
            run_dir: Destination directory; created if it does not exist.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(self.results.to_dict(), indent=2))

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
