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
        """Render the per-metric table as a GitHub-flavoured Markdown table.

        Requires the ``tabulate`` package (not a hard dependency). Install with
        ``uv add tabulate`` or ``pip install tabulate`` before calling.

        Returns:
            A Markdown string with one row per metric entry (dict-valued metrics
            are expanded by :meth:`to_pandas`).

        Raises:
            ImportError: When ``tabulate`` is not installed. The message names
                the missing package and the install command.
        """
        try:
            import tabulate  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MetricResultsView.to_markdown() requires the 'tabulate' package. "
                "Install with: uv add tabulate"
            ) from exc
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

    # ---- plot helpers ----------------------------------------------------

    def plot_calibration_curve(self, ax: Any = None) -> Any:
        """Plot the bootstrap SE calibration curve over k.

        Args:
            ax: Existing matplotlib Axes to draw into. When None, a new
                figure and axes are created via ``plt.subplots()``.

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

        if "calibration_curve" not in self.results.scores:
            raise KeyError("calibration_curve is not in results.scores")
        curve = self.results.scores["calibration_curve"]
        if ax is None:
            _, ax = plt.subplots()
        ks = sorted(curve.keys())
        ax.plot(ks, [curve[k] for k in ks], marker="o")
        ax.set_xlabel("k (number of judge passes)")
        ax.set_ylabel("Bootstrap SE of agreement")
        ax.set_title("Calibration curve")
        return ax

    def plot_dimension_confusion(self, ax: Any = None) -> Any:
        """Stacked bar chart of per-dimension confusion counts.

        Args:
            ax: Existing matplotlib Axes to draw into. When None, a new
                figure and axes are created via ``plt.subplots()``.

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

        if "per_dimension_confusion" not in self.results.scores:
            raise KeyError("per_dimension_confusion is not in results.scores")
        confusion = self.results.scores["per_dimension_confusion"]
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
