"""Tests for MetricResultsView."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.meta_evaluation import MetricResults
from healthbench_agent.llm_eval.meta_eval_results import MetricResultsView


@pytest.fixture
def view() -> MetricResultsView:
    results = MetricResults(
        scores={"gold_score": 0.873, "cohens_kappa": 0.612},
        n_samples_graded=100,
        n_rubrics_graded=287,
        judge_metadata={"judge_model": "openai/gpt-4.1", "k": 7},
    )
    return MetricResultsView(results=results)


def test_repr_includes_judge_model_and_counts(view):
    text = repr(view)
    assert "openai/gpt-4.1" in text
    assert "100" in text


def test_summary_has_one_line_per_metric_plus_header(view):
    summary = view.summary()
    lines = summary.strip().splitlines()
    assert "gold_score" in summary
    assert "cohens_kappa" in summary
    assert any("METRIC" in line for line in lines)


def test_repr_html_starts_with_table_tag(view):
    assert view._repr_html_().lstrip().startswith("<table")


def test_to_pandas_returns_dataframe_with_score_rows(view):
    df = view.to_pandas()
    assert "gold_score" in df["metric"].tolist()
    assert "cohens_kappa" in df["metric"].tolist()


def test_save_and_load_round_trip(tmp_path: Path, view):
    view.save(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    reloaded = MetricResultsView.load(tmp_path)
    assert reloaded.results.scores == view.results.scores


def test_load_raises_on_unknown_schema_version(tmp_path: Path, view):
    payload = view.results.to_dict()
    payload["schema_version"] = 999
    (tmp_path / "metrics.json").write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        MetricResultsView.load(tmp_path)


def test_load_missing_metrics_json_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MetricResultsView.load(tmp_path)


def test_to_markdown_raises_helpful_error_when_tabulate_missing(monkeypatch, view):
    """to_markdown should raise a clear ImportError naming tabulate."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("No module named 'tabulate'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="tabulate"):
        view.to_markdown()


def test_compare_two_views_returns_diff_dataframe(view):
    other = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.812, "cohens_kappa": 0.589},
            n_samples_graded=100,
            n_rubrics_graded=287,
            judge_metadata={"judge_model": "google/gemini-2.5"},
        )
    )
    diff = view.compare(other)
    assert "metric" in diff.columns
    assert "self" in diff.columns
    assert "other" in diff.columns
    assert "delta" in diff.columns
    assert len(diff) == 2


def test_plot_calibration_curve_returns_axes_when_data_present():
    view = MetricResultsView(
        results=MetricResults(
            scores={"calibration_curve": {1: 0.08, 3: 0.06, 5: 0.05, 7: 0.04}},
            n_samples_graded=10,
            n_rubrics_graded=10,
            judge_metadata={},
        )
    )
    ax = view.plot_calibration_curve()
    assert ax is not None


def test_plot_dimension_confusion_returns_axes_when_data_present():
    view = MetricResultsView(
        results=MetricResults(
            scores={"per_dimension_confusion": {"accuracy": {"tp": 1, "fp": 0, "tn": 1, "fn": 0}}},
            n_samples_graded=2,
            n_rubrics_graded=2,
            judge_metadata={},
        )
    )
    ax = view.plot_dimension_confusion()
    assert ax is not None


def test_plot_calibration_curve_raises_keyerror_when_missing(view):
    with pytest.raises(KeyError):
        view.plot_calibration_curve()
