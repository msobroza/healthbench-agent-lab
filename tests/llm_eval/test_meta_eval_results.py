"""Tests for MetricResultsView, results_io, and results_plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.meta_evaluation import MetricResults
from healthbench_agent.llm_eval.meta_eval.results_io import load_results, save_results
from healthbench_agent.llm_eval.meta_eval.results_plots import (
    plot_calibration_curve,
    plot_dimension_confusion,
)
from healthbench_agent.llm_eval.meta_eval.results_view import MetricResultsView


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


def test_to_pandas_returns_dataframe_with_score_rows(view):
    df = view.to_pandas()
    assert "gold_score" in df["metric"].tolist()
    assert "cohens_kappa" in df["metric"].tolist()


def test_save_and_load_round_trip(tmp_path: Path, view):
    save_results(view.results, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    reloaded = load_results(tmp_path)
    assert reloaded.results.scores == view.results.scores


def test_load_raises_on_unknown_schema_version(tmp_path: Path, view):
    payload = view.results.to_dict()
    payload["schema_version"] = 999
    (tmp_path / "metrics.json").write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        load_results(tmp_path)


def test_load_missing_metrics_json_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)


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
    assert diff["metric"].tolist() == ["cohens_kappa", "gold_score"]


def test_plot_calibration_curve_returns_axes_when_data_present():
    results = MetricResults(
        scores={"calibration_curve": {1: 0.08, 3: 0.06, 5: 0.05, 7: 0.04}},
        n_samples_graded=10,
        n_rubrics_graded=10,
        judge_metadata={},
    )
    ax = plot_calibration_curve(results)
    assert ax is not None


def test_plot_dimension_confusion_returns_axes_when_data_present():
    results = MetricResults(
        scores={"per_dimension_confusion": {"accuracy": {"tp": 1, "fp": 0, "tn": 1, "fn": 0}}},
        n_samples_graded=2,
        n_rubrics_graded=2,
        judge_metadata={},
    )
    ax = plot_dimension_confusion(results)
    assert ax is not None


def test_plot_calibration_curve_raises_keyerror_when_missing(view):
    with pytest.raises(KeyError):
        plot_calibration_curve(view.results)


def test_plot_calibration_curve_after_save_load_round_trip_uses_numeric_order(tmp_path: Path):
    """After save→load, calibration curve keys are str. Plot must sort numerically."""
    results = MetricResults(
        scores={"calibration_curve": {1: 0.04, 3: 0.08, 5: 0.06, 7: 0.05, 10: 0.03}},
        n_samples_graded=10,
        n_rubrics_graded=10,
        judge_metadata={},
    )
    save_results(results, tmp_path)
    reloaded = load_results(tmp_path)
    ax = plot_calibration_curve(reloaded.results)
    xdata = list(ax.lines[0].get_xdata())
    # Coerce x values to int so that lex-sorted str keys ('1','10','3','5','7')
    # become the numerically-misordered [1,10,3,5,7] and the assertion fails.
    numeric_x = [int(x) for x in xdata]
    # If sorted numerically, the values should be monotonically increasing.
    assert numeric_x == sorted(numeric_x)
    # And the count must include all 5 k values, not just the lex-sortable subset.
    assert len(xdata) == 5


def test_plot_dimension_confusion_raises_keyerror_when_missing(view):
    """The dimension-confusion plot must fail loudly when its score is absent."""
    with pytest.raises(KeyError):
        plot_dimension_confusion(view.results)


def test_plot_calibration_curve_falls_back_to_string_sort_on_mixed_keys():
    """Mixed-type keys (some non-coercible) fall through to lexicographic sort."""
    results = MetricResults(
        scores={"calibration_curve": {"a": 0.1, "b": 0.2}},
        n_samples_graded=2,
        n_rubrics_graded=2,
        judge_metadata={},
    )
    ax = plot_calibration_curve(results)
    xdata = list(ax.lines[0].get_xdata())
    assert xdata == sorted(xdata)


def test_summary_marks_unregistered_metrics_with_question_mark():
    """summary() must not crash when a score key is not in the metric registry."""
    results = MetricResults(
        scores={"not_registered": 0.5},
        n_samples_graded=1,
        n_rubrics_graded=1,
        judge_metadata={"judge_model": "fake"},
    )
    view_local = MetricResultsView(results=results)
    assert "?" in view_local.summary()


def test_to_pandas_flattens_dict_valued_metrics_and_tags_unknown_metric():
    """to_pandas flattens dict metrics into sub-rows and degrades gracefully on unknown names."""
    results = MetricResults(
        scores={
            "per_dimension_confusion": {"axis1": {"tp": 1, "fp": 0}},
            "unknown_metric": 0.25,
        },
        n_samples_graded=1,
        n_rubrics_graded=1,
        judge_metadata={},
    )
    df = MetricResultsView(results=results).to_pandas()
    # Dict-valued metric got flattened into sub-keys with dotted metric names.
    assert "per_dimension_confusion.axis1" in df["metric"].tolist()
    # Unknown registry entry still appears, tagged with a '?' level.
    unknown_row = df[df["metric"] == "unknown_metric"].iloc[0]
    assert unknown_row["level"] == "?"


def test_compare_returns_see_details_sentinel_for_dict_metrics():
    """Dict-valued metric deltas are rendered as the 'see details' sentinel string."""
    a = MetricResultsView(
        results=MetricResults(
            scores={"per_dimension_confusion": {"x": {"tp": 1}}},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={},
        )
    )
    b = MetricResultsView(
        results=MetricResults(
            scores={"per_dimension_confusion": {"x": {"tp": 2}}},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={},
        )
    )
    diff = a.compare(b)
    assert diff.iloc[0]["delta"] == "see details"


def test_verdicts_loads_parquet_lazily_and_memoises(tmp_path: Path):
    """verdicts() should read parquet once and reuse the cached DataFrame."""
    import pandas as pd

    parquet = tmp_path / "verdicts.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(parquet)
    results = MetricResults(
        scores={"gold_score": 0.5},
        n_samples_graded=2,
        n_rubrics_graded=2,
        judge_metadata={},
        verdicts_path=parquet,
    )
    view_local = MetricResultsView(results=results)
    first = view_local.verdicts()
    second = view_local.verdicts()
    assert list(first.columns) == ["a", "b"]
    # Memoised: the second call returns the same DataFrame instance.
    assert first is second


def test_verdicts_raises_file_not_found_when_path_missing():
    """Views constructed in-memory (no output_dir) raise on verdicts()."""
    results = MetricResults(
        scores={"gold_score": 0.5},
        n_samples_graded=1,
        n_rubrics_graded=1,
        judge_metadata={},
    )
    view_local = MetricResultsView(results=results)
    with pytest.raises(FileNotFoundError, match="verdicts_path is None"):
        view_local.verdicts()
