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
