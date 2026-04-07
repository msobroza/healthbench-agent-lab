"""Tests for the LabelledSample / MetricResults domain types."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.meta_evaluation import (
    SCHEMA_VERSION,
    LabelledSample,
    MetricResults,
)
from healthbench_agent.domain.rubric import RubricItem


def test_labelled_sample_required_fields_only():
    sample = LabelledSample(
        prompt_id="p1",
        prompt=[{"role": "user", "content": "hello"}],
        rubrics=[RubricItem(criterion="says hi", points=1.0)],
    )
    assert sample.gold_response is None
    assert sample.expected == {}
    assert sample.language is None
    assert sample.specialty is None
    assert sample.user_persona is None
    assert sample.metadata == {}


def test_labelled_sample_full_spec_md_fields_round_trip():
    sample = LabelledSample(
        prompt_id="p2",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[],
        gold_response="ideal answer",
        expected={"says hi": True},
        language="en",
        specialty="cardiology",
        user_persona="patient",
        metadata={"clinical_urgency": "emergency"},
    )
    assert sample.gold_response == "ideal answer"
    assert sample.expected == {"says hi": True}
    assert sample.language == "en"
    assert sample.specialty == "cardiology"
    assert sample.user_persona == "patient"
    assert sample.metadata == {"clinical_urgency": "emergency"}


def test_metric_results_to_from_dict_round_trip():
    results = MetricResults(
        scores={"gold_score": 0.873, "cohens_kappa": 0.612},
        n_samples_graded=100,
        n_rubrics_graded=287,
        judge_metadata={"judge_model": "openai/gpt-4.1", "k": 7},
    )
    payload = results.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "verdicts_path" not in payload  # None is omitted
    rebuilt = MetricResults.from_dict(payload)
    assert rebuilt == results


def test_metric_results_to_dict_coerces_path_to_string():
    results = MetricResults(
        scores={},
        n_samples_graded=0,
        n_rubrics_graded=0,
        judge_metadata={},
        verdicts_path=Path("/tmp/run/verdicts.parquet"),
    )
    payload = results.to_dict()
    assert payload["verdicts_path"] == "/tmp/run/verdicts.parquet"
    rebuilt = MetricResults.from_dict(payload)
    assert rebuilt.verdicts_path == Path("/tmp/run/verdicts.parquet")


def test_metric_results_from_dict_rejects_unknown_schema_version():
    payload = {
        "scores": {},
        "n_samples_graded": 0,
        "n_rubrics_graded": 0,
        "judge_metadata": {},
        "schema_version": SCHEMA_VERSION + 1,
    }
    with pytest.raises(ValueError, match="schema_version"):
        MetricResults.from_dict(payload)


def test_metric_results_carries_no_ux_methods():
    """Domain dataclass must stay stdlib-only — no plot_* / to_pandas / load."""
    forbidden = {"plot_calibration_curve", "plot_dimension_confusion", "to_pandas", "load"}
    assert forbidden.isdisjoint(set(dir(MetricResults)))
