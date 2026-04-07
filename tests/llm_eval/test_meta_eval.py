"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""

from __future__ import annotations

import pandas as pd
import pytest

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.meta_eval import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    MetricLevel,
    MetricSpec,
    axis_filter,
    get_meta_metric,
    gold_score,
    metadata_filter,
    register_meta_metric,
    registered_meta_metrics,
    specialty_filter,
)


def test_metric_level_enum_values():
    assert MetricLevel.SAMPLE.value == "sample"
    assert MetricLevel.RUBRIC.value == "rubric"
    assert MetricLevel.ANY.value == "any"


def test_register_meta_metric_round_trip():
    @register_meta_metric("dummy_metric", level=MetricLevel.ANY, description="dummy")
    def _fn(df):
        return 0.0

    spec = get_meta_metric("dummy_metric")
    assert isinstance(spec, MetricSpec)
    assert spec.name == "dummy_metric"
    assert spec.level is MetricLevel.ANY
    assert spec.description == "dummy"
    assert spec.fn is _fn
    # Cleanup
    del registered_meta_metrics()["dummy_metric"]


def test_register_meta_metric_requires_level_and_description():
    with pytest.raises(TypeError):

        @register_meta_metric("missing_args")  # type: ignore[call-arg]
        def _bad(df):
            return 0.0


def test_get_meta_metric_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_meta_metric("totally_unknown_metric")


def test_axis_tag_prefix_includes_trailing_space():
    assert AXIS_TAG_PREFIX == "axis: "


def test_axis_filter_matches_tag():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: accuracy"]))
    assert not f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_axis_filter_matches_category_field():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, category="accuracy"))


def test_axis_filter_accepts_multiple_axes():
    f = axis_filter("accuracy", "emergency")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_metadata_filter_top_level_fields():
    f = metadata_filter(language="en", specialty="cardiology")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], language="en", specialty="cardiology"
    )
    assert f(sample)
    other = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], language="fr", specialty="cardiology"
    )
    assert not f(other)


def test_metadata_filter_metadata_dict_keys():
    f = metadata_filter(clinical_urgency="emergency")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "emergency"}
    )
    assert f(sample)
    other = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "routine"}
    )
    assert not f(other)


def test_specialty_filter_matches_any():
    f = specialty_filter("cardiology", "pediatrics")
    assert f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="cardiology"))
    assert not f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="general"))


def test_empty_filter_error_carries_filter_reprs():
    err = EmptyFilterError(sample_filter="metadata_filter(language='fr')", rubric_filter=None)
    assert "language='fr'" in str(err)


def _sample_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_gold_score_perfect_judge_returns_one():
    df = _sample_rows(
        [
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c1",
                "points": 1.0,
                "observed_met": True,
            },
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c2",
                "points": 2.0,
                "observed_met": True,
            },
        ]
    )
    assert gold_score(df) == pytest.approx(1.0)


def test_gold_score_clips_per_conversation_to_zero_when_negative():
    df = _sample_rows(
        [
            # raw = -3 / 1 = -3 → clipped to 0
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c1",
                "points": 1.0,
                "observed_met": False,
            },
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c2",
                "points": -3.0,
                "observed_met": True,
            },
            # second conversation: perfect, raw = 1
            {
                "prompt_id": "p2",
                "sample_k": 1,
                "criterion": "c3",
                "points": 1.0,
                "observed_met": True,
            },
        ]
    )
    # mean(0.0, 1.0) == 0.5
    assert gold_score(df) == pytest.approx(0.5)


def test_gold_score_empty_dataframe_returns_zero():
    df = pd.DataFrame(columns=["prompt_id", "sample_k", "criterion", "points", "observed_met"])
    assert gold_score(df) == 0.0


def test_gold_score_registered_with_sample_level():
    spec = get_meta_metric("gold_score")
    assert spec.level is MetricLevel.SAMPLE
