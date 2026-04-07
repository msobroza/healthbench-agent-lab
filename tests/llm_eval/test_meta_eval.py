"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""

from __future__ import annotations

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
