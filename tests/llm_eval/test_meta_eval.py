"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""

from __future__ import annotations

import pytest

from healthbench_agent.llm_eval.meta_eval import (
    MetricLevel,
    MetricSpec,
    get_meta_metric,
    register_meta_metric,
    registered_meta_metrics,
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
