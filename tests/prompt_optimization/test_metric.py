"""Tests for EndToEndMetric."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.metric import EndToEndMetric


def _make_sample(prompt_id: str = "test-1") -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id=prompt_id,
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


def _make_pipeline(response_text: str) -> MagicMock:
    """Build a mock AgentPipeline whose .generate returns the given text."""
    pipeline = MagicMock()
    pipeline.generate = AsyncMock(return_value=response_text)
    return pipeline


def _make_config() -> MagicMock:
    """Build a mock RootAgentPipelineConfig that records model_copy calls."""
    config = MagicMock()
    config.model_copy.return_value = config
    return config


class TestEndToEndMetric:
    def test_returns_aggregate_score_when_criteria_met(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Aspirin is used for pain relief.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            score = metric("You are a health assistant.")

        assert score == pytest.approx(1.0)

    def test_returns_zero_when_no_criteria_met(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=False)
        ]
        pipeline = _make_pipeline("I don't know.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            score = metric("Bad prompt.")

        assert score == pytest.approx(0.0)

    def test_passes_instruction_override_to_create_pipeline(self):
        """The candidate prompt must reach the agent via instruction_override.

        Regression test for the bug where model_copy(update=...) silently
        dropped the override and every trial scored the original prompt.
        """
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ) as mock_create:
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            metric("New candidate prompt.")

        config.model_copy.assert_called_once_with(
            update={"instruction_override": "New candidate prompt."}
        )
        mock_create.assert_called_once_with(config)

    def test_pipeline_generate_is_called_with_sample_prompt(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            metric("Some prompt.")

        pipeline.generate.assert_awaited_once_with(sample.prompt)
