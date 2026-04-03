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


class TestEndToEndMetric:
    def test_returns_aggregate_score(self):
        sample = _make_sample()
        mock_judge = MagicMock()
        mock_judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]

        mock_pipeline = AsyncMock()
        mock_pipeline.generate.return_value = "Aspirin is used for pain relief."

        mock_config = MagicMock()
        mock_config.model_copy.return_value = mock_config

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=mock_pipeline,
        ):
            metric = EndToEndMetric(
                agent_config=mock_config,
                judge=mock_judge,
                samples=[sample],
            )
            score = metric("You are a health assistant.")

        assert score == pytest.approx(1.0)
        mock_config.model_copy.assert_called_once()

    def test_returns_zero_when_no_criteria_met(self):
        sample = _make_sample()
        mock_judge = MagicMock()
        mock_judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=False)
        ]

        mock_pipeline = AsyncMock()
        mock_pipeline.generate.return_value = "I don't know."

        mock_config = MagicMock()
        mock_config.model_copy.return_value = mock_config

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=mock_pipeline,
        ):
            metric = EndToEndMetric(
                agent_config=mock_config,
                judge=mock_judge,
                samples=[sample],
            )
            score = metric("Bad prompt.")

        assert score == pytest.approx(0.0)

    def test_does_not_mutate_original_config(self):
        sample = _make_sample()
        mock_judge = MagicMock()
        mock_judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]

        mock_pipeline = AsyncMock()
        mock_pipeline.generate.return_value = "Response."

        mock_config = MagicMock()
        copied_config = MagicMock()
        mock_config.model_copy.return_value = copied_config

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=mock_pipeline,
        ) as mock_create:
            metric = EndToEndMetric(
                agent_config=mock_config,
                judge=mock_judge,
                samples=[sample],
            )
            metric("New prompt text.")

        # create_pipeline should receive the copy, not the original
        mock_create.assert_called_once_with(copied_config)
