"""Tests for the TextGradOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.adapters.textgrad_adapter import (
    TextGradOptimizer,
)
from healthbench_agent.prompt_optimization.config import TextGradConfig


def _make_sample() -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id="test-1",
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


class TestTextGradOptimizer:
    def test_requires_samples(self):
        config = TextGradConfig()
        optimizer = TextGradOptimizer(config)
        mock_metric = MagicMock(return_value=0.5)

        with pytest.raises(ValueError, match="samples"):
            optimizer.optimize("prompt", samples=None, metric=mock_metric)

    def test_requires_metric(self):
        config = TextGradConfig()
        optimizer = TextGradOptimizer(config)
        sample = _make_sample()

        with pytest.raises(ValueError, match="metric"):
            optimizer.optimize("prompt", samples=[sample], metric=None)

    @patch("healthbench_agent.prompt_optimization.adapters.textgrad_adapter.textgrad")
    def test_optimization_runs_steps(self, mock_tg):
        config = TextGradConfig(steps=3)
        optimizer = TextGradOptimizer(config)

        # Mock TextGrad Variable — needs .value attribute
        mock_var = MagicMock()
        mock_var.value = "Optimized via TextGrad."
        mock_tg.Variable.return_value = mock_var

        mock_engine = MagicMock()
        mock_tg.get_engine.return_value = mock_engine
        mock_optimizer = MagicMock()
        mock_tg.TGD.return_value = mock_optimizer

        mock_metric = MagicMock(return_value=0.7)
        sample = _make_sample()

        result = optimizer.optimize(
            current_prompt="Original prompt.",
            samples=[sample],
            metric=mock_metric,
        )

        assert result.optimizer_name == "textgrad"
        assert result.optimized_prompt == "Optimized via TextGrad."
        assert result.config["steps"] == 3

    @patch("healthbench_agent.prompt_optimization.adapters.textgrad_adapter.textgrad")
    def test_config_stored_in_result(self, mock_tg):
        config = TextGradConfig(steps=5)
        optimizer = TextGradOptimizer(config)

        mock_var = MagicMock()
        mock_var.value = "Result."
        mock_tg.Variable.return_value = mock_var
        mock_tg.get_engine.return_value = MagicMock()
        mock_tg.TGD.return_value = MagicMock()

        mock_metric = MagicMock(return_value=0.5)
        sample = _make_sample()

        result = optimizer.optimize("Original.", samples=[sample], metric=mock_metric)

        assert result.config["optimizer"] == "textgrad"
        assert result.config["steps"] == 5
