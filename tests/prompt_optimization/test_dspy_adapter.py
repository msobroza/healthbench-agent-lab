"""Tests for the DSPyOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.adapters.dspy_adapter import (
    DSPyOptimizer,
)
from healthbench_agent.prompt_optimization.config import DSPyConfig


def _make_sample() -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id="test-1",
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


class TestDSPyOptimizer:
    def test_requires_samples(self):
        config = DSPyConfig()
        optimizer = DSPyOptimizer(config)
        mock_metric = MagicMock(return_value=0.5)

        with pytest.raises(ValueError, match="samples"):
            optimizer.optimize("prompt", samples=None, metric=mock_metric)

    def test_requires_metric(self):
        config = DSPyConfig()
        optimizer = DSPyOptimizer(config)
        sample = _make_sample()

        with pytest.raises(ValueError, match="metric"):
            optimizer.optimize("prompt", samples=[sample], metric=None)

    @patch("healthbench_agent.prompt_optimization.adapters.dspy_adapter.dspy")
    def test_copro_optimization(self, mock_dspy):
        config = DSPyConfig(dspy_optimizer="copro", max_trials=5)
        optimizer = DSPyOptimizer(config)

        mock_compiled = MagicMock()
        mock_compiled.generate.signature.instructions = "Optimized health prompt."
        mock_dspy.COPRO.return_value.compile.return_value = mock_compiled

        mock_metric = MagicMock(return_value=0.7)
        sample = _make_sample()

        result = optimizer.optimize(
            current_prompt="Original prompt.",
            samples=[sample],
            metric=mock_metric,
        )

        assert result.optimizer_name == "dspy"
        assert result.optimized_prompt == "Optimized health prompt."
        assert result.config["dspy_optimizer"] == "copro"

    @patch("healthbench_agent.prompt_optimization.adapters.dspy_adapter.dspy")
    def test_miprov2_optimization(self, mock_dspy):
        config = DSPyConfig(dspy_optimizer="miprov2", max_trials=5)
        optimizer = DSPyOptimizer(config)

        mock_compiled = MagicMock()
        mock_compiled.generate.signature.instructions = "MIPROv2 optimized."
        mock_dspy.MIPROv2.return_value.compile.return_value = mock_compiled

        mock_metric = MagicMock(return_value=0.8)
        sample = _make_sample()

        result = optimizer.optimize(
            current_prompt="Original.",
            samples=[sample],
            metric=mock_metric,
        )

        assert result.optimized_prompt == "MIPROv2 optimized."
        assert result.config["dspy_optimizer"] == "miprov2"
