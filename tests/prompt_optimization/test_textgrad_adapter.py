"""Tests for the TextGradOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.adapters import textgrad_adapter
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


def _patch_textgrad(prompt_value: str) -> MagicMock:
    """Build a mocked textgrad module whose Variable exposes ``prompt_value``."""
    mock_tg = MagicMock()
    mock_var = MagicMock()
    mock_var.value = prompt_value
    mock_tg.Variable.return_value = mock_var
    mock_tg.get_engine.return_value = MagicMock()
    mock_tg.TGD.return_value = MagicMock()
    return mock_tg


class TestTextGradOptimizerValidation:
    def test_requires_samples(self):
        config = TextGradConfig()
        optimizer = TextGradOptimizer(config)
        with patch.object(textgrad_adapter, "textgrad", MagicMock()):
            with pytest.raises(ValueError, match="samples"):
                optimizer.optimize("prompt", samples=None, metric=MagicMock(return_value=0.5))

    def test_requires_metric(self):
        config = TextGradConfig()
        optimizer = TextGradOptimizer(config)
        with patch.object(textgrad_adapter, "textgrad", MagicMock()):
            with pytest.raises(ValueError, match="metric"):
                optimizer.optimize("prompt", samples=[_make_sample()], metric=None)

    def test_raises_when_textgrad_not_installed(self):
        config = TextGradConfig()
        optimizer = TextGradOptimizer(config)
        with patch.object(textgrad_adapter, "textgrad", None):
            with pytest.raises(ImportError, match="optimization"):
                optimizer.optimize(
                    "prompt", samples=[_make_sample()], metric=MagicMock(return_value=0.5)
                )


class TestTextGradOptimizerCompile:
    def test_optimization_runs_steps(self):
        config = TextGradConfig(steps=3)
        optimizer = TextGradOptimizer(config)
        mock_tg = _patch_textgrad("Optimized via TextGrad.")
        # baseline=0.5, then 3 step scores increasing so a step strictly beats baseline.
        end_metric = MagicMock(side_effect=[0.5, 0.6, 0.8, 0.9])

        with patch.object(textgrad_adapter, "textgrad", mock_tg):
            result = optimizer.optimize(
                current_prompt="Original prompt.",
                samples=[_make_sample()],
                metric=end_metric,
            )

        assert result.optimizer_name == "textgrad"
        assert result.optimized_prompt == "Optimized via TextGrad."
        assert result.config["steps"] == 3

    def test_config_stored_in_result(self):
        config = TextGradConfig(steps=5)
        optimizer = TextGradOptimizer(config)
        mock_tg = _patch_textgrad("Result.")

        with patch.object(textgrad_adapter, "textgrad", mock_tg):
            result = optimizer.optimize(
                "Original.",
                samples=[_make_sample()],
                metric=MagicMock(return_value=0.5),
            )

        assert result.config["optimizer"] == "textgrad"
        assert result.config["steps"] == 5


class TestTextGradOptimizerBudget:
    """Verifies the max_trials cap and strict-improvement best tracking."""

    def test_steps_capped_by_max_trials(self):
        # steps=10 but max_trials=3 → only 3 iterations should run
        config = TextGradConfig(steps=10, max_trials=3)
        optimizer = TextGradOptimizer(config)
        mock_tg = _patch_textgrad("Final value.")

        # baseline + 3 capped steps = 4 metric calls expected
        end_metric = MagicMock(return_value=0.5)

        with patch.object(textgrad_adapter, "textgrad", mock_tg):
            result = optimizer.optimize(
                current_prompt="Baseline.",
                samples=[_make_sample()],
                metric=end_metric,
            )

        assert result.num_trials == 3
        assert end_metric.call_count == 4  # 1 baseline + 3 steps
        # TGD.step should also be called exactly once per trial
        assert mock_tg.TGD.return_value.step.call_count == 3

    def test_best_uses_strict_improvement(self):
        """Ties should not overwrite the best — strict ``>``."""
        config = TextGradConfig(steps=3)
        optimizer = TextGradOptimizer(config)
        mock_tg = _patch_textgrad("Latest value.")

        # baseline=0.5, then step1=0.5 (tie, should NOT update),
        # step2=0.5 (tie), step3=0.5 (tie). Best stays at baseline.
        end_metric = MagicMock(side_effect=[0.5, 0.5, 0.5, 0.5])

        with patch.object(textgrad_adapter, "textgrad", mock_tg):
            result = optimizer.optimize(
                current_prompt="Baseline.",
                samples=[_make_sample()],
                metric=end_metric,
            )

        # Baseline wins because no step strictly improved on it.
        assert result.optimized_prompt == "Baseline."
        assert result.optimized_score == pytest.approx(0.5)
        assert result.baseline_score == pytest.approx(0.5)
        assert result.improvement == pytest.approx(0.0)
