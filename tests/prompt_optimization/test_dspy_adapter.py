"""Tests for the DSPyOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.adapters import dspy_adapter
from healthbench_agent.prompt_optimization.adapters.dspy_adapter import (
    DSPyOptimizer,
    _BudgetExceededError,
)
from healthbench_agent.prompt_optimization.config import DSPyConfig


def _make_sample() -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id="test-1",
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


class TestDSPyOptimizerValidation:
    def test_requires_samples(self):
        config = DSPyConfig()
        optimizer = DSPyOptimizer(config)
        with patch.object(dspy_adapter, "dspy", MagicMock()):
            with pytest.raises(ValueError, match="samples"):
                optimizer.optimize("prompt", samples=None, metric=MagicMock(return_value=0.5))

    def test_requires_metric(self):
        config = DSPyConfig()
        optimizer = DSPyOptimizer(config)
        with patch.object(dspy_adapter, "dspy", MagicMock()):
            with pytest.raises(ValueError, match="metric"):
                optimizer.optimize("prompt", samples=[_make_sample()], metric=None)

    def test_raises_when_dspy_not_installed(self):
        config = DSPyConfig()
        optimizer = DSPyOptimizer(config)
        with patch.object(dspy_adapter, "dspy", None):
            with pytest.raises(ImportError, match="optimization"):
                optimizer.optimize(
                    "prompt", samples=[_make_sample()], metric=MagicMock(return_value=0.5)
                )

    def test_raises_on_unknown_dspy_optimizer(self):
        config = DSPyConfig(dspy_optimizer="bogus")
        optimizer = DSPyOptimizer(config)
        with patch.object(dspy_adapter, "dspy", MagicMock()):
            with pytest.raises(ValueError, match="Unsupported dspy_optimizer"):
                optimizer.optimize(
                    "prompt", samples=[_make_sample()], metric=MagicMock(return_value=0.5)
                )


def _build_mock_dspy(compiled_instruction: str) -> MagicMock:
    """Build a mocked dspy module that returns a controllable compiled module."""
    mock_dspy = MagicMock()
    mock_compiled = MagicMock()
    mock_compiled.generate.signature.instructions = compiled_instruction
    return mock_dspy, mock_compiled


class TestDSPyOptimizerCompile:
    def test_copro_returns_optimized_prompt(self):
        config = DSPyConfig(dspy_optimizer="copro", max_trials=10)
        optimizer = DSPyOptimizer(config)
        mock_dspy, mock_compiled = _build_mock_dspy("Optimized health prompt.")
        mock_dspy.COPRO.return_value.compile.return_value = mock_compiled
        # baseline=0.5, then final compiled instruction scores 0.9 → strict win.
        end_metric = MagicMock(side_effect=[0.5, 0.9])

        with patch.object(dspy_adapter, "dspy", mock_dspy):
            result = optimizer.optimize(
                current_prompt="Original prompt.",
                samples=[_make_sample()],
                metric=end_metric,
            )

        assert result.optimizer_name == "dspy"
        assert result.optimized_prompt == "Optimized health prompt."
        assert result.config["dspy_optimizer"] == "copro"

    def test_miprov2_returns_optimized_prompt(self):
        config = DSPyConfig(dspy_optimizer="miprov2", max_trials=10)
        optimizer = DSPyOptimizer(config)
        mock_dspy, mock_compiled = _build_mock_dspy("MIPROv2 optimized.")
        mock_dspy.MIPROv2.return_value.compile.return_value = mock_compiled
        end_metric = MagicMock(side_effect=[0.6, 0.95])

        with patch.object(dspy_adapter, "dspy", mock_dspy):
            result = optimizer.optimize(
                current_prompt="Original.",
                samples=[_make_sample()],
                metric=end_metric,
            )

        assert result.optimized_prompt == "MIPROv2 optimized."
        assert result.config["dspy_optimizer"] == "miprov2"


class TestDSPyOptimizerCachingAndBudget:
    """Verifies the per-instruction cache and the max_trials early-stop budget."""

    def test_caches_per_instruction(self):
        config = DSPyConfig(dspy_optimizer="copro", max_trials=20)
        optimizer = DSPyOptimizer(config)
        mock_dspy, mock_compiled = _build_mock_dspy("final_instruction")
        end_metric = MagicMock(side_effect=[0.50, 0.70, 0.90, 0.60])

        def fake_compile(module, **_kwargs):
            # Capture the dspy_metric closure that was handed to COPRO.
            dspy_metric = mock_dspy.COPRO.call_args.kwargs["metric"]
            module.generate.signature.instructions = "candidate_a"
            dspy_metric(MagicMock(), MagicMock())  # cache miss → 0.70
            dspy_metric(MagicMock(), MagicMock())  # cache hit → 0.70
            dspy_metric(MagicMock(), MagicMock())  # cache hit → 0.70
            module.generate.signature.instructions = "candidate_b"
            dspy_metric(MagicMock(), MagicMock())  # cache miss → 0.90
            return mock_compiled

        mock_dspy.COPRO.return_value.compile.side_effect = fake_compile

        with patch.object(dspy_adapter, "dspy", mock_dspy):
            result = optimizer.optimize(
                current_prompt="baseline",
                samples=[_make_sample()],
                metric=end_metric,
            )

        # Underlying metric should be called only on cache misses:
        # baseline + candidate_a + candidate_b + final_instruction = 4
        assert end_metric.call_count == 4
        assert result.num_trials == 4
        # Best across all trials is 0.90 (candidate_b)
        assert result.optimized_score == pytest.approx(0.90)
        assert result.optimized_prompt == "candidate_b"
        assert result.baseline_score == pytest.approx(0.50)
        assert result.improvement == pytest.approx(0.40)

    def test_budget_early_stop_returns_best_so_far(self):
        config = DSPyConfig(dspy_optimizer="copro", max_trials=2)
        optimizer = DSPyOptimizer(config)
        mock_dspy, mock_compiled = _build_mock_dspy("final_instruction")
        # Only 2 calls allowed; the 3rd should raise _BudgetExceededError.
        end_metric = MagicMock(side_effect=[0.40, 0.80, 0.99, 0.99])

        def fake_compile(module, **_kwargs):
            dspy_metric = mock_dspy.COPRO.call_args.kwargs["metric"]
            module.generate.signature.instructions = "first"
            dspy_metric(MagicMock(), MagicMock())  # trial 2 (baseline was trial 1)
            module.generate.signature.instructions = "second"
            # This call should raise _BudgetExceededError because trial_history
            # already has 2 entries (baseline + first).
            with pytest.raises(_BudgetExceededError):
                dspy_metric(MagicMock(), MagicMock())
            raise _BudgetExceededError()

        mock_dspy.COPRO.return_value.compile.side_effect = fake_compile

        with patch.object(dspy_adapter, "dspy", mock_dspy):
            result = optimizer.optimize(
                current_prompt="baseline",
                samples=[_make_sample()],
                metric=end_metric,
            )

        # Underlying metric called only twice (baseline + first); both
        # "second" attempts hit the budget exception.
        assert end_metric.call_count == 2
        assert result.num_trials == 2
        # Best is "first" with 0.80; baseline was 0.40.
        assert result.optimized_prompt == "first"
        assert result.optimized_score == pytest.approx(0.80)
