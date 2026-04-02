"""Tests for PromptOptimizer ABC and result dataclasses."""

from __future__ import annotations

from typing import Any

import pytest

from healthbench_agent.prompt_optimization.optimizer import (
    OptimizationResult,
    PromptOptimizer,
    TrialRecord,
)


class TestTrialRecord:
    """Tests for TrialRecord frozen dataclass."""

    def test_create_with_score(self):
        record = TrialRecord(
            trial_id=1,
            prompt="You are a helpful assistant.",
            score=0.85,
            timestamp="2026-04-03T10:00:00",
        )
        assert record.trial_id == 1
        assert record.prompt == "You are a helpful assistant."
        assert record.score == 0.85
        assert record.timestamp == "2026-04-03T10:00:00"

    def test_create_with_none_score(self):
        record = TrialRecord(
            trial_id=0,
            prompt="test",
            score=None,
            timestamp="2026-04-03T10:00:00",
        )
        assert record.score is None

    def test_frozen(self):
        record = TrialRecord(
            trial_id=1, prompt="test", score=0.5, timestamp="t"
        )
        with pytest.raises(AttributeError):
            record.score = 0.9  # type: ignore[misc]


class TestOptimizationResult:
    """Tests for OptimizationResult frozen dataclass."""

    def test_create(self):
        trial = TrialRecord(
            trial_id=0, prompt="p", score=0.5, timestamp="t"
        )
        result = OptimizationResult(
            optimized_prompt="optimized",
            baseline_score=0.4,
            optimized_score=0.6,
            improvement=0.2,
            num_trials=1,
            trial_history=[trial],
            optimizer_name="test",
            config={"key": "value"},
        )
        assert result.optimized_prompt == "optimized"
        assert result.improvement == pytest.approx(0.2)
        assert result.num_trials == 1
        assert len(result.trial_history) == 1
        assert result.optimizer_name == "test"

    def test_frozen(self):
        result = OptimizationResult(
            optimized_prompt="p",
            baseline_score=0.0,
            optimized_score=0.0,
            improvement=0.0,
            num_trials=0,
            trial_history=[],
            optimizer_name="test",
            config={},
        )
        with pytest.raises(AttributeError):
            result.optimized_prompt = "new"  # type: ignore[misc]


class TestPromptOptimizerABC:
    """Tests for PromptOptimizer abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            PromptOptimizer()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        class DummyOptimizer(PromptOptimizer):
            def optimize(self, current_prompt, samples, metric):
                return OptimizationResult(
                    optimized_prompt=current_prompt,
                    baseline_score=0.0,
                    optimized_score=0.0,
                    improvement=0.0,
                    num_trials=0,
                    trial_history=[],
                    optimizer_name="dummy",
                    config={},
                )

        optimizer = DummyOptimizer()
        result = optimizer.optimize("prompt", None, None)
        assert result.optimized_prompt == "prompt"
