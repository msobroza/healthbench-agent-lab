"""Tests for the prompt optimizer registry."""

from __future__ import annotations

import pytest

from healthbench_agent.prompt_optimization.config import (
    BaseOptimizationConfig,
)
from healthbench_agent.prompt_optimization.optimizer import (
    OptimizationResult,
    PromptOptimizer,
)
from healthbench_agent.prompt_optimization.optimizer_registry import (
    _PROMPT_OPTIMIZER_REGISTRY,
    create_prompt_optimizer,
    register_prompt_optimizer,
    registered_prompt_optimizers,
)


class _StubConfig(BaseOptimizationConfig):
    optimizer: str = "stub"


class TestRegisterPromptOptimizer:
    def test_register_and_lookup(self):
        @register_prompt_optimizer("stub_test", _StubConfig)
        class StubOptimizer(PromptOptimizer):
            def optimize(self, current_prompt, samples, metric):
                return OptimizationResult(
                    optimized_prompt=current_prompt,
                    baseline_score=0.0,
                    optimized_score=0.0,
                    improvement=0.0,
                    num_trials=0,
                    trial_history=[],
                    optimizer_name="stub_test",
                    config={},
                )

        assert "stub_test" in _PROMPT_OPTIMIZER_REGISTRY
        config_cls, opt_cls = _PROMPT_OPTIMIZER_REGISTRY["stub_test"]
        assert config_cls is _StubConfig
        assert opt_cls is StubOptimizer
        del _PROMPT_OPTIMIZER_REGISTRY["stub_test"]

    def test_duplicate_registration_raises(self):
        @register_prompt_optimizer("dup_test", _StubConfig)
        class First(PromptOptimizer):
            def optimize(self, current_prompt, samples, metric): ...

        with pytest.raises(ValueError, match="already registered"):

            @register_prompt_optimizer("dup_test", _StubConfig)
            class Second(PromptOptimizer):
                def optimize(self, current_prompt, samples, metric): ...

        del _PROMPT_OPTIMIZER_REGISTRY["dup_test"]


class TestCreatePromptOptimizer:
    def test_create_registered_optimizer(self):
        @register_prompt_optimizer("factory_test", _StubConfig)
        class FactoryStub(PromptOptimizer):
            def __init__(self, config):
                self.config = config

            def optimize(self, current_prompt, samples, metric):
                return OptimizationResult(
                    optimized_prompt=current_prompt,
                    baseline_score=0.0,
                    optimized_score=0.0,
                    improvement=0.0,
                    num_trials=0,
                    trial_history=[],
                    optimizer_name="factory_test",
                    config={},
                )

        config = _StubConfig(optimizer="factory_test")
        optimizer = create_prompt_optimizer(config)
        assert isinstance(optimizer, FactoryStub)
        del _PROMPT_OPTIMIZER_REGISTRY["factory_test"]

    def test_unknown_optimizer_raises(self):
        config = _StubConfig(optimizer="nonexistent")
        with pytest.raises(ValueError, match="Unknown prompt optimizer"):
            create_prompt_optimizer(config)


class TestRegisteredPromptOptimizers:
    def test_returns_copy(self):
        result = registered_prompt_optimizers()
        assert isinstance(result, dict)
        result["fake"] = (BaseOptimizationConfig, PromptOptimizer)
        assert "fake" not in _PROMPT_OPTIMIZER_REGISTRY
