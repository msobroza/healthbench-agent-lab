"""Tests for prompt optimization configuration classes."""

from __future__ import annotations

import pytest

from healthbench_agent.prompt_optimization.config import (
    CritiqueRefineConfig,
    DSPyConfig,
    TextGradConfig,
)


class TestBaseOptimizationConfig:
    def test_dspy_config_defaults(self):
        config = DSPyConfig()
        assert config.optimizer == "dspy"
        assert config.max_trials == 50
        assert config.sample_size == 20
        assert config.seed == 42
        assert config.meta_model == "gemini-2.5-flash"
        assert config.meta_provider == "gemini"

    def test_max_trials_validation(self):
        with pytest.raises(ValueError):
            DSPyConfig(max_trials=0)

    def test_sample_size_validation(self):
        with pytest.raises(ValueError):
            DSPyConfig(sample_size=0)


class TestDSPyConfig:
    def test_defaults(self):
        config = DSPyConfig()
        assert config.dspy_optimizer == "copro"
        assert config.max_bootstrapped_demos == 0

    def test_miprov2(self):
        config = DSPyConfig(dspy_optimizer="miprov2")
        assert config.dspy_optimizer == "miprov2"


class TestTextGradConfig:
    def test_defaults(self):
        config = TextGradConfig()
        assert config.optimizer == "textgrad"
        assert config.steps == 10

    def test_steps_validation(self):
        with pytest.raises(ValueError):
            TextGradConfig(steps=0)


class TestCritiqueRefineConfig:
    def test_defaults(self):
        config = CritiqueRefineConfig()
        assert config.optimizer == "critique_refine"
        assert config.mutation_rounds == 3
        assert config.refine_iterations == 3
        assert config.style_variations == 5

    def test_mutation_rounds_validation(self):
        with pytest.raises(ValueError):
            CritiqueRefineConfig(mutation_rounds=0)

    def test_refine_iterations_validation(self):
        with pytest.raises(ValueError):
            CritiqueRefineConfig(refine_iterations=0)

    def test_style_variations_validation(self):
        with pytest.raises(ValueError):
            CritiqueRefineConfig(style_variations=0)


class TestConfigEnvOverride:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPTIM_MAX_TRIALS", "100")
        monkeypatch.setenv("OPTIM_SEED", "99")
        config = DSPyConfig()
        assert config.max_trials == 100
        assert config.seed == 99
