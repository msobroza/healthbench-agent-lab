# Prompt Optimization Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `prompt_optimization` module with three APE backends (DSPy, TextGrad, critique-refine) behind a common `PromptOptimizer` abstraction, integrated with the existing evaluation pipeline and experiment tracker.

**Architecture:** Registry-based adapter pattern. A `PromptOptimizer` ABC defines `optimize()`. Each backend self-registers via `@register_prompt_optimizer`. `EndToEndMetric` bridges optimizers to the existing agent pipeline + LLM judge. Config is per-framework via `BaseOptimizationConfig` subclasses.

**Tech Stack:** Python 3.11+, pydantic-settings, DSPy, TextGrad, MLflow, pytest

**Design Spec:** `docs/superpowers/specs/2026-04-03-prompt-optimization-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/healthbench_agent/prompt_optimization/__init__.py` | Public API exports |
| Create | `src/healthbench_agent/prompt_optimization/optimizer.py` | `PromptOptimizer` ABC, `OptimizationResult`, `TrialRecord` |
| Create | `src/healthbench_agent/prompt_optimization/config.py` | `BaseOptimizationConfig`, `DSPyConfig`, `TextGradConfig`, `CritiqueRefineConfig` |
| Create | `src/healthbench_agent/prompt_optimization/metric.py` | `EndToEndMetric` |
| Create | `src/healthbench_agent/prompt_optimization/optimizer_registry.py` | `register_prompt_optimizer`, `create_prompt_optimizer` |
| Create | `src/healthbench_agent/prompt_optimization/adapters/__init__.py` | Import all adapters to trigger registration |
| Create | `src/healthbench_agent/prompt_optimization/adapters/dspy_adapter.py` | `DSPyOptimizer` |
| Create | `src/healthbench_agent/prompt_optimization/adapters/textgrad_adapter.py` | `TextGradOptimizer` |
| Create | `src/healthbench_agent/prompt_optimization/adapters/critique_refine_adapter.py` | `CritiqueRefineOptimizer` |
| Create | `tests/prompt_optimization/__init__.py` | Test package |
| Create | `tests/prompt_optimization/test_optimizer.py` | Tests for ABC, dataclasses |
| Create | `tests/prompt_optimization/test_config.py` | Tests for all config classes |
| Create | `tests/prompt_optimization/test_metric.py` | Tests for `EndToEndMetric` |
| Create | `tests/prompt_optimization/test_optimizer_registry.py` | Tests for registry |
| Create | `tests/prompt_optimization/test_dspy_adapter.py` | Tests for DSPy adapter |
| Create | `tests/prompt_optimization/test_textgrad_adapter.py` | Tests for TextGrad adapter |
| Create | `tests/prompt_optimization/test_critique_refine_adapter.py` | Tests for critique-refine adapter |
| Modify | `pyproject.toml` | Add `dspy`, `textgrad` as optional deps; add `optimize-prompt` CLI entry point |

---

### Task 1: Core Data Types — `TrialRecord` and `OptimizationResult`

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/optimizer.py`
- Create: `tests/prompt_optimization/__init__.py`
- Create: `tests/prompt_optimization/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_optimizer.py
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
```

- [ ] **Step 2: Create empty test package**

```bash
touch tests/prompt_optimization/__init__.py
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_optimizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'healthbench_agent.prompt_optimization'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/optimizer.py
"""Prompt optimizer abstraction and result types.

Defines the contract that any prompt optimization backend (DSPy, TextGrad,
critique-refine) must implement. Result types are frozen dataclasses for
immutability and easy serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from .metric import EndToEndMetric


@dataclass(frozen=True)
class TrialRecord:
    """Single optimization trial.

    Attributes:
        trial_id: Sequential trial number.
        prompt: The candidate prompt evaluated in this trial.
        score: Evaluation score, or None in mutation-only mode.
        timestamp: ISO 8601 timestamp of when the trial completed.
    """

    trial_id: int
    prompt: str
    score: float | None
    timestamp: str


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a prompt optimization run.

    Attributes:
        optimized_prompt: The best prompt found during optimization.
        baseline_score: Score of the original prompt before optimization.
        optimized_score: Score of the best prompt found.
        improvement: Score delta (optimized_score - baseline_score).
        num_trials: Total number of candidate prompts evaluated.
        trial_history: Per-trial details for reproducibility.
        optimizer_name: Identifier of the optimizer used.
        config: Serialized optimizer configuration for reproducibility.
    """

    optimized_prompt: str
    baseline_score: float
    optimized_score: float
    improvement: float
    num_trials: int
    trial_history: list[TrialRecord]
    optimizer_name: str
    config: dict[str, Any]


class PromptOptimizer(ABC):
    """Abstract base for prompt optimizers.

    Subclasses implement the optimization logic for a specific framework
    (DSPy, TextGrad, critique-refine) and return an OptimizationResult.
    """

    @abstractmethod
    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt against a scoring metric.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required for DSPy/TextGrad,
                optional for critique-refine (mutation-only mode).
            metric: Callable that scores a prompt end-to-end.
                Required for DSPy/TextGrad, optional for critique-refine.

        Returns:
            OptimizationResult with the best prompt and trial history.
        """
        ...
```

Also create the package `__init__.py` (empty for now, will be populated in a later task):

```python
# src/healthbench_agent/prompt_optimization/__init__.py
"""Automatic prompt engineering for agent system prompts.

Provides a registry-based adapter pattern for optimizing agent prompts
using different backends (DSPy, TextGrad, critique-refine) behind a
common PromptOptimizer abstraction.
"""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_optimizer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/__init__.py \
       src/healthbench_agent/prompt_optimization/optimizer.py \
       tests/prompt_optimization/__init__.py \
       tests/prompt_optimization/test_optimizer.py
git commit -m "feat(prompt_optimization): add PromptOptimizer ABC and result dataclasses"
```

---

### Task 2: Configuration Classes

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/config.py`
- Create: `tests/prompt_optimization/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_config.py
"""Tests for prompt optimization configuration classes."""

from __future__ import annotations

import pytest

from healthbench_agent.prompt_optimization.config import (
    BaseOptimizationConfig,
    CritiqueRefineConfig,
    DSPyConfig,
    TextGradConfig,
)


class TestBaseOptimizationConfig:
    """Tests for shared base config."""

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
    """Tests for DSPy-specific config."""

    def test_defaults(self):
        config = DSPyConfig()
        assert config.dspy_optimizer == "copro"
        assert config.max_bootstrapped_demos == 0

    def test_miprov2(self):
        config = DSPyConfig(dspy_optimizer="miprov2")
        assert config.dspy_optimizer == "miprov2"


class TestTextGradConfig:
    """Tests for TextGrad-specific config."""

    def test_defaults(self):
        config = TextGradConfig()
        assert config.optimizer == "textgrad"
        assert config.steps == 10

    def test_steps_validation(self):
        with pytest.raises(ValueError):
            TextGradConfig(steps=0)


class TestCritiqueRefineConfig:
    """Tests for critique-refine-specific config."""

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
    """Tests for env var override via OPTIM_ prefix."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPTIM_MAX_TRIALS", "100")
        monkeypatch.setenv("OPTIM_SEED", "99")
        config = DSPyConfig()
        assert config.max_trials == 100
        assert config.seed == 99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/config.py
"""Configuration classes for prompt optimization.

Each optimizer backend has its own config subclass with framework-specific
fields. All share a common base with env var override (prefix ``OPTIM_``).
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseOptimizationConfig(BaseSettings):
    """Shared settings for all prompt optimizers.

    Settings are resolved in priority order:
        1. Explicit constructor kwargs
        2. Environment variables (prefixed ``OPTIM_``)
        3. ``.env`` dotenv file
        4. Field defaults

    Attributes:
        optimizer: Backend identifier used by the registry to dispatch.
        max_trials: Maximum candidate prompts to evaluate.
        sample_size: Number of HealthBench samples per evaluation.
        seed: Random seed for reproducible sampling.
        meta_model: Model used by the optimizer to propose/critique prompts.
        meta_provider: Provider for the meta model.
        google_api_key: Google API key for Gemini-based optimization.
        openai_api_key: OpenAI API key for OpenAI-based optimization.
    """

    model_config = SettingsConfigDict(env_prefix="OPTIM_", env_file=".env")

    optimizer: str = "dspy"
    max_trials: int = Field(50, ge=1)
    sample_size: int = Field(20, ge=1)
    seed: int = 42
    meta_model: str = "gemini-2.5-flash"
    meta_provider: str = "gemini"

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )


class DSPyConfig(BaseOptimizationConfig):
    """DSPy-specific optimization config.

    Attributes:
        dspy_optimizer: DSPy optimizer to use — ``"copro"`` or ``"miprov2"``.
        max_bootstrapped_demos: Number of bootstrapped demos. Set to 0 for
            instruction-only optimization (no few-shot examples).
    """

    optimizer: str = "dspy"
    dspy_optimizer: str = "copro"
    max_bootstrapped_demos: int = 0


class TextGradConfig(BaseOptimizationConfig):
    """TextGrad-specific optimization config.

    Attributes:
        steps: Number of text-gradient descent steps.
    """

    optimizer: str = "textgrad"
    steps: int = Field(10, ge=1)


class CritiqueRefineConfig(BaseOptimizationConfig):
    """Critique-refine (PromptWizard algorithm) optimization config.

    Attributes:
        mutation_rounds: Number of prompt mutation rounds per iteration.
        refine_iterations: Number of critique-and-refine cycles.
        style_variations: Number of thinking-style variations per mutation.
    """

    optimizer: str = "critique_refine"
    mutation_rounds: int = Field(3, ge=1)
    refine_iterations: int = Field(3, ge=1)
    style_variations: int = Field(5, ge=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_config.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/config.py \
       tests/prompt_optimization/test_config.py
git commit -m "feat(prompt_optimization): add config classes for all optimizer backends"
```

---

### Task 3: Optimizer Registry

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/optimizer_registry.py`
- Create: `tests/prompt_optimization/test_optimizer_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_optimizer_registry.py
"""Tests for the prompt optimizer registry."""

from __future__ import annotations

import pytest

from healthbench_agent.prompt_optimization.config import (
    BaseOptimizationConfig,
    DSPyConfig,
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
    """Tests for the @register_prompt_optimizer decorator."""

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

        # Cleanup
        del _PROMPT_OPTIMIZER_REGISTRY["stub_test"]

    def test_duplicate_registration_raises(self):
        @register_prompt_optimizer("dup_test", _StubConfig)
        class First(PromptOptimizer):
            def optimize(self, current_prompt, samples, metric):
                ...

        with pytest.raises(ValueError, match="already registered"):

            @register_prompt_optimizer("dup_test", _StubConfig)
            class Second(PromptOptimizer):
                def optimize(self, current_prompt, samples, metric):
                    ...

        # Cleanup
        del _PROMPT_OPTIMIZER_REGISTRY["dup_test"]


class TestCreatePromptOptimizer:
    """Tests for the create_prompt_optimizer factory."""

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

        # Cleanup
        del _PROMPT_OPTIMIZER_REGISTRY["factory_test"]

    def test_unknown_optimizer_raises(self):
        config = _StubConfig(optimizer="nonexistent")
        with pytest.raises(ValueError, match="Unknown prompt optimizer"):
            create_prompt_optimizer(config)


class TestRegisteredPromptOptimizers:
    """Tests for the registered_prompt_optimizers listing."""

    def test_returns_copy(self):
        result = registered_prompt_optimizers()
        assert isinstance(result, dict)
        # Mutating the copy should not affect the registry
        result["fake"] = (BaseOptimizationConfig, PromptOptimizer)
        assert "fake" not in _PROMPT_OPTIMIZER_REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_optimizer_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/optimizer_registry.py
"""Registry for prompt optimizer backends.

Maps optimizer names to (config_class, optimizer_class) tuples. Adding a
new optimizer requires only a new adapter file with
``@register_prompt_optimizer`` — no changes to this module (Open/Closed).
"""

from __future__ import annotations

from typing import Any

from .config import BaseOptimizationConfig
from .optimizer import PromptOptimizer

_PROMPT_OPTIMIZER_REGISTRY: dict[
    str, tuple[type[BaseOptimizationConfig], type[PromptOptimizer]]
] = {}


def register_prompt_optimizer(
    name: str,
    config_class: type[BaseOptimizationConfig],
) -> Any:
    """Class decorator that registers an optimizer with its config type.

    Args:
        name: Unique identifier for this optimizer backend.
        config_class: The config subclass associated with this optimizer.

    Returns:
        The original class, unchanged.

    Raises:
        ValueError: If an optimizer with the same name is already registered.
    """

    def decorator(cls: type[PromptOptimizer]) -> type[PromptOptimizer]:
        if name in _PROMPT_OPTIMIZER_REGISTRY:
            existing = _PROMPT_OPTIMIZER_REGISTRY[name][1]
            raise ValueError(
                f"Prompt optimizer '{name}' is already registered "
                f"(existing: {existing.__module__}.{existing.__name__})"
            )
        _PROMPT_OPTIMIZER_REGISTRY[name] = (config_class, cls)
        return cls

    return decorator


def create_prompt_optimizer(
    config: BaseOptimizationConfig,
) -> PromptOptimizer:
    """Create an optimizer from config, dispatching via registry.

    Args:
        config: Optimizer configuration. The ``optimizer`` field selects
            which backend to instantiate.

    Returns:
        A fully configured PromptOptimizer instance.

    Raises:
        ValueError: If the optimizer name is not registered.
    """
    if config.optimizer not in _PROMPT_OPTIMIZER_REGISTRY:
        available = list(_PROMPT_OPTIMIZER_REGISTRY) or ["(none)"]
        raise ValueError(
            f"Unknown prompt optimizer: {config.optimizer!r}. "
            f"Available: {available}"
        )
    _, optimizer_class = _PROMPT_OPTIMIZER_REGISTRY[config.optimizer]
    return optimizer_class(config)


def registered_prompt_optimizers() -> (
    dict[str, tuple[type[BaseOptimizationConfig], type[PromptOptimizer]]]
):
    """Return a copy of the current optimizer registry.

    Returns:
        Dict mapping optimizer names to (config_class, optimizer_class) tuples.
    """
    return dict(_PROMPT_OPTIMIZER_REGISTRY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_optimizer_registry.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/optimizer_registry.py \
       tests/prompt_optimization/test_optimizer_registry.py
git commit -m "feat(prompt_optimization): add optimizer registry with decorator pattern"
```

---

### Task 4: EndToEndMetric

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/metric.py`
- Create: `tests/prompt_optimization/test_metric.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_metric.py
"""Tests for EndToEndMetric."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import CriterionVerdict, SingleEvalResult
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
    """Tests for EndToEndMetric scoring callable."""

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_metric.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/metric.py
"""End-to-end evaluation metric for prompt optimization.

Bridges the optimizer to the existing agent pipeline and LLM judge by
wrapping the full evaluation flow (generate responses, grade, score)
into a single callable: ``metric(prompt) -> float``.
"""

from __future__ import annotations

import asyncio

from healthbench_agent.agent import RootAgentPipelineConfig, create_pipeline
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.scoring import aggregate_scores, calculate_score


class EndToEndMetric:
    """Scores a prompt by running agent generation + LLM judge grading.

    Stateless with respect to the prompt being tested — builds a fresh
    pipeline per call so the prompt change takes effect. Reuses the same
    judge instance across calls to avoid re-initialization overhead.

    Attributes:
        agent_config: Base agent config to copy and patch with candidate prompts.
        judge: Grader that evaluates conversations against rubrics.
        samples: Fixed evaluation dataset for fair comparison across trials.
    """

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
    ) -> None:
        self.agent_config = agent_config
        self.judge = judge
        self.samples = samples

    def __call__(self, prompt: str) -> float:
        """Evaluate a candidate prompt end-to-end.

        Creates a copy of the agent config with the candidate prompt
        injected, builds a fresh pipeline, generates responses for all
        samples, grades them via the judge, and returns the aggregate
        score.

        Args:
            prompt: The candidate prompt text to evaluate.

        Returns:
            Aggregate HealthBench score in [0.0, 1.0].
        """
        patched_config = self.agent_config.model_copy(
            update={"instruction_override": prompt}
        )
        pipeline = create_pipeline(patched_config)

        responses = asyncio.run(
            asyncio.gather(
                *[pipeline.generate(sample.prompt) for sample in self.samples]
            )
        )

        results = []
        for sample, response in zip(self.samples, responses):
            conversation = list(sample.prompt) + [
                {"role": "assistant", "content": response}
            ]
            verdicts = self.judge.grade(conversation, sample.rubrics)
            score = calculate_score(sample.rubrics, verdicts)
            from healthbench_agent.domain.evaluation import SingleEvalResult

            results.append(SingleEvalResult(score=score, metrics={}))

        return aggregate_scores(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_metric.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/metric.py \
       tests/prompt_optimization/test_metric.py
git commit -m "feat(prompt_optimization): add EndToEndMetric scoring callable"
```

---

### Task 5: CritiqueRefineOptimizer Adapter (No External Dependencies)

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/adapters/__init__.py`
- Create: `src/healthbench_agent/prompt_optimization/adapters/critique_refine_adapter.py`
- Create: `tests/prompt_optimization/test_critique_refine_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_critique_refine_adapter.py
"""Tests for the CritiqueRefineOptimizer adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.sampler import SamplerResponse
from healthbench_agent.prompt_optimization.adapters.critique_refine_adapter import (
    THINKING_STYLES,
    CritiqueRefineOptimizer,
)
from healthbench_agent.prompt_optimization.config import CritiqueRefineConfig


def _make_sample() -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id="test-1",
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


class TestThinkingStyles:
    """Tests for the thinking styles constant."""

    def test_styles_is_nonempty_list(self):
        assert isinstance(THINKING_STYLES, list)
        assert len(THINKING_STYLES) > 0

    def test_styles_are_strings(self):
        for style in THINKING_STYLES:
            assert isinstance(style, str)
            assert len(style) > 0


class TestCritiqueRefineOptimizerMutationOnly:
    """Tests for mutation-only mode (no metric)."""

    def test_mutation_only_returns_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=2,
            style_variations=2,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Improved prompt: You are an expert health advisor.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="You are a health assistant.",
                samples=None,
                metric=None,
            )

        assert result.optimizer_name == "critique_refine"
        assert result.num_trials > 0
        assert result.optimized_prompt != ""
        assert result.baseline_score == 0.0
        assert result.optimized_score == 0.0

    def test_mutation_only_trial_scores_are_none(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=1,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Mutated prompt.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize("Original.", samples=None, metric=None)

        for trial in result.trial_history:
            assert trial.score is None


class TestCritiqueRefineOptimizerWithMetric:
    """Tests for critique-refine mode with metric."""

    def test_with_metric_returns_scored_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=2,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        call_count = 0

        def sampler_side_effect(message_list):
            nonlocal call_count
            call_count += 1
            # Alternate between mutation and critique/refine responses
            return SamplerResponse(
                response_text=f"Variant {call_count}: Be a thorough health expert.",
                actual_queried_message_list=message_list,
                response_metadata={},
            )

        mock_sampler.side_effect = sampler_side_effect

        mock_metric = MagicMock()
        mock_metric.side_effect = [0.5, 0.7, 0.8]  # scores for candidates

        sample = _make_sample()

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize(
                current_prompt="You are a health assistant.",
                samples=[sample],
                metric=mock_metric,
            )

        assert result.optimized_score >= result.baseline_score or result.num_trials > 0
        assert result.optimizer_name == "critique_refine"
        assert len(result.trial_history) > 0

    def test_config_stored_in_result(self):
        config = CritiqueRefineConfig(
            mutation_rounds=1,
            style_variations=1,
            refine_iterations=1,
        )
        optimizer = CritiqueRefineOptimizer(config)

        mock_sampler = MagicMock()
        mock_sampler.return_value = SamplerResponse(
            response_text="Mutated.",
            actual_queried_message_list=[],
            response_metadata={},
        )

        with patch(
            "healthbench_agent.prompt_optimization.adapters.critique_refine_adapter.create_sampler",
            return_value=mock_sampler,
        ):
            result = optimizer.optimize("prompt", samples=None, metric=None)

        assert result.config["optimizer"] == "critique_refine"
        assert result.config["mutation_rounds"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_critique_refine_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/adapters/__init__.py
"""Prompt optimizer adapters.

Importing this package triggers registration of all built-in adapters.
"""

from . import critique_refine_adapter  # noqa: F401
```

```python
# src/healthbench_agent/prompt_optimization/adapters/critique_refine_adapter.py
"""Critique-refine prompt optimizer inspired by Microsoft PromptWizard.

Uses an LLM to mutate prompts via thinking-style injection, then
iteratively critiques and refines candidates. Has no external
dependencies — uses the existing SamplerBase for all LLM calls.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from healthbench_agent.domain.sampler import SamplerBase
from healthbench_agent.llm_eval.samplers import create_sampler as _create_llm_sampler

from ..config import CritiqueRefineConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric

THINKING_STYLES: list[str] = [
    "Think step by step, breaking the problem into smaller parts.",
    "Consider edge cases and unusual scenarios.",
    "Reason from first principles rather than relying on heuristics.",
    "Think about what a domain expert would prioritize.",
    "Focus on clarity and actionability for the end user.",
    "Consider safety implications and potential harms.",
    "Emphasize precision and cite established medical consensus.",
    "Adopt the perspective of a patient seeking reassurance.",
    "Prioritize completeness — cover all relevant aspects.",
    "Think about what information is most time-sensitive.",
]


def create_sampler(config: CritiqueRefineConfig) -> SamplerBase:
    """Build a sampler for the meta-LLM from optimization config.

    Args:
        config: Optimization config with meta_model and meta_provider.

    Returns:
        A SamplerBase instance for the configured meta model.
    """
    from healthbench_agent.llm_eval.config_grader import JudgeConfig

    judge_config = JudgeConfig(
        provider=config.meta_provider,
        model=config.meta_model,
        temperature=0.7,
        google_api_key=config.google_api_key,
        openai_api_key=config.openai_api_key,
    )
    return _create_llm_sampler(judge_config)


@register_prompt_optimizer("critique_refine", CritiqueRefineConfig)
class CritiqueRefineOptimizer(PromptOptimizer):
    """Prompt optimizer using mutation + critique-refine loop.

    Two modes:
        - **With metric**: Mutate → score → critique failures → refine.
          Repeats for ``refine_iterations`` cycles.
        - **Without metric** (mutation-only): Mutate via thinking-style
          injection and return the best variant by meta-LLM self-ranking.

    Attributes:
        config: Critique-refine specific configuration.
    """

    def __init__(self, config: CritiqueRefineConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Run critique-refine optimization.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Optional — when None, runs
                mutation-only mode.
            metric: Scoring callable. Optional — when None, runs
                mutation-only mode.

        Returns:
            OptimizationResult with the best prompt found.
        """
        sampler = create_sampler(self.config)
        rng = random.Random(self.config.seed)
        trial_history: list[TrialRecord] = []
        trial_id = 0

        # Phase 1: Mutation — generate style variations
        candidates: list[str] = []
        for _ in range(self.config.mutation_rounds):
            styles = rng.sample(
                THINKING_STYLES,
                min(self.config.style_variations, len(THINKING_STYLES)),
            )
            for style in styles:
                mutated = self._mutate_prompt(sampler, current_prompt, style)
                candidates.append(mutated)
                score = metric(mutated) if metric is not None else None
                trial_history.append(
                    TrialRecord(
                        trial_id=trial_id,
                        prompt=mutated,
                        score=score,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
                trial_id += 1

        # Select best candidate
        if metric is not None:
            scored_trials = [t for t in trial_history if t.score is not None]
            best_trial = max(scored_trials, key=lambda t: t.score)  # type: ignore[arg-type]
            best_prompt = best_trial.prompt
            best_score = best_trial.score or 0.0
        else:
            best_prompt = candidates[0] if candidates else current_prompt
            best_score = 0.0

        # Phase 2: Critique-refine (only when metric is available)
        if metric is not None:
            for _ in range(self.config.refine_iterations):
                critique = self._critique_prompt(
                    sampler, best_prompt, best_score
                )
                refined = self._refine_prompt(sampler, best_prompt, critique)
                refined_score = metric(refined)
                trial_history.append(
                    TrialRecord(
                        trial_id=trial_id,
                        prompt=refined,
                        score=refined_score,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                )
                trial_id += 1
                if refined_score > best_score:
                    best_prompt = refined
                    best_score = refined_score

        baseline_score = metric(current_prompt) if metric is not None else 0.0

        return OptimizationResult(
            optimized_prompt=best_prompt,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="critique_refine",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )

    def _mutate_prompt(
        self, sampler: SamplerBase, prompt: str, style: str
    ) -> str:
        """Generate a style-variant of the prompt.

        Args:
            sampler: Meta-LLM sampler.
            prompt: Current prompt to mutate.
            style: Thinking style to inject.

        Returns:
            Mutated prompt text.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"Rewrite the following system prompt to incorporate this "
                    f"thinking approach: '{style}'\n\n"
                    f"Original prompt:\n{prompt}\n\n"
                    f"Return ONLY the rewritten prompt, nothing else."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()

    def _critique_prompt(
        self, sampler: SamplerBase, prompt: str, score: float
    ) -> str:
        """Ask the meta-LLM to critique the prompt.

        Args:
            sampler: Meta-LLM sampler.
            prompt: Current best prompt.
            score: Current score of the prompt.

        Returns:
            Critique text with suggestions for improvement.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"This system prompt for a health assistant scored "
                    f"{score:.3f} out of 1.0 on a medical evaluation rubric.\n\n"
                    f"Prompt:\n{prompt}\n\n"
                    f"Analyze what might be causing the prompt to lose points. "
                    f"Consider: accuracy, safety, communication quality, "
                    f"instruction following, and completeness. "
                    f"Be specific about weaknesses and suggest improvements."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()

    def _refine_prompt(
        self, sampler: SamplerBase, prompt: str, critique: str
    ) -> str:
        """Refine the prompt based on critique.

        Args:
            sampler: Meta-LLM sampler.
            prompt: Current prompt to refine.
            critique: Critique with improvement suggestions.

        Returns:
            Refined prompt text.
        """
        message_list = [
            {
                "role": "user",
                "content": (
                    f"Improve this health assistant system prompt based on "
                    f"the following critique.\n\n"
                    f"Current prompt:\n{prompt}\n\n"
                    f"Critique:\n{critique}\n\n"
                    f"Return ONLY the improved prompt, nothing else."
                ),
            }
        ]
        response = sampler(message_list)
        return response.response_text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_critique_refine_adapter.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/adapters/__init__.py \
       src/healthbench_agent/prompt_optimization/adapters/critique_refine_adapter.py \
       tests/prompt_optimization/test_critique_refine_adapter.py
git commit -m "feat(prompt_optimization): add CritiqueRefineOptimizer adapter"
```

---

### Task 6: DSPyOptimizer Adapter

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/adapters/dspy_adapter.py`
- Create: `tests/prompt_optimization/test_dspy_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_dspy_adapter.py
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
    """Tests for DSPyOptimizer with mocked DSPy calls."""

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

        # Mock DSPy's COPRO optimizer
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_dspy_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/adapters/dspy_adapter.py
"""DSPy prompt optimizer adapter.

Uses DSPy's COPRO or MIPROv2 teleprompters for instruction-only
optimization. DSPy is lazily imported so the dependency is only
required when this adapter is used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..config import DSPyConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric


@register_prompt_optimizer("dspy", DSPyConfig)
class DSPyOptimizer(PromptOptimizer):
    """Prompt optimizer using DSPy's COPRO or MIPROv2 teleprompters.

    Wraps the current prompt as a DSPy Signature instruction, runs the
    selected optimizer in instruction-only mode (no few-shot demos), and
    extracts the optimized instruction.

    Attributes:
        config: DSPy-specific optimization configuration.
    """

    def __init__(self, config: DSPyConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Run DSPy optimization on the prompt.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required.
            metric: Scoring callable. Required.

        Returns:
            OptimizationResult with the DSPy-optimized prompt.

        Raises:
            ValueError: If samples or metric is None.
        """
        if samples is None:
            raise ValueError(
                "DSPyOptimizer requires samples for evaluation. "
                "Pass a list of HealthBenchSample instances."
            )
        if metric is None:
            raise ValueError(
                "DSPyOptimizer requires a metric for scoring. "
                "Pass an EndToEndMetric instance."
            )

        import dspy

        # Configure DSPy LM
        lm = dspy.LM(
            model=f"{self.config.meta_provider}/{self.config.meta_model}",
        )
        dspy.configure(lm=lm)

        # Define a minimal DSPy module with the current instruction
        class HealthAgentModule(dspy.Module):
            def __init__(self, instruction: str) -> None:
                super().__init__()
                self.generate = dspy.Predict(
                    dspy.Signature(
                        "conversation -> response",
                        instructions=instruction,
                    )
                )

            def forward(self, conversation: str) -> Any:
                return self.generate(conversation=conversation)

        module = HealthAgentModule(current_prompt)

        # Build DSPy metric wrapper
        def dspy_metric(
            example: Any, prediction: Any, trace: Any = None
        ) -> float:
            return metric(prediction.response if hasattr(prediction, "response") else current_prompt)

        # Build trainset from samples
        trainset = [
            dspy.Example(
                conversation=str(sample.prompt),
                response="",
            ).with_inputs("conversation")
            for sample in samples
        ]

        # Select and run optimizer
        baseline_score = metric(current_prompt)
        trial_history: list[TrialRecord] = []

        if self.config.dspy_optimizer == "copro":
            teleprompter = dspy.COPRO(
                metric=dspy_metric,
                verbose=False,
            )
        elif self.config.dspy_optimizer == "miprov2":
            teleprompter = dspy.MIPROv2(
                metric=dspy_metric,
                num_threads=1,
            )
        else:
            raise ValueError(
                f"Unknown DSPy optimizer: {self.config.dspy_optimizer!r}. "
                f"Supported: 'copro', 'miprov2'."
            )

        compiled_module = teleprompter.compile(
            module,
            trainset=trainset,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=0,
        )

        # Extract optimized instruction
        optimized_prompt = compiled_module.generate.signature.instructions
        optimized_score = metric(optimized_prompt)

        trial_history.append(
            TrialRecord(
                trial_id=0,
                prompt=optimized_prompt,
                score=optimized_score,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            improvement=optimized_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="dspy",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_dspy_adapter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update adapters `__init__.py` to register DSPy**

Add to `src/healthbench_agent/prompt_optimization/adapters/__init__.py`:

```python
"""Prompt optimizer adapters.

Importing this package triggers registration of all built-in adapters.
"""

from . import critique_refine_adapter  # noqa: F401
from . import dspy_adapter  # noqa: F401
```

- [ ] **Step 6: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/adapters/dspy_adapter.py \
       src/healthbench_agent/prompt_optimization/adapters/__init__.py \
       tests/prompt_optimization/test_dspy_adapter.py
git commit -m "feat(prompt_optimization): add DSPyOptimizer adapter"
```

---

### Task 7: TextGradOptimizer Adapter

**Files:**
- Create: `src/healthbench_agent/prompt_optimization/adapters/textgrad_adapter.py`
- Create: `tests/prompt_optimization/test_textgrad_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/prompt_optimization/test_textgrad_adapter.py
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
    """Tests for TextGradOptimizer with mocked TextGrad calls."""

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

        # Mock TextGrad Variable
        mock_var = MagicMock()
        mock_var.value = "Optimized via TextGrad."
        mock_tg.Variable.return_value = mock_var

        # Mock TextGrad engine and optimizer
        mock_engine = MagicMock()
        mock_tg.get_engine.return_value = mock_engine
        mock_optimizer = MagicMock()
        mock_tg.TGD.return_value = mock_optimizer

        # Mock loss
        mock_loss_var = MagicMock()
        mock_loss_var.value = "0.3"
        mock_tg.Variable.side_effect = [mock_var, mock_loss_var]

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prompt_optimization/test_textgrad_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/prompt_optimization/adapters/textgrad_adapter.py
"""TextGrad prompt optimizer adapter.

Uses TextGrad's text-gradient descent to iteratively refine a prompt.
TextGrad is lazily imported so the dependency is only required when
this adapter is used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..config import TextGradConfig
from ..optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from ..optimizer_registry import register_prompt_optimizer

if TYPE_CHECKING:
    from healthbench_agent.domain.dataset import HealthBenchSample

    from ..metric import EndToEndMetric


@register_prompt_optimizer("textgrad", TextGradConfig)
class TextGradOptimizer(PromptOptimizer):
    """Prompt optimizer using TextGrad's text-gradient descent.

    Wraps the prompt as a TextGrad Variable, defines loss as
    ``1.0 - metric(prompt)``, and runs gradient descent steps to
    iteratively improve the prompt.

    Attributes:
        config: TextGrad-specific optimization configuration.
    """

    def __init__(self, config: TextGradConfig) -> None:
        self.config = config

    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Run TextGrad optimization on the prompt.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required.
            metric: Scoring callable. Required.

        Returns:
            OptimizationResult with the TextGrad-optimized prompt.

        Raises:
            ValueError: If samples or metric is None.
        """
        if samples is None:
            raise ValueError(
                "TextGradOptimizer requires samples for evaluation. "
                "Pass a list of HealthBenchSample instances."
            )
        if metric is None:
            raise ValueError(
                "TextGradOptimizer requires a metric for scoring. "
                "Pass an EndToEndMetric instance."
            )

        import textgrad

        # Configure TextGrad engine
        engine = textgrad.get_engine(
            f"{self.config.meta_provider}/{self.config.meta_model}"
        )

        # Create optimizable prompt variable
        prompt_var = textgrad.Variable(
            current_prompt,
            role_description="System prompt for a health assistant agent",
            requires_grad=True,
        )

        # Create optimizer
        optimizer = textgrad.TGD(parameters=[prompt_var], engine=engine)

        baseline_score = metric(current_prompt)
        trial_history: list[TrialRecord] = []
        best_prompt = current_prompt
        best_score = baseline_score

        for step in range(self.config.steps):
            # Compute loss (TextGrad minimizes, so invert the score)
            current_score = metric(prompt_var.value)
            loss_value = 1.0 - current_score
            loss = textgrad.Variable(
                str(loss_value),
                role_description=(
                    "Loss value for prompt optimization. "
                    "Lower is better. This represents 1.0 minus the "
                    "health evaluation rubric score."
                ),
            )

            # Backward pass and update
            loss.backward(engine)
            optimizer.step()

            # Record trial
            new_score = metric(prompt_var.value)
            trial_history.append(
                TrialRecord(
                    trial_id=step,
                    prompt=prompt_var.value,
                    score=new_score,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )

            if new_score > best_score:
                best_prompt = prompt_var.value
                best_score = new_score

        return OptimizationResult(
            optimized_prompt=best_prompt,
            baseline_score=baseline_score,
            optimized_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            trial_history=trial_history,
            optimizer_name="textgrad",
            config=self.config.model_dump(exclude={"google_api_key", "openai_api_key"}),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prompt_optimization/test_textgrad_adapter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update adapters `__init__.py` to register TextGrad**

```python
# src/healthbench_agent/prompt_optimization/adapters/__init__.py
"""Prompt optimizer adapters.

Importing this package triggers registration of all built-in adapters.
"""

from . import critique_refine_adapter  # noqa: F401
from . import dspy_adapter  # noqa: F401
from . import textgrad_adapter  # noqa: F401
```

- [ ] **Step 6: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/adapters/textgrad_adapter.py \
       src/healthbench_agent/prompt_optimization/adapters/__init__.py \
       tests/prompt_optimization/test_textgrad_adapter.py
git commit -m "feat(prompt_optimization): add TextGradOptimizer adapter"
```

---

### Task 8: Public API `__init__.py`

**Files:**
- Modify: `src/healthbench_agent/prompt_optimization/__init__.py`

- [ ] **Step 1: Write the public API**

```python
# src/healthbench_agent/prompt_optimization/__init__.py
"""Automatic prompt engineering for agent system prompts.

Provides a registry-based adapter pattern for optimizing agent prompts
using different backends (DSPy, TextGrad, critique-refine) behind a
common PromptOptimizer abstraction.

Public API:
    - PromptOptimizer: Abstract base for all optimizers.
    - OptimizationResult: Frozen result of an optimization run.
    - TrialRecord: Per-trial details.
    - EndToEndMetric: Agent + judge scoring callable.
    - BaseOptimizationConfig: Shared config base.
    - DSPyConfig, TextGradConfig, CritiqueRefineConfig: Per-framework configs.
    - register_prompt_optimizer: Decorator for registering new optimizers.
    - create_prompt_optimizer: Factory that dispatches on config.optimizer.
    - registered_prompt_optimizers: List registered backends.
"""

from .config import (
    BaseOptimizationConfig,
    CritiqueRefineConfig,
    DSPyConfig,
    TextGradConfig,
)
from .metric import EndToEndMetric
from .optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from .optimizer_registry import (
    create_prompt_optimizer,
    register_prompt_optimizer,
    registered_prompt_optimizers,
)

# Import adapters to trigger registration
from . import adapters as _adapters  # noqa: F401

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "TrialRecord",
    "EndToEndMetric",
    "BaseOptimizationConfig",
    "DSPyConfig",
    "TextGradConfig",
    "CritiqueRefineConfig",
    "register_prompt_optimizer",
    "create_prompt_optimizer",
    "registered_prompt_optimizers",
]
```

- [ ] **Step 2: Verify all tests pass**

Run: `uv run pytest tests/prompt_optimization/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/__init__.py
git commit -m "feat(prompt_optimization): add public API exports"
```

---

### Task 9: Dependencies and CLI Entry Point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional dependencies and CLI entry point**

Add to `pyproject.toml`:

In `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
notebooks = [
    "jupyterlab>=4.3.0",
    "ipywidgets>=8.1.0",
]
optimization = [
    "dspy>=2.5.0",
    "textgrad>=0.2.0",
]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]
```

In `[project.scripts]`:
```toml
[project.scripts]
download-healthbench = "healthbench_agent.dataset.loader:_cli"
track-experiment = "healthbench_agent.llm_eval.cli:main"
optimize-prompt = "healthbench_agent.prompt_optimization.cli:main"
```

- [ ] **Step 2: Create the CLI module**

```python
# src/healthbench_agent/prompt_optimization/cli.py
"""CLI entry point for prompt optimization.

Usage::

    uv run optimize-prompt \
        --agent-config config/agents/baseline_agent.yaml \
        --optimizer critique_refine \
        --sample-size 20 \
        --max-trials 50 \
        --subset consensus \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Run prompt optimization from the command line."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Optimize agent system prompts via automatic prompt engineering."
    )
    parser.add_argument(
        "--agent-config",
        required=True,
        help="Path to agent YAML config (e.g. config/agents/baseline_agent.yaml).",
    )
    parser.add_argument(
        "--optimizer",
        default="critique_refine",
        choices=["dspy", "textgrad", "critique_refine"],
        help="Optimizer backend to use (default: critique_refine).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of conversations for evaluation (default: 20).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="Maximum candidate prompts to evaluate (default: 50).",
    )
    parser.add_argument(
        "--subset",
        default="consensus",
        choices=["main", "hard", "consensus"],
        help="HealthBench subset to evaluate (default: consensus).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--mutation-only",
        action="store_true",
        help="Run critique-refine in mutation-only mode (no evaluation).",
    )
    args = parser.parse_args()

    from healthbench_agent.agent import RootAgentPipelineConfig, create_pipeline
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample
    from healthbench_agent.llm_eval import EvalRunner, JudgeConfig, create_judge
    from healthbench_agent.prompt_optimization import (
        CritiqueRefineConfig,
        DSPyConfig,
        EndToEndMetric,
        TextGradConfig,
        create_prompt_optimizer,
    )

    # Load agent config and extract current prompt
    agent_config = RootAgentPipelineConfig.from_yaml(args.agent_config)
    from healthbench_agent.agent.prompt import load_instruction

    current_prompt = load_instruction(agent_config.prompt_path, agent_config.prompt_key)
    logger.info("Agent: %s, model: %s", agent_config.name, agent_config.model)
    logger.info("Current prompt length: %d chars", len(current_prompt))

    # Build optimizer config
    config_map = {
        "dspy": DSPyConfig,
        "textgrad": TextGradConfig,
        "critique_refine": CritiqueRefineConfig,
    }
    config_class = config_map[args.optimizer]
    optim_config = config_class(
        max_trials=args.max_trials,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    # Build optimizer
    optimizer = create_prompt_optimizer(optim_config)

    # Load samples and build metric (unless mutation-only)
    samples = None
    metric = None
    if not args.mutation_only:
        dataset = load_dataset(subset=args.subset)
        sampled = stratified_sample(
            dataset, n=args.sample_size, tag_prefix="theme", seed=args.seed
        )
        samples = list(sampled.samples)
        logger.info(
            "Loaded %d samples from %s subset", len(samples), args.subset
        )

        judge_config = JudgeConfig()
        judge = create_judge(judge_config)
        metric = EndToEndMetric(
            agent_config=agent_config,
            judge=judge,
            samples=samples,
        )

    # Run optimization
    logger.info("Running %s optimizer...", args.optimizer)
    result = optimizer.optimize(
        current_prompt=current_prompt,
        samples=samples,
        metric=metric,
    )

    # Report results
    logger.info(
        "Optimization complete: %.4f -> %.4f (%+.4f)",
        result.baseline_score,
        result.optimized_score,
        result.improvement,
    )
    logger.info("Trials evaluated: %d", result.num_trials)

    # Save optimized prompt
    prompt_dir = Path(agent_config.prompt_path).parent
    output_path = prompt_dir / "v2_optimized.yaml"
    _save_optimized_prompt(
        output_path=output_path,
        optimized_prompt=result.optimized_prompt,
        result=result,
        parent_version=agent_config.prompt_version,
    )
    logger.info("Optimized prompt saved to %s", output_path)

    # Save trial history
    trials_path = prompt_dir / "optimization_trials.json"
    trials_data = {
        "optimizer": result.optimizer_name,
        "baseline_score": result.baseline_score,
        "optimized_score": result.optimized_score,
        "improvement": result.improvement,
        "num_trials": result.num_trials,
        "trials": [
            {
                "trial_id": t.trial_id,
                "score": t.score,
                "timestamp": t.timestamp,
                "prompt_length": len(t.prompt),
            }
            for t in result.trial_history
        ],
    }
    trials_path.write_text(json.dumps(trials_data, indent=2))
    logger.info("Trial history saved to %s", trials_path)


def _save_optimized_prompt(
    output_path: Path,
    optimized_prompt: str,
    result: object,
    parent_version: str,
) -> None:
    """Save the optimized prompt as a versioned YAML file.

    Args:
        output_path: Path to write the YAML file.
        optimized_prompt: The optimized prompt text.
        result: OptimizationResult for metadata.
        parent_version: Version of the parent prompt.
    """
    import yaml

    from healthbench_agent.prompt_optimization.optimizer import OptimizationResult

    assert isinstance(result, OptimizationResult)

    data = {
        "version": "2.0.0",
        "created": __import__("datetime").date.today().isoformat(),
        "parent_version": parent_version,
        "architecture": "Optimized via APE",
        "rationale": (
            f"Automatically optimized using {result.optimizer_name}. "
            f"Score: {result.baseline_score:.4f} -> {result.optimized_score:.4f} "
            f"({result.improvement:+.4f}). "
            f"Trials: {result.num_trials}."
        ),
        "instruction": optimized_prompt,
    }
    output_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
```

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/prompt_optimization/ -v`
Expected: All tests PASS

- [ ] **Step 4: Sync dependencies**

Run: `uv sync --all-extras`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml \
       src/healthbench_agent/prompt_optimization/cli.py
git commit -m "feat(prompt_optimization): add CLI entry point and optional dependencies"
```

---

### Task 10: Run Full Test Suite and Lint

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All existing tests still pass, all new tests pass.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/healthbench_agent/prompt_optimization/ tests/prompt_optimization/`
Expected: No lint errors.

- [ ] **Step 3: Run formatter**

Run: `uv run ruff format src/healthbench_agent/prompt_optimization/ tests/prompt_optimization/`

- [ ] **Step 4: Run type checker**

Run: `uv run mypy src/healthbench_agent/prompt_optimization/`
Expected: No type errors (or only expected ones from mocked dependencies).

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "chore: lint and format prompt_optimization module"
```

- [ ] **Step 6: Push all commits to the feature branch**

```bash
git push
```
