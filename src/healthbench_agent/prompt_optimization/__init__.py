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
