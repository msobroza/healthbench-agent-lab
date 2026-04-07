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

# Import adapters to trigger registration
from . import adapters as _adapters  # noqa: F401
from .config import (
    BaseOptimizationConfig,
    CritiqueRefineConfig,
    DSPyConfig,
    TextGradConfig,
)
from .metric import (
    EndToEndMetric,
    accepts_instruction_override,
    find_agent_node,
    list_agent_names,
    locate_target,
)
from .optimizer import OptimizationResult, PromptOptimizer, TrialRecord
from .optimizer_registry import (
    create_prompt_optimizer,
    get_optimizer_config_class,
    register_prompt_optimizer,
    registered_prompt_optimizers,
)

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "TrialRecord",
    "EndToEndMetric",
    "accepts_instruction_override",
    "find_agent_node",
    "list_agent_names",
    "locate_target",
    "BaseOptimizationConfig",
    "DSPyConfig",
    "TextGradConfig",
    "CritiqueRefineConfig",
    "register_prompt_optimizer",
    "create_prompt_optimizer",
    "get_optimizer_config_class",
    "registered_prompt_optimizers",
]
