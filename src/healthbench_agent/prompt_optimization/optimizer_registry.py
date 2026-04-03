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
        raise ValueError(f"Unknown prompt optimizer: {config.optimizer!r}. Available: {available}")
    _, optimizer_class = _PROMPT_OPTIMIZER_REGISTRY[config.optimizer]
    return optimizer_class(config)


def registered_prompt_optimizers() -> dict[
    str, tuple[type[BaseOptimizationConfig], type[PromptOptimizer]]
]:
    """Return a copy of the current optimizer registry.

    Returns:
        Dict mapping optimizer names to (config_class, optimizer_class) tuples.
    """
    return dict(_PROMPT_OPTIMIZER_REGISTRY)
