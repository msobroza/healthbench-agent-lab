"""Callback registry for agent pipelines.

Provides a decorator-based registry that maps callback names (strings)
to callable functions. Agent configs reference callbacks by name; the
registry resolves them to the actual functions at pipeline construction.

Usage::

    from healthbench_agent.agent.callback_registry import (
        register_callback,
        get_callback,
    )

    @register_callback("safety_filter")
    def safety_filter(callback_context, llm_response):
        ...

    # Later, resolve names from config:
    callback = get_callback("safety_filter")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_callback(name: str) -> Callable:
    """Decorator to register a callback function by name.

    Args:
        name: Unique string identifier for the callback.

    Returns:
        The original function, unchanged.

    Raises:
        ValueError: If a callback with the same name is already registered.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(
                f"Callback '{name}' is already registered "
                f"(existing: {_REGISTRY[name].__module__}.{_REGISTRY[name].__name__})"
            )
        _REGISTRY[name] = func
        return func

    return decorator


def get_callback(name: str) -> Callable[..., Any]:
    """Look up a registered callback by name.

    Args:
        name: The registered name of the callback.

    Returns:
        The callback function.

    Raises:
        KeyError: If the callback name is not registered.
    """
    if name not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Callback '{name}' is not registered. Available callbacks: {registered}")
    return _REGISTRY[name]


def registered_callbacks() -> dict[str, Callable[..., Any]]:
    """Return a copy of the current callback registry.

    Returns:
        Dict mapping callback names to functions.
    """
    return dict(_REGISTRY)
