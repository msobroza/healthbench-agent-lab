"""Tool registry for agent pipelines.

Provides a decorator-based registry that maps tool names (strings) to
callable tool functions. Agent configs reference tools by name; the
registry resolves them to the actual functions at pipeline construction.

Usage::

    from healthbench_agent.agent.tool_registry import register_tool, get_tools

    @register_tool("drug_reference")
    def drug_reference(drug_name: str) -> dict:
        ...

    # Later, resolve names from config:
    tools = get_tools(["drug_reference", "symptom_checker"])
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_tool(name: str) -> Callable:
    """Decorator to register a tool function by name.

    Args:
        name: Unique string identifier for the tool.

    Returns:
        The original function, unchanged.

    Raises:
        ValueError: If a tool with the same name is already registered.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(
                f"Tool '{name}' is already registered "
                f"(existing: {_REGISTRY[name].__module__}.{_REGISTRY[name].__name__})"
            )
        _REGISTRY[name] = func
        return func

    return decorator


def get_tool(name: str) -> Callable[..., Any]:
    """Look up a registered tool by name.

    Args:
        name: The registered name of the tool.

    Returns:
        The tool function.

    Raises:
        KeyError: If the tool name is not registered.
    """
    if name not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Tool '{name}' is not registered. Available tools: {registered}"
        )
    return _REGISTRY[name]


def get_tools(names: list[str]) -> list[Callable[..., Any]]:
    """Resolve a list of tool names to their registered functions.

    Args:
        names: List of registered tool names.

    Returns:
        List of tool functions in the same order.

    Raises:
        KeyError: If any tool name is not registered.
    """
    return [get_tool(name) for name in names]


def registered_tools() -> dict[str, Callable[..., Any]]:
    """Return a copy of the current tool registry.

    Returns:
        Dict mapping tool names to functions.
    """
    return dict(_REGISTRY)
