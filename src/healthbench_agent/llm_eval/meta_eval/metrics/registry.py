"""Meta-evaluation metric registry.

Holds the decorator, specification dataclass, level enum, and lookup
helpers. Metric implementations live in the sibling
:mod:`.agreement`, :mod:`.stratified`, and :mod:`.adversarial`
modules; importing the package at load time is what populates the
registry via their ``@register_meta_metric`` decorators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class MetricLevel(StrEnum):
    """Which row subset a metric operates on."""

    SAMPLE = "sample"
    RUBRIC = "rubric"
    ANY = "any"


Metric = Callable[["pd.DataFrame"], Any]


@dataclass(frozen=True)
class MetricSpec:
    """Registered metric metadata.

    Attributes:
        name: Unique identifier (used by --metrics CLI flag and metrics.json).
        fn: The pure metric function. Takes a level-filtered DataFrame.
        level: Which gold_source rows the runner passes to ``fn``.
        description: One-line human-readable summary for ``list-metrics``.
    """

    name: str
    fn: Metric
    level: MetricLevel
    description: str


_METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_meta_metric(
    name: str,
    *,
    level: MetricLevel,
    description: str,
) -> Callable[[Metric], Metric]:
    """Decorator that registers a meta-evaluation metric by name + level.

    Args:
        name: Unique identifier used by the CLI ``--metrics`` flag and
            persisted under ``metrics.json``.
        level: Which gold_source row subset (``sample`` / ``rubric`` /
            ``any``) the runner hands to the wrapped function.
        description: One-line human-readable summary shown by
            ``list-metrics``.

    Returns:
        A decorator that stores the wrapped metric in the module-level
        registry and returns it unchanged.
    """

    def decorator(fn: Metric) -> Metric:
        _METRIC_REGISTRY[name] = MetricSpec(name=name, fn=fn, level=level, description=description)
        return fn

    return decorator


def get_meta_metric(name: str) -> MetricSpec:
    """Look up a registered metric by name.

    Args:
        name: Metric identifier previously passed to
            :func:`register_meta_metric`.

    Returns:
        The :class:`MetricSpec` stored in the registry under ``name``.

    Raises:
        KeyError: When ``name`` is not registered.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric {name!r} is not registered. Available: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def registered_meta_metrics() -> dict[str, MetricSpec]:
    """Return the live meta-metric registry.

    Returns:
        The live ``dict[str, MetricSpec]`` backing the registry. Mutating
        the returned mapping affects global registration state — callers
        that only need a read-only view should copy it.
    """
    return _METRIC_REGISTRY
