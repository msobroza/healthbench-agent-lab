"""Sample and rubric filter predicates used by ``run_meta_eval``.

Filters are pure closures over their configured arguments so the
runner stays decoupled from filter semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem

AXIS_TAG_PREFIX = "axis: "
"""Prefix used by HealthBench's stratified-sample helper for axis tags.

The trailing space matches the form written by
``healthbench_agent.dataset.split_utils._extract_stratum``. Shared with
the CLI's ``axis_extractor`` so the two helpers cannot drift.
"""


class EmptyFilterError(ValueError):
    """Raised when a sample/rubric filter combination eliminates all rows.

    Carries the names (or repr) of the active filters so the CLI can show
    the user exactly which flags caused the empty result.
    """

    def __init__(self, sample_filter: Any, rubric_filter: Any) -> None:
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter
        super().__init__(
            f"Empty filter result. sample_filter={sample_filter!r}, rubric_filter={rubric_filter!r}"
        )


def axis_filter(*axes: str) -> Callable[[RubricItem], bool]:
    """Keep rubrics whose ``category`` or ``axis: <name>`` tag matches any of *axes*.

    Args:
        *axes: One or more axis names to match. A rubric item is kept
            when either its ``category`` attribute equals one of these
            names, or one of its ``tags`` is ``"axis: <name>"`` with
            ``<name>`` in the requested set.

    Returns:
        Predicate that accepts a :class:`RubricItem` and returns True
        when any of the configured axes matches.
    """
    wanted = set(axes)

    def predicate(item: RubricItem) -> bool:
        if item.category in wanted:
            return True
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX) and tag[len(AXIS_TAG_PREFIX) :].strip() in wanted:
                return True
        return False

    return predicate


def metadata_filter(**conditions: Any) -> Callable[[LabelledSample], bool]:
    """Keep samples where every metadata key equals the given value.

    Top-level LabelledSample attributes (``language``, ``specialty``,
    ``user_persona``) are checked against the attribute; other keys are
    looked up in ``sample.metadata``.

    Args:
        **conditions: ``key=value`` constraints to enforce. A key matching
            one of the promoted top-level attributes is compared against
            that attribute; every other key is looked up in
            ``sample.metadata``.

    Returns:
        Predicate that accepts a :class:`LabelledSample` and returns True
        only when every configured constraint matches.
    """
    top_level = {"language", "specialty", "user_persona"}

    def predicate(sample: LabelledSample) -> bool:
        for key, value in conditions.items():
            if key in top_level:
                if getattr(sample, key) != value:
                    return False
            else:
                if sample.metadata.get(key) != value:
                    return False
        return True

    return predicate


def specialty_filter(*specialties: str) -> Callable[[LabelledSample], bool]:
    """Keep samples whose ``specialty`` field is in *specialties*.

    Args:
        *specialties: One or more specialty names to match against the
            ``LabelledSample.specialty`` attribute.

    Returns:
        Predicate that accepts a :class:`LabelledSample` and returns True
        when its specialty is in the requested set.
    """
    wanted = set(specialties)

    def predicate(sample: LabelledSample) -> bool:
        return sample.specialty in wanted

    return predicate
