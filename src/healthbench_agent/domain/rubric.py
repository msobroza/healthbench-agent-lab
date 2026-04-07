"""HealthBench rubric item domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricItem:
    """One graded criterion within a rubric.

    The original HealthBench fields (criterion, points, tags) are unchanged.
    The SPEC.md schema fields below are optional with safe defaults so
    HealthBench loaders work unchanged.

    Attributes:
        criterion: Human-readable statement of what the criterion checks.
        points: Points awarded (positive) or deducted (negative) when met.
        tags: HealthBench-style tag list (e.g. ['axis: accuracy']).
        criterion_id: Stable id from the SPEC.md schema. None for HealthBench.
        category: Explicit category/axis name from SPEC.md.
        example_meets: Adversarial known-good response. When present,
            meta-eval grades it expecting criteria_met=True.
        example_fails: Adversarial known-bad response. When present,
            meta-eval grades it expecting criteria_met=False.
    """

    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)
    criterion_id: str | None = None
    category: str | None = None
    example_meets: str | None = None
    example_fails: str | None = None

    def __str__(self) -> str:
        return f"[{self.points}] {self.criterion}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            Dict that always includes ``criterion``, ``points``, and ``tags``.
            Any of ``criterion_id``, ``category``, ``example_meets``, and
            ``example_fails`` whose value is not None are also included;
            None-valued optional fields are omitted.
        """
        payload: dict[str, Any] = {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }
        for key in ("criterion_id", "category", "example_meets", "example_fails"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricItem:
        """Deserialize from a JSON-compatible dict.

        Tolerates missing ``tags`` (SPEC.md rows may omit it) and reads the
        optional SPEC.md fields when present.

        Args:
            data: Dict with keys 'criterion', 'points', and optionally 'tags'
                plus any SPEC.md fields.

        Returns:
            A RubricItem instance.
        """
        return cls(
            criterion=data["criterion"],
            points=data["points"],
            tags=data.get("tags", []),
            criterion_id=data.get("criterion_id"),
            category=data.get("category"),
            example_meets=data.get("example_meets"),
            example_fails=data.get("example_fails"),
        )
