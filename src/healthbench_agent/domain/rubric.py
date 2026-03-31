"""HealthBench rubric item domain model.

Mirrors simple-evals RubricItem. Nothing in this module imports from the
rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricItem:
    """One graded criterion within a HealthBench rubric.

    Mirrors simple-evals RubricItem with fields: criterion, points, tags.

    Attributes:
        criterion: Human-readable statement of what the criterion checks.
        points: Points awarded (positive) or deducted (negative) when met.
            Range: [-10, 10]. Emergency/safety criteria carry the highest weights.
        tags: Theme and axis labels this criterion belongs to (e.g. 'accuracy',
            'emergency_referral').
    """

    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.points}] {self.criterion}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricItem:
        """Deserialize from a JSON-compatible dict.

        Args:
            data: Dict with keys 'criterion', 'points', 'tags'.

        Returns:
            A RubricItem instance.
        """
        return cls(
            criterion=data["criterion"],
            points=data["points"],
            tags=data["tags"],
        )
