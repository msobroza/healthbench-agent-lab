"""HealthBench dataset domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .meta_evaluation import LabelledSample
from .rubric import RubricItem

DatasetSubset = Literal["main", "hard", "consensus"]


@dataclass
class HealthBenchSample(LabelledSample):
    """One sample loaded from a HealthBench JSONL dataset file.

    Inherits prompt_id, prompt, rubrics, gold_response, expected,
    language, specialty, user_persona, and metadata from LabelledSample.
    Adds HealthBench-specific fields.

    Attributes:
        example_tags: Dataset-level tags for stratified scoring.
        ideal_completions_data: Physician ideal completion data when
            available. Used to populate gold_response/expected at
            meta-eval time via the CLI.
        canary: Dataset integrity signature.
    """

    example_tags: list[str] = field(default_factory=list)
    ideal_completions_data: dict[str, Any] | None = None
    canary: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthBenchSample:
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            rubrics=[RubricItem.from_dict(r) for r in data["rubrics"]],
            example_tags=data["example_tags"],
            ideal_completions_data=data.get("ideal_completions_data"),
            canary=data.get("canary"),
        )


@dataclass
class HealthBenchDataset:
    """A loaded HealthBench dataset subset with its samples and metadata."""

    subset: DatasetSubset
    samples: list[HealthBenchSample]
    source_path: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
