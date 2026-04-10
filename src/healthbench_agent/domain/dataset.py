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
        """Deserialize a HealthBench JSONL row into a sample.

        Args:
            data: Parsed JSONL row. Must contain ``prompt_id``, ``prompt``,
                ``rubrics``, and ``example_tags``. May optionally contain
                ``ideal_completions_data`` and ``canary``.

        Returns:
            A populated ``HealthBenchSample``. ``gold_response`` and
            ``expected`` are left at their ``LabelledSample`` defaults
            (``None`` / ``{}``); they are populated downstream by the
            meta-evaluation pipeline.

        Raises:
            KeyError: When a required JSONL key is missing.
        """
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
    """Loaded HealthBench dataset for one subset.

    Attributes:
        subset: Subset name — one of ``main``, ``hard``, ``consensus``.
        samples: Ordered list of ``HealthBenchSample`` rows from the JSONL.
        source_path: Local path the dataset was loaded from.
    """

    subset: DatasetSubset
    samples: list[HealthBenchSample]
    source_path: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
