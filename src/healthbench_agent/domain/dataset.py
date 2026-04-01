"""HealthBench dataset domain types.

Defines DatasetSubset, HealthBenchSample, and HealthBenchDataset — the types
that represent a loaded HealthBench JSONL dataset. Mirrors the JSONL row schema
from simple-evals healthbench_eval.py.

Depends only on rubric types within this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .conversation import MessageList
from .rubric import RubricItem

# ---------------------------------------------------------------------------
# Dataset subset type alias
# ---------------------------------------------------------------------------

DatasetSubset = Literal["main", "hard", "consensus"]


# ---------------------------------------------------------------------------
# JSONL row and dataset container
# ---------------------------------------------------------------------------


@dataclass
class HealthBenchSample:
    """One sample loaded from a HealthBench JSONL dataset file.

    Field names match the JSONL keys used in simple-evals healthbench_eval.py.

    Attributes:
        prompt_id: Unique identifier for this prompt.
        prompt: Conversation history as a MessageList (role + content dicts).
        rubrics: Graded criteria used to score a response to this prompt.
        example_tags: Dataset-level tags for stratified scoring (themes, axes).
        ideal_completions_data: Physician ideal completion data when available,
            including the completion group and reference responses. None for
            samples without physician annotations.
        canary: Dataset integrity signature embedded by the benchmark authors.
            Format: 'healthbench:<uuid>'. Present in all records; not used for
            scoring.
    """

    prompt_id: str
    prompt: MessageList
    rubrics: list[RubricItem]
    example_tags: list[str]
    ideal_completions_data: dict[str, Any] | None = None
    canary: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthBenchSample:
        """Deserialize from a JSONL row dict.

        Args:
            data: Parsed JSONL row with keys prompt_id, prompt, rubrics,
                example_tags, and optionally ideal_completions_data and canary.

        Returns:
            A HealthBenchSample instance with rubrics deserialized as RubricItem.
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
    """A loaded HealthBench dataset subset with its samples and metadata.

    Attributes:
        subset: Which subset was loaded — 'main', 'hard', or 'consensus'.
        samples: All samples in the dataset, one per JSONL row.
        source_path: Absolute path to the JSONL file that was loaded.
    """

    subset: DatasetSubset
    samples: list[HealthBenchSample]
    source_path: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
