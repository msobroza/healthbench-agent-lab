"""Dataset-agnostic types for meta-evaluation of rubric-grading judges.

Defines ``LabelledSample`` (the parent class for any rubric-graded sample)
and ``MetricResults`` (the persisted shape of a meta-evaluation run).
Both are stdlib-only — no matplotlib, pandas, or pyarrow imports — so the
domain layer keeps its narrow dependency policy. The rich UX wrapper for
``MetricResults`` lives one layer out in ``llm_eval/meta_eval_results.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conversation import MessageList
from .rubric import RubricItem

SCHEMA_VERSION: int = 1
"""Bumped when MetricResults persistence format changes incompatibly."""


@dataclass
class LabelledSample:
    """A rubric-graded sample with optional gold labels for meta-evaluation.

    Acts as the dataset-agnostic shape for any rubric grading task. Concrete
    benchmarks subclass this and add their own fields. Meta-evaluation
    operates on lists of LabelledSample (or any subclass) without knowing
    which dataset they came from.

    Attributes:
        prompt_id: Unique identifier for joining samples across runs.
        prompt: Conversation history before the response to be graded.
        rubrics: Rubric items the judge will score the response against.
        gold_response: Known-good response text to grade for meta-evaluation.
            None for unlabelled samples (the typical agent-eval case).
        expected: Expected verdict per rubric criterion text. True = the
            criterion should be met by ``gold_response``, False = should not.
        language: ISO language code from SPEC.md schema. Optional.
        specialty: Medical specialty from SPEC.md schema. Optional.
        user_persona: 'patient' | 'healthcare professional'. Optional.
        metadata: Free-form per-sample metadata dict.
    """

    prompt_id: str
    prompt: MessageList
    rubrics: list[RubricItem]
    gold_response: str | None = None
    expected: dict[str, bool] = field(default_factory=dict)
    language: str | None = None
    specialty: str | None = None
    user_persona: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResults:
    """Aggregate meta-evaluation result for one judge run.

    Pure data — no methods that touch matplotlib, pandas, or the
    filesystem. Round-trips to JSON via ``to_dict()`` / ``from_dict()``.
    Wrap in :class:`MetricResultsView` (in
    ``llm_eval/meta_eval_results.py``) to get the rich UX surface.

    Attributes:
        scores: Mapping of metric name to its computed value.
        n_samples_graded: Number of LabelledSamples that produced verdicts.
        n_rubrics_graded: Total (sample, rubric) pairs across all k passes.
        judge_metadata: Run-level header (judge_model, temperature, k, ...).
        schema_version: Stamped from SCHEMA_VERSION at write time.
        verdicts_path: Path of the parquet that produced these scores.
            Populated by MetricResultsView.load; None for in-memory results.
    """

    scores: dict[str, Any]
    n_samples_graded: int
    n_rubrics_graded: int
    judge_metadata: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    verdicts_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict — coerces verdicts_path to a string."""
        payload: dict[str, Any] = {
            "scores": self.scores,
            "n_samples_graded": self.n_samples_graded,
            "n_rubrics_graded": self.n_rubrics_graded,
            "judge_metadata": self.judge_metadata,
            "schema_version": self.schema_version,
        }
        if self.verdicts_path is not None:
            payload["verdicts_path"] = str(self.verdicts_path)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResults:
        """Inverse of :meth:`to_dict`. Validates schema_version.

        Args:
            data: Dict previously produced by :meth:`to_dict`.

        Returns:
            A new MetricResults instance with fields populated from data.

        Raises:
            ValueError: If schema_version is newer than SCHEMA_VERSION.
        """
        version = data.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"MetricResults schema_version={version} is newer than this "
                f"build understands (max {SCHEMA_VERSION}). Upgrade healthbench-agent."
            )
        verdicts_path = data.get("verdicts_path")
        return cls(
            scores=data["scores"],
            n_samples_graded=data["n_samples_graded"],
            n_rubrics_graded=data["n_rubrics_graded"],
            judge_metadata=data["judge_metadata"],
            schema_version=version,
            verdicts_path=Path(verdicts_path) if verdicts_path is not None else None,
        )
