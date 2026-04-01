"""Experiment tracking metadata types — pure data, no I/O.

Defines the structured parameter and metric containers that describe
an evaluation run. These are consumed by the MLflow logging layer
in ``evaluation/experiment_tracker.py``.

Dependency order: only standard-library types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RunParams:
    """Parameters logged at the start of every evaluation run.

    Captures the full configuration needed to reproduce results:
    agent identity, judge settings, prompt fingerprint, and eval mode.

    Attributes:
        agent_name: Which architecture (baseline, tool, multi).
        prompt_version: Prompt YAML version string.
        model: Agent LLM model string.
        sample_size: Number of conversations evaluated.
        grader_provider: Judge LLM provider (openai or gemini).
        grader_model: Judge LLM model string.
        grader_temperature: Judge temperature setting.
        grader_prompt_version: Grader prompt YAML version.
        grader_prompt_sha256: SHA-256 of the raw grader template.
        eval_mode: Evaluation mode (async or batch).
        seed: Random seed used for dataset sampling.
    """

    agent_name: str
    prompt_version: str
    model: str
    sample_size: int
    grader_provider: str = "openai"
    grader_model: str = "gpt-4.1-2025-04-14"
    grader_temperature: float = 0.0
    grader_prompt_version: str = "1.0.0"
    grader_prompt_sha256: str = ""
    eval_mode: str = "async"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dictionary suitable for mlflow.log_params.

        Adds a UTC timestamp automatically so every run is time-stamped
        without the caller needing to manage it.

        Returns:
            Flat dict with all fields plus an ISO-format ``timestamp``.
        """
        return {
            "agent_name": self.agent_name,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "sample_size": self.sample_size,
            "timestamp": datetime.now(UTC).isoformat(),
            "grader_provider": self.grader_provider,
            "grader_model": self.grader_model,
            "grader_temperature": self.grader_temperature,
            "grader_prompt_version": self.grader_prompt_version,
            "grader_prompt_sha256": self.grader_prompt_sha256,
            "eval_mode": self.eval_mode,
            "seed": self.seed,
        }


@dataclass
class RunMetrics:
    """Metrics logged for an evaluation run.

    Stores overall plus per-dimension scores. The ``to_flat_dict()``
    method applies the ``theme/`` and ``axis/`` prefixes expected
    by the experiment tracking layer.

    Attributes:
        overall_score: Aggregate HealthBench score (0-100 scale).
        theme_scores: Per-theme mean scores keyed by theme name.
        axis_scores: Per-axis mean scores keyed by axis name.
    """

    overall_score: float
    theme_scores: dict[str, float] = field(default_factory=dict)
    axis_scores: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        """Serialise to a flat dictionary suitable for mlflow.log_metrics.

        Theme scores are prefixed with ``theme/`` and axis scores with
        ``axis/``.

        Returns:
            Flat dict mapping metric names to float values.
        """
        metrics: dict[str, float] = {"overall_score": self.overall_score}
        for theme, score in self.theme_scores.items():
            metrics[f"theme/{theme}/mean"] = score
        for axis, score in self.axis_scores.items():
            metrics[f"axis/{axis}/mean"] = score
        return metrics
