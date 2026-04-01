"""MLflow experiment tracking for HealthBench agent evaluation.

Wraps MLflow to log parameters, metrics, and artifacts for each
evaluation run as specified in SPEC §5.4. Each run captures the full
configuration needed to reproduce results: agent identity, judge
settings, prompt fingerprint, and per-dimension scores.

Usage::

    uv run python -m evaluation.experiment_tracker \
        --agent baseline_agent --sample-size 100
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "healthbench-agent-eval"


@dataclass
class RunParams:
    """Parameters logged at the start of every MLflow run.

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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dictionary suitable for mlflow.log_params."""
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
        }


@dataclass
class RunMetrics:
    """Metrics logged for an evaluation run.

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
        """
        metrics: dict[str, float] = {"overall_score": self.overall_score}
        for theme, score in self.theme_scores.items():
            metrics[f"theme/{theme}/mean"] = score
        for axis, score in self.axis_scores.items():
            metrics[f"axis/{axis}/mean"] = score
        return metrics


def log_evaluation_run(
    params: RunParams,
    metrics: RunMetrics,
    results_json: dict[str, Any] | None = None,
    prompt_yaml_path: Path | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log a complete evaluation run to MLflow.

    Creates (or reuses) an MLflow experiment, starts a run, logs all
    parameters and metrics, and optionally logs artifacts.

    Args:
        params: Run parameters (agent config, judge config).
        metrics: Run metrics (overall + per-dimension scores).
        results_json: Full per-conversation results to log as artifact.
        prompt_yaml_path: Path to the prompt YAML to log as artifact.
        experiment_name: MLflow experiment name.

    Returns:
        The MLflow run ID string.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run %s", run_id)

        mlflow.log_params(params.to_dict())
        mlflow.log_metrics(metrics.to_flat_dict())

        if results_json is not None:
            _log_json_artifact(results_json, "results.json")

        if prompt_yaml_path is not None and prompt_yaml_path.exists():
            mlflow.log_artifact(str(prompt_yaml_path))

        logger.info(
            "Logged run %s: agent=%s, overall_score=%.4f",
            run_id,
            params.agent_name,
            metrics.overall_score,
        )

    return run_id


def _log_json_artifact(data: dict[str, Any], filename: str) -> None:
    """Write a dict as JSON to a temp file and log it as an MLflow artifact."""
    tmp_dir = tempfile.mkdtemp()
    try:
        path = Path(tmp_dir) / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        mlflow.log_artifact(str(path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def log_comparison(
    run_id_a: str,
    run_id_b: str,
    bootstrap_ci: dict[str, Any],
    t_test: dict[str, Any],
    cohens_d: float,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log a statistical comparison between two runs.

    Args:
        run_id_a: MLflow run ID of the baseline agent.
        run_id_b: MLflow run ID of the challenger agent.
        bootstrap_ci: Bootstrap CI result as dict.
        t_test: Paired t-test result as dict.
        cohens_d: Cohen's d effect size.
        experiment_name: MLflow experiment name.

    Returns:
        The MLflow run ID of the comparison run.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "comparison_type": "paired",
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        mlflow.log_metrics({
            "mean_difference": bootstrap_ci.get("mean_difference", 0.0),
            "ci_lower": bootstrap_ci.get("ci_lower", 0.0),
            "ci_upper": bootstrap_ci.get("ci_upper", 0.0),
            "t_statistic": t_test.get("t_statistic", 0.0),
            "p_value": t_test.get("p_value", 1.0),
            "cohens_d": cohens_d,
        })

        comparison_data = {
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "bootstrap_ci": bootstrap_ci,
            "t_test": t_test,
            "cohens_d": cohens_d,
        }
        _log_json_artifact(comparison_data, "comparison.json")

        logger.info(
            "Logged comparison run %s: %s vs %s, d=%.3f",
            run_id,
            run_id_a,
            run_id_b,
            cohens_d,
        )

    return run_id
