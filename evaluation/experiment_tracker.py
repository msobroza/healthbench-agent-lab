"""MLflow experiment tracking for HealthBench agent evaluation.

Wraps MLflow to log parameters, metrics, and artifacts for each
evaluation run as specified in SPEC §5.4. Each run captures the full
configuration needed to reproduce results: agent identity, judge
settings, prompt fingerprint, and per-dimension scores.

The pure data types (``RunParams``, ``RunMetrics``) live in
``healthbench_agent.domain.experiment``. This module provides the
MLflow I/O layer plus the ``build_run_params`` factory that bridges
``JudgeConfig`` + ``RootAgentPipelineConfig`` → ``RunParams``
(Dependency Inversion).

Usage::

    uv run python -m evaluation.experiment_tracker \
        --agent-config config/agents/baseline_agent.yaml \
        --sample-size 100 --seed 42
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from healthbench_agent.agent import RootAgentPipelineConfig
from healthbench_agent.domain.experiment import RunMetrics, RunParams
from healthbench_agent.llm_eval.config_grader import JudgeConfig
from healthbench_agent.llm_eval.grader import load_grader_prompt

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "healthbench-agent-eval"


def build_run_params(
    judge_config: JudgeConfig,
    agent_config: RootAgentPipelineConfig,
    sample_size: int,
    seed: int = 0,
) -> RunParams:
    """Build RunParams from a JudgeConfig and RootAgentPipelineConfig.

    Extracts grader settings from the judge config, agent identity from
    the agent config, and loads the prompt fingerprint (version + SHA-256)
    from the configured grader prompt file. This keeps ``RunParams`` as a
    pure domain type while the bridge logic lives in the outer layer
    (Dependency Inversion).

    Args:
        judge_config: Judge configuration with provider, model, and
            prompt path.
        agent_config: Agent configuration with name, model, and
            prompt version.
        sample_size: Number of conversations to evaluate.
        seed: Random seed used for sampling.

    Returns:
        A fully populated RunParams ready for ``log_evaluation_run``.
    """
    _, grader_prompt_version, grader_prompt_sha256 = load_grader_prompt(
        Path(judge_config.prompt_path)
    )
    return RunParams(
        agent_name=agent_config.name,
        prompt_version=agent_config.prompt_version,
        model=agent_config.model,
        sample_size=sample_size,
        grader_provider=judge_config.provider,
        grader_model=judge_config.model,
        grader_temperature=judge_config.temperature,
        grader_prompt_version=grader_prompt_version,
        grader_prompt_sha256=grader_prompt_sha256,
        eval_mode=str(judge_config.mode),
        seed=seed,
    )


def build_run_metrics(results: list[Any]) -> RunMetrics:
    """Build RunMetrics from a list of SingleEvalResult instances.

    Computes overall aggregate score and per-dimension breakdowns
    using the domain scoring functions.

    Args:
        results: Per-conversation SingleEvalResult instances.

    Returns:
        RunMetrics with overall, theme, and axis scores.
    """
    from healthbench_agent.domain.scoring import (
        aggregate_scores,
        stratified_scores,
    )

    overall = aggregate_scores(results)
    stratified = stratified_scores(results)

    theme_scores = {
        k.removeprefix("theme:"): v for k, v in stratified.items() if k.startswith("theme:")
    }
    axis_scores = {
        k.removeprefix("axis:"): v for k, v in stratified.items() if k.startswith("axis:")
    }
    return RunMetrics(
        overall_score=overall,
        theme_scores=theme_scores,
        axis_scores=axis_scores,
    )


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

    return str(run_id)


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

        mlflow.log_params(
            {
                "comparison_type": "paired",
                "run_id_a": run_id_a,
                "run_id_b": run_id_b,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        mlflow.log_metrics(
            {
                "mean_difference": bootstrap_ci.get("mean_difference", 0.0),
                "ci_lower": bootstrap_ci.get("ci_lower", 0.0),
                "ci_upper": bootstrap_ci.get("ci_upper", 0.0),
                "t_statistic": t_test.get("t_statistic", 0.0),
                "p_value": t_test.get("p_value", 1.0),
                "cohens_d": cohens_d,
            }
        )

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

    return str(run_id)


def _log_json_artifact(data: dict[str, Any], filename: str) -> None:
    """Write a dict as JSON to a temp file and log it as an MLflow artifact."""
    tmp_dir = tempfile.mkdtemp()
    try:
        path = Path(tmp_dir) / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        mlflow.log_artifact(str(path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _cli() -> None:
    """CLI entry point for running evaluation and logging to MLflow.

    Loads agent and judge configuration from YAML files, samples N
    conversations (stratified by theme, deterministic via ``--seed``),
    evaluates each sample with the configured LLM judge, and logs
    params + metrics + artifacts to MLflow.

    Usage::

        uv run python -m evaluation.experiment_tracker \
            --agent-config config/agents/baseline_agent.yaml \
            --sample-size 100 --seed 42
    """
    import argparse
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    from healthbench_agent.agent import create_pipeline
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample
    from healthbench_agent.llm_eval.runner import EvalRunner

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=("Run HealthBench evaluation and log results to MLflow."),
    )
    parser.add_argument(
        "--agent-config",
        required=True,
        help=("Path to agent YAML config (e.g. config/agents/baseline_agent.yaml)."),
    )
    parser.add_argument(
        "--judge-config",
        default=None,
        help=("Path to judge YAML config. Uses JudgeConfig defaults if not provided."),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of conversations to evaluate (default: 100).",
    )
    parser.add_argument(
        "--subset",
        default="main",
        choices=["main", "hard", "consensus"],
        help="HealthBench subset to evaluate (default: main).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 52).",
    )
    args = parser.parse_args()

    # Load configs from YAML
    agent_config = RootAgentPipelineConfig.from_yaml(args.agent_config)
    if args.judge_config:
        judge_config = JudgeConfig.from_yaml(args.judge_config)
    else:
        judge_config = JudgeConfig()

    logger.info(
        "Agent: name=%s, model=%s",
        agent_config.name,
        agent_config.model,
    )
    logger.info(
        "Judge: provider=%s, model=%s, mode=%s",
        judge_config.provider,
        judge_config.model,
        judge_config.mode,
    )

    # Load and sample dataset (deterministic via seed)
    dataset = load_dataset(subset=args.subset)
    sampled = stratified_sample(
        dataset,
        n=args.sample_size,
        tag_prefix="theme",
        seed=args.seed,
    )
    logger.info(
        "Loaded %d samples from %s subset (sampled %d, seed=%d)",
        len(dataset),
        args.subset,
        len(sampled),
        args.seed,
    )

    # Import tool modules so @register_tool decorators fire before pipeline creation.
    # tools/ is a working directory — importing tools.tools triggers all registrations.
    import tools.tools  # noqa: F401

    # Build agent pipeline and runner
    pipeline = create_pipeline(agent_config)
    runner = EvalRunner.from_config(judge_config)
    logger.info("Generating agent responses and evaluating...")
    results = asyncio.run(runner.evaluate_pipeline(pipeline, list(sampled.samples)))
    logger.info("Evaluated %d samples", len(results))

    # Build params and metrics
    params = build_run_params(
        judge_config=judge_config,
        agent_config=agent_config,
        sample_size=len(sampled),
        seed=args.seed,
    )
    metrics = build_run_metrics(results)

    # Build results JSON for artifact
    results_json = {
        "agent": agent_config.name,
        "subset": args.subset,
        "sample_size": len(sampled),
        "seed": args.seed,
        "results": [
            {
                "prompt_id": (r.example_level_metadata.get("prompt_id", "")),
                "score": r.score,
                "verdicts": (r.example_level_metadata.get("verdicts", [])),
            }
            for r in results
            if r.example_level_metadata
        ],
    }

    prompt_path = Path(judge_config.prompt_path)
    run_id = log_evaluation_run(
        params=params,
        metrics=metrics,
        results_json=results_json,
        prompt_yaml_path=prompt_path if prompt_path.exists() else None,
    )
    logger.info("MLflow run ID: %s", run_id)
    logger.info(
        "Overall score: %.4f | Themes: %s | Axes: %s",
        metrics.overall_score,
        {k: f"{v:.3f}" for k, v in metrics.theme_scores.items()},
        {k: f"{v:.3f}" for k, v in metrics.axis_scores.items()},
    )


if __name__ == "__main__":
    _cli()
