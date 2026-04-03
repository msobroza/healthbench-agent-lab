"""CLI entry point for prompt optimization.

Usage::

    uv run optimize-prompt \
        --agent-config config/agents/baseline_agent.yaml \
        --optimizer critique_refine \
        --sample-size 20 \
        --max-trials 50 \
        --subset consensus \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main() -> None:
    """Run prompt optimization from the command line."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Optimize agent system prompts via automatic prompt engineering."
    )
    parser.add_argument(
        "--agent-config",
        required=True,
        help="Path to agent YAML config (e.g. config/agents/baseline_agent.yaml).",
    )
    parser.add_argument(
        "--optimizer",
        default="critique_refine",
        choices=["dspy", "textgrad", "critique_refine"],
        help="Optimizer backend to use (default: critique_refine).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of conversations for evaluation (default: 20).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="Maximum candidate prompts to evaluate (default: 50).",
    )
    parser.add_argument(
        "--subset",
        default="consensus",
        choices=["main", "hard", "consensus"],
        help="HealthBench subset to evaluate (default: consensus).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--mutation-only",
        action="store_true",
        help="Run critique-refine in mutation-only mode (no evaluation).",
    )
    args = parser.parse_args()

    from healthbench_agent.agent import RootAgentPipelineConfig
    from healthbench_agent.agent.prompt import load_instruction
    from healthbench_agent.prompt_optimization import (
        CritiqueRefineConfig,
        DSPyConfig,
        EndToEndMetric,
        TextGradConfig,
        create_prompt_optimizer,
    )

    # Load agent config and extract current prompt
    agent_config = RootAgentPipelineConfig.from_yaml(args.agent_config)
    current_prompt = load_instruction(agent_config.prompt_path, agent_config.prompt_key)
    logger.info("Agent: %s, model: %s", agent_config.name, agent_config.model)
    logger.info("Current prompt length: %d chars", len(current_prompt))

    # Build optimizer config
    config_map: dict = {
        "dspy": DSPyConfig,
        "textgrad": TextGradConfig,
        "critique_refine": CritiqueRefineConfig,
    }
    config_class = config_map[args.optimizer]
    optim_config = config_class(
        max_trials=args.max_trials,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    # Build optimizer
    optimizer = create_prompt_optimizer(optim_config)

    # Load samples and build metric (unless mutation-only)
    samples = None
    metric = None
    if not args.mutation_only:
        from healthbench_agent.dataset.loader import load_dataset
        from healthbench_agent.dataset.split_utils import stratified_sample
        from healthbench_agent.llm_eval import JudgeConfig, create_judge

        dataset = load_dataset(subset=args.subset)
        sampled = stratified_sample(dataset, n=args.sample_size, tag_prefix="theme", seed=args.seed)
        samples = list(sampled.samples)
        logger.info("Loaded %d samples from %s subset", len(samples), args.subset)

        judge_config = JudgeConfig()
        judge = create_judge(judge_config)
        metric = EndToEndMetric(
            agent_config=agent_config,
            judge=judge,
            samples=samples,
        )

    # Run optimization
    logger.info("Running %s optimizer...", args.optimizer)
    result = optimizer.optimize(
        current_prompt=current_prompt,
        samples=samples,
        metric=metric,
    )

    # Report results
    logger.info(
        "Optimization complete: %.4f -> %.4f (%+.4f)",
        result.baseline_score,
        result.optimized_score,
        result.improvement,
    )
    logger.info("Trials evaluated: %d", result.num_trials)

    # Save optimized prompt
    prompt_dir = Path(agent_config.prompt_path).parent
    output_path = prompt_dir / "v2_optimized.yaml"
    prompt_data = {
        "version": "2.0.0",
        "created": date.today().isoformat(),
        "parent_version": agent_config.prompt_version,
        "architecture": "Optimized via APE",
        "rationale": (
            f"Automatically optimized using {result.optimizer_name}. "
            f"Score: {result.baseline_score:.4f} -> {result.optimized_score:.4f} "
            f"({result.improvement:+.4f}). "
            f"Trials: {result.num_trials}."
        ),
        "instruction": result.optimized_prompt,
    }
    output_path.write_text(yaml.dump(prompt_data, default_flow_style=False, sort_keys=False))
    logger.info("Optimized prompt saved to %s", output_path)

    # Save trial history
    trials_path = prompt_dir / "optimization_trials.json"
    trials_data = {
        "optimizer": result.optimizer_name,
        "baseline_score": result.baseline_score,
        "optimized_score": result.optimized_score,
        "improvement": result.improvement,
        "num_trials": result.num_trials,
        "trials": [
            {
                "trial_id": t.trial_id,
                "score": t.score,
                "timestamp": t.timestamp,
                "prompt_length": len(t.prompt),
            }
            for t in result.trial_history
        ],
    }
    trials_path.write_text(json.dumps(trials_data, indent=2))
    logger.info("Trial history saved to %s", trials_path)
