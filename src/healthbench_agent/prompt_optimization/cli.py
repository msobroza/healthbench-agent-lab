"""CLI entry point for prompt optimization.

Usage::

    uv run optimize-prompt \
        --agent-config config/agents/baseline_agent.yaml \
        --optimizer critique_refine \
        --sample-size 20 \
        --max-trials 50 \
        --subset consensus \
        --seed 42

Multi-agent pipelines contain several prompts (one per sub-agent,
identified by ``prompt_key``). Optimize a specific sub-agent with
``--target-agent``::

    uv run optimize-prompt \
        --agent-config config/agents/multi_agent.yaml \
        --target-agent reviewer_agent \
        --optimizer critique_refine
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

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
        "--target-agent",
        default=None,
        help=(
            "Name of the sub-agent whose prompt should be optimized. "
            "Required when the pipeline has multiple prompts (e.g. multi_agent)."
        ),
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
    parser.add_argument(
        "--prompt-path",
        default=None,
        help=(
            "Path to a YAML file with critique-refine templates + "
            "thinking-styles (critique_refine optimizer only). "
            "Defaults to prompts/prompt_optimization/v1_critique_refine.yaml."
        ),
    )
    args = parser.parse_args()

    from healthbench_agent.agent import RootAgentPipelineConfig
    from healthbench_agent.agent.prompt import load_instruction
    from healthbench_agent.prompt_optimization import (
        EndToEndMetric,
        create_prompt_optimizer,
        get_optimizer_config_class,
    )
    from healthbench_agent.prompt_optimization.metric import (
        accepts_instruction_override,
        list_agent_names,
        locate_target,
    )

    # Load agent config and resolve the prompt to optimize
    agent_config = RootAgentPipelineConfig.from_yaml(args.agent_config)

    if args.target_agent is not None:
        target_node, effective_prompt_path = locate_target(agent_config, args.target_agent)
        current_prompt = load_instruction(effective_prompt_path, target_node.prompt_key)
        target_prompt_key = target_node.prompt_key
        target_prompt_path = effective_prompt_path
        logger.info(
            "Optimizing sub-agent %r (prompt_key=%s, prompt_path=%s)",
            target_node.name,
            target_node.prompt_key,
            effective_prompt_path,
        )
    else:
        # Refuse to optimize a composite root: its instruction_override
        # is silently dropped at build time and no sub-agent would ever
        # see the candidate prompt. Force the caller to pick a concrete
        # target so every trial exercises a real instruction.
        if not accepts_instruction_override(agent_config):
            available = ", ".join(list_agent_names(agent_config))
            parser.error(
                f"Root agent '{agent_config.name}' has orchestration "
                f"'{agent_config.orchestration}' with sub_agents — its "
                "instruction_override is silently dropped at build time. "
                "Pass --target-agent to pick a concrete sub-agent to "
                f"optimize. Available: {available}"
            )
        current_prompt = load_instruction(agent_config.prompt_path, agent_config.prompt_key)
        target_prompt_key = agent_config.prompt_key
        target_prompt_path = agent_config.prompt_path

    logger.info("Agent: %s, model: %s", agent_config.name, agent_config.model)
    logger.info("Current prompt length: %d chars", len(current_prompt))

    # Build optimizer config — registry is the single source of truth.
    # --prompt-path only applies to critique_refine; reject it otherwise
    # so typos against the wrong optimizer fail loudly instead of being
    # silently ignored.
    config_kwargs: dict[str, Any] = {
        "optimizer": args.optimizer,
        "max_trials": args.max_trials,
        "sample_size": args.sample_size,
        "seed": args.seed,
    }
    if args.prompt_path is not None:
        if args.optimizer != "critique_refine":
            parser.error(
                f"--prompt-path is only supported for --optimizer critique_refine "
                f"(got {args.optimizer!r})"
            )
        config_kwargs["prompt_path"] = args.prompt_path

    config_class = get_optimizer_config_class(args.optimizer)
    optim_config = config_class(**config_kwargs)

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
            target_agent_name=args.target_agent,
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

    # Save optimized prompt under the target's original prompt_key so that
    # the output YAML can be diffed/merged against the source file. When a
    # target agent is specified its name is included in the filename to
    # avoid clobbering other sub-agent optimizations in the same directory.
    prompt_dir = Path(target_prompt_path).parent
    if args.target_agent is not None:
        output_path = prompt_dir / f"v2_optimized_{args.target_agent}.yaml"
    else:
        output_path = prompt_dir / "v2_optimized.yaml"

    prompt_data = {
        "version": "2.0.0",
        "created": date.today().isoformat(),
        "parent_version": agent_config.prompt_version,
        "parent_prompt_path": target_prompt_path,
        "target_agent": args.target_agent or agent_config.name,
        "architecture": "Optimized via APE",
        "rationale": (
            f"Automatically optimized using {result.optimizer_name}. "
            f"Score: {result.baseline_score:.4f} -> {result.optimized_score:.4f} "
            f"({result.improvement:+.4f}). "
            f"Trials: {result.num_trials}."
        ),
        target_prompt_key: result.optimized_prompt,
    }
    output_path.write_text(yaml.dump(prompt_data, default_flow_style=False, sort_keys=False))
    logger.info("Optimized prompt saved to %s", output_path)

    # Save trial history
    trials_path = prompt_dir / (
        f"optimization_trials_{args.target_agent}.json"
        if args.target_agent is not None
        else "optimization_trials.json"
    )
    trials_data = {
        "optimizer": result.optimizer_name,
        "target_agent": args.target_agent or agent_config.name,
        "target_prompt_key": target_prompt_key,
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
