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


def main(argv: list[str] | None = None) -> None:
    """Run prompt optimization from the command line.

    Args:
        argv: Optional command-line arguments (excluding the program name).
            When ``None``, defaults to ``sys.argv[1:]`` via ``parse_args``.
    """
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Optimize agent system prompts via automatic prompt engineering."
    )
    parser.add_argument(
        "--agent-config",
        default=None,
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
    parser.add_argument(
        "--prompt-domain",
        choices=("agent", "judge"),
        default="agent",
        help="Whether to optimize an agent prompt (default) or a judge prompt.",
    )
    parser.add_argument(
        "--judge-config",
        default=None,
        help="Path to JudgeConfig YAML — required when --prompt-domain judge.",
    )
    parser.add_argument(
        "--rubric-axis",
        action="append",
        default=[],
        help="Restrict meta-eval to rubric items tagged with this axis. Can repeat.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Restrict meta-eval samples by metadata KEY=VALUE. Can repeat.",
    )
    parser.add_argument(
        "--fitness",
        default="gold_score",
        help="Name of the meta-eval metric whose score drives optimization.",
    )
    args = parser.parse_args(argv)

    if args.prompt_domain == "agent" and args.agent_config is None:
        parser.error("--agent-config is required when --prompt-domain agent (the default)")

    if args.prompt_domain == "judge":
        judge_result = _run_judge_optimization(args)
        judge_output_path = _save_judge_yaml(args, judge_result)
        logger.info(
            "Optimized judge prompt saved to %s (%.4f -> %.4f, %+.4f across %d trials)",
            judge_output_path,
            judge_result["baseline_score"],
            judge_result["optimized_score"],
            judge_result["improvement"],
            judge_result["num_trials"],
        )
        return

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


def main_argv(argv: list[str]) -> None:
    """Convenience alias for tests passing an explicit argv list.

    Args:
        argv: Command-line arguments (excluding the program name).
    """
    main(argv)


def _judge_save_path(judge_config_path: str) -> Path:
    """Return the default output path for an optimized judge YAML.

    Args:
        judge_config_path: Path to the source JudgeConfig YAML.

    Returns:
        ``<same_dir>/v2_optimized.yaml``.
    """
    return Path(judge_config_path).parent / "v2_optimized.yaml"


def _run_judge_optimization(args: argparse.Namespace) -> dict[str, Any]:
    """Run the judge-prompt optimization path and return a result dict.

    Mirrors the structure of the agent path but builds a
    :class:`JudgeAgreementMetric` over a labelled HealthBench subset
    instead of :class:`EndToEndMetric` over an agent pipeline.

    Args:
        args: Parsed CLI namespace from the ``optimize-prompt`` parser.

    Returns:
        Dict with keys ``optimized_prompt``, ``baseline_score``,
        ``optimized_score``, ``improvement``, ``num_trials``,
        ``optimizer_name``, ``trials``, and ``target_prompt_path``.

    Raises:
        SystemExit: If ``--judge-config`` is missing.
    """
    from healthbench_agent.llm_eval.cli.meta_eval import (
        _build_filters,
        _load_consensus_labelled,
    )
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.prompt_optimization import (
        create_prompt_optimizer,
        get_optimizer_config_class,
    )
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    if args.judge_config is None:
        raise SystemExit("--prompt-domain judge requires --judge-config")
    cfg = JudgeConfig.from_yaml(args.judge_config)
    samples, axis_extractor = _load_consensus_labelled(
        subset="consensus", sample_size=args.sample_size, seed=args.seed
    )
    sample_filter, rubric_filter = _build_filters(args)
    metric = JudgeAgreementMetric(
        judge_config=cfg,
        labelled=samples,
        dimension_extractor=axis_extractor,
        n_samples=3,
        fitness=args.fitness,
        sample_filter=sample_filter,
        rubric_filter=rubric_filter,
    )
    config_class = get_optimizer_config_class(args.optimizer)
    optim_config = config_class(
        optimizer=args.optimizer,
        max_trials=args.max_trials,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    optimizer = create_prompt_optimizer(optim_config)
    # JudgeAgreementMetric expects the RAW template string, not a compiled
    # Jinja2 Template, so read the YAML directly rather than going through
    # load_grader_prompt(). Matches what the adapters pass to metric().
    data = yaml.safe_load(Path(cfg.prompt_path).read_text())
    current_template = data["template"].strip()
    # JudgeAgreementMetric satisfies the OptimizationMetric Protocol
    # structurally; PromptOptimizer.optimize still annotates its metric
    # parameter as EndToEndMetric | None, so cast at the call site.
    result = optimizer.optimize(
        current_prompt=current_template,
        samples=None,  # labelled set lives on the metric
        metric=metric,  # type: ignore[arg-type]
    )
    return {
        "optimized_prompt": result.optimized_prompt,
        "baseline_score": result.baseline_score,
        "optimized_score": result.optimized_score,
        "improvement": result.improvement,
        "num_trials": result.num_trials,
        "optimizer_name": result.optimizer_name,
        "trials": result.trial_history,
        "target_prompt_path": str(cfg.prompt_path),
    }


def _save_judge_yaml(args: argparse.Namespace, result: dict[str, Any]) -> Path:
    """Write an optimized judge prompt YAML next to the source config.

    Args:
        args: Parsed CLI namespace; ``args.judge_config`` selects the
            output directory.
        result: Dict returned from :func:`_run_judge_optimization`.

    Returns:
        The path the YAML was written to.
    """
    out = _judge_save_path(args.judge_config)
    payload = {
        "version": "2.0.0",
        "created": date.today().isoformat(),
        "parent_version": "1.0.0",
        "parent_prompt_path": result["target_prompt_path"],
        "template": result["optimized_prompt"],
        "rationale": (
            f"Automatically optimized using {result['optimizer_name']}. "
            f"Score: {result['baseline_score']:.4f} -> "
            f"{result['optimized_score']:.4f} "
            f"({result['improvement']:+.4f}). "
            f"Trials: {result['num_trials']}."
        ),
    }
    out.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))
    return out
