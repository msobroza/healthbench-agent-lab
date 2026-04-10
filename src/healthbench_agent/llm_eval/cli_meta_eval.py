"""``meta-evaluate-judge`` CLI.

argparse subcommands: run / regenerate / compare / list-metrics /
list-metadata-keys / clear-cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.meta_eval import (
    EmptyFilterError,
    FakeJudge,  # noqa: F401  -- re-exported for tests
    axis_filter,
    metadata_filter,
    run_meta_eval,
)
from healthbench_agent.llm_eval.meta_eval_results import MetricResultsView  # noqa: F401
from healthbench_agent.llm_eval.verdict_cache import VerdictCache

logger = logging.getLogger(__name__)


def _load_consensus_labelled(
    subset: str,
    sample_size: int,
    seed: int,
) -> tuple[list[LabelledSample], Callable[[RubricItem], str | None]]:
    """Load + populate the labelled set, return it plus a HealthBench axis_extractor."""
    from healthbench_agent.dataset.extraction import extract_ideal_completion_text
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample
    from healthbench_agent.llm_eval.meta_eval import AXIS_TAG_PREFIX

    dataset = load_dataset(subset=cast(Any, subset))
    sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)
    samples: list[LabelledSample] = []
    for sample in sampled.samples:
        gold = extract_ideal_completion_text(sample.ideal_completions_data)
        if gold is None:
            continue
        sample.gold_response = gold
        sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}
        samples.append(sample)
    if not samples:
        sys.stderr.write(
            "All samples dropped during gold-label extraction. The selected subset may "
            "not ship physician ideal completions.\n"
        )
        raise SystemExit(2)

    def axis_extractor(item: RubricItem) -> str | None:
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX):
                return tag[len(AXIS_TAG_PREFIX) :].strip()
        return item.category

    return samples, axis_extractor


def _build_judge_for_cli(config_path: str, temperature: float) -> tuple[Any, str, str]:
    from healthbench_agent.llm_eval.config_grader import JudgeConfig
    from healthbench_agent.llm_eval.grader import create_judge, load_grader_prompt

    cfg = JudgeConfig.from_yaml(config_path)
    cfg = cfg.model_copy(update={"temperature": temperature})
    judge = create_judge(cfg)
    fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
    _, _, prompt_sha = load_grader_prompt(cfg.prompt_path)
    return judge, fingerprint, prompt_sha


def _build_filters(
    args: argparse.Namespace,
) -> tuple[Callable[[LabelledSample], bool] | None, Callable[[RubricItem], bool] | None]:
    rf = axis_filter(*args.rubric_axis) if args.rubric_axis else None
    parsed: dict[str, str] = {}
    for entry in args.metadata or []:
        if "=" not in entry:
            raise SystemExit(f"--metadata expects KEY=VALUE, got {entry!r}")
        key, value = entry.split("=", 1)
        parsed[key] = value
    sf = metadata_filter(**parsed) if parsed else None
    return sf, rf


def _add_run_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("run", help="Grade a labelled set with one judge")
    p.add_argument("--judge-config", required=True)
    p.add_argument("--subset", default="consensus")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--metrics", default="")
    p.add_argument("--rubric-axis", action="append", default=[])
    p.add_argument("--metadata", action="append", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _cmd_run(args: argparse.Namespace) -> None:
    samples, axis_extractor = _load_consensus_labelled(args.subset, args.sample_size, args.seed)
    sample_filter, rubric_filter = _build_filters(args)
    judge, fingerprint, prompt_sha = _build_judge_for_cli(args.judge_config, args.temperature)
    cache = VerdictCache(enabled=not args.no_cache)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        view = run_meta_eval(
            judge=judge,
            labelled=samples,
            dimension_extractor=axis_extractor,
            metric_names=metric_names,
            n_samples=args.n_samples,
            sample_filter=sample_filter,
            rubric_filter=rubric_filter,
            output_dir=output_dir,
            judge_metadata={
                "judge_model": fingerprint,
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "subset": args.subset,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "judge_prompt_sha": prompt_sha,
                "filter_axis": ",".join(args.rubric_axis),
                "filter_metadata": ",".join(args.metadata),
            },
            cache=cache,
            model_fingerprint=fingerprint,
            judge_prompt_sha=prompt_sha,
            progress=not args.no_progress,
        )
    except EmptyFilterError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    print(view.summary())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
