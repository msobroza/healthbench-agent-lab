"""``meta-evaluate-judge`` CLI.

argparse entry point for the meta-evaluation harness.  Subcommands:

* ``run``                — grade a labelled set with one judge and compute metrics.
* ``regenerate``         — recompute metrics from an existing ``verdicts.parquet``.
* ``compare``            — side-by-side metric diff between two run directories.
* ``list-metrics``       — dump every registered meta metric with level/description.
* ``list-metadata-keys`` — discover metadata keys present in the consensus subset.
* ``clear-cache``        — wipe the on-disk verdict cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.cache import VerdictCache
from healthbench_agent.llm_eval.meta_eval import (
    EmptyFilterError,
    axis_filter,
    metadata_filter,
    run_meta_eval,
)

logger = logging.getLogger(__name__)


def _load_consensus_labelled(
    subset: str,
    sample_size: int,
    seed: int,
) -> tuple[list[LabelledSample], Callable[[RubricItem], str | None]]:
    """Load labelled samples and return them with a HealthBench axis extractor.

    Delegates dataset loading and gold-label population to
    :func:`_load_subset_for_meta_eval`, then layers a CLI-friendly empty-check
    and an ``axis_extractor`` closure on top.

    Args:
        subset: HealthBench subset name (e.g. ``"consensus"``).
        sample_size: Number of samples to draw via stratified sampling.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(labelled_samples, axis_extractor)`` where the extractor
        resolves a rubric item's axis from its tags or category.

    Raises:
        SystemExit: Exits with code 2 when the subset has no ideal-completion
            gold labels available (all samples dropped).
    """
    from healthbench_agent.llm_eval.meta_eval import (
        AXIS_TAG_PREFIX,
        _load_subset_for_meta_eval,
    )

    samples = _load_subset_for_meta_eval(subset, sample_size, seed)
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
    """Build a judge from a config file path and return its caching keys.

    Thin CLI wrapper around :func:`_build_judge_for_meta_eval` that narrows the
    accepted input to a filesystem path string (the library helper also accepts
    a pre-built ``JudgeConfig`` instance).

    Args:
        config_path: Filesystem path to a YAML judge config.
        temperature: Sampling temperature override.

    Returns:
        Tuple of ``(judge, model_fingerprint, prompt_sha)``.
    """
    from healthbench_agent.llm_eval.meta_eval import _build_judge_for_meta_eval

    return _build_judge_for_meta_eval(config_path, temperature)


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


def _default_cache_for_cli() -> VerdictCache:
    """Return the default verdict cache for CLI operations.

    Returns:
        A :class:`VerdictCache` pointing at the default XDG cache root,
        with caching enabled.
    """
    return VerdictCache(enabled=True)


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


def _add_other_parsers(subparsers: Any) -> None:
    # list-metrics
    subparsers.add_parser("list-metrics", help="Print all registered meta metrics")

    # clear-cache
    subparsers.add_parser("clear-cache", help="Wipe the on-disk verdict cache")

    # regenerate
    regen = subparsers.add_parser(
        "regenerate",
        help="Recompute metrics from an existing verdicts.parquet",
    )
    regen.add_argument("run_dir", help="Directory containing verdicts.parquet and metrics.json")

    # compare
    cmp = subparsers.add_parser(
        "compare",
        help="Side-by-side metric diff between two run directories",
    )
    cmp.add_argument("run_dir_a", help="First run directory")
    cmp.add_argument("run_dir_b", help="Second run directory")
    cmp.add_argument(
        "--output",
        default=None,
        help="Optional path to write a markdown comparison table",
    )

    # list-metadata-keys
    lmk = subparsers.add_parser(
        "list-metadata-keys",
        help="Discover metadata keys present in the consensus subset",
    )
    lmk.add_argument("--subset", default="consensus")
    lmk.add_argument("--sample-size", type=int, default=100)
    lmk.add_argument("--seed", type=int, default=0)


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


def _cmd_list_metrics(args: argparse.Namespace) -> None:  # noqa: ARG001
    from healthbench_agent.llm_eval.meta_eval.registry import registered_meta_metrics

    registry = registered_meta_metrics()
    print(f"{'NAME':<30} {'LEVEL':<8} DESCRIPTION")
    print("-" * 70)
    for name, spec in sorted(registry.items()):
        print(f"{name:<30} {spec.level.value:<8} {spec.description}")


def _cmd_clear_cache(args: argparse.Namespace) -> None:  # noqa: ARG001
    cache = _default_cache_for_cli()
    cache.clear()
    print(f"Cleared cache at {cache.root}")


def _cmd_regenerate(args: argparse.Namespace) -> None:
    import pandas as pd

    from healthbench_agent.llm_eval.meta_eval.registry import (
        MetricLevel,
        registered_meta_metrics,
    )
    from healthbench_agent.llm_eval.meta_eval.results_io import load_results, save_results

    run_dir = Path(args.run_dir)
    parquet_path = run_dir / "verdicts.parquet"
    if not parquet_path.exists():
        sys.stderr.write(f"verdicts.parquet not found in {run_dir}\n")
        raise SystemExit(2)

    df = pd.read_parquet(parquet_path)
    sample_rows = df[df["gold_source"] == "ideal_completion"]
    rubric_rows = df[df["gold_source"].isin(["example_meets", "example_fails"])]

    registry = registered_meta_metrics()
    scores: dict[str, Any] = {}
    for name, spec in registry.items():
        if spec.level is MetricLevel.SAMPLE:
            subset = sample_rows
        elif spec.level is MetricLevel.RUBRIC:
            subset = rubric_rows
        else:
            subset = df
        if len(subset) == 0:
            logger.info(
                "skipping metric %s (level=%s) — no matching rows",
                name,
                spec.level.value,
            )
            continue
        scores[name] = spec.fn(subset)

    # Load the existing results to preserve judge_metadata header, then update scores.
    view = load_results(run_dir)
    view.results.scores = scores
    save_results(view.results, run_dir)
    print(view.summary())


def _cmd_compare(args: argparse.Namespace) -> None:
    from healthbench_agent.llm_eval.meta_eval.results_io import load_results

    a_view = load_results(args.run_dir_a)
    b_view = load_results(args.run_dir_b)
    diff_df = a_view.compare(b_view)
    print(diff_df.to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(diff_df.to_markdown(index=False))
        print(f"\nMarkdown table written to {output_path}")


def _cmd_list_metadata_keys(args: argparse.Namespace) -> None:
    import collections

    samples, _ = _load_consensus_labelled(args.subset, args.sample_size, args.seed)
    top_level_keys = ("language", "specialty", "user_persona")
    counters: dict[str, collections.Counter[str]] = {
        k: collections.Counter() for k in top_level_keys
    }
    metadata_counter: collections.Counter[str] = collections.Counter()

    for sample in samples:
        for key in top_level_keys:
            value = getattr(sample, key, None)
            if value is not None:
                counters[key][str(value)] += 1
        for meta_key in sample.metadata:
            metadata_counter[meta_key] += 1

    print("Top-level fields:")
    for key in top_level_keys:
        values = counters[key]
        if values:
            print(f"  {key}: {dict(values)}")
        else:
            print(f"  {key}: (none)")

    print("\nmetadata dict keys:")
    if metadata_counter:
        for key, count in sorted(metadata_counter.items()):
            print(f"  {key}: {count} samples")
    else:
        print("  (none)")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``meta-evaluate-judge`` CLI.

    Parses argv, dispatches to the matching subcommand handler, and exits.
    Subcommands: ``run``, ``regenerate``, ``compare``, ``list-metrics``,
    ``list-metadata-keys``, ``clear-cache``.

    Args:
        argv: Command-line arguments to parse. Defaults to ``sys.argv[1:]``
            when None.

    Raises:
        SystemExit: Exits with code 1 when no subcommand is provided, or
            code 2 when the selected run's filters produced an empty set.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    _add_other_parsers(sub)
    args = parser.parse_args(argv)

    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "run": _cmd_run,
        "regenerate": _cmd_regenerate,
        "compare": _cmd_compare,
        "list-metrics": _cmd_list_metrics,
        "list-metadata-keys": _cmd_list_metadata_keys,
        "clear-cache": _cmd_clear_cache,
    }

    handler = handlers.get(args.command or "")
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
