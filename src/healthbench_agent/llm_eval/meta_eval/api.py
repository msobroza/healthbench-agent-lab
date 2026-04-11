"""High-level ``meta_evaluate`` API plus HealthBench loader/judge helpers.

This module layers a convenient single-call API around
:func:`run_meta_eval` — it handles dataset loading, gold-label
extraction, judge construction, and optional caching.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem

from .runner import run_meta_eval

if TYPE_CHECKING:
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    from .results import MetricResultsView


def _load_subset_for_meta_eval(
    subset: str,
    sample_size: int,
    seed: int,
) -> list[LabelledSample]:
    """Load a HealthBench subset and populate gold-label fields in place.

    Imports dataset utilities lazily so the meta_eval module stays
    importable without the full dataset stack.

    Args:
        subset: HealthBench subset name (e.g. ``"consensus"``).
        sample_size: Number of samples to draw via stratified sampling.
        seed: Random seed for reproducibility.

    Returns:
        Labelled samples with ``gold_response`` and ``expected`` populated.
    """
    from healthbench_agent.dataset.extraction import extract_ideal_completion_text
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample

    dataset = load_dataset(subset=cast(Any, subset))
    sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)
    out: list[LabelledSample] = []
    for sample in sampled.samples:
        gold = extract_ideal_completion_text(sample.ideal_completions_data)
        if gold is None:
            continue
        sample.gold_response = gold
        sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}
        out.append(sample)
    return out


def _build_judge_for_meta_eval(
    config: Any,
    temperature: float,
) -> tuple[JudgeGrader, str, str]:
    """Build the judge and return its (model_fingerprint, prompt_sha) for caching.

    Accepts either a file path (str/Path) or a pre-built ``JudgeConfig``.

    Args:
        config: Path to a YAML judge config or a ``JudgeConfig`` instance.
        temperature: Sampling temperature override.

    Returns:
        Tuple of ``(judge, model_fingerprint, prompt_sha)``.

    Raises:
        TypeError: When ``config`` is not a recognized type.
    """
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.grading.judge import create_judge, load_grader_prompt

    if isinstance(config, (str, Path)):
        cfg = JudgeConfig.from_yaml(str(config))
    elif isinstance(config, JudgeConfig):
        cfg = config
    else:
        raise TypeError(f"judge_config must be a path or JudgeConfig, got {type(config)}")
    cfg = cfg.model_copy(update={"temperature": temperature})
    judge = create_judge(cfg)
    fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
    _, _, prompt_sha = load_grader_prompt(cfg.prompt_path)
    return judge, fingerprint, prompt_sha


def meta_evaluate(
    judge_config: Any,
    *,
    subset: str = "consensus",
    sample_size: int = 100,
    n_samples: int = 7,
    temperature: float = 1.0,
    metric_names: list[str] | None = None,
    sample_filter: Callable[[LabelledSample], bool] | None = None,
    rubric_filter: Callable[[RubricItem], bool] | None = None,
    output_dir: Path | str | None = None,
    cache: bool | VerdictCache = True,
    progress: bool | None = None,
    seed: int = 0,
) -> MetricResultsView:
    """Meta-evaluate one judge end-to-end with sensible defaults.

    Loads a HealthBench subset, builds the judge from config, and delegates
    to :func:`run_meta_eval` for grading + metric computation.

    Args:
        judge_config: Path to a YAML judge config file or a
            ``JudgeConfig`` instance.
        subset: HealthBench subset to evaluate on.
        sample_size: Number of samples to draw via stratified sampling.
        n_samples: Number of k-passes per (sample, flow) combination.
        temperature: Sampling temperature for the judge model.
        metric_names: Names of meta metrics to compute. Defaults to all.
        sample_filter: Keep only samples where the predicate is True.
        rubric_filter: Within each sample, keep only matching rubrics.
        output_dir: When set, persist artifacts to this directory.
        cache: True to use a default VerdictCache, False to disable,
            or a pre-built VerdictCache instance.
        progress: Tri-state progress bar toggle.
        seed: Random seed for stratified sampling.

    Returns:
        A :class:`MetricResultsView` wrapping the aggregated results.
    """
    from healthbench_agent.llm_eval.cache.store import (
        VerdictCache as _VerdictCacheImpl,
    )

    samples = _load_subset_for_meta_eval(subset, sample_size, seed)
    judge, fingerprint, prompt_sha = _build_judge_for_meta_eval(judge_config, temperature)

    if isinstance(cache, _VerdictCacheImpl):
        cache_obj: _VerdictCacheImpl | None = cache
    elif cache is True:
        cache_obj = _VerdictCacheImpl(enabled=True)
    else:
        cache_obj = None

    return run_meta_eval(
        judge=judge,
        labelled=samples,
        dimension_extractor=lambda r: r.category,
        metric_names=metric_names,
        n_samples=n_samples,
        sample_filter=sample_filter,
        rubric_filter=rubric_filter,
        output_dir=Path(output_dir) if output_dir is not None else None,
        judge_metadata={
            "judge_model": fingerprint,
            "temperature": temperature,
            "n_samples": n_samples,
            "subset": subset,
            "sample_size": sample_size,
            "seed": seed,
        },
        cache=cache_obj,
        model_fingerprint=fingerprint,
        judge_prompt_sha=prompt_sha,
        progress=progress,
    )
