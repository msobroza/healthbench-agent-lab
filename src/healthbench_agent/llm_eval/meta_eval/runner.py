"""High-level meta-eval orchestrator: filter, grade, dispatch metrics, persist.

Not to be confused with :mod:`healthbench_agent.llm_eval.runner`, which
owns :class:`EvalRunner` for production grading. This module is for
meta-evaluation (scoring the judge itself).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.contrib.concurrent import thread_map

from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample, MetricResults
from healthbench_agent.domain.rubric import RubricItem

from .filters import EmptyFilterError
from .metrics.registry import _METRIC_REGISTRY, MetricLevel, get_meta_metric
from .verdicts import _VERDICT_COLUMNS, _build_verdict_rows

if TYPE_CHECKING:
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    from .results import MetricResultsView

logger = logging.getLogger(__name__)


def run_meta_eval(
    judge: JudgeGrader,
    labelled: list[LabelledSample],
    dimension_extractor: Callable[[RubricItem], str | None],
    metric_names: list[str] | None = None,
    n_samples: int = 7,
    sample_filter: Callable[[LabelledSample], bool] | None = None,
    rubric_filter: Callable[[RubricItem], bool] | None = None,
    output_dir: Path | None = None,
    judge_metadata: dict[str, Any] | None = None,
    cache: VerdictCache | None = None,
    model_fingerprint: str | None = None,
    judge_prompt_sha: str | None = None,
    meta_eval_max_workers: int = 16,
    progress: bool | None = None,
) -> MetricResultsView:
    """Grade labelled samples with one judge, compute metrics, and persist.

    Resolves sample/rubric filters, runs ``k=1..n_samples`` grading passes
    in parallel, dispatches each requested metric against its level-matching
    row subset, and returns a rich-UX view of the aggregated results.

    Args:
        judge: Judge grader to evaluate. Wrapped in
            :class:`CachedJudgeGrader` when ``cache`` is provided.
        labelled: Dataset-agnostic labelled samples to grade.
        dimension_extractor: Maps a rubric item to its dimension tag (e.g.
            axis name). The returned value lands in the ``dimension``
            column and feeds ``per_dimension_confusion``.
        metric_names: Names of meta metrics to compute. Defaults to every
            registered metric.
        n_samples: Number of k-passes per (sample, flow) combination.
        sample_filter: Keep only samples where the predicate is True.
        rubric_filter: Within each surviving sample, keep only rubrics
            where the predicate is True.
        output_dir: When set, persist ``verdicts.parquet`` and
            ``metrics.json`` under this directory.
        judge_metadata: Extra key/value pairs to stamp on the result header.
        cache: Verdict cache wrapper to deduplicate judge calls across runs.
        model_fingerprint: Required when ``cache`` is set.
        judge_prompt_sha: Required when ``cache`` is set.
        meta_eval_max_workers: Max worker threads for the parallel grader.
        progress: Tri-state progress bar toggle. ``None`` = auto (isatty).

    Returns:
        A :class:`MetricResultsView` wrapping a populated
        :class:`MetricResults`.

    Raises:
        EmptyFilterError: When every sample or every rubric is filtered
            out, or when every requested metric was skipped for lack of
            matching rows.
        ValueError: When ``cache`` is provided without ``model_fingerprint``
            and ``judge_prompt_sha``.
    """
    import pandas as pd

    from healthbench_agent.llm_eval.cache.cached_judge import CachedJudgeGrader

    from .results import MetricResultsView

    # --- Step 1: sample filter ------------------------------------------
    if sample_filter is not None:
        kept = [s for s in labelled if sample_filter(s)]
    else:
        kept = list(labelled)
    if not kept:
        raise EmptyFilterError(sample_filter=sample_filter, rubric_filter=rubric_filter)

    # --- Step 2: rubric filter (produces shallow-copied samples) ---------
    if rubric_filter is not None:
        surviving: list[LabelledSample] = []
        for sample in kept:
            new_rubrics = [r for r in sample.rubrics if rubric_filter(r)]
            if not new_rubrics:
                continue
            surviving.append(
                LabelledSample(
                    prompt_id=sample.prompt_id,
                    prompt=sample.prompt,
                    rubrics=new_rubrics,
                    gold_response=sample.gold_response,
                    expected=sample.expected,
                    language=sample.language,
                    specialty=sample.specialty,
                    user_persona=sample.user_persona,
                    metadata=sample.metadata,
                )
            )
    else:
        surviving = kept
    if not surviving:
        raise EmptyFilterError(sample_filter=sample_filter, rubric_filter=rubric_filter)

    # --- Step 3: build per-k grader factory -----------------------------
    if cache is not None:
        if model_fingerprint is None or judge_prompt_sha is None:
            raise ValueError(
                "run_meta_eval(cache=...) requires model_fingerprint and judge_prompt_sha"
            )
        cache_obj: VerdictCache = cache
        fingerprint: str = model_fingerprint
        prompt_sha: str = judge_prompt_sha

        def grade_for_k(k: int) -> JudgeGrader:
            return CachedJudgeGrader(judge, cache_obj, fingerprint, prompt_sha, k)
    else:

        def grade_for_k(k: int) -> JudgeGrader:
            return judge

    # --- Step 4: grade every (sample, k) pair in parallel ----------------
    pairs: list[tuple[LabelledSample, int]] = [
        (sample, k) for k in range(1, n_samples + 1) for sample in surviving
    ]

    def work(pair: tuple[LabelledSample, int]) -> list[dict[str, Any]]:
        sample, k = pair
        return _build_verdict_rows(grade_for_k(k), [sample], dimension_extractor, k)

    show_progress = progress if progress is not None else sys.stdout.isatty()
    chunks: list[list[dict[str, Any]]]
    if show_progress:
        chunks = list(
            thread_map(
                work,
                pairs,
                max_workers=meta_eval_max_workers,
                desc="Grading samples",
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=meta_eval_max_workers) as pool:
            chunks = list(pool.map(work, pairs))

    rows: list[dict[str, Any]] = [row for chunk in chunks for row in chunk]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_VERDICT_COLUMNS)

    # --- Step 5: partition by gold_source -------------------------------
    sample_rows = df[df["gold_source"] == "ideal_completion"]
    rubric_rows = df[df["gold_source"].isin(["example_meets", "example_fails"])]

    # --- Step 6: dispatch metrics by level ------------------------------
    requested = metric_names or list(_METRIC_REGISTRY.keys())
    scores: dict[str, Any] = {}
    skipped: list[str] = []
    for name in requested:
        spec = get_meta_metric(name)
        if spec.level is MetricLevel.SAMPLE:
            subset = sample_rows
        elif spec.level is MetricLevel.RUBRIC:
            subset = rubric_rows
        else:
            subset = df
        if len(subset) == 0:
            logger.info(
                "skipping metric %s (level=%s) - no matching rows in this run",
                name,
                spec.level.value,
            )
            skipped.append(name)
            continue
        scores[name] = spec.fn(subset)

    if not scores:
        raise EmptyFilterError(
            sample_filter=f"every metric skipped: {skipped}",
            rubric_filter=rubric_filter,
        )

    # --- Step 7: build MetricResults + persist --------------------------
    metadata = dict(judge_metadata or {})
    cache_stats = cache.stats() if cache is not None else {"hits": 0, "misses": 0}
    # Cache counters are ground truth for this run — overwrite any caller-provided
    # "cache_hits"/"cache_misses" so the persisted metrics reflect real cache activity.
    for key in ("cache_hits", "cache_misses"):
        if key in metadata:
            logger.warning(
                "judge_metadata[%r]=%r overwritten by real cache stats", key, metadata[key]
            )
    metadata["cache_hits"] = cache_stats["hits"]
    metadata["cache_misses"] = cache_stats["misses"]

    results = MetricResults(
        scores=scores,
        n_samples_graded=len(surviving),
        n_rubrics_graded=len(df),
        judge_metadata=metadata,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "verdicts.parquet")
        results.verdicts_path = output_dir / "verdicts.parquet"
        (output_dir / "metrics.json").write_text(json.dumps(results.to_dict(), indent=2))

    return MetricResultsView(results=results)
