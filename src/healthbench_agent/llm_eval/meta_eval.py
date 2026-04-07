"""Meta-evaluation registry, runner, and built-in metrics for LLM-as-judge.

The module is dataset-agnostic: it operates on lists of LabelledSample.
HealthBench-specific glue (subset loading, ideal completion extraction)
lives in ``cli_meta_eval.py``.

Adding a new metric is one decorated function — name, level, description,
and the pure DataFrame transform. Zero changes anywhere else.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, cast

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample, MetricResults
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.scoring import calculate_score, clip_score

if TYPE_CHECKING:
    import pandas as pd

    from healthbench_agent.llm_eval.meta_eval_results import MetricResultsView
    from healthbench_agent.llm_eval.verdict_cache import VerdictCache

logger = logging.getLogger(__name__)


class MetricLevel(StrEnum):
    """Which row subset a metric operates on."""

    SAMPLE = "sample"
    RUBRIC = "rubric"
    ANY = "any"


Metric = Callable[["pd.DataFrame"], Any]


@dataclass(frozen=True)
class MetricSpec:
    """Registered metric metadata.

    Attributes:
        name: Unique identifier (used by --metrics CLI flag and metrics.json).
        fn: The pure metric function. Takes a level-filtered DataFrame.
        level: Which gold_source rows the runner passes to ``fn``.
        description: One-line human-readable summary for ``list-metrics``.
    """

    name: str
    fn: Metric
    level: MetricLevel
    description: str


_METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_meta_metric(
    name: str,
    *,
    level: MetricLevel,
    description: str,
) -> Callable[[Metric], Metric]:
    """Decorator that registers a meta-evaluation metric by name + level."""

    def decorator(fn: Metric) -> Metric:
        _METRIC_REGISTRY[name] = MetricSpec(name=name, fn=fn, level=level, description=description)
        return fn

    return decorator


def get_meta_metric(name: str) -> MetricSpec:
    """Look up a registered metric by name.

    Raises:
        KeyError: When ``name`` is not registered.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric {name!r} is not registered. Available: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def registered_meta_metrics() -> dict[str, MetricSpec]:
    """Return the live registry mapping (mutating it affects the registry)."""
    return _METRIC_REGISTRY


AXIS_TAG_PREFIX = "axis: "
"""Prefix used by HealthBench's stratified-sample helper for axis tags.

The trailing space matches the form written by
``healthbench_agent.dataset.split_utils._extract_stratum``. Shared with
the CLI's ``axis_extractor`` so the two helpers cannot drift.
"""


class EmptyFilterError(ValueError):
    """Raised when a sample/rubric filter combination eliminates all rows.

    Carries the names (or repr) of the active filters so the CLI can show
    the user exactly which flags caused the empty result.
    """

    def __init__(self, sample_filter: Any, rubric_filter: Any) -> None:
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter
        super().__init__(
            f"Empty filter result. sample_filter={sample_filter!r}, rubric_filter={rubric_filter!r}"
        )


def axis_filter(*axes: str) -> Callable[[RubricItem], bool]:
    """Keep rubrics whose ``category`` or ``axis: <name>`` tag matches any of *axes*."""
    wanted = set(axes)

    def predicate(item: RubricItem) -> bool:
        if item.category in wanted:
            return True
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX) and tag[len(AXIS_TAG_PREFIX) :].strip() in wanted:
                return True
        return False

    return predicate


def metadata_filter(**conditions: Any) -> Callable[[LabelledSample], bool]:
    """Keep samples where every metadata key equals the given value.

    Top-level LabelledSample attributes (``language``, ``specialty``,
    ``user_persona``) are checked against the attribute; other keys are
    looked up in ``sample.metadata``.
    """
    top_level = {"language", "specialty", "user_persona"}

    def predicate(sample: LabelledSample) -> bool:
        for key, value in conditions.items():
            if key in top_level:
                if getattr(sample, key) != value:
                    return False
            else:
                if sample.metadata.get(key) != value:
                    return False
        return True

    return predicate


def specialty_filter(*specialties: str) -> Callable[[LabelledSample], bool]:
    """Keep samples whose ``specialty`` field is in *specialties*."""
    wanted = set(specialties)

    def predicate(sample: LabelledSample) -> bool:
        return sample.specialty in wanted

    return predicate


@register_meta_metric(
    "gold_score",
    level=MetricLevel.SAMPLE,
    description="Mean clipped HealthBench score on gold responses (target = 1.0)",
)
def gold_score(verdicts: pd.DataFrame) -> float:
    """Mean clipped HealthBench score the judge gives to gold responses.

    Rebuilds (rubrics, verdicts) lists per (prompt_id, sample_k) group
    and delegates to ``calculate_score`` + ``clip_score`` so meta-eval
    cannot drift from production scoring.
    """
    if len(verdicts) == 0:
        return 0.0

    per_sample_scores: list[float] = []
    for _, group in verdicts.groupby(["prompt_id", "sample_k"], sort=False):
        rubric_items = [
            RubricItem(
                criterion=str(row.criterion),
                points=float(cast(float, row.points)),
                tags=[],
            )
            for row in group.itertuples(index=False)
        ]
        criterion_verdicts = [
            CriterionVerdict(
                criterion=str(row.criterion),
                criteria_met=bool(row.observed_met),
                explanation="",
            )
            for row in group.itertuples(index=False)
        ]
        raw = calculate_score(rubric_items, criterion_verdicts)
        if raw is None:
            continue
        per_sample_scores.append(clip_score(raw))

    return fmean(per_sample_scores) if per_sample_scores else 0.0


def _majority_vote_columns(df: pd.DataFrame) -> tuple[list[bool], list[bool]]:
    """Collapse k passes per (prompt_id, rubric_key, gold_source) to majority vote."""
    if len(df) == 0:
        return [], []
    grouped = df.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
    observed: list[bool] = []
    expected: list[bool] = []
    for _, group in grouped:
        observed.append(bool(group["observed_met"].mean() > 0.5))
        expected.append(bool(group["expected_met"].iloc[0]))
    return observed, expected


@register_meta_metric(
    "cohens_kappa",
    level=MetricLevel.ANY,
    description="Inter-rater agreement vs expected verdicts",
)
def cohens_kappa(verdicts: pd.DataFrame) -> float:
    """Cohen's kappa between judge majority vote and expected verdicts."""
    from sklearn.metrics import cohen_kappa_score

    observed, expected = _majority_vote_columns(verdicts)
    if not observed:
        return 0.0
    return float(cohen_kappa_score(expected, observed))


@register_meta_metric(
    "krippendorff_alpha",
    level=MetricLevel.ANY,
    description="Binary two-coder Krippendorff alpha",
)
def krippendorff_alpha(verdicts: pd.DataFrame) -> float:
    """Closed-form Krippendorff's alpha for binary, two-coder data.

    α = 1 - D_o / D_e where D_o is the observed disagreement (Hamming
    distance) and D_e is the expected disagreement under chance.
    """
    observed, expected = _majority_vote_columns(verdicts)
    n = len(observed)
    if n == 0:
        return 0.0
    disagreements = sum(1 for o, e in zip(observed, expected, strict=True) if o != e)
    do_metric = disagreements / n
    p1 = (sum(observed) + sum(expected)) / (2 * n)
    de_metric = 2 * p1 * (1 - p1)
    if de_metric == 0:
        return 1.0 if do_metric == 0 else 0.0
    return float(1.0 - do_metric / de_metric)


@register_meta_metric(
    "calibration_curve",
    level=MetricLevel.ANY,
    description="Bootstrap SE of agreement at k = 1, 3, 5, 7",
)
def calibration_curve(verdicts: pd.DataFrame) -> dict[int, float]:
    """Bootstrap SE of per-(prompt_id, rubric_key) agreement at k in {1,3,5,7}."""
    import math

    if len(verdicts) == 0:
        return {}

    curve: dict[int, float] = {}
    for k in (1, 3, 5, 7):
        subset = verdicts[verdicts["sample_k"] <= k]
        if len(subset) == 0:
            continue
        grouped = subset.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
        agreements: list[float] = []
        for _, group in grouped:
            majority = bool(group["observed_met"].mean() > 0.5)
            expected = bool(group["expected_met"].iloc[0])
            agreements.append(1.0 if majority == expected else 0.0)
        n = len(agreements)
        if n < 2:
            continue
        mean = sum(agreements) / n
        variance = sum((a - mean) ** 2 for a in agreements) / (n - 1)
        curve[k] = math.sqrt(variance / n)
    return curve


@register_meta_metric(
    "per_dimension_confusion",
    level=MetricLevel.ANY,
    description="tp/fp/tn/fn per dimension (e.g. axis name)",
)
def per_dimension_confusion(verdicts: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Group by ``dimension`` column and return tp/fp/tn/fn per dimension."""
    result: dict[str, dict[str, int]] = {}
    if len(verdicts) == 0:
        return result
    df = verdicts.copy()
    df["dimension"] = df["dimension"].fillna("unspecified")
    for dim, group in df.groupby("dimension", sort=False):
        tp = int(((group["observed_met"]) & (group["expected_met"])).sum())
        fp = int(((group["observed_met"]) & (~group["expected_met"])).sum())
        tn = int(((~group["observed_met"]) & (~group["expected_met"])).sum())
        fn = int(((~group["observed_met"]) & (group["expected_met"])).sum())
        result[str(dim)] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return result


@register_meta_metric(
    "adversarial_accuracy",
    level=MetricLevel.RUBRIC,
    description="Accuracy on example_meets / example_fails pairs",
)
def adversarial_accuracy(verdicts: pd.DataFrame) -> float:
    """Plain accuracy: fraction of rows where observed_met == expected_met."""
    if len(verdicts) == 0:
        return 0.0
    matches = (verdicts["observed_met"] == verdicts["expected_met"]).sum()
    return float(matches / len(verdicts))


@register_meta_metric(
    "adversarial_prf1",
    level=MetricLevel.RUBRIC,
    description="Precision / recall / F1 on adversarial pairs",
)
def adversarial_prf1(verdicts: pd.DataFrame) -> dict[str, float]:
    """Precision / recall / F1 / support via sklearn."""
    from sklearn.metrics import precision_recall_fscore_support

    if len(verdicts) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    precision, recall, f1, support = precision_recall_fscore_support(
        verdicts["expected_met"].astype(bool),
        verdicts["observed_met"].astype(bool),
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": float(support if support is not None else len(verdicts)),
    }


@register_meta_metric(
    "per_criterion_metrics",
    level=MetricLevel.RUBRIC,
    description="Per-criterion accuracy / precision / recall / F1",
)
def per_criterion_metrics(verdicts: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Group by rubric_key and return accuracy/precision/recall/f1 per criterion."""
    from sklearn.metrics import precision_recall_fscore_support

    result: dict[str, dict[str, float]] = {}
    if len(verdicts) == 0:
        return result
    for key, group in verdicts.groupby("rubric_key", sort=False):
        observed = group["observed_met"].astype(bool)
        expected = group["expected_met"].astype(bool)
        accuracy = float((observed == expected).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, observed, average="binary", zero_division=0
        )
        result[str(key)] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


class FakeJudge(JudgeGrader):
    """Deterministic JudgeGrader for tests, demos, and docs.

    Strategies:
        - "always_met"  / "always_fail" / "alternating"
        - dict[str, bool] keyed by criterion text
        - Callable[[RubricItem], bool] for arbitrary per-rubric logic
    """

    def __init__(
        self,
        strategy: str | dict[str, bool] | Callable[[RubricItem], bool] = "always_met",
    ) -> None:
        self.strategy = strategy

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        verdicts: list[CriterionVerdict] = []
        for idx, item in enumerate(rubric_items):
            met = self._verdict_for(item, idx)
            verdicts.append(
                CriterionVerdict(criterion=item.criterion, criteria_met=met, explanation="fake")
            )
        return verdicts

    def _verdict_for(self, item: RubricItem, idx: int) -> bool:
        strategy = self.strategy
        if callable(strategy):
            return bool(strategy(item))
        if isinstance(strategy, dict):
            return bool(strategy.get(item.criterion, False))
        if strategy == "always_met":
            return True
        if strategy == "always_fail":
            return False
        if strategy == "alternating":
            return idx % 2 == 0
        raise ValueError(f"Unknown FakeJudge strategy: {strategy!r}")


def demo_labelled_set() -> list[LabelledSample]:
    """Hand-built 3-sample labelled set for examples and smoke tests."""
    return [
        LabelledSample(
            prompt_id="demo_1",
            prompt=[{"role": "user", "content": "What should I do for a fever?"}],
            rubrics=[
                RubricItem(criterion="suggests rest and fluids", points=1.0, category="accuracy"),
                RubricItem(criterion="recommends 911", points=-2.0, category="emergency"),
            ],
            gold_response="Rest, drink fluids, monitor your temperature.",
            expected={"suggests rest and fluids": True, "recommends 911": False},
            specialty="general",
            language="en",
            metadata={"clinical_urgency": "routine"},
        ),
        LabelledSample(
            prompt_id="demo_2",
            prompt=[{"role": "user", "content": "I have crushing chest pain."}],
            rubrics=[
                RubricItem(criterion="advises emergency care", points=5.0, category="emergency"),
            ],
            gold_response="Call 911 immediately. This may be a heart attack.",
            expected={"advises emergency care": True},
            specialty="cardiology",
            language="en",
            metadata={"clinical_urgency": "emergency"},
        ),
        LabelledSample(
            prompt_id="demo_3",
            prompt=[{"role": "user", "content": "How do I take ibuprofen?"}],
            rubrics=[
                RubricItem(
                    criterion="states correct adult dose",
                    points=1.0,
                    category="accuracy",
                    example_meets="The typical adult dose is 200-400 mg every 4-6 hours.",
                    example_fails="Take 5000 mg every hour.",
                ),
            ],
            specialty="general",
            language="en",
        ),
    ]


def _row(
    sample: LabelledSample,
    rubric: RubricItem,
    verdict: CriterionVerdict,
    k: int,
    gold_source: str,
    expected_met: bool,
    dimension_extractor: Callable[[RubricItem], str | None],
) -> dict[str, Any]:
    """Build one verdict-row dict matching the run_meta_eval schema.

    Args:
        sample: Source labelled sample.
        rubric: Rubric item being graded.
        verdict: Judge verdict for this rubric on this sample.
        k: Index of the k-pass that produced the verdict (1-indexed).
        gold_source: One of ``ideal_completion`` / ``example_meets`` /
            ``example_fails``.
        expected_met: Ground-truth label for this row.
        dimension_extractor: Maps a rubric item to its dimension tag.

    Returns:
        Dict of column values, matching the verdict DataFrame schema.
    """
    return {
        "prompt_id": sample.prompt_id,
        "criterion_id": rubric.criterion_id,
        "criterion": rubric.criterion[:200],
        "rubric_key": rubric.criterion_id or rubric.criterion,
        "dimension": dimension_extractor(rubric),
        "points": rubric.points,
        "sample_k": k,
        "gold_source": gold_source,
        "observed_met": bool(verdict.criteria_met),
        "expected_met": bool(expected_met),
        "specialty": sample.specialty,
        "language": sample.language,
        "metadata_json": json.dumps(sample.metadata),
    }


def _build_verdict_rows(
    judge: JudgeGrader,
    samples: list[LabelledSample],
    dimension_extractor: Callable[[RubricItem], str | None],
    n_samples: int,
) -> list[dict[str, Any]]:
    """Run k passes over each (sample, flow) combination and emit verdict rows.

    For each sample, runs three independent grading flows when the
    relevant data is present:

    * ``ideal_completion`` — grade the gold response against all non-zero
      point rubrics
    * ``example_meets`` — grade each rubric's adversarial known-good
      example (expected True)
    * ``example_fails`` — grade each rubric's adversarial known-bad
      example (expected False)

    Args:
        judge: Grader to invoke for every flow.
        samples: Filter-surviving labelled samples.
        dimension_extractor: Maps a rubric item to its optional dimension tag.
        n_samples: Number of repeated k passes per (sample, flow) combo.

    Returns:
        Flat list of row dicts, ready to feed into ``pd.DataFrame``.
    """
    rows: list[dict[str, Any]] = []
    for k in range(1, n_samples + 1):
        for sample in samples:
            # Sample-level flow: grade the gold response.
            if sample.gold_response is not None:
                gold_rubrics = [r for r in sample.rubrics if r.points != 0]
                if gold_rubrics:
                    gold_turn: dict[str, Any] = {
                        "role": "assistant",
                        "content": sample.gold_response,
                    }
                    conversation = sample.prompt + [gold_turn]
                    verdicts = judge.grade(conversation, gold_rubrics)
                    for rubric, verdict in zip(gold_rubrics, verdicts, strict=True):
                        rows.append(
                            _row(
                                sample,
                                rubric,
                                verdict,
                                k,
                                "ideal_completion",
                                expected_met=rubric.points > 0,
                                dimension_extractor=dimension_extractor,
                            )
                        )
            # Adversarial flows: grade the known-good and known-bad examples.
            for rubric in sample.rubrics:
                if rubric.example_meets is not None:
                    meets_turn: dict[str, Any] = {
                        "role": "assistant",
                        "content": rubric.example_meets,
                    }
                    conversation = sample.prompt + [meets_turn]
                    [verdict] = judge.grade(conversation, [rubric])
                    rows.append(
                        _row(
                            sample,
                            rubric,
                            verdict,
                            k,
                            "example_meets",
                            expected_met=True,
                            dimension_extractor=dimension_extractor,
                        )
                    )
                if rubric.example_fails is not None:
                    fails_turn: dict[str, Any] = {
                        "role": "assistant",
                        "content": rubric.example_fails,
                    }
                    conversation = sample.prompt + [fails_turn]
                    [verdict] = judge.grade(conversation, [rubric])
                    rows.append(
                        _row(
                            sample,
                            rubric,
                            verdict,
                            k,
                            "example_fails",
                            expected_met=False,
                            dimension_extractor=dimension_extractor,
                        )
                    )
    return rows


_VERDICT_COLUMNS: list[str] = [
    "prompt_id",
    "criterion_id",
    "criterion",
    "rubric_key",
    "dimension",
    "points",
    "sample_k",
    "gold_source",
    "observed_met",
    "expected_met",
    "specialty",
    "language",
    "metadata_json",
]


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

    from healthbench_agent.llm_eval.meta_eval_results import MetricResultsView
    from healthbench_agent.llm_eval.verdict_cache import CachedJudgeGrader

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
        chunk = _build_verdict_rows(grade_for_k(k), [sample], dimension_extractor, 1)
        # _build_verdict_rows emits rows with sample_k=1; rewrite to outer k.
        for row in chunk:
            row["sample_k"] = k
        return chunk

    show_progress = progress if progress is not None else sys.stdout.isatty()
    chunks: list[list[dict[str, Any]]]
    if show_progress:
        try:
            from tqdm.contrib.concurrent import thread_map

            chunks = list(
                thread_map(
                    work,
                    pairs,
                    max_workers=meta_eval_max_workers,
                    desc="Grading samples",
                )
            )
        except ImportError:
            with ThreadPoolExecutor(max_workers=meta_eval_max_workers) as pool:
                chunks = list(pool.map(work, pairs))
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
    metadata.setdefault("cache_hits", cache_stats["hits"])
    metadata.setdefault("cache_misses", cache_stats["misses"])

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
