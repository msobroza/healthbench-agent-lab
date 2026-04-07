# Meta-Evaluation of the LLM-as-Judge Grader

**Date:** 2026-04-07
**Status:** Draft
**Issue:** [msobroza/healthbench-agent-lab#5](https://github.com/msobroza/healthbench-agent-lab/issues/5)
**Scope:** New `meta_eval.py` module inside `src/healthbench_agent/llm_eval/`, a new `LabelledSample` parent class for `HealthBenchSample` in `domain/`, optional SPEC.md schema fields on `RubricItem` and `LabelledSample` (adversarial `example_meets`/`example_fails`, conversation metadata), and a small extension to `prompt_optimization/` so both the grader prompt and agent prompts can be optimized with arbitrary sample-level and rubric-level filters.

## Goal

HealthBench scores are only as trustworthy as the LLM judge that produces them. Today the judge ([`LLMJudgeGrader`](../../../src/healthbench_agent/llm_eval/grader.py)) returns single-shot verdicts at temperature 0.0, and we have no numbers on how well those verdicts agree with humans, with each other across providers, or with themselves across resamples. Without those numbers, any score delta from `prompt_optimization/` could be noise in the grader rather than a real agent improvement.

This spec adds a small, dataset-agnostic meta-evaluation module that grades a fixed labelled set with a configurable judge, computes a registry of pluggable metrics, and persists durable artifacts (raw verdicts + summary scores) for offline comparison and for use as a fitness function when optimising the grader prompt.

## Design Principles

- **Reuse first.** Use the existing `EvalRunner`, `JudgeConfig`, `create_judge`, `stratified_sample`, and `evaluation/stats.py` helpers. Add the smallest possible amount of new code.
- **Domain types live in `domain/`.** Pure data types belong in the domain layer alongside `RubricItem`, `CriterionVerdict`, `HealthBenchSample`. The `meta_eval.py` module imports them but does not own them.
- **One generic input type via inheritance.** `LabelledSample` is the parent class. `HealthBenchSample` inherits from it and adds HealthBench-specific fields. Any function that accepts `LabelledSample` automatically accepts `HealthBenchSample` (Liskov). No `labelled_from_*` builder functions.
- **Optional SPEC.md fields, backwards compatible.** `RubricItem` and `LabelledSample` accept the SPEC.md schema fields (`criterion_id`, `category`, `example_meets`, `example_fails`, `language`, `specialty`, `metadata`) as **optional** with safe defaults. HealthBench loaders ignore them; SPEC.md-format loaders populate them. No existing call site breaks.
- **Filters compose, metrics stay pure.** Sample-level (`Callable[[LabelledSample], bool]`) and rubric-level (`Callable[[RubricItem], bool]`) filters are applied **upstream** of the metric registry, never inside metric functions. Metric functions remain pure DataFrame operations and stay reusable across any combination of filters.
- **Dataset-agnostic core.** `meta_eval.py` knows nothing about HealthBench. It operates on `list[LabelledSample]`. The HealthBench-specific glue (loading the consensus subset, populating the meta-eval fields from `ideal_completions_data`, extracting axis tags) lives in the CLI, not the library module.
- **Pluggable metrics with declared level.** Metrics are functions registered with a `@register_meta_metric` decorator (mirroring `@register_tool`, `@register_callback`, `@register_prompt_optimizer`, `@register_analysis`). Each registration declares a `MetricLevel` (`SAMPLE`, `RUBRIC`, or `ANY`) and a one-line description, which the runner uses to filter rows and the CLI surfaces via `--list-metrics`. Adding a new metric is one decorated function — no changes to the runner, the result type, the CLI, or the artifact schema. The level declaration keeps `gold_source` filtering out of metric bodies and out of user-facing argument flags.
- **Single judge per artifact.** A meta-eval run targets one judge, produces one parquet + one JSON. Cross-judge comparison is done offline by joining two parquets on `(prompt_id, criterion)`. The runner never mixes judges.
- **Raw verdicts persisted.** Every (sample, criterion, k-pass) verdict is written to parquet so any future metric is a pure function over the file — no need to re-call the judge.

## Non-Goals

- Anthropic / Claude judge sampler (deferred — meta-eval is provider-agnostic via `JudgeConfig`).
- Hand-labelled gold set in this issue (the field exists on `LabelledSample` so a future loader can populate it without any further refactor).
- Auto-feedback from meta-eval into agent prompt-optimization runs (the "noise floor hook" — separate follow-up issue).
- Batch-API mode for the judge during meta-eval (existing async mode is sufficient at this scale).
- Optimising the rubric items themselves.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Ground-truth source | Two flows: (1) sample-level via `gold_response` populated from physician ideal completion or hand labels; (2) rubric-level via `example_meets` / `example_fails` adversarial pairs from the SPEC.md schema | Adversarial gold (one positive + one negative example per criterion) yields precision/recall/F1 directly without needing a full ideal response, and works on datasets that don't ship physician completions |
| Filters | Two optional callables: `sample_filter: LabelledSample -> bool` and `rubric_filter: RubricItem -> bool`. Helper builders: `axis_filter`, `metadata_filter`, `specialty_filter` | Lets prompt-opt drive fitness on a specific axis × specialty × metadata slice without changing metric code |
| Empty post-filter set | Hard error with a message naming the active filters | Catches typos in CLI flags early; silent per-sample drops are still allowed |
| Judges per run | One | Cleaner artifacts, reproducible, cross-judge comparison done offline |
| Calibration sampling | Override `temperature` → 1.0 for the meta-eval grader; sweep `n_samples ∈ {1..k}` | Production grader stays deterministic at 0.0; meta-eval temperature is explicit and visible in MLflow |
| Artifact format | Raw `verdicts.parquet` + summary `metrics.json` | New metrics become reanalyses, not re-runs |
| Sampling strategy | Stratified by **theme** via existing `stratified_sample(..., tag_prefix="theme")` | `axis:*` tags live on rubric items, not on samples; the existing helper only supports sample-level tags. Theme stratification gives even coverage across the 7 HealthBench themes; per-axis confusion still works because every rubric on every sampled conversation is graded |
| Default sample size | 100 | Cheap enough for routine "did the grader prompt drift?" checks |
| Metric API | `@register_meta_metric(name, *, level, description)` registry | Consistent with the rest of the project; open/closed; level declared at registration so the runner filters once per run and the CLI can list metrics with their evaluation level |
| Domain placement | `LabelledSample` and `MetricResults` live in `domain/`; `HealthBenchSample` inherits from `LabelledSample` | Pure data types belong in the domain layer; inheritance gives Liskov-clean dataset-agnosticism |
| Prompt-opt integration | New `JudgeAgreementMetric` + `--target {agent, judge}` flag on existing `optimize-prompt` CLI | Reuses all three optimizer adapters unchanged |
| Module placement | `src/healthbench_agent/llm_eval/meta_eval.py` (single file) | Co-located with the judge it evaluates; no new package directory |

## Module Structure

```
src/healthbench_agent/domain/
    meta_evaluation.py      # NEW — LabelledSample, MetricResults
    rubric.py               # EDIT — RubricItem gains optional SPEC.md fields
    dataset.py              # EDIT — HealthBenchSample now inherits from LabelledSample

src/healthbench_agent/llm_eval/
    meta_eval.py            # NEW — registry + 8 built-in metrics + run_meta_eval + composite_fitness + filter helpers + EmptyFilterError
    cli_meta_eval.py        # NEW — argparse → run_meta_eval; HealthBench glue lives here

src/healthbench_agent/prompt_optimization/
    metric.py               # EDIT — add JudgeAgreementMetric, extend EndToEndMetric with sample_filter/rubric_filter, re-export EmptyFilterError
    cli.py                  # EDIT — add --target {agent, judge}, --rubric-axis, --metadata flags

tests/llm_eval/
    test_meta_eval.py       # NEW — pure-metric ZOMBIES tests + runner with fake judge

tests/domain/
    test_meta_evaluation.py # NEW — LabelledSample/HealthBenchSample inheritance tests

notebooks/
    04_judge_meta_evaluation.ipynb   # NEW — load parquet, plot, inter-judge join

pyproject.toml              # EDIT — register meta-evaluate-judge console script
CLAUDE.md                   # EDIT — add meta_evaluation.py + meta_eval.py to project layout block
```

## Dependency Graph

```
domain/meta_evaluation.py
    -> domain/conversation, domain/rubric    (sibling files only)
    -> NO other modules

domain/dataset.py  (modified)
    -> domain/meta_evaluation.py    (inherits LabelledSample)
    -> domain/conversation, domain/rubric    (existing)

llm_eval/meta_eval.py
    -> domain/      (LabelledSample, MetricResults, RubricItem, JudgeGrader)
    -> llm_eval/    (sibling files: JudgeConfig, create_judge, LLMJudgeGrader)

llm_eval/cli_meta_eval.py
    -> llm_eval/meta_eval.py
    -> domain/, dataset/      (HealthBench loading)
    -> evaluation/, mlflow    (logging)

prompt_optimization/metric.py
    -> llm_eval/meta_eval.py     (JudgeAgreementMetric only)
```

No circular edges. `prompt_optimization → llm_eval` already exists for `JudgeConfig`. The new edge `domain/dataset.py → domain/meta_evaluation.py` is sibling-to-sibling within `domain/`.

## Core Abstractions

### `LabelledSample` — the parent class

```python
# src/healthbench_agent/domain/meta_evaluation.py

@dataclass
class LabelledSample:
    """A rubric-graded sample with optional gold labels for meta-evaluation.

    Acts as the dataset-agnostic shape for any rubric grading task. Concrete
    benchmarks subclass this and add their own fields. Meta-evaluation
    operates on lists of LabelledSample (or any subclass) without knowing
    which dataset they came from.

    The gold-label fields (`gold_response` and `expected`) default to
    "unlabelled" — they are populated only when the sample is being used
    for meta-evaluation with a sample-level gold response. The SPEC.md
    metadata fields are also optional so that HealthBench loaders work
    unchanged.

    Attributes:
        prompt_id: Unique identifier for joining samples across runs.
        prompt: Conversation history before the response to be graded.
            For multi-turn datasets this may already contain assistant turns;
            the response that gets graded is appended at evaluation time.
        rubrics: Rubric items the judge will score the response against.
        gold_response: Known-good response text to grade for meta-evaluation.
            None for unlabelled samples (the typical agent-eval case).
        expected: Expected verdict per rubric criterion text. True = the
            criterion should be met by `gold_response`, False = should not.
            Empty for unlabelled samples.
        language: ISO language code from SPEC.md schema. Optional.
        specialty: Medical specialty from SPEC.md schema. Optional.
        user_persona: 'patient' | 'healthcare professional' from SPEC.md.
            Optional.
        metadata: Free-form per-sample metadata dict (clinical_urgency,
            health_literacy_level, cultural_context, etc. — any keys from
            the SPEC.md `metadata` block, or any extension a future dataset
            adds). Used by `metadata_filter`.
    """
    prompt_id: str
    prompt: MessageList
    rubrics: list[RubricItem]
    gold_response: str | None = None
    expected: dict[str, bool] = field(default_factory=dict)
    language: str | None = None
    specialty: str | None = None
    user_persona: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `RubricItem` — gains optional SPEC.md fields

```python
# src/healthbench_agent/domain/rubric.py  (edited)

@dataclass
class RubricItem:
    """One graded criterion within a rubric.

    The original HealthBench fields (`criterion`, `points`, `tags`) are
    unchanged and still required. The SPEC.md schema fields below are
    optional with safe defaults so HealthBench loaders work unchanged.

    Attributes:
        criterion: Human-readable statement of what the criterion checks.
        points: Points awarded (positive) or deducted (negative) when met.
        tags: HealthBench-style tag list (e.g. ['axis:accuracy', 'level:cluster']).
        criterion_id: Stable id from the SPEC.md schema. None for HealthBench.
        category: Explicit category/axis name from SPEC.md (parallel to the
            `axis:*` tag convention used by HealthBench). The default
            `axis_filter` helper checks both this field and the `tags` list.
        example_meets: Adversarial known-good response text. When present,
            meta-eval grades it and expects criteria_met=True.
        example_fails: Adversarial known-bad response text. When present,
            meta-eval grades it and expects criteria_met=False.
    """
    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)
    criterion_id: str | None = None
    category: str | None = None
    example_meets: str | None = None
    example_fails: str | None = None
```

`from_dict` is extended to read the new fields when present and ignore them when absent.

### `HealthBenchSample` — now inherits from `LabelledSample`

```python
# src/healthbench_agent/domain/dataset.py  (edited)

@dataclass
class HealthBenchSample(LabelledSample):
    """One sample loaded from a HealthBench JSONL dataset file.

    Inherits prompt_id, prompt, rubrics, gold_response, and expected from
    LabelledSample. Adds HealthBench-specific fields.

    Attributes:
        example_tags: Dataset-level tags for stratified scoring (themes,
            physician categories).
        ideal_completions_data: Physician ideal completion data when available.
            Used to populate gold_response/expected at meta-eval time.
        canary: Dataset integrity signature.
    """
    example_tags: list[str] = field(default_factory=list)
    ideal_completions_data: dict[str, Any] | None = None
    canary: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthBenchSample:
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            rubrics=[RubricItem.from_dict(r) for r in data["rubrics"]],
            example_tags=data["example_tags"],
            ideal_completions_data=data.get("ideal_completions_data"),
            canary=data.get("canary"),
        )
```

**Migration note:** All existing call sites that construct `HealthBenchSample(...)` positionally with `example_tags` as the 4th argument must be migrated to keyword arguments. The `from_dict` classmethod already uses keywords, so JSONL loading is unaffected. Tests and fixtures that build samples directly are the only places to audit.

### `MetricResults` — also lives in `domain/`

```python
# src/healthbench_agent/domain/meta_evaluation.py  (same file as LabelledSample)

@dataclass(frozen=True)
class MetricResults:
    """Aggregate meta-evaluation result for one judge run.

    Round-trips to JSON via dataclasses.asdict + json.dump.

    Attributes:
        scores: Mapping of metric name to its computed value. Value type
            depends on the metric (float for kappa, dict for confusion).
        n_samples_graded: Number of LabelledSamples that produced verdicts.
        n_rubrics_graded: Total (sample, rubric) pairs across all k passes.
        judge_metadata: Run-level header — judge_model, temperature,
            judge_prompt_sha, n_samples (k), seed.
    """
    scores: dict[str, Any]
    n_samples_graded: int
    n_rubrics_graded: int
    judge_metadata: dict[str, Any]
```

### Metric registry

Each metric declares **at registration time** which evaluation level it consumes. The runner uses this declaration to (a) filter the verdict DataFrame to the correct row subset before calling the metric, (b) skip metrics whose level is absent from the actual data with a single info-log line, and (c) expose the level + description in `meta-evaluate-judge --list-metrics` so users see what each metric does without reading source.

```python
# src/healthbench_agent/llm_eval/meta_eval.py

from enum import Enum

class MetricLevel(str, Enum):
    """Which row subset a metric operates on.

    SAMPLE — sample-level rows only (gold_source == "ideal_completion").
        Metric is meaningful only when the dataset ships a gold_response.
    RUBRIC — rubric-level adversarial rows only
        (gold_source ∈ {"example_meets", "example_fails"}).
        Metric is meaningful only when rubrics carry example_meets/example_fails.
    ANY    — metric is well-defined on either subset and on the union;
        the runner passes whatever rows the dataset produces.
    """
    SAMPLE = "sample"
    RUBRIC = "rubric"
    ANY = "any"


Metric = Callable[[pd.DataFrame], Any]

@dataclass(frozen=True)
class MetricSpec:
    """Registered metric metadata.

    Attributes:
        name: Unique identifier (used by --metrics CLI flag and metrics.json).
        fn: The pure metric function. Takes a (level-filtered) DataFrame,
            returns a JSON-serialisable score.
        level: Which gold_source rows the runner passes to ``fn``.
        description: One-line human-readable summary shown by --list-metrics.
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
    """Decorator that registers a meta-evaluation metric by name + level.

    Both ``level`` and ``description`` are required. The level tells the
    runner which row subset to feed the metric (it filters before calling),
    so individual metric functions never write `gold_source` filters by hand.
    The description is surfaced by ``meta-evaluate-judge --list-metrics``.
    """

def get_meta_metric(name: str) -> MetricSpec: ...
def registered_meta_metrics() -> dict[str, MetricSpec]: ...
```

Adding a new metric (e.g. Fleiss' κ, ROC-AUC, judge-vs-judge agreement) is one decorated function — name, level, description, and the pure DataFrame transform. Zero changes anywhere else.

### Built-in metrics

Every built-in metric is registered with its `level` so the user can see at a glance which gold flow it belongs to.

| Name | Level | Returns | Semantics |
|---|---|---|---|
| `gold_score` | `SAMPLE` | `float` | Mean HealthBench score the judge gives to the gold response. For each (prompt_id, sample_k), reproduces the production `calculate_score` formula on the verdicts: `sum(points where observed_met) / sum(max(0, points))`, clipped to [0, 1] via `clip_score`. Averages across (prompt_id, sample_k) pairs. **A perfectly calibrated judge returns 1.0** because the physician ideal response should meet every positive rubric and avoid every penalty rubric. Mirrors production scoring exactly — same formula, same clipping, same aggregation as `aggregate_scores`. |
| `cohens_kappa` | `ANY` | `float` | Collapse k passes to majority vote per (prompt_id, rubric_key, gold_source), compute Cohen's κ vs `expected_met`. Wraps `sklearn.metrics.cohen_kappa_score`. |
| `krippendorff_alpha` | `ANY` | `float` | Same input, closed-form binary two-coder α. ~15 lines, no new dependency. |
| `calibration_curve` | `ANY` | `dict[int, float]` | For each k in {1, 3, 5, 7}, take the first k passes, collapse to majority, return bootstrap SE of the per-(prompt_id, rubric_key) agreement rate. |
| `per_dimension_confusion` | `ANY` | `dict[str, dict[str, int]]` | Group by `dimension` column → `{tp, fp, tn, fn}` per dimension. Rows where dimension is None aggregated under `"unspecified"`. |
| `adversarial_accuracy` | `RUBRIC` | `float` | Plain accuracy: fraction of rows where `observed_met == expected_met`. |
| `adversarial_prf1` | `RUBRIC` | `dict[str, float]` | Precision/recall/F1/support computed by `sklearn.metrics.precision_recall_fscore_support` with `expected_met` as ground truth. Returned as `{"precision": ..., "recall": ..., "f1": ..., "support": ...}`. |
| `per_criterion_metrics` | `RUBRIC` | `dict[str, dict[str, float]]` | Group by `rubric_key`, return `{"accuracy", "precision", "recall", "f1"}` per criterion. |

`rubric_key` is a derived DataFrame column populated as `criterion_id or criterion` so the same metric implementation works on both HealthBench rows (no `criterion_id`) and SPEC.md rows (stable id). It is added in step 4 of `run_meta_eval` and is not persisted to parquet — readers can rebuild it.

`gold_score` is the **default** built-in metric and the **default fitness** for `JudgeAgreementMetric` when sample-level gold is available. When a dataset only ships adversarial pairs (no `gold_response`), the default automatically falls back to `adversarial_prf1["f1"]`. Both metrics target 1.0 for a perfectly calibrated judge. The κ/α/calibration metrics remain available for cases where the user wants per-rubric agreement statistics rather than per-sample score calibration.

#### How the runner uses `level`

1. After building the verdict DataFrame, `run_meta_eval` partitions it once into `sample_rows` (where `gold_source == "ideal_completion"`), `rubric_rows` (where `gold_source ∈ {"example_meets", "example_fails"}`), and `all_rows` (the full DataFrame).
2. For each requested metric, it looks up the `MetricSpec.level` and passes the matching subset to the metric function:
   - `SAMPLE` → `sample_rows`
   - `RUBRIC` → `rubric_rows`
   - `ANY` → `all_rows`
3. If the chosen subset is empty (e.g. user requested `gold_score` on an adversarial-only dataset), the runner logs `INFO: skipping metric 'gold_score' (level=SAMPLE) — no sample-level rows in this run` and omits the metric from `MetricResults.scores`. The run does **not** fail — other applicable metrics still produce numbers.
4. If **every** requested metric gets skipped this way, `run_meta_eval` raises `EmptyFilterError` because the run produced nothing useful — silent empty results would be worse than a clear error.

This pushes all `gold_source` filtering into one place (the runner) and out of every individual metric function, keeping metrics as pure DataFrame transforms.

#### Default metric selection

When the user does not pass `--metrics`, the CLI chooses defaults based on what the dataset actually contains:

| Dataset has | Default metrics |
|---|---|
| Sample-level gold only | `gold_score`, `cohens_kappa`, `calibration_curve`, `per_dimension_confusion` |
| Adversarial pairs only | `adversarial_prf1`, `adversarial_accuracy`, `per_criterion_metrics`, `cohens_kappa`, `per_dimension_confusion` |
| Both | union of the above two rows |

`meta-evaluate-judge --list-metrics` prints `name | level | description` for every registered metric so users discover this without reading the spec. Example output:

```
$ uv run meta-evaluate-judge --list-metrics
NAME                       LEVEL    DESCRIPTION
gold_score                 SAMPLE   Mean clipped HealthBench score on gold responses (target = 1.0)
cohens_kappa               ANY      Inter-rater agreement vs expected verdicts
krippendorff_alpha         ANY      Binary two-coder Krippendorff alpha
calibration_curve          ANY      Bootstrap SE of agreement at k = 1, 3, 5, 7
per_dimension_confusion    ANY      tp/fp/tn/fn per dimension (e.g. axis name)
adversarial_accuracy       RUBRIC   Accuracy on example_meets / example_fails pairs
adversarial_prf1           RUBRIC   Precision / recall / F1 on adversarial pairs
per_criterion_metrics      RUBRIC   Per-criterion accuracy / precision / recall / F1
```

#### Two gold-evaluation flows

`run_meta_eval` runs two complementary flows over the same labelled set, emitting one DataFrame:

| Flow | Triggered when | What gets graded | `gold_source` value | `expected_met` |
|---|---|---|---|---|
| Sample-level | `sample.gold_response is not None` | Each rubric on the sample with `points != 0`, against `(prompt + gold_response)` | `"ideal_completion"` | True for `points > 0`, False for `points < 0`. Zero-point rubrics are skipped. |
| Adversarial — meets | `rubric.example_meets is not None` | Just that one rubric, against `(prompt + example_meets)` | `"example_meets"` | always True |
| Adversarial — fails | `rubric.example_fails is not None` | Just that one rubric, against `(prompt + example_fails)` | `"example_fails"` | always False |

A sample with both a `gold_response` and rubrics carrying `example_meets`/`example_fails` produces rows for all three flows, joinable by `(prompt_id, rubric_key)`. A sample with only adversarial pairs produces only flow rows 2-3 — and the metric set automatically restricts to those rows.

The conversation passed to the judge for adversarial flows is `sample.prompt + [{"role": "assistant", "content": example_text}]` — the original user prompt is preserved so context-dependent criteria stay evaluable.

#### `gold_score` reference implementation

```python
@register_meta_metric(
    "gold_score",
    level=MetricLevel.SAMPLE,
    description="Mean clipped HealthBench score on gold responses (target = 1.0)",
)
def gold_score(verdicts: pd.DataFrame) -> float:
    """Mean clipped HealthBench score the judge gives to gold responses.

    The runner has already filtered ``verdicts`` to sample-level rows
    (gold_source == "ideal_completion") because of level=SAMPLE, so this
    function does not need to filter by gold_source itself.

    Reproduces calculate_score + clip_score + aggregate_scores from
    domain/scoring.py over the (already filtered) DataFrame, treating
    each (prompt_id, sample_k) as one conversation. A perfectly
    calibrated judge returns 1.0.
    """
    def per_group(g: pd.DataFrame) -> float:
        total_possible = g.loc[g["points"] > 0, "points"].sum()
        if total_possible == 0:
            return float("nan")
        achieved = g.loc[g["observed_met"], "points"].sum()
        return achieved / total_possible

    raw = verdicts.groupby(["prompt_id", "sample_k"]).apply(per_group)
    clipped = raw.dropna().clip(0.0, 1.0)
    return float(clipped.mean()) if len(clipped) > 0 else 0.0
```

### `run_meta_eval` — the only function with I/O

```python
# src/healthbench_agent/llm_eval/meta_eval.py

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
) -> MetricResults:
    """Grade labelled samples k times with one judge, compute metrics, optionally persist.

    Steps:
        1. Apply ``sample_filter`` to drop samples that should not contribute.
           Log per-sample drops; raise EmptyFilterError if zero remain.
        2. For each surviving sample, apply ``rubric_filter`` to drop rubrics
           that should not contribute. Drop the sample if no rubrics survive.
           Raise EmptyFilterError if no (sample, rubric) pair remains.
        3. Loop k = 1..n_samples. For each k, fan out grading via a
           ThreadPoolExecutor (same pattern as EvalRunner.run_async). For each
           surviving sample:
             a. Sample-level flow — if sample.gold_response is not None, grade
                each surviving rubric with points != 0 against
                (prompt + gold_response). Emit one row per rubric with
                gold_source="ideal_completion".
             b. Adversarial flow — for each surviving rubric with example_meets
                or example_fails, grade just that one rubric against
                (prompt + example_*). Emit one row with
                gold_source="example_meets" or "example_fails".
        4. Build a single pandas DataFrame with columns:
           prompt_id, criterion_id, criterion, dimension, points, sample_k,
           gold_source, observed_met, expected_met, specialty, language,
           metadata_json. Add a derived in-memory column
           ``rubric_key = criterion_id or criterion`` used by metrics.
           Partition once into ``sample_rows``, ``rubric_rows``, ``all_rows``.
        5. For each requested metric, look up its ``MetricSpec.level`` and
           pass the matching subset (SAMPLE→sample_rows, RUBRIC→rubric_rows,
           ANY→all_rows). Skip with an INFO log if the subset is empty.
           Collect numeric/dict scores into ``MetricResults.scores``. If
           every requested metric was skipped, raise ``EmptyFilterError``.
        6. If output_dir is set, write verdicts.parquet (DataFrame) and
           metrics.json (dataclasses.asdict(result)).
        7. Return the MetricResults instance.

    Args:
        judge: Any JudgeGrader implementation. The CLI builds an LLMJudgeGrader
            with temperature overridden to >0 for k>1 to be meaningful.
        labelled: Dataset-agnostic input. Caller is responsible for sampling
            and stratification.
        dimension_extractor: Required. Function mapping a RubricItem to a
            dimension label (e.g. axis name). The CLI passes its own — no
            HealthBench-specific default lives in this module.
        metric_names: Names of registered metrics to compute. None = all registered.
        n_samples: How many independent grading passes to run per sample.
        sample_filter: Optional sample-level predicate. None means keep all.
        rubric_filter: Optional rubric-level predicate. None means keep all.
        output_dir: Where to write verdicts.parquet + metrics.json. None = no I/O.
        judge_metadata: Run-level header captured in MetricResults and metrics.json.

    Returns:
        MetricResults with one entry per metric in scores.

    Raises:
        EmptyFilterError: If sample_filter or rubric_filter eliminate all
            samples or all rubrics. Error message names the active filters.
    """
```

The `dimension_extractor` parameter is **required, no default**, so `meta_eval.py` carries no HealthBench knowledge.

### Filter helpers

```python
# src/healthbench_agent/llm_eval/meta_eval.py

class EmptyFilterError(ValueError):
    """Raised when a sample/rubric filter combination eliminates all rows.

    Carries the names (or repr) of the active filters so the CLI can show
    the user exactly which flags caused the empty result.
    """
    def __init__(self, sample_filter: Any, rubric_filter: Any) -> None: ...


def axis_filter(*axes: str) -> Callable[[RubricItem], bool]:
    """Keep rubrics whose `category` field or `axis:*` tag matches any of *axes*.

    Checks both the SPEC.md `category` field and the HealthBench `axis:*` tag
    convention so the same helper works on both schemas.
    """

def metadata_filter(**conditions: Any) -> Callable[[LabelledSample], bool]:
    """Keep samples where every metadata key equals the given value.

    Top-level fields (`language`, `specialty`, `user_persona`) are checked
    against attributes; other keys are looked up in `sample.metadata`.

    Example:
        metadata_filter(clinical_urgency="emergency", language="en")
    """

def specialty_filter(*specialties: str) -> Callable[[LabelledSample], bool]:
    """Keep samples whose `specialty` field is in *specialties*."""
```

Each helper is ~5 lines of code. They are conveniences for CLI use; users can write arbitrary lambdas for anything more complex. `EmptyFilterError` is re-exported from `prompt_optimization/metric.py` so callers of `JudgeAgreementMetric` and `EndToEndMetric` see the same exception type.

### `composite_fitness` — single scalar for prompt-optimization

```python
# src/healthbench_agent/llm_eval/meta_eval.py

def composite_fitness(
    results: MetricResults,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum over numeric scores in MetricResults.

    Default weights anchor on the calibration target (gold_score == 1.0)
    and use κ as a tie-breaker for per-rubric agreement:
        {"gold_score": 0.7, "cohens_kappa": 0.3}
    """
```

The default fitness for `JudgeAgreementMetric` is the bare `gold_score` metric (not `composite`) because (i) it has a clear target value of 1.0, (ii) it directly reflects production scoring, and (iii) it's the simplest signal for the optimizer to push on. Composite remains available for users who want to combine it with κ or other metrics.

## Parquet Schema (`verdicts.parquet`)

| column | type | meaning |
|---|---|---|
| `prompt_id` | str | from `LabelledSample.prompt_id` |
| `criterion_id` | str \| None | from `RubricItem.criterion_id` (None for HealthBench) |
| `criterion` | str | rubric criterion text, truncated to 200 chars on write |
| `dimension` | str \| None | output of `dimension_extractor(rubric)` |
| `points` | float | rubric points |
| `sample_k` | int | which of the n_samples passes (1..n_samples) |
| `gold_source` | str | one of `"ideal_completion"`, `"example_meets"`, `"example_fails"` |
| `observed_met` | bool | judge verdict |
| `expected_met` | bool | True for `points > 0` on sample-level rows, True/False on adversarial rows |
| `specialty` | str \| None | from `LabelledSample.specialty` |
| `language` | str \| None | from `LabelledSample.language` |
| `metadata_json` | str | `json.dumps(LabelledSample.metadata)` for downstream filtering and joins |

Run-level fields (`judge_model`, `temperature`, `judge_prompt_sha`, `n_samples`, `seed`) live in `metrics.json`'s `judge_metadata`, not on every row. `metadata` is serialised as JSON text rather than a struct column to keep the parquet schema flat and tolerant of evolving keys.

## CLI

### `meta-evaluate-judge`

```
uv run meta-evaluate-judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --subset consensus \
    --sample-size 100 \
    --n-samples 7 \
    --temperature 1.0 \
    --metrics gold_score,cohens_kappa,adversarial_prf1,per_dimension_confusion \
    --rubric-axis accuracy \
    --metadata clinical_urgency=emergency \
    --metadata language=en \
    --output-dir runs/meta_eval/2026-04-07_openai/
```

`--rubric-axis` is repeatable and builds an `axis_filter`. `--metadata KEY=VALUE` is repeatable and builds a `metadata_filter`. Both are optional; omitting them keeps every sample and rubric.

`cli_meta_eval.py` flow (~100 lines):

1. Parse args, including repeatable `--rubric-axis` and `--metadata KEY=VALUE` flags.
2. `dataset = load_dataset("consensus")`.
3. `sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)`.
4. **Populate gold-label fields in place** for each `HealthBenchSample` in `sampled.samples` that has `ideal_completions_data`:
   - `sample.gold_response = _extract_ideal_completion_text(sample.ideal_completions_data)`
   - `sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}`
   - Drop samples where extraction fails.
5. Define an inline `axis_extractor`:
   ```python
   def axis_extractor(item: RubricItem) -> str | None:
       for tag in item.tags:
           if tag.startswith("axis:"):
               return tag[len("axis:"):]
       return item.category
   ```
6. Build filter callables:
   - `rubric_filter = axis_filter(*args.rubric_axis) if args.rubric_axis else None`
   - `sample_filter = metadata_filter(**parsed_metadata) if parsed_metadata else None`
7. Build a `JudgeConfig` from the YAML; override `temperature` from CLI.
8. `judge = create_judge(config)`.
9. `results = run_meta_eval(judge, sampled.samples, dimension_extractor=axis_extractor, sample_filter=sample_filter, rubric_filter=rubric_filter, ...)`.
10. Catch `EmptyFilterError` → exit with a clear message naming the active filters.
11. Optional MLflow logging (params + scalar metrics + artifacts, tagged `run_type=meta_eval`). Filter args are logged as MLflow params (`filter_axis`, `filter_metadata`) so runs are reproducible from the MLflow UI alone.
12. Print summary.

Default `--metrics` is empty, meaning "auto-select based on what the dataset contains" using the table in the metric registry section above. `meta-evaluate-judge --list-metrics` prints `name | level | description` for every registered metric and exits — useful for discovering what is available without reading source.

### `optimize-prompt --target judge`

```
uv run optimize-prompt --target judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --optimizer critique_refine \
    --fitness gold_score \
    --sample-size 50 \
    --max-trials 10 \
    --rubric-axis accuracy \
    --metadata clinical_urgency=emergency
```

`--target agent` (default) keeps today's behaviour (`EndToEndMetric`). `--target judge` requires `--judge-config`, loads + populates a labelled set the same way `cli_meta_eval.py` does, builds a `JudgeAgreementMetric`, and hands it to the chosen optimizer adapter unchanged. The three adapters (DSPy, TextGrad, critique-refine) are not modified.

The same `--rubric-axis` and `--metadata` flags accepted by `meta-evaluate-judge` are accepted here. They are forwarded into `JudgeAgreementMetric` (via `sample_filter` / `rubric_filter`) so that the optimizer's fitness signal is restricted to the slice of interest. This is what enables **per-axis / per-specialty / per-metadata judge prompt optimization** without any optimizer changes.

For `--target agent`, `EndToEndMetric` also gains the same two filter parameters so that agent prompts can be optimised on the same slice the user expects to evaluate them on. The CLI passes them to whichever metric `--target` selects, so the user-facing flag set is identical for both targets.

The labelled-set construction is shared between `cli_meta_eval.py` and `cli.py --target judge`. To avoid duplication, the population step (steps 2-5 above) is extracted into a small helper inside `cli_meta_eval.py` (`load_consensus_labelled(sample_size, seed) -> tuple[list[LabelledSample], Callable]`) which `cli.py` imports. Filter parsing (`--rubric-axis`, `--metadata`) is extracted into a second helper (`build_filters(args) -> tuple[sample_filter, rubric_filter]`) shared between both CLIs.

### `JudgeAgreementMetric`

```python
# src/healthbench_agent/prompt_optimization/metric.py  (additions)

class JudgeAgreementMetric:
    """Fitness metric that scores a candidate grader prompt by running
    meta-evaluation against a fixed labelled set.

    Mirrors the EndToEndMetric callable shape so any registered
    PromptOptimizer works without modification. Optional sample/rubric
    filters restrict the fitness signal to a slice (e.g. "accuracy axis,
    emergency cases") so the optimizer specialises the judge prompt for
    that slice.
    """
    def __init__(
        self,
        judge_config: JudgeConfig,
        labelled: list[LabelledSample],
        dimension_extractor: Callable[[RubricItem], str | None],
        n_samples: int = 3,
        fitness: str = "gold_score",
        weights: dict[str, float] | None = None,
        sample_filter: Callable[[LabelledSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None: ...

    def __call__(self, candidate_template: str) -> float:
        """Build a one-off LLMJudgeGrader using the candidate template,
        call run_meta_eval with the configured filters, return the
        fitness scalar.
        """
```

When `fitness == "composite"`, the call returns `composite_fitness(results, weights)`. Otherwise it returns `float(results.scores[fitness])`.

`n_samples` defaults to 3 here (vs 7 in the meta-eval CLI default) because each optimizer trial calls the metric once and the fitness signal needs to be cheap; the meta-eval CLI runs once and can afford 7.

### `EndToEndMetric` extension

`prompt_optimization/metric.py::EndToEndMetric` gains the **same two filter parameters** (`sample_filter` and `rubric_filter`). The agent pipeline still runs against every sample in the input set, but the per-rubric scoring step skips rubrics that fail `rubric_filter`, and the aggregate skips samples that fail `sample_filter`. This is what lets `optimize-prompt --target agent` produce a per-axis or per-metadata-slice fitness signal without changing any optimizer adapter.

Both metrics raise `EmptyFilterError` (re-exported from `meta_eval.py`) if a filter combination eliminates every (sample, rubric) pair, so the user sees a clear failure rather than silently optimising against an empty signal.

## MLflow Logging

`cli_meta_eval.py` (default ON; `--no-mlflow` to disable):
- `mlflow.set_experiment("meta_eval")`
- `mlflow.log_params({"judge_model", "temperature", "n_samples", "sample_size", "seed", "judge_prompt_sha"})`
- For numeric scores: `mlflow.log_metric("gold_score", v)`, `mlflow.log_metric("cohens_kappa", v)`, `mlflow.log_metric("krippendorff_alpha", v)`. Calibration-curve dict flattened as `cal_se_k1`, `cal_se_k3`, etc. Non-numeric scores (e.g. `per_dimension_confusion`) are NOT logged as metrics — they live in `metrics.json` and the parquet artifact.
- `mlflow.log_artifact(verdicts.parquet)`, `mlflow.log_artifact(metrics.json)`
- `mlflow.set_tag("run_type", "meta_eval")` — keeps these out of agent-comparison views.

For `--target judge` optimization runs, the existing prompt-optimization MLflow integration logs each trial as today; no extra wiring.

## Testing Strategy

### `tests/llm_eval/test_meta_eval.py`

ZOMBIES coverage on the pure metric functions using synthetic verdict DataFrames.

| Test | Expected |
|---|---|
| `gold_score` on perfect verdicts (all positive rubrics met, no penalties) | 1.0 |
| `gold_score` clips per-conversation scores to [0, 1] | conversation with negative raw score contributes 0.0 to mean |
| `gold_score` registered with `level=SAMPLE` | runner routes only sample-level rows to it (verified via spy) |
| Full agreement (observed == expected for all rows) | κ = 1.0, α = 1.0 |
| Inverse (observed == not expected) | κ = -1.0 |
| Random / orthogonal labels (50/50) | κ ≈ 0.0 |
| Single class on both sides (all True or all False) | κ = NaN or 0 — document sklearn behaviour and assert |
| Empty DataFrame | each metric returns sensible default or raises with a clear message |
| Calibration curve at k=1 vs k=7 with synthetic noisy verdicts | SE@7 < SE@1 |
| `per_dimension_confusion` with two dimensions | correct tp/fp/tn/fn per dimension |
| `adversarial_accuracy` on a row mix where 3/4 match | 0.75 |
| `adversarial_prf1` returns dict with precision/recall/f1/support | values match sklearn ground truth |
| `per_criterion_metrics` groups by `criterion_id` | per-criterion dicts have all 4 keys |
| Registry: `register_meta_metric(name, level, description)` + `get_meta_metric` round-trip | returns `MetricSpec` with all four fields populated |
| Registering without `level` or `description` | TypeError |
| `run_meta_eval` partitions DataFrame by `gold_source` and routes correct subset to each metric by `level` | metric receives only matching rows (assert via spy) |
| `run_meta_eval` skips a SAMPLE-level metric on adversarial-only data and logs INFO | metric absent from `MetricResults.scores`, log captured |
| `run_meta_eval` raises `EmptyFilterError` when **every** requested metric is skipped due to level mismatch | error message mentions skipped metric names |
| `run_meta_eval` with a fake `JudgeGrader` | produces verdicts.parquet and metrics.json with the requested `scores` keys |
| `run_meta_eval` skips samples where `gold_response is None` | dropped count is correct |
| `run_meta_eval` emits adversarial rows when `example_meets`/`example_fails` set | one row per pair, correct `gold_source` |
| `run_meta_eval` with `sample_filter` that rejects all samples | raises `EmptyFilterError` mentioning the filter |
| `run_meta_eval` with `rubric_filter` that rejects all rubrics | raises `EmptyFilterError` mentioning the filter |
| `run_meta_eval` with `axis_filter("accuracy")` | only accuracy-tagged rubrics graded; row count matches expected |
| `metadata_filter(clinical_urgency="emergency")` keeps only matching samples | filtered count is correct |
| `JudgeAgreementMetric.__call__` with a fake judge | returns expected scalar |
| `JudgeAgreementMetric` honours `sample_filter` + `rubric_filter` | fitness equals manually filtered run |
| `EndToEndMetric` honours `sample_filter` + `rubric_filter` | fitness equals manually filtered run |
| CLI smoke test: argparse → mocked `run_meta_eval` | dispatches with correct params |
| CLI smoke test: `--rubric-axis accuracy --metadata language=en` | filter callables forwarded to `run_meta_eval` |
| CLI smoke test: `--list-metrics` exits 0 and prints every registered metric with name + level + description | stdout contains all 8 built-ins, no judge call made |
| CLI smoke test: `EmptyFilterError` → CLI exits non-zero with clear message | exit code != 0, message mentions both filters |

### `tests/domain/test_meta_evaluation.py`

| Test | Expected |
|---|---|
| `LabelledSample` constructs with required fields, defaults populate | ok |
| `LabelledSample` constructs with full SPEC.md fields (language, specialty, user_persona, metadata) | values round-trip |
| `HealthBenchSample` is a `LabelledSample` (`isinstance` check) | True |
| `HealthBenchSample.from_dict(jsonl_row)` populates inherited + own fields | matches expected |
| `RubricItem.from_dict` reads optional `criterion_id`, `category`, `example_meets`, `example_fails` when present | values populated; defaults remain `None` when absent |
| Existing `HealthBenchSample` keyword construction still works | ok |
| A function annotated `def foo(s: LabelledSample)` accepts a `HealthBenchSample` | mypy passes |
| Setting `gold_response` and `expected` on a loaded `HealthBenchSample` works | mutation succeeds (dataclass not frozen) |

Coverage targets: 100% on pure metric functions, ≥80% module-wide per project policy.

## Notebook — `notebooks/04_judge_meta_evaluation.ipynb`

Five cells:
1. Load `verdicts.parquet` + `metrics.json` for one judge run.
2. Plot the calibration curve (matplotlib).
3. Plot per-dimension confusion as a heatmap (seaborn).
4. Load a second `verdicts.parquet` (different judge), join on `(prompt_id, criterion)`, compute inter-judge κ, show the disagreement table sorted by `points`.
5. Print `composite_fitness` for both judges.

No new analysis-registry entries — meta-eval artifacts live in `runs/meta_eval/`, not the analysis output directory.

## Dependencies

No new third-party packages.

| Need | Reuse |
|---|---|
| Cohen's κ | `sklearn.metrics.cohen_kappa_score` (already a dep) |
| Bootstrap SE | `scipy.stats` + `numpy` (already deps) |
| Krippendorff's α | Inline closed-form for binary, two-coder case (~15 lines) |
| Concurrent grading | `concurrent.futures.ThreadPoolExecutor` (stdlib) |
| Parquet I/O | `pandas.DataFrame.to_parquet` + `pyarrow` (already in deps for analysis) |

## Migration Risk: `HealthBenchSample` Inheritance

Adding `LabelledSample` as a parent of `HealthBenchSample` introduces two extra inherited fields (`gold_response`, `expected`) with defaults. Risks:

1. **Positional construction.** Any code that builds `HealthBenchSample(prompt_id, prompt, rubrics, example_tags)` positionally will break because `example_tags` is no longer the 4th positional argument. Audit and migrate to keyword form. The `from_dict` classmethod is already keyword-based.
2. **Field order in dataclass inheritance.** All inherited fields without defaults must come before fields with defaults. `LabelledSample` has 3 no-default + 2 default; `HealthBenchSample` adds 3 default fields. Order is consistent.
3. **`asdict` / serialization** of `HealthBenchSample` now produces the 5 inherited fields plus the 3 own fields. Any code that round-trips samples through `asdict` and back via `from_dict` is unaffected because `from_dict` ignores unknown keys, but code that compares `asdict(sample)` against a fixed dict will need updating.
4. **Test fixtures.** `tests/conftest.py` and dataset-related test files build sample objects directly. Audit and migrate.

The migration is mechanical (keyword args + a few extra `gold_response=None` defaults that already exist) but must be in the implementation plan as its own step before the meta-eval module is wired.

## Open Questions (resolved during brainstorming)

1. ~~Ground-truth source~~ → two flows: sample-level via physician ideal completion (or future hand labels) populating `gold_response`, plus rubric-level adversarial pairs via `example_meets`/`example_fails` from the SPEC.md schema.
2. ~~How many judges per run~~ → exactly one; cross-judge done offline by joining parquets.
3. ~~Calibration temperature~~ → meta-eval overrides judge to temperature=1.0; production grader stays at 0.0.
4. ~~Artifact format~~ → raw parquet + metrics.json.
5. ~~Sampling strategy~~ → stratified by theme (sample-level), default 100.
6. ~~Prompt-optimization integration~~ → grader prompt is also optimisable via `--target judge`; agent prompt optimization gains the same filter flags.
7. ~~Fitness function~~ → configurable, default `gold_score` (falls back to `adversarial_prf1["f1"]` when no sample-level gold exists), composite available with weights.
8. ~~Module placement~~ → single file inside `llm_eval/`; pure data types in `domain/`.
9. ~~Type relationships~~ → `HealthBenchSample` inherits from `LabelledSample`; no `labelled_from_*` builder functions.
10. ~~Slice-restricted optimization~~ → both `JudgeAgreementMetric` and `EndToEndMetric` accept optional `sample_filter` and `rubric_filter` (Option A: two separate `Callable` parameters), exposed on the CLI as repeatable `--rubric-axis` and `--metadata KEY=VALUE` flags.
11. ~~Empty filter behaviour~~ → `EmptyFilterError` (subclass of `ValueError`) raised by `run_meta_eval`; CLI catches and exits non-zero with the active filter names.
12. ~~SPEC.md schema fields~~ → optional with safe defaults on both `RubricItem` and `LabelledSample`; HealthBench loaders ignore them, future SPEC.md loaders populate them.
13. ~~Metric level discovery~~ → each metric declares `MetricLevel ∈ {SAMPLE, RUBRIC, ANY}` at registration; runner filters rows per level; CLI exposes `meta-evaluate-judge --list-metrics` so users see the level + a one-line description without reading source.

## Out of Scope (Follow-up Issues)

- **Anthropic / Claude judge sampler.** Meta-eval is provider-agnostic via `JudgeConfig`; adding a third sampler is a separate ~60-line PR.
- **Hand-labelled gold set loader.** The `LabelledSample.gold_response` and `LabelledSample.expected` fields already exist for this; the future loader just needs to populate them from a JSONL.
- **Noise-floor hook into `prompt_optimization/`.** Auto-warn when an agent-optimization trial's score delta is below the meta-eval-measured judge variance.
- **Batch-API mode** for the judge during meta-eval. Async ThreadPool is sufficient at the projected sample sizes.
- **Optimising the rubric items themselves.**
