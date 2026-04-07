# Meta-Evaluation of the LLM-as-Judge Grader

**Date:** 2026-04-07
**Status:** Draft
**Issue:** [msobroza/healthbench-agent-lab#5](https://github.com/msobroza/healthbench-agent-lab/issues/5)
**Scope:** New `meta_eval.py` module inside `src/healthbench_agent/llm_eval/`, a new `LabelledSample` parent class for `HealthBenchSample` in `domain/`, plus a small extension to `prompt_optimization/` so the grader prompt itself can be optimized.

## Goal

HealthBench scores are only as trustworthy as the LLM judge that produces them. Today the judge ([`LLMJudgeGrader`](../../../src/healthbench_agent/llm_eval/grader.py)) returns single-shot verdicts at temperature 0.0, and we have no numbers on how well those verdicts agree with humans, with each other across providers, or with themselves across resamples. Without those numbers, any score delta from `prompt_optimization/` could be noise in the grader rather than a real agent improvement.

This spec adds a small, dataset-agnostic meta-evaluation module that grades a fixed labelled set with a configurable judge, computes a registry of pluggable metrics, and persists durable artifacts (raw verdicts + summary scores) for offline comparison and for use as a fitness function when optimising the grader prompt.

## Design Principles

- **Reuse first.** Use the existing `EvalRunner`, `JudgeConfig`, `create_judge`, `stratified_sample`, and `evaluation/stats.py` helpers. Add the smallest possible amount of new code.
- **Domain types live in `domain/`.** Pure data types belong in the domain layer alongside `RubricItem`, `CriterionVerdict`, `HealthBenchSample`. The `meta_eval.py` module imports them but does not own them.
- **One generic input type via inheritance.** `LabelledSample` is the parent class. `HealthBenchSample` inherits from it and adds HealthBench-specific fields. Any function that accepts `LabelledSample` automatically accepts `HealthBenchSample` (Liskov). No `labelled_from_*` builder functions.
- **Dataset-agnostic core.** `meta_eval.py` knows nothing about HealthBench. It operates on `list[LabelledSample]`. The HealthBench-specific glue (loading the consensus subset, populating the meta-eval fields from `ideal_completions_data`, extracting axis tags) lives in the CLI, not the library module.
- **Pluggable metrics.** Metrics are functions registered with a `@register_meta_metric` decorator (mirroring `@register_tool`, `@register_callback`, `@register_prompt_optimizer`, `@register_analysis`). Adding a new metric is one decorated function — no changes to the runner, the result type, the CLI, or the artifact schema.
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
| Ground-truth source | Physician-ideal completion proxy (primary); the future hand-label loader populates the same `expected` field on `LabelledSample` | Zero-cost gold available today; the typed parent class accommodates real labels later with no API change |
| Judges per run | One | Cleaner artifacts, reproducible, cross-judge comparison done offline |
| Calibration sampling | Override `temperature` → 1.0 for the meta-eval grader; sweep `n_samples ∈ {1..k}` | Production grader stays deterministic at 0.0; meta-eval temperature is explicit and visible in MLflow |
| Artifact format | Raw `verdicts.parquet` + summary `metrics.json` | New metrics become reanalyses, not re-runs |
| Sampling strategy | Stratified by **theme** via existing `stratified_sample(..., tag_prefix="theme")` | `axis:*` tags live on rubric items, not on samples; the existing helper only supports sample-level tags. Theme stratification gives even coverage across the 7 HealthBench themes; per-axis confusion still works because every rubric on every sampled conversation is graded |
| Default sample size | 100 | Cheap enough for routine "did the grader prompt drift?" checks |
| Metric API | `@register_meta_metric` registry | Consistent with the rest of the project; open/closed |
| Domain placement | `LabelledSample` and `MetricResults` live in `domain/`; `HealthBenchSample` inherits from `LabelledSample` | Pure data types belong in the domain layer; inheritance gives Liskov-clean dataset-agnosticism |
| Prompt-opt integration | New `JudgeAgreementMetric` + `--target {agent, judge}` flag on existing `optimize-prompt` CLI | Reuses all three optimizer adapters unchanged |
| Module placement | `src/healthbench_agent/llm_eval/meta_eval.py` (single file) | Co-located with the judge it evaluates; no new package directory |

## Module Structure

```
src/healthbench_agent/domain/
    meta_evaluation.py      # NEW — LabelledSample, MetricResults
    dataset.py              # EDIT — HealthBenchSample now inherits from LabelledSample

src/healthbench_agent/llm_eval/
    meta_eval.py            # NEW — registry + 4 metrics + run_meta_eval + composite_fitness
    cli_meta_eval.py        # NEW — argparse → run_meta_eval; HealthBench glue lives here

src/healthbench_agent/prompt_optimization/
    metric.py               # EDIT — add JudgeAgreementMetric (~30 lines)
    cli.py                  # EDIT — add --target {agent, judge} flag

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
    for meta-evaluation. Regular agent-evaluation runs leave them empty.

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
    """
    prompt_id: str
    prompt: MessageList
    rubrics: list[RubricItem]
    gold_response: str | None = None
    expected: dict[str, bool] = field(default_factory=dict)
```

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

```python
# src/healthbench_agent/llm_eval/meta_eval.py

Metric = Callable[[pd.DataFrame], Any]
_METRIC_REGISTRY: dict[str, Metric] = {}

def register_meta_metric(name: str) -> Callable[[Metric], Metric]:
    """Decorator that registers a meta-evaluation metric by name."""

def get_meta_metric(name: str) -> Metric: ...
def registered_meta_metrics() -> dict[str, Metric]: ...
```

Adding a new metric (e.g. Fleiss' κ, ROC-AUC, judge-vs-judge agreement) is one decorated function. Zero changes anywhere else.

### Built-in metrics

| Name | Returns | Semantics |
|---|---|---|
| `cohens_kappa` | `float` | Collapse k passes to majority vote per (prompt_id, criterion), compute Cohen's κ vs `expected_met`. Wraps `sklearn.metrics.cohen_kappa_score`. |
| `krippendorff_alpha` | `float` | Same input, closed-form binary two-coder α. ~15 lines, no new dependency. |
| `calibration_curve` | `dict[int, float]` | For each k in {1, 3, 5, 7}, take the first k passes, collapse to majority, return bootstrap SE of the per-criterion agreement rate. |
| `per_dimension_confusion` | `dict[str, dict[str, int]]` | Group by `dimension` column → `{tp, fp, tn, fn}` per dimension. Rows where dimension is None aggregated under `"unspecified"`. |

### `run_meta_eval` — the only function with I/O

```python
# src/healthbench_agent/llm_eval/meta_eval.py

def run_meta_eval(
    judge: JudgeGrader,
    labelled: list[LabelledSample],
    dimension_extractor: Callable[[RubricItem], str | None],
    metric_names: list[str] | None = None,
    n_samples: int = 7,
    output_dir: Path | None = None,
    judge_metadata: dict[str, Any] | None = None,
) -> MetricResults:
    """Grade labelled samples k times with one judge, compute metrics, optionally persist.

    Steps:
        1. Filter to samples where gold_response is not None and expected is non-empty.
           Log how many were dropped.
        2. Loop k = 1..n_samples. Each iteration grades every sample by calling
           judge.grade(prompt + [{"role": "assistant", "content": gold_response}], rubrics)
           using a ThreadPoolExecutor (the same fan-out pattern as EvalRunner.run_async).
        3. Build a single pandas DataFrame with columns:
           prompt_id, criterion, dimension, points, sample_k, observed_met, expected_met.
        4. Run each requested metric over the DataFrame and collect results
           into MetricResults.scores.
        5. If output_dir is set, write verdicts.parquet (DataFrame) and
           metrics.json (dataclasses.asdict(result)).
        6. Return the MetricResults instance.

    Args:
        judge: Any JudgeGrader implementation. The CLI builds an LLMJudgeGrader
            with temperature overridden to >0 for k>1 to be meaningful.
        labelled: Dataset-agnostic input. Caller is responsible for sampling,
            stratification, and populating gold_response/expected.
        dimension_extractor: Required. Function mapping a RubricItem to a
            dimension label (e.g. axis name). The CLI passes its own — no
            HealthBench-specific default lives in this module.
        metric_names: Names of registered metrics to compute. None = all registered.
        n_samples: How many independent grading passes to run per sample.
        output_dir: Where to write verdicts.parquet + metrics.json. None = no I/O.
        judge_metadata: Run-level header captured in MetricResults and metrics.json.

    Returns:
        MetricResults with one entry per metric in scores.
    """
```

The `dimension_extractor` parameter is **required, no default**, so `meta_eval.py` carries no HealthBench knowledge.

### `composite_fitness` — single scalar for prompt-optimization

```python
# src/healthbench_agent/llm_eval/meta_eval.py

def composite_fitness(
    results: MetricResults,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum over numeric scores in MetricResults.

    Default weights privilege correctness over consistency:
        {"cohens_kappa": 0.6, "krippendorff_alpha": 0.4}
    """
```

## Parquet Schema (`verdicts.parquet`)

| column | type | meaning |
|---|---|---|
| `prompt_id` | str | from `LabelledSample.prompt_id` |
| `criterion` | str | rubric criterion text, truncated to 200 chars on write |
| `dimension` | str \| None | output of `dimension_extractor(rubric)` |
| `points` | float | rubric points |
| `sample_k` | int | which of the n_samples passes (1..n_samples) |
| `observed_met` | bool | judge verdict |
| `expected_met` | bool | from `LabelledSample.expected[criterion]` |

Run-level fields (`judge_model`, `temperature`, `judge_prompt_sha`, `n_samples`, `seed`) live in `metrics.json`'s `judge_metadata`, not on every row.

## CLI

### `meta-evaluate-judge`

```
uv run meta-evaluate-judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --subset consensus \
    --sample-size 100 \
    --n-samples 7 \
    --temperature 1.0 \
    --metrics cohens_kappa,krippendorff_alpha,calibration_curve,per_dimension_confusion \
    --output-dir runs/meta_eval/2026-04-07_openai/
```

`cli_meta_eval.py` flow (~80 lines):

1. Parse args.
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
       return None
   ```
6. Build a `JudgeConfig` from the YAML; override `temperature` from CLI.
7. `judge = create_judge(config)`.
8. `results = run_meta_eval(judge, sampled.samples, dimension_extractor=axis_extractor, ...)`.
9. Optional MLflow logging (params + scalar metrics + artifacts, tagged `run_type=meta_eval`).
10. Print summary.

Default `--metrics` is empty, meaning "all registered metrics".

### `optimize-prompt --target judge`

```
uv run optimize-prompt --target judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --optimizer critique_refine \
    --fitness cohens_kappa \
    --sample-size 50 \
    --max-trials 10
```

`--target agent` (default) keeps today's behaviour (`EndToEndMetric`). `--target judge` requires `--judge-config`, loads + populates a labelled set the same way `cli_meta_eval.py` does, builds a `JudgeAgreementMetric`, and hands it to the chosen optimizer adapter unchanged. The three adapters (DSPy, TextGrad, critique-refine) are not modified.

The labelled-set construction is shared between `cli_meta_eval.py` and `cli.py --target judge`. To avoid duplication, the population step (steps 2-5 above) is extracted into a small helper inside `cli_meta_eval.py` (`load_consensus_labelled(sample_size, seed) -> tuple[list[LabelledSample], Callable]`) which `cli.py` imports.

### `JudgeAgreementMetric`

```python
# src/healthbench_agent/prompt_optimization/metric.py  (additions)

class JudgeAgreementMetric:
    """Fitness metric that scores a candidate grader prompt by running
    meta-evaluation against a fixed labelled set.

    Mirrors the EndToEndMetric callable shape so any registered
    PromptOptimizer works without modification.
    """
    def __init__(
        self,
        judge_config: JudgeConfig,
        labelled: list[LabelledSample],
        dimension_extractor: Callable[[RubricItem], str | None],
        n_samples: int = 3,
        fitness: str = "cohens_kappa",
        weights: dict[str, float] | None = None,
    ) -> None: ...

    def __call__(self, candidate_template: str) -> float:
        """Build a one-off LLMJudgeGrader using the candidate template,
        call run_meta_eval, return the configured fitness scalar.
        """
```

When `fitness == "composite"`, the call returns `composite_fitness(results, weights)`. Otherwise it returns `float(results.scores[fitness])`.

`n_samples` defaults to 3 here (vs 7 in the meta-eval CLI default) because each optimizer trial calls the metric once and the fitness signal needs to be cheap; the meta-eval CLI runs once and can afford 7.

## MLflow Logging

`cli_meta_eval.py` (default ON; `--no-mlflow` to disable):
- `mlflow.set_experiment("meta_eval")`
- `mlflow.log_params({"judge_model", "temperature", "n_samples", "sample_size", "seed", "judge_prompt_sha"})`
- For numeric scores: `mlflow.log_metric("cohens_kappa", v)`, `mlflow.log_metric("krippendorff_alpha", v)`. Calibration-curve dict flattened as `cal_se_k1`, `cal_se_k3`, etc. Non-numeric scores (e.g. `per_dimension_confusion`) are NOT logged as metrics — they live in `metrics.json` and the parquet artifact.
- `mlflow.log_artifact(verdicts.parquet)`, `mlflow.log_artifact(metrics.json)`
- `mlflow.set_tag("run_type", "meta_eval")` — keeps these out of agent-comparison views.

For `--target judge` optimization runs, the existing prompt-optimization MLflow integration logs each trial as today; no extra wiring.

## Testing Strategy

### `tests/llm_eval/test_meta_eval.py`

ZOMBIES coverage on the pure metric functions using synthetic verdict DataFrames.

| Test | Expected |
|---|---|
| Full agreement (observed == expected for all rows) | κ = 1.0, α = 1.0 |
| Inverse (observed == not expected) | κ = -1.0 |
| Random / orthogonal labels (50/50) | κ ≈ 0.0 |
| Single class on both sides (all True or all False) | κ = NaN or 0 — document sklearn behaviour and assert |
| Empty DataFrame | each metric returns sensible default or raises with a clear message |
| Calibration curve at k=1 vs k=7 with synthetic noisy verdicts | SE@7 < SE@1 |
| `per_dimension_confusion` with two dimensions | correct tp/fp/tn/fn per dimension |
| Registry: `register_meta_metric` + `get_meta_metric` round-trip | works |
| `run_meta_eval` with a fake `JudgeGrader` | produces verdicts.parquet and metrics.json with the requested `scores` keys |
| `run_meta_eval` skips samples where `gold_response is None` | dropped count is correct |
| `JudgeAgreementMetric.__call__` with a fake judge | returns expected scalar |
| CLI smoke test: argparse → mocked `run_meta_eval` | dispatches with correct params |

### `tests/domain/test_meta_evaluation.py`

| Test | Expected |
|---|---|
| `LabelledSample` constructs with required fields, defaults populate | ok |
| `HealthBenchSample` is a `LabelledSample` (`isinstance` check) | True |
| `HealthBenchSample.from_dict(jsonl_row)` populates inherited + own fields | matches expected |
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

1. ~~Ground-truth source~~ → physician-ideal completion proxy primary; the same `expected` field on `LabelledSample` accommodates a future hand-labels loader.
2. ~~How many judges per run~~ → exactly one; cross-judge done offline by joining parquets.
3. ~~Calibration temperature~~ → meta-eval overrides judge to temperature=1.0; production grader stays at 0.0.
4. ~~Artifact format~~ → raw parquet + metrics.json.
5. ~~Sampling strategy~~ → stratified by theme (sample-level), default 100.
6. ~~Prompt-optimization integration~~ → grader prompt is also optimisable via `--target judge`.
7. ~~Fitness function~~ → configurable, default `cohens_kappa`, composite available with weights.
8. ~~Module placement~~ → single file inside `llm_eval/`; pure data types in `domain/`.
9. ~~Type relationships~~ → `HealthBenchSample` inherits from `LabelledSample`; no `labelled_from_*` builder functions.

## Out of Scope (Follow-up Issues)

- **Anthropic / Claude judge sampler.** Meta-eval is provider-agnostic via `JudgeConfig`; adding a third sampler is a separate ~60-line PR.
- **Hand-labelled gold set loader.** The `LabelledSample.gold_response` and `LabelledSample.expected` fields already exist for this; the future loader just needs to populate them from a JSONL.
- **Noise-floor hook into `prompt_optimization/`.** Auto-warn when an agent-optimization trial's score delta is below the meta-eval-measured judge variance.
- **Batch-API mode** for the judge during meta-eval. Async ThreadPool is sufficient at the projected sample sizes.
- **Optimising the rubric items themselves.**
