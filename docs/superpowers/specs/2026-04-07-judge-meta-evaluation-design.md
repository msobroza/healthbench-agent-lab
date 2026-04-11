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
| Domain placement | `LabelledSample` + lightweight `MetricResults` dataclass live in `domain/`; UX wrapper `MetricResultsView` (plot/load/summary/compare) lives in `llm_eval/`; `HealthBenchSample` inherits from `LabelledSample` | Pure data types belong in the domain layer (stdlib only). Methods that touch matplotlib, pandas, or the filesystem live one layer out so the domain dependency graph stays "stdlib only" and the rich UX still ships out of the box |
| Prompt-opt integration | New `JudgeAgreementMetric` + `--prompt-domain {agent, judge}` flag on existing `optimize-prompt` CLI (renamed from `--target` to avoid clashing with the existing `--target-agent`) | Reuses all three optimizer adapters unchanged; the rename keeps the two flags from being silently confused |
| Module placement | `src/healthbench_agent/llm_eval/meta_eval.py` (registry + runner) and `meta_eval_results.py` (rich UX wrapper) | Co-located with the judge it evaluates; no new package directory; UX wrapper sits next to the runner so it can import `MetricSpec.level` without crossing layers |
| Verdict cache | File-based `VerdictCache` keyed by `sha256(model_fingerprint ‖ prompt_sha ‖ conv_hash ‖ k_index ‖ rubric_text)`; integrated via a `CachedJudgeGrader` proxy that wraps any `JudgeGrader` and is built fresh per k-pass; OFF by default in library use, ON by default in CLI use (`--no-cache` to disable) | Calibration sweeps and prompt-opt loops re-grade the same `(judge, prompt, sample)` triple repeatedly. The proxy keeps `JudgeGrader.grade()` unchanged (no ABC edits) — the runner injects fingerprint + prompt_sha + k_index at construction so the cache key is deterministic without introspecting the inner judge |
| Thread pool composition | One `ThreadPoolExecutor(meta_eval_max_workers, default 16)` in `run_meta_eval` fans out across `(sample, k)` pairs; each task calls `inner_judge.grade()` which uses its own existing `grader_max_workers` pool (default 8). Total bound: `meta_eval_max_workers × grader_max_workers ≈ 128` threads | Reuses the existing per-rubric pool inside `LLMJudgeGrader` instead of nesting a third pool; both knobs are explicit and tunable so users can size the run for their judge's rate limits |
| Cost preview | `--dry-run` flag on the CLI prints an estimate table (samples × rubrics × n_samples × $/1k tokens) and exits before any LLM call | Inspired by `terraform plan`; the meta-eval grader is the most expensive thing in the project, so users deserve to see the bill before paying it |
| Happy-path API | Top-level `meta_evaluate(judge_config, ...)` function that wires dataset → judge → runner → MetricResultsView in one call | sklearn `cross_val_score` analog: most users just want "give me the numbers", not "build the runner yourself". Power users still get `run_meta_eval` for full control |
| MetricResults UX | Two-class split: lightweight `MetricResults(scores, n_samples_graded, ..., schema_version)` dataclass in `domain/` (json round-trip only); `MetricResultsView` wrapper in `llm_eval/meta_eval_results.py` adding `__repr__`, `summary()`, `_repr_html_`, `to_pandas`, `to_markdown`, `load()`, `verdicts()`, `compare(other)`, `plot_calibration_curve()`, `plot_dimension_confusion()` | Following sklearn / transformers conventions: results object should be inspectable in the REPL, render nicely in Jupyter, save/load round-trip, and produce its own plots — *without* dragging matplotlib/pandas/pyarrow into the domain layer |
| Saving optimized judge prompts | `prompts/llm_grader/v2_optimized.yaml` written in the existing `{version, template, parent_version, parent_prompt_path, rationale}` shape that `load_grader_prompt` reads. No `prompt_key` because the grader YAML is single-template, not multi-key | Mirrors the existing agent-side save path (`prompts/{agent_dir}/v2_optimized.yaml`) but follows the grader YAML schema so the file can be passed straight back through `JudgeConfig.prompt_path` without conversion |
| Test/demo helpers | `OracleJudge` (deterministic strategies) and `demo_labelled_set()` (3 hand-built samples with gold + adversarial pairs) ship in the library, not just in tests | Lets users try the API end-to-end with no API key; sklearn ships `make_classification`, transformers ships `pipeline("sentiment-analysis", model=…)` smoke samples for the same reason |
| Progress + errors | `tqdm` progress bar (auto-disabled in non-TTY); error messages name the active filters and suggest the next action (e.g. "no rubrics matched `axis_filter('accuracy')` — try `meta-evaluate-judge list-metadata-keys`") | DX polish; CI logs stay clean while interactive runs get feedback; errors that name the missing knob halve the time-to-first-success |
| CLI verbs | `argparse` subcommands: `run` / `regenerate` / `compare` / `list-metrics` / `list-metadata-keys` / `clear-cache` | Six verbs is enough to justify subparsers over a flat flag namespace; mirrors `git`, `docker`, `mlflow` |

## Module Structure

```
src/healthbench_agent/domain/
    meta_evaluation.py      # NEW — LabelledSample, lightweight MetricResults dataclass, SCHEMA_VERSION (stdlib only — no matplotlib / pandas / pyarrow imports)
    rubric.py               # EDIT — RubricItem gains optional SPEC.md fields; from_dict uses data.get("tags", [])
    dataset.py              # EDIT — HealthBenchSample now inherits from LabelledSample

src/healthbench_agent/llm_eval/
    meta_eval.py            # NEW — registry + 8 built-in metrics + run_meta_eval + filter helpers + EmptyFilterError + meta_evaluate() happy path + OracleJudge + demo_labelled_set
    meta_eval_results.py    # NEW — MetricResultsView UX wrapper around MetricResults: __repr__, _repr_html_, summary, to_pandas, to_markdown, load, verdicts, compare, plot_* (matplotlib / pandas / pyarrow live here, not in domain/)
    verdict_cache.py        # NEW — file-based VerdictCache keyed by sha256(judge_model || prompt_sha || conv_hash || k_index || rubric_text), plus CachedJudgeGrader proxy that wraps any JudgeGrader without changing the ABC
    cli_meta_eval.py        # NEW — argparse with subcommands: run / regenerate / compare / list-metrics / list-metadata-keys / clear-cache

src/healthbench_agent/prompt_optimization/
    optimizer.py            # EDIT — add OptimizationMetric Protocol shared by EndToEndMetric and JudgeAgreementMetric
    metric.py               # EDIT — add JudgeAgreementMetric, extend EndToEndMetric with sample_filter/rubric_filter, re-export EmptyFilterError
    cli.py                  # EDIT — add --prompt-domain {agent, judge} (note: rename from --target to avoid clashing with the existing --target-agent flag), --rubric-axis, --metadata flags

tests/llm_eval/
    test_meta_eval.py       # NEW — pure-metric ZOMBIES tests + runner with OracleJudge
    test_meta_eval_results.py  # NEW — MetricResultsView UX tests (summary, _repr_html_, load, compare, plot_*)
    test_verdict_cache.py   # NEW — cache hit/miss/disabled paths + CachedJudgeGrader proxy tests

tests/domain/
    test_meta_evaluation.py # NEW — LabelledSample/HealthBenchSample inheritance + MetricResults dataclass round-trip tests

config/judges/              # NEW directory — JudgeConfig YAMLs (e.g. openai_gpt41.yaml, gemini_25.yaml). Mirrors config/agents/. Convention introduced by this feature.

notebooks/
    04_judge_meta_evaluation.ipynb   # NEW — uses MetricResultsView.load + plot helpers; no manual matplotlib

pyproject.toml              # EDIT — register meta-evaluate-judge console script; add tqdm dependency
CLAUDE.md                   # EDIT — add meta_evaluation.py + meta_eval.py + meta_eval_results.py + verdict_cache.py + config/judges/ to project layout block
```

## Dependency Graph

```
domain/meta_evaluation.py
    -> domain/conversation, domain/rubric    (sibling files only)
    -> stdlib only (dataclasses, pathlib type hints)
    -> NO matplotlib, NO pandas, NO pyarrow — keeps the domain layer pure

domain/dataset.py  (modified)
    -> domain/meta_evaluation.py    (inherits LabelledSample)
    -> domain/conversation, domain/rubric    (existing)

llm_eval/meta_eval.py
    -> domain/      (LabelledSample, MetricResults, RubricItem, JudgeGrader, CriterionVerdict)
    -> llm_eval/    (sibling files: JudgeConfig, create_judge, LLMJudgeGrader, VerdictCache, CachedJudgeGrader)

llm_eval/meta_eval_results.py
    -> domain/meta_evaluation.py    (MetricResults dataclass)
    -> llm_eval/meta_eval.py        (MetricSpec / MetricLevel for summary() formatting)
    -> matplotlib, pandas, pyarrow  (lazy-imported inside the methods that need them)

llm_eval/verdict_cache.py
    -> domain/evaluation     (CriterionVerdict only)
    -> domain/judge          (JudgeGrader for the CachedJudgeGrader proxy)
    -> domain/rubric         (RubricItem for the proxy's grade() signature)
    -> stdlib only otherwise (hashlib, json, pathlib)

llm_eval/cli_meta_eval.py
    -> llm_eval/meta_eval.py
    -> llm_eval/meta_eval_results.py    (printing MetricResultsView in CLI output)
    -> llm_eval/verdict_cache.py        (build VerdictCache + CachedJudgeGrader)
    -> domain/, dataset/                (HealthBench loading)
    -> evaluation/, mlflow              (logging)

prompt_optimization/optimizer.py
    -> typing.Protocol (stdlib)         (OptimizationMetric Protocol shared by both metrics)

prompt_optimization/metric.py
    -> llm_eval/meta_eval.py            (JudgeAgreementMetric only)
    -> llm_eval/meta_eval_results.py    (MetricResultsView for inspection helpers)
    -> prompt_optimization/optimizer.py (OptimizationMetric Protocol)
```

No circular edges. `prompt_optimization → llm_eval` already exists for `JudgeConfig`. The new edge `domain/dataset.py → domain/meta_evaluation.py` is sibling-to-sibling within `domain/`. The split between `domain/meta_evaluation.py` and `llm_eval/meta_eval_results.py` is what keeps matplotlib, pandas, and pyarrow out of the domain layer — the rich UX wrapper imports them lazily inside its own methods, never at the domain layer.

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

### `MetricResults` — lightweight dataclass in `domain/`

`MetricResults` is the persisted shape of a meta-eval run: scores, counts, run-level metadata, and a schema version. It is intentionally **data-only** so that the `domain/` layer keeps its "stdlib only" dependency policy — no matplotlib, no pandas, no pyarrow imports. The rich user-facing helpers (`summary`, `to_pandas`, `compare`, `plot_*`) live one layer out in `MetricResultsView`.

`run_meta_eval` and `meta_evaluate` always **return a `MetricResultsView`** (which wraps a `MetricResults`) so users get the rich UX out of the box; the dataclass is what's serialised to disk.

```python
# src/healthbench_agent/domain/meta_evaluation.py  (same file as LabelledSample)

SCHEMA_VERSION: int = 1
"""Bumped when MetricResults persistence format changes incompatibly."""


@dataclass
class MetricResults:
    """Aggregate meta-evaluation result for one judge run.

    Pure data — no methods that touch matplotlib, pandas, or the
    filesystem. Round-trips to JSON via ``to_dict()`` / ``from_dict()``.
    Wrap in :class:`MetricResultsView` (in ``llm_eval/meta_eval_results.py``)
    to get the rich UX surface (summary, plots, compare, load).

    Attributes:
        scores: Mapping of metric name to its computed value. Value type
            depends on the metric (float for kappa, dict for confusion).
        n_samples_graded: Number of LabelledSamples that produced verdicts.
        n_rubrics_graded: Total (sample, rubric) pairs across all k passes.
        judge_metadata: Run-level header — judge_model, temperature,
            judge_prompt_sha, n_samples (k), seed, dataset name + size,
            active filter reprs, cache_hits, cache_misses.
        schema_version: Stamped from SCHEMA_VERSION at write time. Readers
            check this on load and raise a clear error on a mismatch.
        verdicts_path: Path of the parquet that produced these scores.
            Populated by ``MetricResultsView.load``; None for fresh
            in-memory results.
    """
    scores: dict[str, Any]
    n_samples_graded: int
    n_rubrics_graded: int
    judge_metadata: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    verdicts_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict.

        Wrapper around :func:`dataclasses.asdict` that converts
        ``verdicts_path`` to a string (or omits it if None) because
        ``Path`` is not natively JSON-serialisable.
        """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResults:
        """Inverse of :meth:`to_dict`. Validates ``schema_version``.

        Raises:
            ValueError: If ``schema_version`` is newer than ``SCHEMA_VERSION``.
        """
```

#### Why two result types — `MetricResults` vs `EvalResult`

`evaluation/` and `domain/evaluation.py` already define `EvalResult` and `SingleEvalResult` for the **end-to-end agent evaluation** pipeline (one row per (sample, rubric) verdict produced by an agent run). `MetricResults` is deliberately **not** that shape: it carries the **aggregate of a meta-evaluation run** (scores keyed by metric name, run-level header, schema version) rather than a list of per-sample verdicts. The two types coexist because they answer different questions:

- `EvalResult` — "what did the agent answer, and how did the judge rate every rubric?"
- `MetricResults` — "how well does this judge agree with ground truth, summarised by k metrics?"

Joining them is what `MetricResultsView.verdicts()` is for: the parquet on disk has the same row-per-verdict shape as `EvalResult`, but the in-memory dataclass only carries the aggregate scores. Keeping them separate prevents bloating `EvalResult` with run-level metadata it does not need and lets meta-eval evolve its schema independently.

### `MetricResultsView` — rich UX wrapper in `llm_eval/`

`MetricResultsView` is the user-facing object: it wraps a `MetricResults` instance and adds the sklearn-style ergonomics (REPL printing, Jupyter rendering, save/load, comparison, plots). It lives in `llm_eval/meta_eval_results.py` so it can import `MetricSpec.level` for `summary()` formatting and lazily import matplotlib / pandas / pyarrow inside the methods that need them — keeping the domain layer pure.

```python
# src/healthbench_agent/llm_eval/meta_eval_results.py

@dataclass
class MetricResultsView:
    """User-facing wrapper around a MetricResults dataclass.

    Adds REPL/Jupyter/IO/plot helpers without dragging matplotlib,
    pandas, or pyarrow into the domain layer. Returned by
    ``meta_evaluate`` and ``run_meta_eval`` so users get rich UX
    by default.

    Attributes:
        results: The underlying lightweight MetricResults dataclass.
            Always populated.
    """
    results: MetricResults

    # ---- pretty printing -------------------------------------------------

    def __repr__(self) -> str:
        """Compact one-line repr — judge model + sample/rubric counts."""

    def summary(self) -> str:
        """Multi-line tabular summary for terminal printing.

        Format mirrors sklearn's classification_report — fixed-width
        columns of name / level / value, with run-level metadata above.
        Reads ``MetricSpec.level`` from the registry to populate the
        LEVEL column. Example:

            MetricResults(judge=openai/gpt-4.1-2025-04-14, k=7, n=100)
            ────────────────────────────────────────────────────────────
            METRIC                  LEVEL    VALUE
            gold_score              SAMPLE   0.873
            cohens_kappa            ANY      0.612
            calibration_curve       ANY      {1: 0.08, 7: 0.04}
            per_dimension_confusion ANY      {accuracy: ..., ...}
        """

    def _repr_html_(self) -> str:
        """Jupyter HTML rendering — same content as summary() in a table.

        Auto-detected by Jupyter so printing the view in a notebook
        cell yields a styled table with no extra import.
        """

    # ---- conversions -----------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """Long-form DataFrame with columns metric, level, value.

        Dict-valued scores (calibration_curve, per_dimension_confusion)
        are exploded into one row per sub-key. Useful for joining results
        from multiple judges. Lazily imports pandas.
        """

    def to_markdown(self) -> str:
        """Markdown table suitable for pasting into a PR description or
        issue comment. Uses pandas' to_markdown under the hood (which
        uses tabulate, already a transitive dep)."""

    # ---- IO --------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path | str) -> MetricResultsView:
        """Reconstruct a MetricResultsView from a run directory.

        Reads ``run_dir/metrics.json`` into a ``MetricResults`` via
        ``MetricResults.from_dict`` (which validates ``schema_version``),
        sets ``verdicts_path = run_dir/verdicts.parquet``, and returns
        the wrapping view so ``.verdicts()`` can lazily load the parquet
        on demand.

        Raises:
            ValueError: If schema_version is newer than this code knows.
            FileNotFoundError: If metrics.json is missing.
        """

    def save(self, run_dir: Path | str) -> None:
        """Write metrics.json (from ``results.to_dict()``) into ``run_dir``.

        ``run_meta_eval`` calls this internally when the user passes
        ``output_dir``; exposed publicly so users can persist an
        in-memory result after the fact.
        """

    def verdicts(self) -> pd.DataFrame:
        """Lazily load verdicts.parquet from the same run directory.

        Returns the raw verdict DataFrame so users can rebuild any
        metric offline. Cached on first call. Lazily imports pandas
        and pyarrow.

        Raises:
            FileNotFoundError: If the view was constructed in-memory
                (``results.verdicts_path is None``).
        """

    # ---- comparison ------------------------------------------------------

    def compare(self, other: MetricResultsView) -> pd.DataFrame:
        """Side-by-side diff of scores against another judge run.

        Returns a DataFrame with columns metric, self, other, delta.
        Only numeric scores are diffed; dict scores are flagged as
        'see details'. Useful for inter-judge comparison.
        """

    # ---- plot helpers ----------------------------------------------------

    def plot_calibration_curve(self, ax: Any = None) -> Any:
        """Plot the calibration curve as bootstrap SE vs k.

        Returns a matplotlib Axes (creates one if ax is None) so plots
        compose into subplots. Mirrors sklearn's RocCurveDisplay style.
        Lazily imports matplotlib.

        Raises:
            ImportError: If matplotlib is not installed (suggests
                ``uv sync --extra viz``).
            KeyError: If ``calibration_curve`` is not in
                ``results.scores``.
        """

    def plot_dimension_confusion(self, ax: Any = None) -> Any:
        """Plot per-dimension confusion counts as a stacked bar chart.

        Returns a matplotlib Axes. Same composition + import handling
        as ``plot_calibration_curve``.
        """
```

The view stays thin — pure formatting/IO/plot wrappers around the underlying `MetricResults` dataclass and the persisted parquet. It is sklearn-style ergonomics, not new business logic. Users who want to round-trip results through JSON without pulling in matplotlib can work directly with `MetricResults.to_dict()` / `from_dict()`.

### Metric registry

Each metric declares **at registration time** which evaluation level it consumes. The runner uses this declaration to (a) filter the verdict DataFrame to the correct row subset before calling the metric, (b) skip metrics whose level is absent from the actual data with a single info-log line, and (c) expose the level + description in `meta-evaluate-judge list-metrics` so users see what each metric does without reading source.

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
        description: One-line human-readable summary shown by the
            ``list-metrics`` subcommand.
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
    The description is surfaced by ``meta-evaluate-judge list-metrics``.
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

`rubric_key` is a derived DataFrame column populated as `criterion_id or criterion` so the same metric implementation works on both HealthBench rows (no `criterion_id`) and SPEC.md rows (stable id). It is added in step 4 of `run_meta_eval` **and persisted to parquet** so `regenerate RUN_DIR` and `MetricResultsView.verdicts()` see the same column without recomputing it from `criterion_id` / `criterion` on the fly.

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

`meta-evaluate-judge list-metrics` (subcommand, not `--list-metrics`; see the CLI section for the default-subcommand workaround) prints `name | level | description` for every registered metric so users discover this without reading the spec. Example output:

```
$ uv run meta-evaluate-judge list-metrics
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

`gold_score` does **not** reimplement the scoring formula — it builds per-`(prompt_id, sample_k)` `(rubrics, verdicts)` pairs straight from the DataFrame columns, delegates to `calculate_score` + `clip_score` from `domain/scoring.py`, and means them with `statistics.fmean`. This is the project's single source of truth for HealthBench scoring; any future change to the formula (e.g. different penalty handling) propagates to meta-eval automatically.

```python
from statistics import fmean

from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.scoring import calculate_score, clip_score


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

    For each (prompt_id, sample_k) group it rebuilds a list of
    ``RubricItem``/``CriterionVerdict`` pairs from the DataFrame columns
    and calls the existing ``calculate_score`` + ``clip_score`` helpers
    from ``domain/scoring.py`` so meta-eval cannot drift from production
    scoring. A perfectly calibrated judge returns 1.0.

    Note: this calls ``calculate_score`` + ``clip_score`` directly
    instead of ``aggregate_scores`` because the latter takes
    ``list[SingleEvalResult]`` (the agent-evaluation domain type),
    while gold_score has DataFrame rows. The mean step is one
    ``statistics.fmean`` call.
    """
    per_sample_scores: list[float] = []
    for _, group in verdicts.groupby(["prompt_id", "sample_k"], sort=False):
        rubric_items = [
            RubricItem(criterion=row.criterion, points=float(row.points), tags=[])
            for row in group.itertuples(index=False)
        ]
        criterion_verdicts = [
            CriterionVerdict(
                criterion=row.criterion,
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
```

The metric is ~20 lines and contains zero scoring arithmetic of its own — it is pure plumbing between the verdict DataFrame and the existing scoring helpers.

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
    cache: VerdictCache | None = None,
    model_fingerprint: str | None = None,
    judge_prompt_sha: str | None = None,
    meta_eval_max_workers: int = 16,
    progress: bool | None = None,
) -> MetricResultsView:
    """Grade labelled samples k times with one judge, compute metrics, optionally persist.

    Steps:
        1. Apply ``sample_filter`` to drop samples that should not contribute.
           Log per-sample drops; raise EmptyFilterError if zero remain.
        2. For each surviving sample, apply ``rubric_filter`` to drop rubrics
           that should not contribute. Drop the sample if no rubrics survive.
           Raise EmptyFilterError if no (sample, rubric) pair remains.
        3. Loop k = 1..n_samples. For each k:
             - When ``cache`` is provided, wrap the inner judge in a fresh
               ``CachedJudgeGrader(judge, cache, model_fingerprint,
               judge_prompt_sha, k)`` so the cache key carries the k-pass
               index. When ``cache`` is None, use the inner judge directly.
             - Fan out grading via a single
               ``ThreadPoolExecutor(max_workers=meta_eval_max_workers)``
               over the (surviving sample, k-pass) pairs. Each task calls
               ``judge.grade(conversation, surviving_rubrics)`` once,
               which itself parallelises across rubrics using the inner
               grader's existing ``grader_max_workers`` pool (default 8).
               Two pool layers, total bound
               ``meta_eval_max_workers × grader_max_workers ≈ 128`` —
               no third nested pool.
             - Wrap the (sample, k) iterator in
               ``tqdm.contrib.concurrent.thread_map`` (or a no-op when
               ``progress=False`` or stdout is not a TTY) so the bar
               composes cleanly with the executor.
             - For each surviving sample:
                a. Sample-level flow — if sample.gold_response is not None,
                   grade each surviving rubric with points != 0 against
                   (prompt + gold_response). Emit one row per rubric with
                   gold_source="ideal_completion".
                b. Adversarial flow — for each surviving rubric with
                   example_meets or example_fails, grade just that one
                   rubric against (prompt + example_*). Emit one row with
                   gold_source="example_meets" or "example_fails".
        4. Build a single pandas DataFrame with columns:
           prompt_id, criterion_id, criterion, rubric_key, dimension,
           points, sample_k, gold_source, observed_met, expected_met,
           specialty, language, metadata_json.
           ``rubric_key = criterion_id or criterion`` is computed once at
           build time so the same value is used both by the in-memory
           metric step and persisted to parquet for reload.
           Partition once into ``sample_rows``, ``rubric_rows``,
           ``all_rows``.
        5. For each requested metric, look up its ``MetricSpec.level`` and
           pass the matching subset (SAMPLE→sample_rows, RUBRIC→rubric_rows,
           ANY→all_rows). Skip with an INFO log if the subset is empty.
           Collect numeric/dict scores into ``MetricResults.scores``. If
           every requested metric was skipped, raise ``EmptyFilterError``.
        6. Construct the lightweight ``MetricResults`` dataclass and wrap
           it in a ``MetricResultsView``. Stamp ``judge_metadata`` with
           ``cache_hits`` / ``cache_misses`` from ``cache.stats()`` (zeros
           when ``cache`` is None). If ``output_dir`` is set, write
           ``verdicts.parquet`` (DataFrame) and ``metrics.json``
           (``view.results.to_dict()``) — never ``dataclasses.asdict``
           directly, because ``verdicts_path`` is a ``Path`` and
           ``to_dict`` is the only place that knows how to coerce it
           to a string.
        7. Return the ``MetricResultsView``. ``results.verdicts_path`` is
           populated when ``output_dir`` was provided so
           ``view.verdicts()`` can lazy-load the parquet later.

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
        cache: Optional VerdictCache. None disables caching (every call hits
            the judge directly, with no proxy in front). Default-on in the
            CLI; default-off in library use to avoid surprising filesystem
            writes. When provided, ``model_fingerprint`` and
            ``judge_prompt_sha`` are required.
        model_fingerprint: Stable identifier for the inner judge (e.g.
            ``"openai/gpt-4.1@1.0"``). Required when ``cache`` is provided
            so the ``CachedJudgeGrader`` can key on it. Ignored otherwise.
        judge_prompt_sha: SHA-256 of the rendered grader template (the
            third return value from ``load_grader_prompt``). Required
            when ``cache`` is provided. Ignored otherwise.
        meta_eval_max_workers: Upper bound on the outer ThreadPoolExecutor
            (over (sample, k) pairs). Default 16. Total worker count is
            bounded by ``meta_eval_max_workers × grader_max_workers``;
            tune for the judge's rate limits.
        progress: Show a tqdm progress bar. None = auto (TTY-attached).
            False to force off (CI / pipes); True to force on.

    Returns:
        MetricResultsView wrapping a MetricResults with one entry per metric
        in scores. ``results.verdicts_path`` is set when output_dir was
        provided so the view can be reloaded later with
        ``MetricResultsView.load()``.

    Raises:
        EmptyFilterError: If sample_filter or rubric_filter eliminate all
            samples or all rubrics, or if every requested metric is skipped
            due to level mismatch. Error message names the active filters
            and the skipped metrics.
        ValueError: If ``cache`` is provided without both
            ``model_fingerprint`` and ``judge_prompt_sha``.
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
    """Keep rubrics whose `category` field or `axis: *` tag matches any of *axes*.

    Checks both the SPEC.md `category` field and the HealthBench
    ``"axis: <name>"`` tag convention (note the **space** after the
    colon — matches the form written by the existing
    ``stratified_sample`` / ``_extract_stratum`` helper in
    ``dataset/split_utils.py``). The constant
    ``AXIS_TAG_PREFIX = "axis: "`` is shared with the CLI's
    ``axis_extractor`` so the two helpers cannot drift.
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

### `meta_evaluate()` — top-level happy-path API

`run_meta_eval` is the power-user surface; most users want a one-liner that wires the judge, the dataset, the filters, and the cache for them. `meta_evaluate()` is that one-liner.

```python
# src/healthbench_agent/llm_eval/meta_eval.py

def meta_evaluate(
    judge_config: str | Path | JudgeConfig,
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

    The hello-world entry point. Loads the named HealthBench subset,
    stratifies + samples, populates gold-label fields from physician
    ideal completions, builds the judge from config, runs the meta-eval,
    and returns a ``MetricResultsView`` you can ``print()`` or
    ``.summary()``.

    Internally constructs the ``model_fingerprint`` and ``judge_prompt_sha``
    from the JudgeConfig (so the cache key is well-defined) and forwards
    them to ``run_meta_eval``. Mirrors sklearn.model_selection.cross_val_score
    in spirit: one call, sensible defaults, returns a rich result object
    that knows how to pretty-print itself.

    Args:
        judge_config: Path to a judge YAML, or a constructed JudgeConfig.
        subset: HealthBench subset name. Default 'consensus'.
        sample_size: How many samples to draw (stratified by theme).
        n_samples: Independent grading passes (k).
        temperature: Override applied to the judge model so k>1 is
            meaningful. Production grader stays deterministic.
        metric_names: Which metrics to compute. None = auto by dataset.
        sample_filter / rubric_filter: Optional callables built via
            axis_filter / metadata_filter / specialty_filter helpers,
            or any user-supplied predicate.
        output_dir: Where to persist verdicts.parquet + metrics.json.
            None = no I/O (in-memory only).
        cache: True = use the default ~/.cache/healthbench_agent cache.
            False = no caching. Or pass a custom VerdictCache instance.
        progress: tqdm bar control (None = auto-TTY).
        seed: Random seed for stratified sampling.

    Returns:
        MetricResultsView that prints itself nicely and can be reloaded
        with ``MetricResultsView.load(output_dir)``.

    Example:
        >>> view = meta_evaluate("config/judges/openai_gpt41.yaml")
        >>> print(view)
        MetricResultsView(judge=openai/gpt-4.1, k=7, n=100)
        >>> print(view.summary())                # tabular text
        >>> view.plot_calibration_curve()        # matplotlib Axes
        >>> reloaded = MetricResultsView.load("runs/meta_eval/...")
    """
```

The function is ~40 lines wrapping the existing pieces — no new logic, just a humane default-rich entry point that hides the `model_fingerprint` / `judge_prompt_sha` plumbing the runner needs for caching.

### `OracleJudge` and `demo_labelled_set` — try without an API key

Both are exported from `meta_eval.py` so users can run the pipeline end-to-end with zero external dependencies.

```python
# src/healthbench_agent/llm_eval/meta_eval.py

class OracleJudge(JudgeGrader):
    """Deterministic JudgeGrader for tests, demos, and docs.

    Implements the JudgeGrader ABC. Each call to grade() returns
    verdicts according to a configurable strategy:

      - "always_met"   — every criterion meets
      - "always_fail"  — no criterion meets
      - "alternating"  — first met, then fail, etc., per call
      - dict[str, bool] — explicit per-criterion verdict map keyed
        by ``RubricItem.criterion`` text. Pass the labelled set's
        union expected map to simulate a perfect judge:
        ``OracleJudge({c: m for s in samples for c, m in s.expected.items()})``
      - Callable[[RubricItem], bool] — for arbitrary per-rubric logic.

    Mirrors sklearn.dummy.DummyClassifier — useful for wiring tests
    without paying for or mocking a real LLM call.

    Note: there is no "match_expected" strategy. The previous design
    tried to read ground truth off ``RubricItem`` directly, but the
    expected verdict lives on ``LabelledSample.expected`` (sample-level)
    or in the ``example_meets`` / ``example_fails`` adversarial fields,
    not on the rubric, and ``JudgeGrader.grade(conversation,
    rubric_items)`` does not see the parent sample. Pass the explicit
    dict at construction time instead.
    """
    def __init__(
        self,
        strategy: str | dict[str, bool] | Callable[[RubricItem], bool] = "always_met",
    ) -> None: ...

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]: ...


def demo_labelled_set() -> list[LabelledSample]:
    """Hand-built 3-sample labelled set for examples and smoke tests.

    Each sample carries a gold_response, populated `expected`, and one
    rubric with example_meets/example_fails so both the sample-level
    and adversarial flows are exercised. Mirrors sklearn.datasets.load_iris
    or sklearn.datasets.make_classification — small, deterministic, no IO.

    Returns:
        Three LabelledSample instances ready to feed into meta_evaluate().
    """
```

Combined, they make the smallest possible smoke test:

```python
from healthbench_agent.llm_eval import (
    OracleJudge, demo_labelled_set, run_meta_eval
)

samples = demo_labelled_set()
# Build a perfect-judge map from the labelled set's own expected verdicts.
perfect = {criterion: met for s in samples for criterion, met in s.expected.items()}

view = run_meta_eval(
    OracleJudge(perfect),
    samples,
    dimension_extractor=lambda r: r.category,
)
print(view)  # gold_score == 1.0, no API key needed
```

### `VerdictCache` — file-based judge call cache

Lives in `llm_eval/verdict_cache.py` so the pure-data `meta_eval.py` does not gain filesystem responsibilities.

```python
# src/healthbench_agent/llm_eval/verdict_cache.py

class VerdictCache:
    """File-based cache for individual judge verdicts.

    Enables iterating on metric definitions, filters, or output formats
    without re-paying the LLM. Cache key is

        sha256(judge_model || judge_prompt_sha || conversation_hash || k_index)

    where conversation_hash is sha256 of the JSON-serialised MessageList +
    rubric criterion text. Each cache entry is one tiny JSON file under
    ``root / first2 / rest.json``, like git's loose object format.

    Attributes:
        root: Cache directory. Defaults to
            ``$XDG_CACHE_HOME/healthbench_agent/verdicts/`` (or
            ``~/.cache/healthbench_agent/verdicts/`` if XDG_CACHE_HOME
            is unset).
        enabled: When False the cache is a no-op (every get returns None,
            put is a no-op). Useful for benchmarking and tests.
    """
    def __init__(
        self,
        root: Path | None = None,
        enabled: bool = True,
    ) -> None: ...

    def make_key(
        self,
        judge_model: str,
        judge_prompt_sha: str,
        conversation: MessageList,
        rubric_text: str,
        k_index: int,
    ) -> str: ...

    def get(self, key: str) -> CriterionVerdict | None: ...
    def put(self, key: str, verdict: CriterionVerdict) -> None: ...
    def clear(self) -> None:
        """Delete every cached verdict. Used by tests and ``--clear-cache``."""

    def stats(self) -> dict[str, int]:
        """Return ``{"hits": int, "misses": int, "size_bytes": int}``."""
```

The cache is **off** when `meta_eval` is called as a library (no surprising filesystem writes), and **on by default** when invoked through the CLI (the `--no-cache` flag opts out). Cache misses fall through to the judge transparently; the `MetricResults.judge_metadata` records `cache_hits` and `cache_misses` so users see the savings on each run.

### `CachedJudgeGrader` — proxy that wires `VerdictCache` into any `JudgeGrader`

`JudgeGrader.grade(conversation, rubric_items)` is the existing ABC and must not gain new parameters — every existing caller (the runner, the optimizer metric, every test) would have to be updated otherwise. Instead, the cache integrates via a thin **proxy** that wraps any `JudgeGrader` and is constructed fresh for each k-pass with the run-level constants (model fingerprint, prompt sha, k_index) baked in.

```python
# src/healthbench_agent/llm_eval/verdict_cache.py  (same file as VerdictCache)

class CachedJudgeGrader(JudgeGrader):
    """JudgeGrader proxy that consults a VerdictCache before delegating.

    Wraps any JudgeGrader and intercepts ``grade()`` to deduplicate
    verdicts across calibration passes, optimizer trials, and re-runs.
    The cache key components that are constant across one k-pass
    (model fingerprint, prompt sha, k_index) are baked in at
    construction time so the proxy can compute the full key from just
    the (conversation, rubric_text) tuple — no introspection of the
    inner grader required.

    The inner ``LLMJudgeGrader`` is unchanged: from its perspective the
    proxy is just another caller. This is what keeps the JudgeGrader ABC
    stable while still letting the runner pass a cache through.

    Attributes:
        inner: The underlying JudgeGrader (typically an LLMJudgeGrader).
        cache: VerdictCache backing this proxy. May be ``enabled=False``
            (then the proxy is a transparent passthrough).
        model_fingerprint: Stable identifier for the inner grader's
            model + sampling temperature, e.g. ``"openai/gpt-4.1@1.0"``.
            The runner builds this from JudgeConfig.
        prompt_sha: SHA-256 of the rendered grader template, returned
            by ``load_grader_prompt(path)[2]``.
        k_index: Which calibration pass this proxy belongs to (1..n_samples).
    """
    def __init__(
        self,
        inner: JudgeGrader,
        cache: VerdictCache,
        model_fingerprint: str,
        prompt_sha: str,
        k_index: int,
    ) -> None: ...

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """For each rubric item, look up the cached verdict; on miss,
        delegate to ``inner.grade`` with just that one rubric and
        store the result. Returns the verdicts in the original order.

        Misses are batched into a single ``inner.grade(conversation,
        miss_items)`` call so the inner grader's per-rubric thread
        pool still gets to parallelise.
        """
```

The runner constructs one `CachedJudgeGrader` per k-pass:

```python
fingerprint = f"{judge_config.provider}/{judge_config.model}@{judge_config.temperature}"
prompt_sha = load_grader_prompt(judge_config.prompt_path)[2]
for k in range(1, n_samples + 1):
    cached_judge = CachedJudgeGrader(inner_judge, cache, fingerprint, prompt_sha, k)
    # ... fan out grading via meta_eval_max_workers ...
```

Constructing a fresh proxy per k-pass is cheap (it just stores three strings) and keeps the `k_index` baked into the cache key without the runner having to compute keys by hand. Crucially, **no edit to the `JudgeGrader` ABC**: every existing caller works unchanged.

### Developer experience: progress bars, dry-run, error messages

Three small but high-leverage UX features that touch the runner and the CLI without complicating the core types.

**Progress bars (tqdm).** `run_meta_eval` and `meta_evaluate` accept `progress: bool | None = None`. The default auto-detects whether stdout is a TTY (`sys.stdout.isatty()`). The bar wraps the per-sample loop and shows `Grading samples 23/100 [k=3/7] ETA 1:23`. In CI / piped output it's silently suppressed, so logs stay clean.

**Cost preview / `--dry-run`.** The CLI accepts `--dry-run`. When set, the runner stops after step 2 of `run_meta_eval` (filter resolution) and prints:

```
Dry run summary
─────────────────────────────────────────────
Judge:           openai/gpt-4.1-2025-04-14
Subset:          consensus  (sample_size=100, seed=0)
Samples:         100  (after filters: 92)
Rubrics:         418  (after filters: 287)
k passes:        7
Adversarial:     43 example_meets, 39 example_fails
Total LLM calls: (92 × 287 + 82) × 7 = 185,402
Cache hits:      0      (use --no-cache to disable cache)
Estimated cost:  ~$2.74  (est. 350 in / 60 out tokens per call @ openai/gpt-4.1)
```

Token estimates come from a small `_estimate_cost(judge_config, n_calls)` helper that knows per-model rates for the supported judges; unknown models print "no rate table — pass --skip-cost". `--dry-run` exits zero — it never calls the judge. Inspired by `terraform plan` and `pip install --dry-run`.

**Helpful error messages.** Every error class in this feature names the next user action.

| Error | Message template |
|---|---|
| `EmptyFilterError(sample, rubric)` | `All N samples were dropped by sample_filter=metadata_filter(language='fr'). The HealthBench consensus subset is English-only — try removing the --metadata language=fr flag. Available metadata keys: {keys}.` |
| `EmptyFilterError` (every metric skipped) | `Requested metrics {names} are all SAMPLE-level, but the dataset has no gold_response. Use --metrics adversarial_prf1,adversarial_accuracy or load a dataset with gold responses.` |
| Missing API key | `OPENAI_API_KEY is not set. Either export it (export OPENAI_API_KEY=sk-...) or pass --openai-api-key on the CLI, or set openai_api_key in your judge YAML.` |
| Schema-version mismatch on `MetricResultsView.load` | `runs/.../metrics.json was written with schema_version=2 but this build understands up to schema_version=1. Upgrade healthbench-agent or pass an older run.` |
| Missing matplotlib for `plot_*` | `Plotting requires matplotlib. Install with: uv sync --extra viz` |

A small `_format_filter_error(sample_filter, rubric_filter, available_keys)` helper builds these messages so the runner stays terse.

## Parquet Schema (`verdicts.parquet`)

| column | type | meaning |
|---|---|---|
| `prompt_id` | str | from `LabelledSample.prompt_id` |
| `criterion_id` | str \| None | from `RubricItem.criterion_id` (None for HealthBench) |
| `criterion` | str | rubric criterion text, truncated to 200 chars on write |
| `rubric_key` | str | derived `criterion_id or criterion` so that grouping/joining metrics work uniformly across HealthBench (no `criterion_id`) and SPEC.md (stable id). Persisted in parquet — `regenerate RUN_DIR` and `MetricResultsView.verdicts()` get the same column without recomputing. |
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

## HealthBench gold-completion extraction

The CLI is the only place that knows about the shape of HealthBench's `ideal_completions_data`. The extraction helper lives in `dataset/extraction.py` (sibling of `dataset/loader.py` and `dataset/split_utils.py`) so the `domain/` and `llm_eval/` layers stay HealthBench-agnostic and so the helper has a unit-test home alongside the rest of the dataset I/O.

```python
# src/healthbench_agent/dataset/extraction.py

def extract_ideal_completion_text(
    ideal_completions_data: dict[str, Any] | None,
) -> str | None:
    """Pull the gold response text out of a HealthBench sample's
    ``ideal_completions_data`` block.

    HealthBench ships physician ideal completions under several
    schema variants depending on subset version
    (``ideal_completion``, ``ideal_completions``, raw string,
    list of {role, content}). This helper normalises them all to
    a single ``str``, returning ``None`` when the block is missing
    or every variant fails to parse.

    Args:
        ideal_completions_data: The raw dict from
            ``HealthBenchSample.ideal_completions_data``. May be None.

    Returns:
        The extracted gold response text, or None when extraction fails.
        ``None`` is the signal the CLI uses to drop the sample at step 4.
    """
```

The CLI imports it under its private alias for back-compat with the original step 4 description (`from healthbench_agent.dataset.extraction import extract_ideal_completion_text as _extract_ideal_completion_text`). Tests live in `tests/dataset/test_extraction.py` and exercise every known schema variant plus the failure cases.

## CLI

### `meta-evaluate-judge`

The CLI uses argparse subparsers so each verb has its own focused flag set.

```
meta-evaluate-judge run [args]                  # grade a labelled set with one judge
meta-evaluate-judge regenerate RUN_DIR          # recompute metrics from a stored parquet
meta-evaluate-judge compare RUN1 RUN2           # side-by-side score diff
meta-evaluate-judge list-metrics                # name | level | description
meta-evaluate-judge list-metadata-keys          # discoverable filter keys + values
meta-evaluate-judge clear-cache                 # delete the verdict cache
```

**Default-subcommand workaround.** argparse does not natively support a default subcommand — `meta-evaluate-judge --judge-config foo.yaml` (no leading verb) errors out unless we patch around it. The CLI does this once at top of `main()`:

```python
parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
subparsers = parser.add_subparsers(dest="command")
# ... add run / regenerate / compare / list-metrics / list-metadata-keys / clear-cache ...

# Default subcommand: when no verb is given OR the first arg starts with '-',
# inject 'run' so existing scripts keep working.
if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
    sys.argv.insert(1, "run")

args = parser.parse_args()
if args.command is None:
    args.command = "run"   # belt-and-braces in case argv was already mutated
```

This is a known argparse limitation; the equivalent workaround appears in `pip`, `mlflow`, and `pytest`. Tests cover both the with-verb and bare-flag invocations.

#### `run` (default)

```
uv run meta-evaluate-judge run \
    --judge-config config/judges/openai_gpt41.yaml \
    --subset consensus \
    --sample-size 100 \
    --n-samples 7 \
    --temperature 1.0 \
    --metrics gold_score,cohens_kappa,adversarial_prf1,per_dimension_confusion \
    --rubric-axis accuracy \
    --metadata clinical_urgency=emergency \
    --metadata language=en \
    --output-dir runs/meta_eval/2026-04-07_openai/ \
    --dry-run                          # estimate cost, exit before any LLM call
    --no-cache                         # disable verdict cache for this run
    --no-progress                      # force tqdm off (default = auto-TTY)
```

`--rubric-axis` is repeatable and builds an `axis_filter`. `--metadata KEY=VALUE` is repeatable and builds a `metadata_filter`. Both are optional; omitting them keeps every sample and rubric.

`cli_meta_eval.py` flow (~120 lines):

1. Parse args, including repeatable `--rubric-axis` and `--metadata KEY=VALUE` flags, and the DX flags `--dry-run`, `--no-cache`, `--no-progress`.
2. `dataset = load_dataset("consensus")`.
3. `sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)`.
4. **Populate gold-label fields in place** for each `HealthBenchSample` in `sampled.samples` that has `ideal_completions_data`:
   - `sample.gold_response = _extract_ideal_completion_text(sample.ideal_completions_data)` (the helper lives in `dataset/extraction.py` — see the *HealthBench gold-completion extraction* section below — so the CLI is the only place that knows about HealthBench's `ideal_completions_data` shape)
   - `sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}`
   - Drop samples where extraction fails (returns None or raises). After dropping, **if zero samples remain** the CLI exits non-zero with `"All N samples dropped during gold-label extraction. The selected subset may not ship physician ideal completions — try --subset consensus or load a labelled set with hand-built gold responses."` so the user is never silently handed an empty meta-eval.
5. Define an inline `axis_extractor` that matches the existing
   `_extract_stratum` convention in `dataset/split_utils.py` (tags use
   the form `"axis: accuracy"` with a colon **and a space**):
   ```python
   AXIS_TAG_PREFIX = "axis: "

   def axis_extractor(item: RubricItem) -> str | None:
       for tag in item.tags:
           if tag.startswith(AXIS_TAG_PREFIX):
               return tag[len(AXIS_TAG_PREFIX):].strip()
       return item.category
   ```
   The space after the colon matters: `"axis:accuracy"` and
   `"axis: accuracy"` are distinct strings, and the existing
   stratified-sample helper writes the latter form. The same
   `AXIS_TAG_PREFIX` constant is shared by `axis_filter` so the two
   helpers can never drift out of sync.
6. Build filter callables:
   - `rubric_filter = axis_filter(*args.rubric_axis) if args.rubric_axis else None`
   - `sample_filter = metadata_filter(**parsed_metadata) if parsed_metadata else None`
7. Build a `JudgeConfig` from the YAML; override `temperature` from CLI.
8. Build the cache: `cache = VerdictCache(enabled=not args.no_cache)`.
9. If `--dry-run`: call `_estimate_cost(...)`, print the dry-run table from the Developer Experience section, exit 0.
10. `judge = create_judge(config)`.
11. `view = run_meta_eval(judge, sampled.samples, dimension_extractor=axis_extractor, sample_filter=sample_filter, rubric_filter=rubric_filter, cache=cache, progress=not args.no_progress, ...)`. (`view` is a `MetricResultsView`; the underlying pure `MetricResults` is `view.results`.)
12. Catch `EmptyFilterError` → exit non-zero with the helpful message described in the Developer Experience section.
13. Optional MLflow logging (params + scalar metrics + artifacts, tagged `run_type=meta_eval`). Filter args logged as MLflow **params** (`filter_axis`, `filter_metadata`) so runs are reproducible from the MLflow UI alone; cache stats (`cache_hits`, `cache_misses`) read from `view.results.judge_metadata` and logged as MLflow **metrics** because they are observed at runtime, not configured up front.
14. `print(view.summary())`.

Default `--metrics` is empty, meaning "auto-select based on what the dataset contains" using the table in the metric registry section above.

#### `regenerate RUN_DIR`

Reads `RUN_DIR/verdicts.parquet`, re-runs the metric registry against it, and overwrites `RUN_DIR/metrics.json` (with the new `schema_version`). Never calls the LLM. The killer feature paired with persisted verdicts: a researcher who just registered a new metric can recompute it across every historical run with one command. Accepts `--metrics` to select a subset.

```
uv run meta-evaluate-judge regenerate runs/meta_eval/2026-04-07_openai/ \
    --metrics adversarial_prf1,per_criterion_metrics
```

#### `compare RUN1 RUN2`

Loads two `MetricResultsView` instances via `MetricResultsView.load`, calls `.compare()`, and prints the resulting DataFrame. Optional `--output FILE.md` writes a markdown table for pasting into a PR.

```
$ uv run meta-evaluate-judge compare runs/openai/ runs/gemini/
METRIC                  OPENAI   GEMINI   Δ
gold_score              0.873    0.812    -0.061
cohens_kappa            0.612    0.589    -0.023
calibration_curve@k=7   0.04     0.05     +0.01
```

#### `list-metrics`

Prints `name | level | description` for every registered metric and exits. Sample output is in the metric registry section above. Discoverability without reading source — like `pytest --markers` or `pip list`.

#### `list-metadata-keys`

Loads the labelled set (using the same `--subset`, `--sample-size`, `--seed` flags as `run`) and prints every distinct key in `LabelledSample.metadata` plus its top-5 most common values. No judge calls. Helps users learn what filters are even possible on a given dataset.

```
$ uv run meta-evaluate-judge list-metadata-keys --subset consensus
KEY                       N    TOP VALUES
language                  100  en (100)
specialty                 100  general (54), pediatrics (18), cardiology (12), ...
clinical_urgency          100  routine (61), urgent (28), emergency (11)
```

#### `clear-cache`

Deletes the verdict cache directory. Prints how many entries / MB were freed.

### `optimize-prompt --prompt-domain judge`

```
uv run optimize-prompt --prompt-domain judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --optimizer critique_refine \
    --fitness gold_score \
    --sample-size 50 \
    --max-trials 10 \
    --rubric-axis accuracy \
    --metadata clinical_urgency=emergency
```

**Why `--prompt-domain` and not `--target`?** The existing `optimize-prompt` CLI already has a `--target-agent <name>` flag that selects which sub-agent's prompt to optimize inside a multi-agent pipeline (see [src/healthbench_agent/prompt_optimization/cli.py:55](src/healthbench_agent/prompt_optimization/cli.py#L55)). A new `--target {agent, judge}` flag would clash semantically — both flags would start with `--target` and both would mean "what to optimize", but they answer two different questions. Renaming the new flag to `--prompt-domain {agent, judge}` keeps the old flag's meaning intact and lets users combine them naturally:

```
# optimize the reviewer sub-agent's prompt
uv run optimize-prompt --prompt-domain agent --target-agent reviewer ...

# optimize the judge prompt (no target-agent)
uv run optimize-prompt --prompt-domain judge --judge-config ... ...
```

`--prompt-domain agent` (default) keeps today's behaviour (`EndToEndMetric`). `--prompt-domain judge` requires `--judge-config`, loads + populates a labelled set the same way `cli_meta_eval.py` does, builds a `JudgeAgreementMetric`, and hands it to the chosen optimizer adapter unchanged. The three adapters (DSPy, TextGrad, critique-refine) are not modified.

The same `--rubric-axis` and `--metadata` flags accepted by `meta-evaluate-judge` are accepted here. They are forwarded into `JudgeAgreementMetric` (via `sample_filter` / `rubric_filter`) so that the optimizer's fitness signal is restricted to the slice of interest. This is what enables **per-axis / per-specialty / per-metadata judge prompt optimization** without any optimizer changes.

For `--prompt-domain agent`, `EndToEndMetric` also gains the same two filter parameters so that agent prompts can be optimised on the same slice the user expects to evaluate them on. The CLI passes them to whichever metric `--prompt-domain` selects, so the user-facing flag set is identical for both domains.

The labelled-set construction is shared between `cli_meta_eval.py` and `cli.py --prompt-domain judge`. To avoid duplication, the population step (steps 2-5 above) is extracted into a small helper inside `cli_meta_eval.py` (`load_consensus_labelled(sample_size, seed) -> tuple[list[LabelledSample], Callable]`) which `cli.py` imports. Filter parsing (`--rubric-axis`, `--metadata`) is extracted into a second helper (`build_filters(args) -> tuple[sample_filter, rubric_filter]`) shared between both CLIs.

#### Saving optimized judge prompts

`optimize-prompt --prompt-domain agent` writes its winning prompt under `prompts/{agent_dir}/v2_optimized.yaml` (or `v2_optimized_{target_agent}.yaml` when a sub-agent is targeted; see [src/healthbench_agent/prompt_optimization/cli.py:222-244](src/healthbench_agent/prompt_optimization/cli.py#L222-L244) for the existing logic). `--prompt-domain judge` mirrors this convention but writes to the judge prompt directory:

```
prompts/llm_grader/v2_optimized.yaml
```

The file shape matches the existing grader YAML schema that `load_grader_prompt` reads (`{version, template, parent_version, parent_prompt_path, rationale}`), so the optimized prompt can be passed straight back through `JudgeConfig.prompt_path` without conversion. There is no `prompt_key` because the grader YAML is single-template, not multi-key — the field would be silently dropped if added. Example output:

```yaml
version: 2.0.0
created: 2026-04-07
parent_version: 1.0.0
parent_prompt_path: prompts/llm_grader/v1_llm_grader.yaml
template: |
  Your job is to look at a conversation and a rubric item ...
rationale: >
  Automatically optimized via critique_refine on the consensus subset
  (n=50, axis=accuracy). Score: 0.682 -> 0.741 (+0.059). Trials: 10.
```

The trial history JSON is similarly written next to it as `prompts/llm_grader/optimization_trials.json`.

### `OptimizationMetric` Protocol

`EndToEndMetric` and `JudgeAgreementMetric` already share the same call shape (`__call__(prompt: str) -> float`) because every registered optimizer adapter expects that contract. Make it explicit with a `typing.Protocol` declared in `prompt_optimization/optimizer.py` (next to `PromptOptimizer`) so that `create_prompt_optimizer` can type-check the metric it receives at construction time and so future fitness functions are obviously interchangeable:

```python
# src/healthbench_agent/prompt_optimization/optimizer.py  (additions)

from typing import Protocol


class OptimizationMetric(Protocol):
    """Callable contract shared by every prompt-optimization fitness function.

    Both ``EndToEndMetric`` and ``JudgeAgreementMetric`` satisfy this
    Protocol structurally (no inheritance required). Any optimizer
    adapter that takes a metric should annotate it as
    ``OptimizationMetric`` rather than the concrete class so the user
    can swap fitness functions without touching adapter code.

    The metric is responsible for whatever expensive work it needs to
    do per call (rebuilding a pipeline, running a meta-eval pass, etc.).
    The optimizer just hands it candidate prompts.
    """

    def __call__(self, prompt: str) -> float:
        """Score a candidate prompt and return a single fitness scalar.

        Higher is better. The optimizer compares the returned value
        across calls; absolute magnitudes are not assumed.
        """
        ...
```

This is a Protocol, not an ABC — neither metric needs to inherit from anything. It is documentation + a type-checker hook.

### `JudgeAgreementMetric`

```python
# src/healthbench_agent/prompt_optimization/metric.py  (additions)

from healthbench_agent.prompt_optimization.optimizer import OptimizationMetric


class JudgeAgreementMetric:
    """Fitness metric that scores a candidate grader prompt by running
    meta-evaluation against a fixed labelled set.

    Mirrors the EndToEndMetric callable shape so any registered
    PromptOptimizer works without modification — both classes satisfy
    the ``OptimizationMetric`` Protocol structurally. Optional
    sample/rubric filters restrict the fitness signal to a slice
    (e.g. "accuracy axis, emergency cases") so the optimizer
    specialises the judge prompt for that slice.
    """
    def __init__(
        self,
        judge_config: JudgeConfig,
        labelled: list[LabelledSample],
        dimension_extractor: Callable[[RubricItem], str | None],
        n_samples: int = 3,
        fitness: str = "gold_score",
        sample_filter: Callable[[LabelledSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None: ...

    def __call__(self, candidate_template: str) -> float:
        """Build a one-off LLMJudgeGrader using the candidate template,
        call ``run_meta_eval`` with the configured filters, and return
        ``float(view.results.scores[self.fitness])`` (``run_meta_eval``
        returns a ``MetricResultsView``; the underlying pure dataclass
        is at ``view.results``).

        The supported ``fitness`` values are exactly the names in the
        meta-eval registry (``gold_score``, ``cohens_kappa``, etc.).
        There is no special-case "composite" mode — composite fitness
        was YAGNI in practice (no real call site beyond the spec) and
        the per-metric Float dispatch keeps this method to three lines.
        Users who want a weighted combination can write a custom
        ``OptimizationMetric`` that wraps two
        ``JudgeAgreementMetric`` instances.
        """
```

`n_samples` defaults to 3 here (vs 7 in the meta-eval CLI default) because each optimizer trial calls the metric once and the fitness signal needs to be cheap; the meta-eval CLI runs once and can afford 7.

### `EndToEndMetric` extension

`prompt_optimization/metric.py::EndToEndMetric` gains the **same two filter parameters** (`sample_filter` and `rubric_filter`) and is annotated as satisfying the `OptimizationMetric` Protocol. The agent pipeline still runs against every sample in the input set, but the per-rubric scoring step skips rubrics that fail `rubric_filter`, and the aggregate skips samples that fail `sample_filter`. This is what lets `optimize-prompt --prompt-domain agent` produce a per-axis or per-metadata-slice fitness signal without changing any optimizer adapter.

Both metrics raise `EmptyFilterError` (re-exported from `meta_eval.py`) if a filter combination eliminates every (sample, rubric) pair, so the user sees a clear failure rather than silently optimising against an empty signal.

## MLflow Logging

`cli_meta_eval.py` (default ON; `--no-mlflow` to disable):
- `mlflow.set_experiment("meta_eval")`
- `mlflow.log_params({"judge_model", "temperature", "n_samples", "sample_size", "seed", "judge_prompt_sha", "filter_axis", "filter_metadata"})`
- For numeric scores: `mlflow.log_metric("gold_score", v)`, `mlflow.log_metric("cohens_kappa", v)`, `mlflow.log_metric("krippendorff_alpha", v)`. Calibration-curve dict flattened as `cal_se_k1`, `cal_se_k3`, etc. Cache stats are logged as metrics: `mlflow.log_metric("cache_hits", n)`, `mlflow.log_metric("cache_misses", n)` — they are observed at run time, not configured. Non-numeric scores (e.g. `per_dimension_confusion`) are NOT logged as metrics — they live in `metrics.json` and the parquet artifact.
- `mlflow.log_artifact(verdicts.parquet)`, `mlflow.log_artifact(metrics.json)`
- `mlflow.set_tag("run_type", "meta_eval")` — keeps these out of agent-comparison views.

For `--prompt-domain judge` optimization runs, the existing prompt-optimization MLflow integration logs each trial as today; no extra wiring.

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
| CLI smoke test: `list-metadata-keys` on a labelled set with mixed `language`, `specialty`, `metadata` keys | stdout lists every observed key + a few example values, no judge call made |
| CLI smoke test: `--dry-run` prints cost-estimate table and exits 0 before any judge call | judge mock receives 0 calls; stdout contains "DRY RUN" + token estimate |
| CLI smoke test: `regenerate RUN_DIR` replays metrics from stored parquet | judge mock receives 0 calls; new `metrics.json` matches old |
| CLI smoke test: `compare RUN1 RUN2` prints a diff table with both judges' scalars | stdout has both judge model names and a delta column |
| CLI smoke test: `clear-cache` removes the cache directory and prints freed-bytes summary | directory is gone, exit code 0 |
| `meta_evaluate(judge_config, ...)` happy-path with `OracleJudge` | returns `MetricResultsView` whose `view.results.scores` is non-empty, runs end-to-end without touching the network |
| `meta_evaluate` with `cache=True` (default in CLI) | second call hits cache (assert via `cache.stats()`); runtime measurably shorter |
| `meta_evaluate` with `cache=False` | every call goes to the judge; cache directory not created |
| `tqdm` progress bar suppressed when stdout is not a TTY (`progress=None` auto-detect) | no `tqdm` instance created (assert via patch) |
| `tqdm` progress bar enabled when `progress=True` even in non-TTY | bar instance created |
| Helpful error: empty `axis_filter` produces an error mentioning the filter and suggesting `list-metadata-keys` | message regex matches |

### `tests/domain/test_meta_evaluation.py`

| Test | Expected |
|---|---|
| `LabelledSample` constructs with required fields, defaults populate | ok |
| `LabelledSample` constructs with full SPEC.md fields (language, specialty, user_persona, metadata) | values round-trip |
| `HealthBenchSample` is a `LabelledSample` (`isinstance` check) | True |
| `HealthBenchSample.from_dict(jsonl_row)` populates inherited + own fields | matches expected |
| `RubricItem.from_dict` reads optional `criterion_id`, `category`, `example_meets`, `example_fails` when present | values populated; defaults remain `None` when absent |
| `RubricItem.from_dict` accepts rows without a `tags` key (uses empty list default) | succeeds; `tags == []` |
| Existing `HealthBenchSample` keyword construction still works | ok |
| A function annotated `def foo(s: LabelledSample)` accepts a `HealthBenchSample` | mypy passes |
| Setting `gold_response` and `expected` on a loaded `HealthBenchSample` works | mutation succeeds (dataclass not frozen) |
| `MetricResults.to_dict()` round-trips through `MetricResults.from_dict(d)` | equal |
| `MetricResults.from_dict()` raises on unknown schema_version | `ValueError` mentioning the version mismatch |
| `MetricResults.to_dict()` coerces `verdicts_path` to a string when set, omits when None | dict has stringified path or no key |
| `MetricResults` carries no methods that touch matplotlib/pandas/pyarrow | introspecting the class confirms only data methods (no plot_* / to_pandas / load) |
| `demo_labelled_set()` returns 3 `LabelledSample` instances with mixed gold + adversarial pairs | counts: gold=2, adversarial=1 (or whatever the demo defines) |

### `tests/llm_eval/test_meta_eval_results.py`

| Test | Expected |
|---|---|
| `MetricResultsView(results).__repr__` is short and includes judge model | matches regex |
| `MetricResultsView.summary()` returns a multi-line string with one row per metric | line count == len(scores) + header |
| `MetricResultsView._repr_html_` returns a string starting with `<table` | True |
| `MetricResultsView.to_pandas()` returns a DataFrame with the score columns | columns match `scores.keys()`; lazy import works without matplotlib installed |
| `MetricResultsView.to_markdown()` returns a string starting with `\|` | True |
| `MetricResultsView.load(run_dir)` reads `metrics.json` + sets `results.verdicts_path` | fields populated; `verdicts()` returns the parquet |
| `MetricResultsView.load(run_dir)` raises when `schema_version` is unknown | `ValueError` mentioning the version mismatch |
| `MetricResultsView.save(run_dir)` writes `metrics.json` round-trippable via `load()` | reload equals original |
| `MetricResultsView.compare(other)` returns a 2-column DataFrame indexed by metric name | columns are both judge model names |
| `MetricResultsView.plot_calibration_curve(ax=None)` returns a matplotlib Axes | not None |
| `MetricResultsView.plot_dimension_confusion(ax=None)` returns a matplotlib Axes | not None |
| `plot_calibration_curve` raises a helpful ImportError when matplotlib is missing | message mentions `uv sync --extra viz` |

### `tests/llm_eval/test_meta_eval.py` — OracleJudge sub-section additions

| Test | Expected |
|---|---|
| `OracleJudge("always_met")` returns `criteria_met=True` for every rubric | True for all |
| `OracleJudge("always_fail")` returns `criteria_met=False` for every rubric | False for all |
| `OracleJudge("alternating")` flips per criterion within one `grade()` call | even-index True, odd-index False |
| `OracleJudge({"crit_a": True, "crit_b": False})` honours the dict mapping | per-criterion verdicts match |
| `OracleJudge` constructed from a labelled set's union expected map yields `gold_score == 1.0` | computed via `run_meta_eval(OracleJudge(perfect_map), demo_labelled_set(), ...)` |
| `OracleJudge` accepts a callable strategy `Callable[[RubricItem], bool]` | per-rubric verdicts match callable's output |

### `tests/llm_eval/test_verdict_cache.py`

| Test | Expected |
|---|---|
| `VerdictCache(enabled=False).get(any_key)` always returns `None` | True |
| `VerdictCache(enabled=False).put(...)` is a no-op (no file written) | directory empty |
| `make_key` is deterministic across instances with same inputs | equal |
| `make_key` differs when any of model / prompt_sha / conv_hash / k_index / rubric_text differ | all 5 distinct |
| `put` then `get` round-trips a `CriterionVerdict` | equal |
| `get` on a missing key returns `None` (cache miss) | True |
| Cache files are sharded by first-2 hex characters | path matches `root/ab/cdef…json` |
| `clear()` removes the cache root directory | directory gone |
| `stats()` reports hits / misses / writes | counters match calls |
| Two `VerdictCache` instances pointing at the same root see each other's writes | True |
| `CachedJudgeGrader(inner, cache, fingerprint, sha, k=1).grade(...)` calls `inner.grade` only on the first invocation | second call hits cache; `inner.grade.call_count == 1` |
| `CachedJudgeGrader` with two different `k_index` values produces two distinct cache entries for the same rubric | both keys present in cache |
| `CachedJudgeGrader` with `cache.enabled=False` always delegates to inner | `inner.grade.call_count` equals number of `grade()` calls |
| `CachedJudgeGrader` batches misses into a single `inner.grade(conversation, miss_items)` call | only one delegated call per multi-rubric `grade()` |
| `CachedJudgeGrader` preserves the original rubric order in its return value | order matches input even when cache hits and misses are interleaved |

Coverage targets: 100% on pure metric functions, ≥80% module-wide per project policy.

## Notebook — `notebooks/04_judge_meta_evaluation.ipynb`

Five cells, all built on `MetricResultsView` methods so the user never touches matplotlib directly:

```python
# Cell 1 — load
from healthbench_agent.llm_eval.meta_eval_results import MetricResultsView
view = MetricResultsView.load("runs/meta_eval/2026-04-07_gpt-4-1")
view  # rich HTML repr in Jupyter

# Cell 2 — calibration
view.plot_calibration_curve()  # returns Axes

# Cell 3 — per-dimension confusion
view.plot_dimension_confusion()  # returns Axes

# Cell 4 — compare two judges
other = MetricResultsView.load("runs/meta_eval/2026-04-07_gemini-2-5")
view.compare(other)  # DataFrame indexed by metric name

# Cell 5 — extract a single fitness scalar for both
view.results.scores["gold_score"], other.results.scores["gold_score"]
```

No manual matplotlib in the notebook, no new analysis-registry entries — meta-eval artifacts live in `runs/meta_eval/`, not the analysis output directory.

## Dependencies

One new required dependency (`tqdm`); matplotlib already ships via the analysis layer.

| Need | Source |
|---|---|
| Cohen's κ | `sklearn.metrics.cohen_kappa_score` (already a dep) |
| Precision / recall / F1 | `sklearn.metrics.precision_recall_fscore_support` (already a dep) |
| Bootstrap SE | `scipy.stats` + `numpy` (already deps) |
| Krippendorff's α | Inline closed-form for binary, two-coder case (~15 lines) |
| Concurrent grading | `concurrent.futures.ThreadPoolExecutor` (stdlib) |
| Parquet I/O | `pandas.DataFrame.to_parquet` + `pyarrow` (already in deps for analysis) |
| Verdict cache I/O | `hashlib` + `json` + `pathlib` (stdlib) |
| Progress bars | **NEW** `tqdm` (small, pure-Python; auto-disables in non-TTY) |
| Plot helpers (`plot_calibration_curve`, `plot_dimension_confusion`) | `matplotlib` (already a dep via the analysis layer) |

## Migration Risk: `HealthBenchSample` Inheritance

Adding `LabelledSample` as a parent of `HealthBenchSample` introduces two extra inherited fields (`gold_response`, `expected`) with defaults. Risks:

1. **Positional construction.** Any code that builds `HealthBenchSample(prompt_id, prompt, rubrics, example_tags)` positionally will break because `example_tags` is no longer the 4th positional argument. Audit and migrate to keyword form. The `from_dict` classmethod is already keyword-based.
2. **Field order in dataclass inheritance.** All inherited fields without defaults must come before fields with defaults. `LabelledSample` has 3 no-default + 2 default; `HealthBenchSample` adds 3 default fields. Order is consistent.
3. **`asdict` / serialization** of `HealthBenchSample` now produces the 5 inherited fields plus the 3 own fields. Any code that round-trips samples through `asdict` and back via `from_dict` is unaffected because `from_dict` ignores unknown keys, but code that compares `asdict(sample)` against a fixed dict will need updating.
4. **Test fixtures.** `tests/conftest.py` and dataset-related test files build sample objects directly. Audit and migrate.
5. **`RubricItem.from_dict` requires `tags`.** The current implementation reads `data["tags"]`, which `KeyError`s on rows that omit the field. SPEC.md-shaped rows can legally omit `tags` (the new optional fields like `category` / `language` / `metadata` may be the only structure on the rubric), so this must become `data.get("tags", [])` as part of the same migration step. Add a regression test that round-trips a tag-less rubric row through `from_dict`.

The migration is mechanical (keyword args + a few extra `gold_response=None` defaults that already exist plus the `tags` default) but must be in the implementation plan as its own step before the meta-eval module is wired.

## Open Questions (resolved during brainstorming)

1. ~~Ground-truth source~~ → two flows: sample-level via physician ideal completion (or future hand labels) populating `gold_response`, plus rubric-level adversarial pairs via `example_meets`/`example_fails` from the SPEC.md schema.
2. ~~How many judges per run~~ → exactly one; cross-judge done offline by joining parquets.
3. ~~Calibration temperature~~ → meta-eval overrides judge to temperature=1.0; production grader stays at 0.0.
4. ~~Artifact format~~ → raw parquet + metrics.json.
5. ~~Sampling strategy~~ → stratified by theme (sample-level), default 100.
6. ~~Prompt-optimization integration~~ → grader prompt is also optimisable via `--prompt-domain judge`; agent prompt optimization gains the same filter flags.
7. ~~Fitness function~~ → configurable, default `gold_score` (falls back to `adversarial_prf1["f1"]` when no sample-level gold exists), composite available with weights.
8. ~~Module placement~~ → single file inside `llm_eval/`; pure data types in `domain/`.
9. ~~Type relationships~~ → `HealthBenchSample` inherits from `LabelledSample`; no `labelled_from_*` builder functions.
10. ~~Slice-restricted optimization~~ → both `JudgeAgreementMetric` and `EndToEndMetric` accept optional `sample_filter` and `rubric_filter` (Option A: two separate `Callable` parameters), exposed on the CLI as repeatable `--rubric-axis` and `--metadata KEY=VALUE` flags.
11. ~~Empty filter behaviour~~ → `EmptyFilterError` (subclass of `ValueError`) raised by `run_meta_eval`; CLI catches and exits non-zero with the active filter names.
12. ~~SPEC.md schema fields~~ → optional with safe defaults on both `RubricItem` and `LabelledSample`; HealthBench loaders ignore them, future SPEC.md loaders populate them.
13. ~~Metric level discovery~~ → each metric declares `MetricLevel ∈ {SAMPLE, RUBRIC, ANY}` at registration; runner filters rows per level; CLI exposes `meta-evaluate-judge list-metrics` (subcommand) so users see the level + a one-line description without reading source.
14. ~~Cost iteration friction~~ → file-based `VerdictCache` keyed by `sha256(judge_model ‖ prompt_sha ‖ conv_hash ‖ k_index)` (OFF in library, ON in CLI), `--dry-run` cost preview, and `regenerate RUN_DIR` subcommand let users tune metrics/filters without re-paying for verdicts.
15. ~~Beginner ergonomics~~ → top-level `meta_evaluate(judge_config, ...)` happy-path function, `OracleJudge` + `demo_labelled_set()` for an offline smoke run, `tqdm` progress bars (auto-detected), and a rich `MetricResultsView` wrapper (in `llm_eval/`) with `__repr__` / `_repr_html_` / `summary()` / `compare()` / `plot_*` methods so beginners do not need to touch matplotlib or pandas to inspect a run. The pure-domain `MetricResults` dataclass stays UX-free so it can live in `domain/`.
16. ~~Discoverability of metadata filters~~ → new CLI subcommand `list-metadata-keys` walks the dataset and prints every observed key with example values, so users can build correct `--metadata KEY=VALUE` flags without reading the dataset by hand.
17. ~~Artifact format versioning~~ → `MetricResults.schema_version` (constant `SCHEMA_VERSION = 1`) is written to `metrics.json`; `MetricResultsView.load` raises a clear `ValueError` on unknown versions so future format breaks fail loud, not silent.
18. ~~Where matplotlib/pandas/pyarrow live~~ → split into a pure-domain `MetricResults` dataclass (`domain/meta_evaluation.py`, stdlib only) and a UX wrapper `MetricResultsView` in `llm_eval/meta_eval_results.py`. Domain stays I/O-free; the view owns `summary()`, `_repr_html_`, `to_pandas`, `to_markdown`, `load`, `save`, `compare`, and the `plot_*` helpers.
19. ~~Cache integration without leaking into the ABC~~ → introduce a `CachedJudgeGrader` proxy in `llm_eval/verdict_cache.py` that wraps any `JudgeGrader`. The proxy bakes the run-level constants (`model_fingerprint`, `judge_prompt_sha`, `k_index`) into the cache key at construction time, so the inner grader stays untouched and `JudgeGrader.grade()` keeps its current signature. The runner constructs one proxy per `k_index` pass.
20. ~~Concurrency vs `LLMJudgeGrader.max_workers`~~ → meta-eval uses one outer `ThreadPoolExecutor(max_workers=meta_eval_max_workers, default 16)` over `(sample, k_index)` pairs; each inner `LLMJudgeGrader.grade()` keeps its own thread pool of `grader_max_workers=8`. Total concurrency is capped at `16 × 8 = 128` threads, which matches the OpenAI default rate limits comfortably; both knobs are exposed on `JudgeConfig` and the CLI for tuning. Outer pool uses `tqdm.contrib.concurrent.thread_map` for progress bars.
21. ~~Optimizer-CLI flag clash~~ → the existing `--target-agent` flag in `optimize-prompt` selects a sub-agent to patch via `instruction_override`; reusing `--target` for the new agent-vs-judge axis would clash. Resolved by introducing `--prompt-domain {agent, judge}` (default `agent`) for the new axis, leaving `--target-agent` untouched.
22. ~~Saving optimized judge prompts~~ → `optimize-prompt --prompt-domain judge` writes the best prompt to `prompts/llm_grader/v2_optimized.yaml` with the same metadata block (`version`, `created`, `parent_version`, `parent_prompt_path`, `rationale`) as the agent path. The directory is determined from `JudgeConfig.prompt_path.parent`, mirroring how the agent path uses the target agent's `prompt_path.parent`.
23. ~~Shared optimizer-metric contract~~ → introduce a tiny `OptimizationMetric` Protocol (`__call__(prompt: str) -> float`) in `prompt_optimization/optimizer.py`; both `EndToEndMetric` and `JudgeAgreementMetric` already match it structurally. Optimizer adapters now type-hint against the Protocol instead of `EndToEndMetric` so adding new metric kinds (e.g. an A/B-test fitness) does not require touching the adapters. No abstract base class is added — the Protocol is purely structural so existing call sites need no change.

## Out of Scope (Follow-up Issues)

- **Anthropic / Claude judge sampler.** Meta-eval is provider-agnostic via `JudgeConfig`; adding a third sampler is a separate ~60-line PR.
- **Hand-labelled gold set loader.** The `LabelledSample.gold_response` and `LabelledSample.expected` fields already exist for this; the future loader just needs to populate them from a JSONL.
- **Noise-floor hook into `prompt_optimization/`.** Auto-warn when an agent-optimization trial's score delta is below the meta-eval-measured judge variance.
- **Batch-API mode** for the judge during meta-eval. Async ThreadPool is sufficient at the projected sample sizes.
- **Optimising the rubric items themselves.**
