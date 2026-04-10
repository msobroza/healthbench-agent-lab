# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Layout

- **Project name (pyproject.toml):** `healthbench-agent` (installable package: `healthbench_agent`)
- **Build backend:** `uv_build` with `src/` layout
- **Installable package:** `src/healthbench_agent/` — importable as `import healthbench_agent`
  - `domain/` — pure domain layer (no I/O, no external deps)
    - `rubric.py` — `RubricItem`
    - `conversation.py` — `Message`, `MessageList`, `ConversationMetadata`, `Conversation`
    - `llm_client.py` — `LLMClient`, `LLMResponse`
    - `evaluation.py` — `CriterionVerdict`, `SingleEvalResult`, `EvalResult`, `Eval`
    - `dataset.py` — `DatasetSubset`, `HealthBenchSample`, `HealthBenchDataset`
    - `scoring.py` — pure scoring functions (`calculate_score`, `clip_score`, `aggregate_scores`, `stratified_scores`)
    - `judge.py` — `JudgeGrader` (ABC — grade conversation against rubric items)
    - `experiment.py` — `RunParams`, `RunMetrics` (experiment tracking metadata dataclasses)
    - `meta_evaluation.py` — `LabelledSample` (labelled HealthBench row for judge meta-eval), `MetricResults` (scores + graded counts + judge metadata), `SCHEMA_VERSION`
  - `agent/` — agent infrastructure (pipeline ABC, config, prompt rendering, tool registry, framework adapters)
    - `__init__.py` — re-exports `AgentPipeline`, `AgentNodeConfig`, `PlannerConfig`, `RootAgentPipelineConfig`, `FrameworkAdapter`, `create_pipeline`, `format_conversation`, `load_instruction`, `render_instruction`, `register_callback`, `get_callback`, `registered_callbacks`, `register_tool`, `get_tool`, `get_tools`, `registered_tools`
    - `agent_pipeline.py` — `AgentPipeline` (ABC — async generate response from conversation)
    - `config.py` — `PlannerConfig` (planner type + thinking params), `AgentNodeConfig` (recursive BaseModel, shared agent fields incl. `framework`, `orchestration`, `planner`, `include_contents`, `disallow_transfer_to_parent`, `disallow_transfer_to_peers`, `global_instruction`, `instruction_override`, callback names), `RootAgentPipelineConfig(AgentNodeConfig, BaseSettings)` (root config with env/YAML support)
    - `framework_adapter.py` — `FrameworkAdapter` (ABC — translates config into runnable `AgentPipeline`)
    - `factory.py` — `create_pipeline()` factory function (dispatches on `config.framework`)
    - `prompt.py` — `render_instruction()` (Jinja2 render of a raw template string), `load_instruction()` (load YAML and render), `format_conversation()` (formats `MessageList`)
    - `tool_registry.py` — `@register_tool` decorator, `get_tool()`, `get_tools()`, `registered_tools()`
    - `callback_registry.py` — `@register_callback` decorator, `get_callback()`, `registered_callbacks()`
    - `adapters/` — framework-specific adapter implementations
      - `adk_adapter.py` — `ADKFrameworkAdapter` (→ `FrameworkAdapter`), `ADKAgentPipeline` (→ `AgentPipeline`, shared `generate()`), `build_agent_node()` (recursive ADK agent tree builder, supports sequential/routing/loop/parallel orchestration, planners, callbacks, multi-agent control), `_resolve_instruction()` (prefers `config.instruction_override` over file load)
  - `dataset/` — HealthBench dataset I/O (→ domain)
    - `loader.py` — `download_dataset`, `download_all_datasets`, `load_dataset`, URL/filename/path constants
    - `split_utils.py` — `sample_dataset`, `stratified_sample` (sklearn-backed proportional sampling)
    - `extraction.py` — `extract_ideal_completion_text` (normalises HealthBench `ideal_completions_data` into a single gold response string)
  - `analysis/` — statistics layer (→ domain)
    - `registry.py` — `@register_analysis` decorator, `run_one`, `run_category`, `run_all`
    - `utils.py` — shared helpers (`series_stats`, `save_csv`, `DEFAULT_PERCENTILES`, `build_rubric_dataframe`, `build_sample_dataframe`)
    - `exploration.py` — 12 descriptive stats analyses (registered under `"exploration"`)
    - `insights.py` — 8 cross-cutting insight analyses (registered under `"insights"`)
    - `visualization.py` — 8 matplotlib visualizations (registered under `"visualization"`)
  - `llm_eval/` — LLM-as-judge evaluation (provider-agnostic, → domain)
    - `runner.py` — `EvalRunner` (depends on `JudgeGrader` and `AgentPipeline` abstractions)
    - `grading/` — prompt template loader + LLM-backed judge implementation
      - `config.py` — `JudgeConfig` (pydantic-settings `BaseSettings`), `EvalMode` enum
      - `judge.py` — `LLMJudgeGrader` (→ `JudgeGrader`), `create_judge()` factory, `GRADER_TEMPLATE`, `grade_sample()`, `format_conversation()`, `parse_grading_response()`, `load_grader_prompt()`
    - `clients/` — provider-specific `LLMClient` implementations and factory
      - `openai.py` — `OpenAIChatClient` (→ `LLMClient`)
      - `gemini.py` — `GeminiChatClient` (→ `LLMClient`)
      - `__init__.py` — `create_llm_client()` factory
    - `cache/` — on-disk verdict cache and cached-judge wrapper
      - `store.py` — `VerdictCache` (thread-safe sha256 loose-object store), `make_verdict_cache_key()` (pure free function)
      - `cached_judge.py` — `CachedJudgeGrader` (→ `JudgeGrader`, wraps any grader with cache reads/writes)
    - `cli/` — console script entry points
      - `track_experiment.py` — `track-experiment` wrapper that delegates to `evaluation.experiment_tracker._cli`
      - `meta_eval.py` — `meta-evaluate-judge` CLI for `run_meta_eval`
    - `meta_eval/` — judge meta-evaluation registry, metrics, runner, and result view
      - `registry.py` — `MetricLevel`, `MetricSpec`, `@register_meta_metric`, `get_meta_metric`, `registered_meta_metrics`
      - `metrics.py` — registered metric functions (`gold_score`, `cohens_kappa`, `krippendorff_alpha`, `calibration_curve`, `per_dimension_confusion`, `adversarial_accuracy`, `adversarial_prf1`, `per_criterion_metrics`)
      - `filters.py` — `axis_filter`, `metadata_filter`, `specialty_filter`, `EmptyFilterError`, `AXIS_TAG_PREFIX`
      - `fixtures.py` — `FakeJudge`, `demo_labelled_set`
      - `verdicts.py` — verdict-row builder helpers shared across runner/metrics
      - `runner.py` — `run_meta_eval` orchestrator (caching, parquet emission, MetricResultsView wrapping)
      - `api.py` — `meta_evaluate` high-level wrapper, `_load_subset_for_meta_eval`, `_build_judge_for_meta_eval`
      - `results_view.py` — `MetricResultsView` (REPL ergonomics around `MetricResults`)
      - `results_io.py` — `save_results`, `load_results` free functions
      - `results_plots.py` — `plot_calibration_curve`, `plot_dimension_confusion` free functions
  - `prompt_optimization/` — automatic prompt engineering (→ domain, agent, llm_eval). **Domain-agnostic**: any vertical-specific text (mutate/critique/refine templates, thinking styles) lives in YAML under `prompts/prompt_optimization/`, never in Python.
    - `optimizer.py` — `PromptOptimizer` ABC, `OptimizationResult`, `TrialRecord` (frozen dataclasses), `OptimizationMetric` Protocol (structural callable contract for fitness functions), `_TrialBudget` + `_BudgetExceededError` (shared cache/history/best/budget helper for adapters), `require_optional()` (optional-dep guard)
    - `config.py` — `BaseOptimizationConfig` (pydantic-settings, prefix `OPTIM_`, `dump_safe()`), `DSPyConfig`, `TextGradConfig`, `CritiqueRefineConfig` (with `prompt_path` field pointing to a YAML template file)
    - `metric.py` — `EndToEndMetric` (runs agent + judge end-to-end via `instruction_override`; supports `sample_filter`/`rubric_filter` kwargs), `JudgeAgreementMetric` (scores a candidate grader template by running `run_meta_eval` against a fixed labelled set), re-exports `EmptyFilterError`
    - `optimizer_registry.py` — `@register_prompt_optimizer` decorator, `create_prompt_optimizer()` factory, `get_optimizer_config_class()`, `registered_prompt_optimizers()`
    - `cli.py` — `optimize-prompt` CLI entry point; `--prompt-domain {agent,judge}` dispatches between agent-prompt optimization (default) and judge-prompt optimization (`_run_judge_optimization` → `_save_judge_yaml` writes `v2_optimized.yaml` next to the source grader YAML)
    - `adapters/` — per-framework optimizer implementations (lazy-imported)
      - `dspy_adapter.py` — `DSPyOptimizer` (COPRO/MIPROv2; uses `_TrialBudget` for cached per-instruction metric)
      - `textgrad_adapter.py` — `TextGradOptimizer` (text-gradient descent; uses `_TrialBudget` for trial bookkeeping)
      - `critique_refine_adapter.py` — `CritiqueRefineOptimizer` (mutation + critique loop, no external dep). Loads `_CritiqueRefinePrompts` (mutate/critique/refine Jinja2 templates + thinking-styles list) from `config.prompt_path` at construction time.
- **Working directories** (not installed, accessed via PYTHONPATH when using `uv run`):
  - `agents/` — ADK agent pipeline definitions (`*_pipeline.py`) and ADK entry points (subdirs with `agent.py` and colocated `*_pipeline.test.json` golden test cases)
  - `tools/` — medical reference tool modules (`drug_reference`, `symptom_checker`, `emergency_flag`)
  - `.github/workflows/` — GitHub Actions CI (`ci.yml` runs ruff check, ruff format check, mypy, pytest with coverage)
  - `evaluation/` — scoring, stats, experiment tracking (`experiment_tracker.py` CLI, `stats.py`)
  - `prompts/` — versioned YAML prompt files (subdirs: `llm_grader/`, `baseline_agent/`, `tool_agent/`, `multi_agent/`, `prompt_optimization/`)
  - `config/` — YAML configuration files for agent pipelines (`config/agents/*.yaml`)

## Commands

### Setup
```bash
uv sync                     # Install all dependencies
cp .env.example .env        # Set GOOGLE_API_KEY
uv run download-healthbench              # download all subsets
uv run download-healthbench --subset hard --force   # single subset, re-download
```

### Running Agents
```bash
uv run adk web agents/baseline_agent    # Web UI
uv run adk run agents/baseline_agent    # Terminal
```

### Evaluation
```bash
uv run adk eval agents/baseline_agent evaluation/test_config.json
uv run pytest tests/ -v
uv run track-experiment --agent-config config/agents/baseline_agent.yaml --sample-size 100
uv run track-experiment --agent-config config/agents/tool_agent.yaml --subset hard --seed 42
```

### Prompt Optimization
```bash
# Critique-refine (no external dep, ships with default templates)
uv run optimize-prompt --optimizer critique_refine \
    --agent-config config/agents/baseline_agent.yaml \
    --sample-size 20 --max-trials 10

# DSPy COPRO/MIPROv2 (requires `uv sync --extra optimization`)
uv run optimize-prompt --optimizer dspy --dspy-optimizer copro \
    --agent-config config/agents/baseline_agent.yaml --sample-size 20

# TextGrad (requires `uv sync --extra optimization`)
uv run optimize-prompt --optimizer textgrad \
    --agent-config config/agents/baseline_agent.yaml --sample-size 20 --steps 5

# Override critique-refine templates with a custom YAML
uv run optimize-prompt --optimizer critique_refine \
    --agent-config config/agents/baseline_agent.yaml \
    --prompt-path prompts/prompt_optimization/v1_critique_refine.yaml
```

### Meta-Evaluation (judge grader)
```bash
# Default run on the consensus subset, k=7 calibration passes
uv run meta-evaluate-judge run \
    --judge-config config/judges/openai_gpt41.yaml \
    --sample-size 100

# Show all registered meta-evaluation metrics
uv run meta-evaluate-judge list-metrics

# Discover metadata keys present in the consensus subset
uv run meta-evaluate-judge list-metadata-keys --subset consensus

# Regenerate metrics from an existing verdicts.parquet
uv run meta-evaluate-judge regenerate runs/openai/

# Compare two judges offline (optionally write markdown diff)
uv run meta-evaluate-judge compare runs/openai/ runs/gemini/

# Optimize the judge prompt itself (slice-restricted)
uv run optimize-prompt --prompt-domain judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --optimizer critique_refine --rubric-axis accuracy
```

### Code Quality
```bash
uv run ruff check .         # Lint
uv run ruff format .        # Format
uv run mypy .               # Type check
```

### Notebooks
```bash
uv run jupyter lab
```

## Architecture

Three agent architectures are built and compared against HealthBench:

1. **Baseline Agent** (`agents/baseline_pipeline.py`) — Single ADK agent (gemini-2.5-flash), no tools, minimal prompt (`prompts/v1_baseline.yaml`). Establishes performance floor.

2. **Tool-Augmented Agent** (`agents/tool_pipeline.py`) — Single ADK agent + custom tools (`tools/`): `drug_reference()`, `symptom_checker()`, `emergency_flag()`. Uses `prompts/v1_clinical.yaml`.

3. **Multi-Agent Pipeline** (`agents/multi_pipeline.py`) — SequentialAgent: Triage → Specialist (Emergency/GeneralHealth) → Reviewer. Uses `prompts/v1_structured.yaml`.

### Agent Config Features
- **Orchestration modes**: `sequential` (SequentialAgent), `routing` (LLM-driven delegation), `loop` (LoopAgent with `max_iterations`), `parallel` (ParallelAgent for concurrent execution)
- **Planners**: `PlannerConfig` with `type="builtin"` (Gemini thinking) or `type="plan_react"` (structured plan-action-reasoning)
- **Callbacks**: Registered by name via `@register_callback`, resolved at build time. Supports `before/after_agent_callback` (all agents), `before/after_model_callback`, `before/after_tool_callback` (LlmAgent only)
- **Multi-agent control**: `include_contents` (`"default"` or `"none"`), `disallow_transfer_to_parent`, `disallow_transfer_to_peers`, `global_instruction`

### Evaluation Pipeline (`evaluation/`)
- `healthbench_adapter.py` — Converts HealthBench conversations ↔ ADK eval format
- `rubric_scorer.py` — LLM-as-judge with 3-sample majority vote per criterion
- `stats.py` — Paired bootstrap CI (n=10,000), t-tests, Cohen's d, Bonferroni correction
- `experiment_tracker.py` — MLflow logging of params, metrics, and artifacts

### ADK Criteria Used
- `rubrics_based_criterion` — Scores each HealthBench rubric as met/not-met
- `tool_trajectory_avg_score` — Verifies tool calls in correct order
- `final_response_match_v2` — Semantic equivalence with reference responses

### Prompt Optimization (`src/healthbench_agent/prompt_optimization/`)
The module provides three optimizer backends behind a common `PromptOptimizer` ABC and a registry-based factory:
- **Critique-refine** — PromptWizard-style mutation + critique loop. **No external dependency**, uses the meta-LLM via the existing `LLMClient`. Templates and thinking-styles are loaded from `prompts/prompt_optimization/v1_critique_refine.yaml` so the Python is **fully domain-agnostic** — any vertical-specific phrasing belongs in YAML, never inside the adapter.
- **DSPy** — wraps COPRO/MIPROv2 teleprompters. The shared `_TrialBudget` helper caches scores per instruction (DSPy calls the metric per-example with one candidate at a time), enforces `max_trials` via `_BudgetExceededError`, and tracks the best prompt across compilation.
- **TextGrad** — wraps text-gradient descent. Also uses `_TrialBudget` for trial bookkeeping; loop is pre-capped at `min(steps, max_trials)`.

DSPy and TextGrad are optional dependencies (install with `uv sync --extra optimization`); the lazy import is guarded by `require_optional()` so the rest of the package works without them. Any optimizer scores candidate prompts via `EndToEndMetric`, which runs the configured agent pipeline + LLM judge end-to-end through `instruction_override`. To specialise the critique-refine optimizer for a non-health vertical, copy the YAML, edit the templates and thinking-styles, then point `CritiqueRefineConfig.prompt_path` (or `--prompt-path` on the CLI) at it.

## Coding Standards

### Naming Conventions (Clean Code)

- **Classes** — `PascalCase`, noun or noun phrase describing what the class *is*: `RubricCriterion`, `EvalResult`, `HealthBenchAdapter`. Avoid suffixes like `Manager`, `Processor`, `Helper`.
- **Functions / methods** — `snake_case`, verb or verb phrase describing what the function *does*: `criterion_score()`, `download_dataset()`, `flag_emergency()`. Boolean-returning functions start with `is_`, `has_`, or `can_`: `is_emergency()`, `has_missing_context()`.
- **Variables** — `snake_case`, descriptive nouns; length proportional to scope (short loop counters `i`, `c` are fine; module-level names must be explicit): `conversation_id`, `met_points`, `bootstrap_samples`.
- **Constants** — `UPPER_SNAKE_CASE`: `MAX_RUBRIC_WEIGHT`, `DEFAULT_SAMPLE_SIZE`.
- **Private symbols** — prefix with `_`: `_compute_max_points()`. Only use `__dunder__` for Python protocols.
- **No abbreviations** — prefer `criterion` over `crit`, `conversation` over `conv`, unless the abbreviation is a domain standard (e.g. `CI` for confidence interval).

### Docstrings (Google format)

Every public function, class, and module must have a Google-style docstring.

```python
def criterion_score(
    criteria: list[RubricCriterion],
    verdicts: list[CriterionVerdict],
) -> float:
    """Compute the HealthBench score for a single conversation.

    Applies the formula: sum(met weights) / sum(max(0, weight)).
    Returns a value in (-inf, 1.0]; negative when penalties dominate.

    Args:
        criteria: All rubric criteria for the conversation.
        verdicts: LLM-judge verdicts, one per criterion.

    Returns:
        Score in (-inf, 1.0]. Returns 0.0 if max_points is zero.

    Raises:
        ValueError: If criteria and verdicts reference different criterion IDs.
    """
```

Rules:
- One-line summary on the first line, no period if it fits on one line.
- Blank line before `Args:` / `Returns:` / `Raises:` sections.
- `Args:` — one entry per parameter; type annotation in the signature, not repeated here.
- `Returns:` — describe the value and its range or shape.
- `Raises:` — only document exceptions the caller needs to handle; omit internal-only ones.
- Module docstring: one paragraph stating *what* the module contains and its role in the dependency graph.

### SOLID Principles

**Single Responsibility** — Each class/module has one reason to change. `scoring.py` only contains the scoring formula; `types.py` only contains data definitions. Do not add I/O, logging, or side effects to either.

**Open / Closed** — Extend behaviour through new functions or subclasses rather than modifying existing ones. Add a new scoring variant as a new function alongside `criterion_score`, not as a flag inside it.

**Liskov Substitution** — Subclasses must honour the contract of their parent. If you subclass `EvalResult`, every function that accepts `EvalResult` must work with the subclass unchanged.

**Interface Segregation** — Prefer narrow, focused function signatures over large god-objects. A function that needs only `criteria` and `verdicts` should not receive the entire `Conversation`.

**Dependency Inversion** — High-level modules (`evaluation/`, `agents/`) depend on abstractions in `healthbench_agent` (types, scoring). They must not import from each other. Concretions (MLflow, ADK, pandas) belong in the outer layers, never in `src/healthbench_agent/`.

## Analysis Convention

Every analysis in `analysis/` must follow this pattern:

1. **One function per analysis** — each analysis is a standalone function with a clear verb name: `plot_score_distribution()`, `compute_specialty_breakdown()`.
2. **Datasets as input** — every analysis function accepts a `list[HealthBenchDataset]`. It must not load or download data internally.
3. **Returns per-dataset stats** — return type is `dict[str, Any]` keyed by `dataset.subset`, so callers always get structured results regardless of whether artefacts are saved.
4. **Output directory as input** — every analysis function accepts an `output_dir: Path` argument. It must not hardcode paths.
5. **`save` flag controls I/O** — every analysis function accepts `save: bool = False`. Figures and artefacts are **only written to disk when `save=True`**. When `False`, the function computes and returns results without any side effects.
6. **Registered with category and datasets** — use `@register_analysis(name, category, datasets)` where `category` is one of `"exploration"`, `"insights"`, or `"visualization"`, and `datasets` is the list of subsets the function applies to.

### Registry pattern

`healthbench_agent.analysis.registry` owns the registry. Use the decorator to register:

```python
from healthbench_agent.analysis import register_analysis

@register_analysis(
    name="score_distribution",
    category="visualization",
    datasets=["main", "hard", "consensus"],
)
def plot_score_distribution(
    datasets: list[HealthBenchDataset],
    output_dir: Path,
    save: bool = False,
) -> dict[str, Any]:
    """Plot the score distribution for each provided dataset.

    Args:
        datasets: Loaded HealthBench datasets to analyse, one per subset.
        output_dir: Directory where output files are saved when save=True.
        save: Write figures and artefacts to disk when True. When False,
            computes and returns stats without any I/O side effects.

    Returns:
        Mapping of dataset subset name to its computed stats. Artefact paths
        are included under the key 'paths' when save=True.
    """
    results = {}
    for dataset in datasets:
        fig, stats = _build_figure(dataset)
        if save:
            path = output_dir / f"score_distribution_{dataset.subset}.png"
            fig.savefig(path)
            stats["paths"] = [str(path)]
        results[dataset.subset] = stats
    return results
```

Running analyses:

```python
from healthbench_agent.analysis import run_all, run_one, run_category

run_all(datasets, output_dir=Path("exports/"), save=True)
run_one("score_distribution", datasets, output_dir=Path("exports/"), save=True)
run_category("exploration", datasets, output_dir=Path("exports/"), save=False)
```

### Rules
- Return `dict[str, Any]` keyed by `dataset.subset` — always return stats even when `save=False`.
- Never mutate a dataset inside an analysis function.
- Analyses are order-independent; do not call one analysis from another.
- Keep each function focused: if a function produces more than one unrelated artefact type, split it.

### DataFrame-first methodology

Analysis functions must convert domain objects to pandas DataFrames as early as possible, then express all logic as DataFrame operations (groupby, pivot_table, agg, filter). Two shared builders in `analysis/utils.py` handle the conversion:

- **`build_rubric_dataframe(dataset)`** — one row per rubric item. Columns: `prompt_id`, `theme`, `axis`, `points`, `criterion`, `criterion_length`. Use for any analysis that operates at the rubric-item level (axis breakdowns, point distributions, penalty heatmaps).
- **`build_sample_dataframe(dataset)`** — one row per sample. Columns: `prompt_id`, `theme`, `rubric_size`, `total_possible_points`, `total_penalty_points`, `penalty_mass_ratio`, `prompt_char_length`, `num_turns`, `num_user_turns`. Use for any analysis at the sample level (theme counts, complexity comparisons, score ceilings).

Rules:
- Call the builder once per dataset, at the top of the analysis function. Do not iterate over `dataset.samples` manually to build rows — that logic belongs in the builder.
- Derive additional columns from existing builder columns using vectorised pandas operations rather than Python loops: `df["turn_type"] = np.where(df["num_turns"] > 2, "multi", "single")`.
- Tag extraction (theme, axis) is handled inside the builders via `tag_value()`. Analysis functions should never call `tag_value()` directly.
- For analyses that need columns not provided by either builder, add the column to the builder if it is reusable, or derive it inline from existing columns if it is one-off.

```python
from healthbench_agent.analysis.utils import build_rubric_dataframe, build_sample_dataframe

# Rubric-level analysis: positive share by axis
df = build_rubric_dataframe(dataset)
pos_by_axis = df[df["points"] > 0].groupby("axis")["points"].sum()
total_by_axis = df["points"].abs().groupby(df["axis"]).sum()
positive_share = (pos_by_axis / total_by_axis).sort_values()

# Sample-level analysis: penalty ratio by theme
df = build_sample_dataframe(dataset)
theme_stats = df.groupby("theme")["penalty_mass_ratio"].agg(["mean", "median"])
```

## Testing Convention (pytest)

### Structure
- One test file per module: `test_scoring.py` mirrors `scoring.py`.
- One test function per behaviour, not per function — a single function may need many tests.
- Name tests as sentences: `test_calculate_score_returns_none_when_no_positive_criteria`.
- Group related tests in a class only when they share fixtures or setup.

### Fixtures over setup/teardown
```python
@pytest.fixture
def sample_rubric_items():
    return [RubricItem("States emergency referral", 1.0, ["emergency"])]

def test_calculate_score_full_credit(sample_rubric_items):
    verdicts = [CriterionVerdict("States emergency referral", criteria_met=True)]
    assert calculate_score(sample_rubric_items, verdicts) == 1.0
```

### Parametrize to avoid repetition
```python
@pytest.mark.parametrize("score,expected", [
    (1.0,  1.0),   # already in range
    (-0.5, 0.0),   # negative clipped to 0
    (1.5,  1.0),   # above 1 clipped to 1
])
def test_clip_score(score, expected):
    assert clip_score(score) == expected
```

### Assert on behaviour, not implementation
```python
# bad — tests internal structure
assert len(result._buckets) == 3

# good — tests the contract
assert result["emergency"] == pytest.approx(0.75, abs=1e-6)
```

### Selecting test case scenarios — ZOMBIES

Use the ZOMBIES heuristic to identify what to test for each function:

| Letter | Meaning | Example for `calculate_score` |
|---|---|---|
| **Z** | Zero | Empty rubric → returns `None` |
| **O** | One | Single criterion met / not met |
| **M** | Many | Multiple criteria, mixed verdicts |
| **B** | Boundaries | All met, all penalties, score exactly 0 |
| **I** | Interface | Mismatched verdict/rubric length raises error |
| **E** | Exceptions | `total_possible_points == 0` edge case |
| **S** | Simple scenarios | Typical case: 3 criteria, 2 met |

### Project-specific test priorities
- **Scoring functions** — boundary values: all met, none met, only penalties, `max_points == 0`.
- **`HealthBenchSample.from_dict`** — real JSONL rows, missing optional fields, malformed rubric.
- **Analysis functions** — `save=False` produces no files and returns correct dict keys per subset.
- **Registry** — `run_category("exploration", ...)` calls only exploration functions.

### Coverage
Run coverage after every test session and enforce a minimum of **80%** per module:

```bash
uv run pytest tests/ --cov=src/healthbench_agent --cov-report=term-missing
```

- If any module falls below 80%, add tests before merging.
- Aim for 100% on pure functions (`scoring.py`) — they have no I/O to mock and no excuse for gaps.
- Uncovered lines must be either tested or explicitly excluded with `# pragma: no cover` (only for unreachable abstract stubs like `raise NotImplementedError`).

### Rules
- Tests must be fast and deterministic — no network calls, fix random seeds with `random.seed(0)`.
- Test the public contract, not private helpers — test privates through the public API.
- Do not test Python built-ins or third-party library behaviour.
- A test that fails must point to exactly one broken behaviour.

## Logging Convention

### Use the standard `logging` module — never `print`
```python
import logging

logger = logging.getLogger(__name__)  # one logger per module, named by module path

logger.info("Downloading dataset subset %r", subset)
logger.warning("Skipping sample %s — missing required field", prompt_id)
logger.error("Download failed for %r: %s", url, exc)
```

### Rules
- **One logger per module**, named `__name__` — resolves to `healthbench_agent.io.downloader`, `evaluation.rubric_scorer`, etc., giving free namespace hierarchy.
- **Use `%s` formatting, not f-strings** — the string is only built if the log level is enabled.
- **Configure once at the entry point** — never call `logging.basicConfig()` or add handlers inside `src/healthbench_agent/` or any library module. Configure only in `scripts/`, notebooks, or `__main__` entry points.
- **Never log inside `src/healthbench_agent/`** — pure functions must have no side effects.
- **Log exceptions with `logger.exception()`** inside `except` blocks — it automatically attaches the traceback.

```python
try:
    path = download_dataset(subset)
except urllib.error.URLError as exc:
    logger.exception("Failed to download %r subset", subset)
    raise
```

### Log levels
| Level | When to use |
|---|---|
| `DEBUG` | Internal state useful during development |
| `INFO` | Normal progress milestones (`"Downloaded 5000 samples"`) |
| `WARNING` | Unexpected but recoverable (`"Skipping malformed row"`) |
| `ERROR` | Operation failed, continuing if possible |
| `CRITICAL` | Application cannot continue |

### Per-module guidance
- `healthbench_agent.io.downloader` — `INFO` for download start/finish; `WARNING` for skipped files.
- `healthbench_agent.io.dataset_loader` — `INFO` on load completion.
- `evaluation/` — `INFO` per scored sample batch, `DEBUG` for individual verdict details.
- `agents/` — `DEBUG` for tool calls, `INFO` for agent run start/finish.

## Phase 2 — Agent Development Subtasks

See `AGENT_DECISIONS.md` for exhaustive pros/cons and design rationale for each decision.

### Subtask 2.1 — Baseline Agent
- `prompts/v1_baseline.yaml` — minimal health instruction with version/rationale metadata
- `agents/baseline_agent/__init__.py` — empty, makes directory a package
- `agents/baseline_pipeline.py` — `root_agent = LlmAgent(name, model, instruction)` loaded from YAML
- Tests: verify prompt loading, agent configuration, root_agent export

### Subtask 2.2 — Medical Tools + Clinical Prompt
- `prompts/v1_clinical.yaml` — clinically-aware prompt with tool-usage guidance (Jinja2 `{{ conversation }}` template)
- `tools/__init__.py`
- `tools/drug_reference.py` — `@register_tool("drug_reference")`, drug lookup function
- `tools/symptom_checker.py` — `@register_tool("symptom_checker")`, symptom analysis function
- `tools/emergency_flag.py` — `@register_tool("emergency_flag")`, emergency detection function
- `tools/tools.py` — imports all tool modules (triggers registration), re-exports
- Tests: each tool function returns correct dict shape, edge cases, keyword matching

### Subtask 2.3 — Tool-Augmented Agent
- `agents/tool_pipeline.py` — `root_agent = LlmAgent(name, model, instruction, tools=[...])` with v1_clinical prompt
- Tests: agent has tools attached, instruction loaded correctly

### Subtask 2.4 — Multi-Agent Prompts + Sub-Agents
- `prompts/v1_structured.yaml` — multi-agent coordination prompts (triage, emergency, general, reviewer, coordinator)
- `config/agents/multi_agent.yaml` — full pipeline config with recursive `sub_agents` definitions
- `agents/multi_agent/__init__.py`
- Tests: each sub-agent has correct model/tools/output_key configuration

### Subtask 2.5 — Multi-Agent Orchestration
- `agents/multi_pipeline.py` — `ADKAgentPipeline` builds agent tree from `RootAgentPipelineConfig.sub_agents`
- Config-driven: recursive `AgentNodeConfig` with `orchestration` (sequential/routing) and `condition`
- Pipeline: triage → coordinator(routing: emergency, general_health) → reviewer
- Tests: routing vs sequential orchestration, condition in description, prompt_path inheritance, unsupported orchestration raises

### Subtask 2.6 — Golden Datasets
- `agents/baseline_agent/baseline_pipeline.test.json` — golden test cases
- `agents/tool_agent/tool_pipeline.test.json` — golden test cases with tool trajectories
- `agents/multi_agent/multi_pipeline.test.json` — golden test cases with delegation
- `evaluation/test_config.json` — ADK eval criteria thresholds

## Phase 3 — Evaluation Framework Subtasks

### Subtask 3.1 — Grader Module
- `src/healthbench_agent/llm_eval/__init__.py`
- `src/healthbench_agent/llm_eval/grading/judge.py` — `GRADER_TEMPLATE`, `grade_sample()`, `format_conversation()`, `parse_grading_response()`, `load_grader_prompt()`
- `prompts/llm_v1_llm_grader.yaml` — verbatim simple-evals grader template with Jinja2 placeholders
- Tests: template rendering, response parsing, grade_sample with mocked sampler

### Subtask 3.2 — LLM Clients
- `src/healthbench_agent/llm_eval/clients/openai.py` — `OpenAIChatClient`
- `src/healthbench_agent/llm_eval/clients/gemini.py` — `GeminiChatClient`
- `src/healthbench_agent/llm_eval/clients/__init__.py` — `create_llm_client()` factory
- `src/healthbench_agent/llm_eval/grading/config.py` — `JudgeConfig` (pydantic-settings)
- Tests: both clients mocked, JudgeConfig validation, env var override

### Subtask 3.3 — Eval Runner
- `src/healthbench_agent/llm_eval/runner.py` — `EvalRunner` with `mode="async"` (ThreadPool) and `mode="batch"` (OpenAI Batch API)
- Tests: async mode with mocked samplers, batch mode with mocked API

### Subtask 3.4 — Evaluation Pipeline
- `evaluation/__init__.py`
- `evaluation/rubric_scorer.py` — adapter wiring `EvalRunner` → `calculate_score` → MLflow
- `evaluation/stats.py` — `paired_bootstrap_ci()`, `paired_t_test()`, `effect_size_cohens_d()`, `bonferroni_correction()`
- `evaluation/experiment_tracker.py` — MLflow logging wrapper
- Tests: stats functions, experiment tracker with mocked MLflow

### Subtask 3.5 — Analysis Notebooks
- `notebooks/02_agent_comparison.ipynb` — comparative analysis with CI plots
- `notebooks/03_evaluation_deep_dive.ipynb` — failure mode analysis, per-theme deep dives

## Key Conventions

**Prompt Versioning** — All system prompts live in `prompts/` as YAML with documented rationale. Always link prompt version to MLflow runs. This applies to **every** subsystem that consumes a prompt: agent instructions (`baseline_agent/`, `tool_agent/`, `multi_agent/`), the LLM judge (`llm_grader/`), and the prompt optimizer (`prompt_optimization/`). Python modules must never embed vertical-specific (medical, legal, financial, etc.) phrasing — keep them domain-agnostic and load any specialised text from a YAML file under `prompts/`.

**Paired Evaluation** — Always run the same sampled conversations across all agents being compared to reduce variance.

**Stratified Sampling** — Sample HealthBench data proportionally by theme (7 themes) and axis (5 axes) to avoid masking failures in critical areas like emergency detection.

**Statistical Decision Rule** — An agent improvement is considered significant only when bootstrap CI excludes zero AND p < 0.05 (with Bonferroni correction for multiple comparisons).

**HealthBench Subsets** — `consensus` (3,671 conversations, physician-validated) vs `hard` (1,000 conversations, max model score ~32%). Use `hard` to stress-test.
