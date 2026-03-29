# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Layout

- **Project name (pyproject.toml):** `healthbench-agent` (installable package: `healthbench_agent`)
- **Build backend:** `uv_build` with `src/` layout
- **Installable package:** `src/healthbench_agent/` — importable as `import healthbench_agent`
  - `models.py` — domain dataclasses (`Conversation`, `Rubric`, `RubricCriterion`, `EvalResult`, …)
  - `scoring.py` — pure HealthBench scoring functions (`criterion_score`, `aggregate_scores`, …)
- **Working directories** (not installed, accessed via PYTHONPATH when using `uv run`):
  - `analysis/` — dataset loading and visualization
  - `agents/` — ADK agent definitions
  - `evaluation/` — scoring, stats, experiment tracking
  - `prompts/` — versioned YAML prompt files

## Commands

### Setup
```bash
uv sync                     # Install all dependencies
cp .env.example .env        # Set GOOGLE_API_KEY
uv run python -c "from analysis.exploration import download_dataset; download_dataset()"
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
uv run python -m evaluation.experiment_tracker --agent baseline_agent --sample-size 100
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

1. **Baseline Agent** (`agents/baseline_agent/`) — Single ADK agent (gemini-2.0-flash), no tools, minimal prompt (`prompts/v1_baseline.yaml`). Establishes performance floor.

2. **Tool-Augmented Agent** (`agents/tool_agent/`) — Single ADK agent + custom tools: `drug_reference()`, `symptom_checker()`, `emergency_flag()`. Uses `prompts/v2_clinical.yaml`.

3. **Multi-Agent Pipeline** (`agents/multi_agent/`) — SequentialAgent: Triage → Specialist (Emergency/GeneralHealth) → Reviewer. Uses `prompts/v3_structured.yaml`.

### Evaluation Pipeline (`evaluation/`)
- `healthbench_adapter.py` — Converts HealthBench conversations ↔ ADK eval format
- `rubric_scorer.py` — LLM-as-judge with 3-sample majority vote per criterion
- `stats.py` — Paired bootstrap CI (n=10,000), t-tests, Cohen's d, Bonferroni correction
- `experiment_tracker.py` — MLflow logging of params, metrics, and artifacts

### ADK Criteria Used
- `rubrics_based_criterion` — Scores each HealthBench rubric as met/not-met
- `tool_trajectory_avg_score` — Verifies tool calls in correct order
- `final_response_match_v2` — Semantic equivalence with reference responses

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

## Key Conventions

**Prompt Versioning** — All system prompts live in `prompts/` as YAML with documented rationale. Always link prompt version to MLflow runs.

**Paired Evaluation** — Always run the same sampled conversations across all agents being compared to reduce variance.

**Stratified Sampling** — Sample HealthBench data proportionally by theme (7 themes) and axis (5 axes) to avoid masking failures in critical areas like emergency detection.

**Statistical Decision Rule** — An agent improvement is considered significant only when bootstrap CI excludes zero AND p < 0.05 (with Bonferroni correction for multiple comparisons).

**HealthBench Subsets** — `consensus` (3,671 conversations, physician-validated) vs `hard` (1,000 conversations, max model score ~32%). Use `hard` to stress-test.
