# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Key Conventions

**Prompt Versioning** — All system prompts live in `prompts/` as YAML with documented rationale. Always link prompt version to MLflow runs.

**Paired Evaluation** — Always run the same sampled conversations across all agents being compared to reduce variance.

**Stratified Sampling** — Sample HealthBench data proportionally by theme (7 themes) and axis (5 axes) to avoid masking failures in critical areas like emergency detection.

**Statistical Decision Rule** — An agent improvement is considered significant only when bootstrap CI excludes zero AND p < 0.05 (with Bonferroni correction for multiple comparisons).

**HealthBench Subsets** — `consensus` (3,671 conversations, physician-validated) vs `hard` (1,000 conversations, max model score ~32%). Use `hard` to stress-test.
