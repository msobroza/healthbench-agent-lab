# healthbench-agent-lab — Project Specification

## 1. Project Overview

**Name:** `healthbench-agent-lab`

**One-liner:** Build, evaluate, and iterate on agentic AI systems for health question answering, benchmarked against HealthBench.

**GitLab Description:**
> Hands-on exploration of agentic system design and evaluation methodology. Covers HealthBench dataset analysis, multiple Google ADK agent architectures (single-agent, tool-augmented, multi-agent), and a rigorous evaluation framework with rubric-based LLM-as-judge scoring, experiment tracking via MLflow, and statistical comparison. Managed with uv.

**Context:** This project serves as preparation for a 75-minute technical interview focused on agentic AI systems for health, covering dataset exploration, agent development with Google ADK, evaluation framework expansion, and conceptual discussion on robust evaluation design.

---

## 2. Objectives

### 2.1 Dataset Analysis
- Load and explore the HealthBench dataset (5,000 conversations, 48,562 rubric criteria)
- Produce descriptive statistics and visualizations across key dimensions
- Identify patterns, failure modes, and areas where models struggle most
- Generate actionable insights that inform agent design decisions

### 2.2 Agent Architectures
- Build at least three agent variants using Google ADK, each with increasing capability
- Demonstrate understanding of ADK's agent model, tool system, and orchestration patterns
- Version prompts as structured YAML files with documented rationale

### 2.3 Evaluation Framework
- Connect HealthBench's rubric format to ADK's built-in eval criteria
- Track experiments across agent variants with MLflow
- Apply statistical methods to determine whether improvements are real
- Design and document a reproducible evaluation pipeline

---

## 3. HealthBench Dataset Reference

### 3.1 Dataset Summary

| Metric                          | Value   |
|---------------------------------|---------|
| Total conversations             | 5,000   |
| Unique rubric criteria          | 48,562  |
| Languages                       | 49      |
| Countries represented           | 60      |
| Medical specialties             | 26      |
| Contributing physicians         | 262     |
| Consensus subset size           | 3,671   |
| Hard subset size                | 1,000   |
| Avg criteria per conversation   | 11.5    |
| Avg turns per conversation      | 2.6     |
| Criteria range per conversation | 2–48    |

### 3.2 Conversation Structure

```json
{
  "conversation_id": "str",
  "language": "str (ISO code)",
  "specialty": "str",
  "user_persona": "str (patient | healthcare professional)",
  "turns": [
    {"role": "user|assistant", "content": "str", "turn_number": "int"}
  ],
  "rubric": {
    "criteria": [
      {
        "criterion_id": "str",
        "description": "str",
        "weight": "float (point value, range [-10, 10])",
        "category": "str (theme/axis)",
        "example_meets": "str",
        "example_fails": "str"
      }
    ],
    "max_score": "float"
  },
  "metadata": {
    "difficulty": "standard | hard",
    "variant": "consensus | hard",
    "language_family": "str",
    "sub_specialty": "str (optional)",
    "health_literacy_level": "low | medium | high | professional",
    "clinical_urgency": "routine | urgent | emergency",
    "cultural_context": "str",
    "validator_specialties": ["str"],
    "adversarial_tested": "bool"
  }
}
```

### 3.3 Scoring Formula

For a single example *i* with criteria *j = 1..M_i*:

```
score_i = sum(met_criteria_points) / sum(max(0, point_value) for all criteria)
```

- Point values range from -10 to +10 (negative = penalize undesirable behavior)
- Per-example score can be negative (if penalties exceed positive points met)
- Overall benchmark score = clipped mean of per-example scores, clipped to [0, 1]
- Scores can be stratified by theme or axis by filtering to relevant criteria only

### 3.4 Seven Themes

| Theme                              | Count   | % of dataset | Description |
|-------------------------------------|---------|-------------|-------------|
| Global health                       | 1,097   | 21.9%       | Adapting to varied healthcare contexts, resource availability, regional disease patterns |
| Responding under uncertainty        | 1,071   | 21.4%       | Recognizing and communicating uncertainty appropriately |
| Expertise-tailored communication    | 919     | 18.4%       | Matching depth and vocabulary to user expertise level |
| Context seeking                     | 594     | 11.9%       | Recognizing missing information and asking for it |
| Emergency referrals                 | 482     | 9.6%        | Recognizing emergencies and directing to care |
| Health data tasks                   | 477     | 9.5%        | Clinical documentation, decision support, research assistance |
| Response depth                      | 360     | 7.2%        | Matching response detail level to task complexity |

### 3.5 Five Axes

| Axis                    | Count   | % of criteria | Description |
|-------------------------|---------|--------------|-------------|
| Completeness            | 22,285  | 39%          | Includes all important information for safety and helpfulness |
| Accuracy                | 18,888  | 33%          | Factually correct, aligned with medical consensus |
| Context awareness       | 8,991   | 16%          | Responds appropriately to provided context and persona |
| Communication quality   | 4,522   | 8%           | Clear, well-structured, appropriate technical depth |
| Instruction following   | 2,551   | 4%           | Follows user instructions and constraints |

### 3.6 Dataset Subsets

**HealthBench Consensus (3,671 examples):** Contains only the 34 pre-defined consensus criteria that multiple physicians agreed are relevant. Higher physician validation, but measures fewer dimensions. Used for meta-evaluation since physician grades exist.

**HealthBench Hard (1,000 examples):** Selected for difficulty across frontier models. Selection method: compute scores for 5 models across providers (o3, Grok 3, Gemini 2.5 Pro, Claude 3.7 Sonnet, Llama 4 Maverick), filter out examples where no model scored positive, then take the 1,000 with lowest average score. No model scored above 32%.

### 3.7 Meta-Evaluation (Grader Trustworthiness)

HealthBench uses GPT-4.1 as the default model-based grader. Its reliability was validated against physician judgments:
- 60,896 meta-examples across 34 consensus criteria
- Model-physician agreement (Macro F1 = 0.709) is comparable to physician-physician agreement
- Model grader exceeds the average physician score in 5/7 themes
- Low run-to-run variability: std ≈ 0.002 across 16 repeated benchmark runs

---

## 4. Agent Architectures

### 4.1 Architecture A — Baseline Single Agent

**Goal:** Establish a performance floor with the simplest possible agent.

```
┌──────────────────────┐
│   User conversation  │
│        turns         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Baseline Agent     │
│   (gemini-2.0-flash) │
│   Minimal prompt     │
│   No tools           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Response           │
└──────────────────────┘
```

**ADK Implementation:**
- Single `Agent()` with basic health instruction
- Prompt version: `v1_baseline.yaml`
- Model: `gemini-2.0-flash`
- No tools, no sub-agents

**Expected strengths:** Reasonable accuracy and communication on straightforward questions.
**Expected weaknesses:** Incomplete responses, poor emergency detection, no context seeking, limited multilingual handling.

### 4.2 Architecture B — Tool-Augmented Agent

**Goal:** Improve completeness and accuracy by giving the agent access to medical reference tools.

```
┌──────────────────────┐
│   User conversation  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Tool Agent         │
│   (gemini-2.0-flash) │
│   Clinical prompt    │
│                      │
│   Tools:             │
│   ├── drug_reference │
│   ├── symptom_checker│
│   └── emergency_flag │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Response           │
└──────────────────────┘
```

**ADK Implementation:**
- Single `Agent()` with clinically-aware prompt
- Prompt version: `v2_clinical.yaml`
- Custom Python tool functions:
  - `drug_reference(drug_name)` → dosage, interactions, contraindications
  - `symptom_checker(symptoms)` → possible conditions with urgency level
  - `emergency_flag(description)` → binary urgent/non-urgent classification
- Model: `gemini-2.0-flash`

**Expected strengths:** Better accuracy on drug/treatment questions, improved emergency detection via tool use.
**Expected weaknesses:** May over-rely on tools, potentially slower, tool trajectory needs validation.

### 4.3 Architecture C — Multi-Agent Pipeline

**Goal:** Maximize coverage across all HealthBench themes via specialized sub-agents.

```
┌──────────────────────┐
│   User conversation  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Triage Agent       │
│   (router)           │
│   Classifies:        │
│   urgency, topic,    │
│   user expertise     │
└──────┬───────────────┘
       │ delegates to
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌──────────────────┐          ┌──────────────────┐
│  Emergency Agent │          │  General Health   │
│  High-urgency    │          │  Agent            │
│  Rapid escalation│          │  + medical tools  │
└──────┬───────────┘          └──────┬───────────┘
       │                              │
       ▼                              ▼
┌──────────────────────────────────────────┐
│   Reviewer Agent                         │
│   Checks: completeness, safety,          │
│   communication quality, context gaps    │
│   May request follow-up questions        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │   Response   │
            └──────────────┘
```

**ADK Implementation:**
- Root `Agent()` acts as triage/router using LLM-driven delegation
- Sub-agents: `EmergencyAgent`, `GeneralHealthAgent`
- `SequentialAgent` wraps the pipeline: triage → specialist → reviewer
- Reviewer agent applies a self-check rubric before emitting the final response
- Prompt version: `v3_structured.yaml`
- Model: `gemini-2.0-flash` (or `gemini-2.5-pro` for reviewer)

**Expected strengths:** Best emergency detection, context seeking via reviewer, expertise-tailored communication via triage classification.
**Expected weaknesses:** Higher latency, more complex to debug, potential for delegation errors.

---

## 5. Evaluation Framework

### 5.1 Evaluation Pipeline

```
HealthBench Dataset
       │
       ▼
┌──────────────────────────┐
│ 1. Sample Selection      │  Select N conversations (stratified by theme/difficulty)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Agent Inference       │  Run agent on conversation turns → generate response
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 3. Rubric Scoring        │  ADK rubrics_based_criterion: LLM judge scores each
│                          │  criterion as met (1.0) / not met (0.0)
│                          │  Majority vote over num_samples=3
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Score Computation     │  HealthBench formula: sum(met weights) / max_score × 100
│                          │  Aggregate by: overall, per-theme, per-axis
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 5. Experiment Logging    │  MLflow: params (agent, prompt, model), metrics (scores),
│                          │  artifacts (full results JSON)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 6. Statistical Comparison│  Paired bootstrap CI, t-test, Cohen's d, Bonferroni
│                          │  correction for multi-dimension comparisons
└──────────────────────────┘
```

### 5.2 ADK Evaluation Criteria Used

| Criterion                    | Purpose                                              | Configuration |
|------------------------------|------------------------------------------------------|---------------|
| `rubrics_based_criterion`    | Score each HealthBench rubric item via LLM-as-judge   | threshold: 0.6, num_samples: 3 |
| `tool_trajectory_avg_score`  | Verify tool-augmented agents call the right tools     | match_type: IN_ORDER, threshold: 0.8 |
| `final_response_match_v2`    | Semantic equivalence with reference (when available)  | threshold: 0.5 |
| `hallucinations_v1`          | Sentence-level grounding check                        | Optional, for safety analysis |
| `safety_v1`                  | Harmlessness scoring (requires Google Cloud project)  | Optional |

### 5.3 HealthBench Adapter

The adapter module (`evaluation/healthbench_adapter.py`) handles:
- Loading HealthBench JSONL files
- Converting conversation + rubric format into ADK eval cases
- Mapping HealthBench criteria to ADK's `rubrics_based_criterion` rubric format
- Computing HealthBench-style scores from ADK's per-criterion verdicts
- Aggregating scores by any metadata dimension (specialty, language, theme, difficulty, urgency)

### 5.4 Experiment Tracking

Each evaluation run is logged to MLflow with:

**Parameters:**
- `agent_name` — which architecture (baseline, tool, multi)
- `prompt_version` — which YAML prompt file
- `model` — LLM model string
- `sample_size` — number of conversations evaluated
- `timestamp` — ISO timestamp

**Metrics:**
- `overall_score` — aggregate HealthBench score (0–100)
- `{theme}/{theme_name}/mean` — per-theme breakdown
- `{axis}/{axis_name}/mean` — per-axis breakdown

**Artifacts:**
- Full per-conversation results JSON
- Prompt YAML used

### 5.5 Statistical Methods

All agent comparisons use paired evaluation (same conversations, different agents) to reduce variance.

| Method | When to use | Implementation |
|--------|-------------|----------------|
| Paired bootstrap CI (n=10,000) | Primary comparison method — determines if improvement CI excludes zero | `evaluation/stats.py::paired_bootstrap_ci()` |
| Paired t-test | Quick significance check, assumes roughly normal score differences | `evaluation/stats.py::paired_t_test()` |
| Cohen's d | Effect size — is the improvement practically meaningful? | `evaluation/stats.py::effect_size_cohens_d()` |
| Bonferroni correction | When testing across multiple themes/axes simultaneously | `evaluation/stats.py::bonferroni_correction()` |

**Decision rule:** An improvement is reported only when the 95% bootstrap CI for the paired difference excludes zero AND p < 0.05 after Bonferroni correction (when testing multiple dimensions).

---

## 6. Tech Stack

| Layer                  | Tool                                      | Version      | Purpose |
|------------------------|-------------------------------------------|-------------|---------|
| Package manager        | uv + uv_build                             | latest       | Fast dependency management, venv creation, build backend |
| Python                 | CPython                                   | 3.11+        | Runtime |
| Agent framework        | google-adk[eval]                          | ≥1.0.0       | Agent definition, tool integration, eval engine |
| Experiment tracking    | MLflow                                    | ≥2.18.0      | Log params, metrics, artifacts across runs |
| Data analysis          | pandas                                    | ≥2.2.0       | Dataset loading, slicing, aggregation |
| Statistics             | scipy                                     | ≥1.14.0      | t-tests, bootstrap, confidence intervals |
| Visualization          | seaborn + matplotlib                      | ≥0.13 / ≥3.9 | Score distributions, heatmaps, comparisons |
| Prompt management      | YAML files + Git                          | —            | Versioned prompts with rationale documentation |
| Eval orchestration     | pytest + pytest-asyncio                   | ≥8.3 / ≥0.24 | ADK AgentEvaluator integration, CI-ready |
| Linting                | ruff                                      | ≥0.8.0       | Code formatting and lint |
| Type checking          | mypy                                      | ≥1.13.0      | Static type analysis |
| Exploration            | JupyterLab                                | ≥4.3.0       | Interactive data analysis |

### 6.1 Excluded Tools (and Why)

| Tool | Reason for exclusion |
|------|---------------------|
| LangSmith / LangFuse | LangChain-ecosystem; mixing frameworks adds complexity without benefit when using ADK |
| FutureAGI / Arize | Production observability — overkill for a development/evaluation project |
| Vector databases (Chroma, Pinecone) | RAG is a possible extension but secondary to the core eval focus |
| PromptLayer | Over-engineered for this scope; Git-versioned YAML files suffice |
| Weights & Biases | Good alternative to MLflow but MLflow is local-first and zero-setup |

---

## 7. Project Structure

```
healthbench-agent-lab/
├── pyproject.toml              # uv project config (name: healthbench-agent), all dependencies
├── uv.lock                     # pinned dependency lockfile
├── .gitignore
├── LICENSE
├── README.md                   # setup guide, quick start, experiment workflow
├── SPEC.md                     # ← this document
├── CLAUDE.md                   # coding conventions for Claude Code
│
├── scripts/
│   └── clean_cache.sh          # removes __pycache__, .mypy_cache, .ruff_cache
│
├── src/
│   └── healthbench_agent/      # installable package (uv_build, src layout)
│       ├── __init__.py         # public API re-exports (all subpackages)
│       │
│       ├── domain/             # pure domain layer — no I/O, no external deps
│       │   ├── __init__.py     # re-exports all public types and scoring functions
│       │   ├── rubric.py       # RubricItem
│       │   ├── conversation.py # Message, MessageList, Conversation, ConversationMetadata
│       │   ├── sampler.py      # SamplerBase, SamplerResponse
│       │   ├── evaluation.py   # CriterionVerdict, SingleEvalResult, EvalResult, Eval
│       │   ├── dataset.py      # DatasetSubset, HealthBenchSample, HealthBenchDataset
│       │   └── scoring.py      # calculate_score, clip_score, aggregate_scores, stratified_scores
│       │
│       ├── dataset/            # I/O layer — download, load, split
│       │   ├── __init__.py     # re-exports loader and split_utils symbols
│       │   ├── loader.py       # download_dataset, download_all_datasets, load_dataset
│       │   └── split_utils.py  # sample_dataset, stratified_sample
│       │
│       └── analysis/           # statistics layer — registered analyses
│           ├── __init__.py     # re-exports registry runners and decorator
│           ├── registry.py     # @register_analysis, run_one, run_category, run_all
│           ├── utils.py        # series_stats, save_csv, DEFAULT_PERCENTILES
│           └── exploration.py  # 12 descriptive stats analyses (category: "exploration")
│
├── data/
│   └── healthbench/            # dataset files (gitignored, downloaded at setup)
│       ├── healthbench.jsonl           # main subset  (~5,000 samples)
│       ├── healthbench_hard.jsonl      # hard subset  (~1,000 samples)
│       └── healthbench_consensus.jsonl # consensus subset (~3,671 samples)
│
├── agents/                     # ADK agent definitions (not installed; via PYTHONPATH)
│   ├── baseline_agent/         # Architecture A: single agent, no tools
│   │   ├── __init__.py
│   │   └── agent.py            # root_agent definition
│   ├── tool_agent/             # Architecture B: single agent + medical tools
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── tools.py            # drug_reference, symptom_checker, emergency_flag
│   └── multi_agent/            # Architecture C: triage → specialist → reviewer
│       ├── __init__.py
│       ├── agent.py            # root_agent with SequentialAgent orchestration
│       └── sub_agents.py       # EmergencyAgent, GeneralHealthAgent, ReviewerAgent
│
├── prompts/                    # versioned YAML prompt files with documented rationale
│   ├── v1_baseline.yaml        # minimal instruction
│   ├── v2_clinical.yaml        # clinically-aware, tool-guidance
│   └── v3_structured.yaml      # structured output, multi-agent coordination
│
├── evaluation/                 # scoring pipeline and experiment tracking (via PYTHONPATH)
│   ├── __init__.py
│   ├── healthbench_adapter.py  # HealthBench ↔ ADK eval format conversion
│   ├── rubric_scorer.py        # map HealthBench rubrics → ADK rubrics_based_criterion
│   ├── stats.py                # bootstrap CI, t-test, Bonferroni, Cohen's d
│   ├── experiment_tracker.py   # MLflow logging wrapper
│   └── test_config.json        # ADK eval criteria configuration
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # descriptive stats, distributions
│   ├── 02_agent_comparison.ipynb       # side-by-side eval results
│   └── 03_evaluation_deep_dive.ipynb   # statistical analysis, failure modes
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # shared fixtures and make_sample factory (all suites)
    ├── domain/
    │   ├── __init__.py
    │   ├── test_models.py      # RubricItem, HealthBenchSample, HealthBenchDataset, CriterionVerdict
    │   └── test_scoring.py     # calculate_score, clip_score, aggregate_scores, stratified_scores
    ├── dataset/
    │   ├── __init__.py
    │   ├── test_downloader.py  # download_dataset, download_all_datasets
    │   ├── test_loader.py      # load_dataset
    │   └── test_split_utils.py # sample_dataset, stratified_sample
    └── analysis/
        ├── __init__.py
        ├── test_exploration.py # all 12 exploration analyses
        ├── test_registry.py    # @register_analysis, run_one, run_category, run_all
        └── test_utils.py       # series_stats, save_csv
```

---

## 8. Analysis Convention

Every analysis function in `analysis/` must follow this contract:

- Accepts one or more `HealthBenchDataset` instances — never loads or downloads data internally.
- **Returns a stats dict** (`dict[str, Any]`) keyed by dataset name, so callers always get per-dataset results regardless of whether artefacts are saved.
- Accepts an `output_dir: Path` argument for where exports are written.
- Accepts a `save: bool = False` parameter — **figures and artefacts are only written to disk when `save=True`**. When `False`, the function computes and returns results without any I/O side effects.
- Registered with `@register_analysis` where the decorator specifies the **category** (`"exploration"`, `"insights"`, or `"visualization"`) and the **list of dataset subsets** the function applies to (`"main"`, `"hard"`, `"consensus"`, or any combination).

### Categories

| Category | Module | Purpose |
|---|---|---|
| `exploration` | `healthbench_agent.analysis.exploration` | Descriptive stats: counts, distributions, language/specialty breakdowns |
| `insights` | `healthbench_agent.analysis.insights` | Cross-dimensional analysis: theme × axis, specialty × language, urgency × difficulty |
| `visualization` | `healthbench_agent.analysis.visualization` | Figures: score distributions, heatmaps, radar charts, criteria weight histograms |

### Decorator signature

```python
@register_analysis(
    name="score_distribution",
    category="visualization",
    datasets=["main", "hard", "consensus"],  # subsets this function applies to
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

### Running the registry

```python
from analysis.registry import run_all, run_one, run_category

# Run every registered analysis across all its declared datasets
run_all(datasets, output_dir=Path("exports/"), save=True)

# Run a single analysis by name
run_one("score_distribution", datasets, output_dir=Path("exports/"), save=True)

# Run all analyses in a category
run_category("exploration", datasets, output_dir=Path("exports/"), save=False)
```

---

## 9. Milestones & Deliverables

### Phase 0 — Core Package (done)
**Deliverables:**
- [x] `src/healthbench_agent/data_models.py` — domain models aligned with simple-evals (`RubricItem`, `HealthBenchSample`, `HealthBenchDataset`, `SamplerBase`, `SingleEvalResult`, `EvalResult`, `Eval`)
- [x] `src/healthbench_agent/scoring.py` — pure scoring functions (`calculate_score`, `clip_score`, `aggregate_scores`, `stratified_scores`)
- [x] `src/healthbench_agent/dataset/downloader.py` — `download_dataset()`, `download_all_datasets()`, URL/filename constants
- [x] `src/healthbench_agent/dataset/loader.py` — `load_dataset() → HealthBenchDataset`
- [x] `tests/test_data_models.py` — 30 tests covering `RubricItem`, `HealthBenchSample`, `HealthBenchDataset`, `CriterionVerdict`, `SamplerBase`, `SamplerResponse`
- [x] `tests/test_scoring.py` — 29 tests covering `calculate_score`, `clip_score`, `aggregate_scores`, `stratified_scores`

### Phase 1 — Dataset Analysis
**Deliverables:**
- [x] `src/healthbench_agent/analysis/registry.py` — `@register_analysis` decorator, `run_one`, `run_category`, `run_all`
- [x] `src/healthbench_agent/analysis/utils.py` — `series_stats`, `save_csv`, `DEFAULT_PERCENTILES`
- [x] `src/healthbench_agent/analysis/exploration.py` — 12 descriptive stats analyses: sample counts, prompt structure, rubric size, points distribution, positive vs penalty, score range, rubric tag frequency, rubric tags per sample, example tag frequency, tag prefix distribution, data quality, subset overlap
- [ ] `src/healthbench_agent/analysis/insights.py` — cross-dimensional breakdowns (theme × axis, specialty × language, urgency × difficulty)
- [ ] `src/healthbench_agent/analysis/visualization.py` — score distribution plots, theme/axis heatmap, criteria weight histogram
- [ ] `notebooks/01_dataset_exploration.ipynb` — complete walkthrough with findings
- [ ] Written summary of 3–5 actionable insights that inform agent design

### Phase 2 — Agent Development
**Deliverables:**
- [ ] `agents/baseline_agent/agent.py` — working baseline, runnable with `uv run adk web`
- [ ] `agents/tool_agent/` — agent + tools, verified tool calls via ADK tracing
- [ ] `agents/multi_agent/` — multi-agent pipeline with triage → specialist → reviewer
- [ ] `prompts/v1–v3.yaml` — versioned prompts with documented rationale
- [ ] Golden datasets captured from ADK web UI for each agent

### Phase 3 — Evaluation Framework
**Deliverables:**
- [ ] `evaluation/rubric_scorer.py` — end-to-end scoring pipeline using `calculate_score`
- [ ] `evaluation/stats.py` — all statistical comparison functions
- [ ] `evaluation/experiment_tracker.py` — MLflow integration
- [ ] `tests/test_*.py` — regression tests for all three agents
- [ ] `notebooks/02_agent_comparison.ipynb` — comparative analysis with CI plots
- [ ] `notebooks/03_evaluation_deep_dive.ipynb` — failure mode analysis, per-theme deep dives

### Phase 4 — Iteration & Documentation
**Deliverables:**
- [ ] At least 2 documented improvement iterations (change → eval → compare → decision)
- [ ] MLflow experiment history showing progression
- [ ] Final `README.md` with results summary
- [ ] This `SPEC.md` kept up to date

---

## 9. Key Design Decisions

### 9.1 Why uv over pip/poetry/conda?
- Extremely fast dependency resolution and install (10–100× faster than pip)
- Built-in venv management (`uv sync` creates and populates venv)
- Single `pyproject.toml` config — no separate lock file management
- Growing ecosystem adoption (recommended by ADK codelabs)

### 9.2 Why MLflow over W&B?
- Local-first: runs entirely on disk, no account or network needed
- Lighter footprint for a development/interview project
- Native Python API, minimal boilerplate
- Easy to export or compare runs

### 9.3 Why YAML prompts over a prompt management platform?
- Prompts are code — they belong in version control
- Each YAML file includes the instruction + documented rationale + expected behavior notes
- Git diff shows exactly what changed between prompt versions
- Linked to MLflow runs via `prompt_version` parameter

### 9.4 Why stratified sampling for eval?
- HealthBench themes are imbalanced (7.2% to 21.9%)
- Random sampling may under-represent Emergency Referrals and Response Depth
- Stratified sampling ensures every theme is evaluated proportionally
- Critical for drawing valid per-theme conclusions

---

## 10. Conceptual Discussion Topics

These are the kinds of questions expected in the interview discussion portion. Preparation notes:

### 10.1 How would you design a robust evaluation framework for health AI?
- Start with rubric-based evaluation (physician-validated criteria, not just vibes)
- Use LLM-as-judge with meta-evaluation to validate the grader
- Report confidence intervals, not point estimates
- Stratify by theme/axis to avoid masking failures in critical areas (e.g., emergency detection)
- Include negative criteria (penalize harmful behaviors, not just reward helpful ones)

### 10.2 What are the limitations of LLM-as-judge?
- Position bias (order of options matters in pairwise comparison)
- Verbosity bias (longer responses rated higher regardless of accuracy)
- Self-preference bias (models may rate their own outputs higher)
- Rubric ambiguity (vague criteria lead to inconsistent grading)
- Mitigation: multiple samples + majority vote, clear rubric phrasing, meta-evaluation against human judges

### 10.3 When can you trust an eval result?
- Low run-to-run variability (HealthBench: std ≈ 0.002 across 16 runs)
- Model-physician agreement comparable to physician-physician agreement (MF1 ≈ 0.71)
- Sufficient sample size per stratum for the conclusions drawn
- Paired evaluation to control for conversation difficulty
- Bonferroni correction when testing multiple hypotheses simultaneously

### 10.4 How do you avoid overfitting to the eval set?
- Use held-out test conversations not seen during development
- Track performance on HealthBench Hard separately (harder, more discriminating)
- Monitor for Goodhart's Law: optimizing for rubric scores ≠ optimizing for real-world health quality
- Periodically validate with new rubric criteria or physician spot-checks

### 10.5 What makes a good agent architecture for health?
- Emergency detection must be first-class (highest clinical stakes)
- Context seeking: the agent should ask for missing info rather than guess
- Expertise calibration: different communication for patients vs clinicians
- Uncertainty communication: explicit about what it doesn't know
- Safety review: a final check before emitting responses on high-stakes topics

---

## 11. Setup Instructions

```bash
# Prerequisites
# - Python 3.11+
# - uv installed: curl -LsSf https://astral.sh/uv/install.sh | sh
# - Gemini API key from https://aistudio.google.com/apikey

# Clone and setup
git clone https://gitlab.com/<namespace>/healthbench-agent-lab.git
cd healthbench-agent-lab
uv sync                        # creates venv + installs all deps
cp .env.example .env           # add GOOGLE_API_KEY

# Run baseline agent
uv run adk web agents/baseline_agent

# Run evaluation
uv run pytest tests/ -v

# Track experiments
uv run python -m evaluation.experiment_tracker \
    --agent baseline_agent --sample-size 100

# Launch notebooks
uv run jupyter lab
```

---

## 12. References

- **HealthBench Paper:** OpenAI, 2025. "HealthBench: A Benchmark for Health Question Answering with LLM Evaluation." arXiv:2505.08775
- **HealthBench Dataset Card:** Provided in project knowledge
- **Google ADK Documentation:** https://google.github.io/adk-docs/
- **ADK Python GitHub:** https://github.com/google/adk-python
- **ADK Evaluation Guide:** https://google.github.io/adk-docs/evaluate/
- **ADK Eval Criteria:** https://google.github.io/adk-docs/evaluate/criteria/
- **ADK Samples (incl. medical-pre-authorization):** https://github.com/google/adk-samples
- **uv Documentation:** https://docs.astral.sh/uv/
- **MLflow Documentation:** https://mlflow.org/docs/latest/