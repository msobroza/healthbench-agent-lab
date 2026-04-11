# 🏥 healthbench-agent-lab

> **Building, evaluating, and iterating on agentic AI systems for health question answering — benchmarked against [HealthBench](https://github.com/openai/healthbench).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)

---

## 🎯 What Is This?

A research experimentation platform for designing, benchmarking, and comparing **AI health agents** against HealthBench — a dataset of **5,000 physician-validated medical conversations** spanning 49 languages and 26 medical specialties.

Three agent architectures are built and rigorously compared:

| 🏗️ Architecture | Description | Purpose |
|---|---|---|
| 🟢 **Baseline** | Single LLM agent, minimal prompt, no tools | Performance floor |
| 🔧 **Tool-Augmented** | Same agent + drug reference, symptom checker, emergency flag | Measures tool impact |
| 🤝 **Multi-Agent** | Triage → Specialist routing → Review pipeline | Tests orchestration gains |

---

## ✨ Key Features

- 🧪 **Three competing agent architectures** — baseline → tool-augmented → multi-agent pipeline
- 📊 **28 registered analyses** — exploration, cross-cutting insights, and visualizations
- ⚖️ **LLM-as-judge evaluation** — provider-agnostic rubric grading (OpenAI / Gemini)
- 🪄 **Automatic prompt optimization** — pluggable backends (DSPy COPRO/MIPROv2, TextGrad, critique-refine) behind a single CLI
- 📈 **Statistical rigor** — bootstrap CI, paired t-tests, Cohen's d, Bonferroni correction
- 🔬 **Experiment tracking** — MLflow logging with prompt fingerprints (SHA-256)
- 📝 **Versioned prompts** — Git-tracked YAML with Jinja2 templating and metadata
- ⚙️ **Config-driven pipelines** — YAML configs define agent trees with orchestration modes
- 🧩 **Clean architecture** — pure domain layer, dependency inversion, SOLID principles
- 🌐 **Domain-agnostic optimization core** — vertical-specific text lives in YAML, never in Python

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| 📦 Package Manager | [uv](https://docs.astral.sh/uv/) + uv_build |
| 🤖 Agent Framework | [Google ADK](https://google.github.io/adk-docs/) (`google-adk`) |
| ✅ Built-in Eval | ADK eval (rubrics, trajectory, response) |
| 📈 Experiment Tracking | [MLflow](https://mlflow.org/) |
| 📊 Data Analysis | pandas, scipy, seaborn, matplotlib |
| 📝 Prompt Management | Git-versioned YAML in `prompts/` |
| 🧪 Testing | pytest + pytest-cov (80% minimum) |
| 🔍 Code Quality | ruff, mypy |
| 📓 Exploration | Jupyter notebooks |

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Gemini API key ([get one here](https://aistudio.google.com/apikey))

### ⚡ Setup

```bash
# Clone the repo
git clone https://github.com/<your-namespace>/healthbench-agent-lab.git
cd healthbench-agent-lab

# Install all dependencies
uv sync

# Download HealthBench datasets
uv run download-healthbench                           # all subsets
uv run download-healthbench --subset hard --force     # single subset
```

### 🤖 Run an Agent

```bash
# Web UI (interactive chat)
uv run adk web agents

# Terminal mode
uv run adk run agents
```

### 🧪 Run Evaluation

```bash
# ADK golden-test eval
uv run adk eval agents/baseline_agent evaluation/test_config.json

# Full test suite
uv run pytest tests/ -v

# HealthBench evaluation with MLflow experiment tracking
uv run track-experiment \
    --agent-config config/agents/baseline_agent.yaml \
    --sample-size 100 --seed 42

# Compare agents on the hard subset
uv run track-experiment \
    --agent-config config/agents/tool_agent.yaml \
    --subset hard --seed 42
```

### 🪄 Optimize a Prompt

```bash
# Critique-refine — no extra deps, ships with a domain-agnostic default template
uv run optimize-prompt \
    --optimizer critique_refine \
    --agent-config config/agents/baseline_agent.yaml \
    --sample-size 20 --max-trials 10

# DSPy COPRO/MIPROv2 — requires the optional `optimization` extra
uv sync --extra optimization
uv run optimize-prompt \
    --optimizer dspy --dspy-optimizer copro \
    --agent-config config/agents/baseline_agent.yaml --sample-size 20

# TextGrad — also under the `optimization` extra
uv run optimize-prompt \
    --optimizer textgrad \
    --agent-config config/agents/baseline_agent.yaml \
    --sample-size 20 --steps 5
```

#### 🎯 Optimizing one prompt inside a multi-agent pipeline

Multi-agent pipelines contain several prompts — one per sub-agent, each
identified by its `prompt_key` in the YAML file. Pass `--target-agent NAME`
to pick exactly which sub-agent's instruction to optimize. Every other
sub-agent keeps loading its own instruction from the YAML file on disk,
so only the targeted node varies across trials.

```bash
# Optimize just the reviewer in the multi-agent pipeline
uv run optimize-prompt \
    --optimizer critique_refine \
    --agent-config config/agents/multi_agent.yaml \
    --target-agent reviewer_agent \
    --sample-size 20 --max-trials 10

# Optimize the triage classifier instead
uv run optimize-prompt \
    --optimizer critique_refine \
    --agent-config config/agents/multi_agent.yaml \
    --target-agent triage_agent
```

The output YAML is written next to the source prompt file as
`v2_optimized_<target>.yaml`, using the target's original `prompt_key`
(e.g. `reviewer_instruction`), so diffing or merging back into the
source file is straightforward.

> ⚠️ **When `--target-agent` is required.** A root agent whose
> `orchestration` is `sequential`, `loop`, or `parallel` builds a pure
> composite (`SequentialAgent` / `LoopAgent` / `ParallelAgent`) with no
> `instruction` field, so a root-level override is silently dropped.
> The CLI refuses to run in that case and lists the available sub-agent
> names. Targetable nodes are either **leaf agents** or non-leaf agents
> with `orchestration: routing`.

> ✨ The critique-refine optimizer is **domain-agnostic** — its mutate / critique / refine
> templates and thinking-styles all live in `prompts/prompt_optimization/v1_critique_refine.yaml`.
> Specialise it for any vertical by copying the YAML, editing the templates, and pointing
> `--prompt-path` at your file:
>
> ```bash
> uv run optimize-prompt --optimizer critique_refine \
>     --agent-config config/agents/baseline_agent.yaml \
>     --prompt-path prompts/prompt_optimization/v1_legal.yaml
> ```

### 📓 Explore the Dataset

```bash
uv run jupyter lab notebooks/01_dataset_exploration.ipynb
```

### 🔍 Code Quality

```bash
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy .                # Type check
uv run pytest tests/ --cov=src/healthbench_agent --cov-report=term-missing
```

---

## 🏛️ Architecture

### 📐 Clean Layered Design

```
┌─────────────────────────────────────────────────────────┐
│  🎯 Entry Points                                       │
│  agents/ · evaluation/ · notebooks/ · tools/            │
├─────────────────────────────────────────────────────────┤
│  🤖 Agent Infrastructure    │  ⚖️ LLM Evaluation       │
│  agent_pipeline.py          │  grader.py                │
│  config.py (recursive)      │  samplers.py              │
│  framework_adapter.py       │  runner.py (async/batch)  │
│  adk_adapter.py             │  config_grader.py         │
├─────────────────────────────────────────────────────────┤
│  💎 Pure Domain Layer (no I/O, no external deps)        │
│  conversation · rubric · scoring · evaluation · dataset │
├─────────────────────────────────────────────────────────┤
│  📊 Analysis Layer          │  💾 I/O Layer             │
│  exploration (12 analyses)  │  downloader.py            │
│  insights (8 analyses)      │  dataset_loader.py        │
│  visualization (8 plots)    │  split_utils.py           │
└─────────────────────────────────────────────────────────┘
```

### 🤖 Agent Architectures

**🟢 Baseline Agent** (`agents/baseline_pipeline.py`)
- Single `LlmAgent` with `gemini-2.5-flash`
- Minimal health instruction from `prompts/baseline_agent/v1_baseline.yaml`
- No tools — establishes the performance floor

**🔧 Tool-Augmented Agent** (`agents/tool_pipeline.py`)
- Same model + clinical prompt from `prompts/tool_agent/v1_clinical.yaml`
- Three medical reference tools:
  - 💊 `drug_reference()` — drug interaction and information lookup
  - 🩺 `symptom_checker()` — symptom analysis and differential suggestions
  - 🚨 `emergency_flag()` — emergency condition detection

**🤝 Multi-Agent Pipeline** (`agents/multi_pipeline.py`)
- `SequentialAgent` orchestration wired from `prompts/multi_agent/v1_structured.yaml`:
  - 🔀 **Triage** — classifies urgency and routes (`triage_instruction`)
  - 🎯 **Coordinator** — routes to Emergency or General Health specialist (`coordinator_instruction`)
  - 🚨 **Emergency specialist** — high-urgency referral guidance (`emergency_instruction`)
  - 💊 **General Health specialist** — tool-backed answers (`general_health_instruction`)
  - 📋 **Reviewer** — validates response quality and safety (`reviewer_instruction`)
- Optimize any one of the five instructions via `--target-agent` (see below).

### ⚙️ Orchestration Modes

| Mode | Agent Type | Use Case |
|---|---|---|
| `sequential` | SequentialAgent | Fixed pipeline stages |
| `routing` | LLM-driven delegation | Dynamic specialist selection |
| `loop` | LoopAgent | Iterative refinement |
| `parallel` | ParallelAgent | Concurrent execution |

---

## 🔄 Experiment Workflow

```
    ┌──────────┐
    │ 📊       │
    │ Analyze  │◄──────────────────────────┐
    └────┬─────┘                           │
         ▼                                 │
    ┌──────────┐                           │
    │ 💡       │                           │
    │Hypothesize│                          │
    └────┬─────┘                           │
         ▼                                 │
    ┌──────────┐                           │
    │ 🛠️       │                           │
    │Implement │                           │
    └────┬─────┘                           │
         ▼                                 │
    ┌──────────┐    ┌──────────┐    ┌──────┴─────┐
    │ 🧪       │───▶│ 📈       │───▶│ 🔄        │
    │ Evaluate │    │ Compare  │    │ Iterate    │
    └──────────┘    └──────────┘    └────────────┘
```

1. **📊 Analyze** — Explore HealthBench to find weak areas (emergency triage, multilingual, uncertainty)
2. **💡 Hypothesize** — Design a change (new prompt, tool, architectural shift)
3. **🛠️ Implement** — Build the agent variant in `agents/`
4. **🧪 Evaluate** — Score against HealthBench, log to MLflow
5. **📈 Compare** — Statistical comparison (paired tests, confidence intervals)
6. **🔄 Iterate** — Pick the next highest-leverage improvement

---

## 📚 Key Concepts

### 📏 HealthBench Scoring

Each conversation has weighted rubric criteria. The score formula:

```
score = sum(met criteria weights) / sum(max(0, weight)) × 100
```

- ✅ Positive weights (0.1–1.0) reward correct information
- ❌ Negative weights (-0.1 to -1.0) penalize harmful/incorrect responses
- 🚨 Emergency/safety criteria carry the highest weights (0.8–1.0)
- 📊 Scores range from negative (penalties dominate) to 100% (all criteria met)

### 📦 HealthBench Subsets

| Subset | Size | Description |
|---|---|---|
| `main` | ~5,000 | Full benchmark dataset |
| `consensus` | 3,671 | Physician-validated (high inter-rater agreement) |
| `hard` | 1,000 | Maximum model score ~32% — stress-test territory |

### ✅ ADK Evaluation Criteria

- **`rubrics_based_criterion`** — LLM-as-judge scores each rubric item as met/not-met
- **`tool_trajectory_avg_score`** — verifies agents call the right tools in the right order
- **`final_response_match_v2`** — semantic equivalence check against reference responses

### 📈 Statistical Rigor

All agent comparisons use:
- 🔁 **Paired bootstrap CI** (n=10,000) for confidence intervals
- 📐 **Paired t-tests** for significance
- 📏 **Cohen's d** for effect size
- 🛡️ **Bonferroni correction** for multiple comparisons

> ⚠️ An improvement is only reported as significant when bootstrap CI excludes zero **AND** p < 0.05 after correction.

---

## 📁 Project Structure

```
healthbench-agent-lab/
├── 📦 pyproject.toml               # Project config & dependencies
├── 📖 README.md
├── 🔑 .env.example                 # API keys template
│
├── 📂 src/healthbench_agent/       # 💎 Installable package
│   ├── domain/                     # Pure types, scoring, abstractions
│   │   ├── rubric.py               #   RubricItem
│   │   ├── conversation.py         #   Message, MessageList, Conversation
│   │   ├── sampler.py              #   SamplerBase, SamplerResponse
│   │   ├── evaluation.py           #   CriterionVerdict, EvalResult
│   │   ├── dataset.py              #   HealthBenchSample, HealthBenchDataset
│   │   ├── scoring.py              #   calculate_score, aggregate_scores
│   │   ├── judge.py                #   JudgeGrader (ABC)
│   │   └── experiment.py           #   RunParams, RunMetrics
│   ├── agent/                      # Agent infra — pipelines, config, adapters
│   │   ├── agent_pipeline.py       #   AgentPipeline (ABC)
│   │   ├── config.py               #   AgentNodeConfig, RootAgentPipelineConfig
│   │   ├── framework_adapter.py    #   FrameworkAdapter (ABC)
│   │   ├── factory.py              #   create_pipeline() factory
│   │   ├── prompt.py               #   load_instruction(), format_conversation()
│   │   ├── tool_registry.py        #   @register_tool decorator
│   │   ├── callback_registry.py    #   @register_callback decorator
│   │   └── adapters/
│   │       └── adk_adapter.py      #   ADKFrameworkAdapter, build_agent_node()
│   ├── io/                         # I/O layer — download & load
│   │   ├── downloader.py           #   download_dataset, URL constants
│   │   └── dataset_loader.py       #   load_dataset
│   ├── analysis/                   # 📊 28 registered analyses
│   │   ├── registry.py             #   @register_analysis, run_one, run_all
│   │   ├── utils.py                #   build_rubric_dataframe, build_sample_dataframe
│   │   ├── exploration.py          #   12 descriptive stats
│   │   ├── insights.py             #   8 cross-cutting insights
│   │   └── visualization.py        #   8 matplotlib visualizations
│   ├── llm_eval/                   # ⚖️ LLM-as-judge evaluation
│   │   ├── config_grader.py        #   JudgeConfig, EvalMode
│   │   ├── grader.py               #   LLMJudgeGrader, make_template, load_grader_prompt
│   │   ├── samplers.py             #   OpenAIChatSampler, GeminiChatSampler
│   │   └── runner.py               #   EvalRunner (async/batch)
│   └── prompt_optimization/        # 🪄 Automatic prompt optimization (domain-agnostic)
│       ├── optimizer.py            #   PromptOptimizer ABC, _TrialBudget, require_optional
│       ├── config.py               #   BaseOptimizationConfig, DSPy/TextGrad/CritiqueRefine
│       ├── metric.py               #   EndToEndMetric (agent + judge end-to-end)
│       ├── optimizer_registry.py   #   @register_prompt_optimizer, factory
│       ├── cli.py                  #   `optimize-prompt` entry point
│       └── adapters/               #   Per-backend adapters (lazy-imported)
│           ├── dspy_adapter.py     #     DSPyOptimizer (COPRO / MIPROv2)
│           ├── textgrad_adapter.py #     TextGradOptimizer (text-gradient descent)
│           └── critique_refine_adapter.py  # PromptWizard mutation + critique loop
│
├── 📂 .github/workflows/           # 🚦 GitHub Actions CI (ruff + mypy + pytest)
│
├── 📂 agents/                      # 🤖 Agent pipeline definitions
│   ├── baseline_pipeline.py        #   Architecture A: single agent
│   ├── tool_pipeline.py            #   Architecture B: agent + tools
│   ├── multi_pipeline.py           #   Architecture C: multi-agent
│   ├── baseline_agent/             #   ADK entry point + golden test cases
│   │   ├── agent.py
│   │   └── baseline_pipeline.test.json
│   ├── tool_agent/                 #   ADK entry point + golden test cases
│   │   ├── agent.py
│   │   └── tool_pipeline.test.json
│   └── multi_agent/                #   ADK entry point + golden test cases
│       ├── agent.py
│       └── multi_pipeline.test.json
│
├── 📂 tools/                       # 🔧 Medical reference tools
│   ├── drug_reference.py           #   💊 Drug lookup
│   ├── symptom_checker.py          #   🩺 Symptom analysis
│   └── emergency_flag.py           #   🚨 Emergency detection
│
├── 📂 prompts/                     # 📝 Versioned YAML prompts
│   ├── baseline_agent/             #   v1_baseline.yaml
│   ├── tool_agent/                 #   v1_clinical.yaml
│   ├── multi_agent/                #   v1_structured.yaml
│   ├── llm_grader/                 #   v1_llm_grader.yaml
│   └── prompt_optimization/        #   v1_critique_refine.yaml (domain-agnostic)
│
├── 📂 config/agents/               # ⚙️ Agent YAML configs
│   ├── baseline_agent.yaml
│   ├── tool_agent.yaml
│   └── multi_agent.yaml
│
├── 📂 evaluation/                  # 📈 Scoring & experiment tracking
│   ├── rubric_scorer.py            #   HealthBench → ADK rubric mapping
│   ├── healthbench_adapter.py      #   Conversation format adapter
│   ├── stats.py                    #   Bootstrap CI, t-tests, Cohen's d
│   ├── experiment_tracker.py       #   MLflow logging wrapper
│   └── test_config.json            #   ADK eval criteria
│
├── 📂 notebooks/                   # 📓 Jupyter analysis notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_agent_comparison.ipynb
│   └── 03_evaluation_deep_dive.ipynb
│
└── 📂 tests/                       # ✅ pytest test suite
    ├── conftest.py                 #   Shared fixtures
    ├── domain/                     #   Domain layer tests
    ├── dataset/                    #   I/O layer tests
    ├── analysis/                   #   Analysis tests
    ├── evaluation/                 #   Evaluation tests
    ├── llm_eval/                   #   Grader & sampler tests
    └── prompt_optimization/        #   Optimizer adapters & metric tests
```

---

## 🤝 Contributing

1. Fork the repo and create a feature branch
2. Follow the coding standards in [CLAUDE.md](CLAUDE.md)
3. Write tests (95% coverage minimum per module)
4. Run `uv run ruff check . && uv run mypy . && uv run pytest tests/ -v`
5. Submit a PR with a clear description

---

## 📄 License

MIT
