# healthbench-agent-lab

**Building, evaluating, and iterating on agentic AI systems for health question answering — benchmarked against HealthBench.**

This project is a hands-on exploration of agentic system design and evaluation methodology. It covers three pillars:

1. **Dataset Analysis** — Extract insights from the [HealthBench](https://github.com/openai/healthbench) evaluation dataset (5,000 physician-validated health conversations across 49 languages and 26 medical specialties)
2. **Agent Architectures** — Build and compare multiple agent pipelines using [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/), from single-agent baselines to multi-agent systems with tool augmentation
3. **Evaluation Framework** — Score agents against HealthBench's weighted rubric criteria using ADK's built-in eval (rubrics-based LLM-as-judge, trajectory matching) and custom statistical analysis

## Tech Stack

| Layer                  | Tool                                      |
|------------------------|-------------------------------------------|
| Package Manager        | [uv](https://docs.astral.sh/uv/)         |
| Agent Framework        | Google ADK (`google-adk`)                 |
| Built-in Eval          | ADK eval (rubrics, trajectory, response)  |
| Experiment Tracking    | MLflow                                    |
| Data Analysis          | pandas, scipy, seaborn, matplotlib        |
| Prompt Management      | Git-versioned YAML files in `prompts/`    |
| Eval Orchestration     | pytest + custom harness                   |
| Exploration            | Jupyter notebooks                         |

## Project Structure

```
healthbench-agent-lab/
├── pyproject.toml              # uv project config & dependencies
├── README.md
├── .python-version             # pinned Python version
├── .env.example                # API keys template
│
├── data/
│   └── healthbench/            # HealthBench dataset files
│
├── analysis/
│   ├── __init__.py
│   ├── exploration.py          # dataset loading & descriptive stats
│   ├── insights.py             # cross-dimensional analysis (theme × specialty × language)
│   └── visualization.py        # score distributions, heatmaps, comparisons
│
├── agents/
│   ├── baseline_agent/         # single ADK agent — minimal prompt, no tools
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── tool_agent/             # agent augmented with medical reference tools
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── tools.py
│   └── multi_agent/            # multi-agent pipeline (triage → specialist → reviewer)
│       ├── __init__.py
│       ├── agent.py
│       └── sub_agents.py
│
├── prompts/
│   ├── v1_baseline.yaml        # initial system prompt
│   ├── v2_clinical.yaml        # clinically-aware prompt
│   └── v3_structured.yaml      # structured output prompt
│
├── evaluation/
│   ├── __init__.py
│   ├── rubric_scorer.py        # map HealthBench rubrics → ADK rubrics_based_criterion
│   ├── healthbench_adapter.py  # load HealthBench conversations into ADK eval format
│   ├── stats.py                # confidence intervals, significance tests, bootstrap
│   ├── experiment_tracker.py   # MLflow logging wrapper
│   └── test_config.json        # ADK eval criteria configuration
│
├── experiments/
│   └── results/                # MLflow artifacts & exported scores
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_agent_comparison.ipynb
│   └── 03_evaluation_deep_dive.ipynb
│
└── tests/
    ├── test_baseline_agent.py  # ADK eval: golden dataset regression
    ├── test_tool_agent.py
    └── test_multi_agent.py
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone the repo
git clone https://gitlab.com/<your-namespace>/healthbench-agent-lab.git
cd healthbench-agent-lab

# Create venv and install all dependencies
uv sync

# Copy env template and add your API key
cp .env.example .env
# Edit .env → set GOOGLE_API_KEY=your-key-here

# Download HealthBench dataset
uv run python -c "from analysis.exploration import download_dataset; download_dataset()"
```

### Run an Agent Locally

```bash
# Start ADK web UI with the baseline agent
uv run adk web agents/baseline_agent

# Or run in terminal
uv run adk run agents/baseline_agent
```

### Run Evaluation

```bash
# Run ADK eval against a golden dataset
uv run adk eval agents/baseline_agent evaluation/test_config.json

# Run the full HealthBench evaluation harness
uv run pytest tests/ -v

# Run with experiment tracking
uv run python -m evaluation.experiment_tracker --agent baseline_agent --sample-size 100
```

### Explore the Dataset

```bash
uv run jupyter lab notebooks/01_dataset_exploration.ipynb
```

## Experiment Workflow

The iteration loop follows this pattern:

1. **Analyze** — Explore HealthBench to identify weak areas (e.g., emergency triage, multilingual, uncertainty)
2. **Hypothesize** — Design an agent change (new prompt, added tool, architectural shift)
3. **Implement** — Build the agent variant in `agents/`
4. **Evaluate** — Run against HealthBench subset, score with rubrics, log to MLflow
5. **Compare** — Statistical comparison against baseline (paired tests, confidence intervals)
6. **Iterate** — Pick the next highest-leverage improvement

## Key Concepts

### HealthBench Scoring
Each conversation has weighted rubric criteria. Score = (sum of met criteria weights) / (max possible score) × 100. Emergency/safety criteria carry the highest weights (0.8–1.0).

### ADK Evaluation Criteria Used
- **`rubrics_based_criterion`** — LLM-as-judge scores each HealthBench criterion as met/not-met
- **`tool_trajectory_avg_score`** — verifies agents use the right tools in the right order
- **`final_response_match_v2`** — semantic equivalence check against reference responses

### Statistical Rigor
All comparisons use bootstrap confidence intervals and paired significance tests. Results are only reported as improvements when p < 0.05 with Bonferroni correction for multiple comparisons.

## License

MIT