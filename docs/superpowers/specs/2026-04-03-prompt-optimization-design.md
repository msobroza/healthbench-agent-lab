# Automatic Prompt Engineering (APE) for Agent System Prompts

**Date:** 2026-04-03
**Status:** Approved
**Scope:** New `prompt_optimization` module in `src/healthbench_agent/`

## Goal

Add support for automatic prompt engineering to optimize agent system prompts
(baseline, tool-augmented, multi-agent) using end-to-end evaluation
(agent generates response, LLM judge grades it, `calculate_score()` aggregates).
Three optimization backends behind a common abstraction: DSPy, TextGrad, and a
custom critique-refine algorithm (inspired by PromptWizard).

## Non-Goals

- Optimizing the LLM judge grading prompt (kept verbatim from simple-evals for comparability).
- Meta-evaluation of the LLM judge (separate concern).
- Batch optimization of multiple agents in a single run.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prompts to optimize | Agent system prompts only | Judge prompt must stay fixed for HealthBench comparability |
| Frameworks | DSPy, TextGrad, critique-refine (custom) | Best cost/performance mix; abstraction keeps door open |
| Evaluation metric | End-to-end (agent + judge + score) | Prompt changes must flow through agent responses |
| Sample handling | User-configurable size, stratified sampling | Reuses existing `stratified_sample()` infrastructure |
| Experiment tracking | Existing MLflow tracker | No reason to duplicate; prompt fingerprinting already built |
| Agents per run | One | YAGNI; script multiple runs externally |
| Config pattern | Separate config per framework + base class | Avoids god-config; each adapter gets typed config |
| Registry pattern | `@register_prompt_optimizer` decorator | Open/Closed; consistent with `tool_registry`, `callback_registry` |

## Module Structure

```
src/healthbench_agent/prompt_optimization/
    __init__.py                    # Public API exports
    optimizer.py                   # PromptOptimizer ABC, OptimizationResult, TrialRecord
    config.py                      # BaseOptimizationConfig, DSPyConfig, TextGradConfig,
                                   #   CritiqueRefineConfig
    metric.py                      # EndToEndMetric
    optimizer_registry.py          # register_prompt_optimizer, create_prompt_optimizer
    adapters/
        __init__.py                # Imports all adapters (triggers registration)
        dspy_adapter.py            # DSPyOptimizer
        textgrad_adapter.py        # TextGradOptimizer
        critique_refine_adapter.py # CritiqueRefineOptimizer
```

## Dependency Graph

```
prompt_optimization/
    depends on    -> domain/ (HealthBenchSample, AgentPipeline, JudgeGrader, scoring)
    depends on    -> agent/ (RootAgentPipelineConfig, create_pipeline)
    depends on    -> llm_eval/ (JudgeConfig, create_judge, SamplerBase)
    does NOT      -> evaluation/, agents/, tools/
```

Same layer as `llm_eval/` -- outer layer depending on domain abstractions (Dependency Inversion).

## Core Abstractions

### TrialRecord

```python
@dataclass(frozen=True)
class TrialRecord:
    """Single optimization trial."""
    trial_id: int
    prompt: str
    score: float | None             # None in mutation-only mode (no metric)
    timestamp: str
```

### OptimizationResult

```python
@dataclass(frozen=True)
class OptimizationResult:
    """Result of a prompt optimization run."""
    optimized_prompt: str
    baseline_score: float
    optimized_score: float
    improvement: float              # optimized_score - baseline_score
    num_trials: int
    trial_history: list[TrialRecord]
    optimizer_name: str
    config: dict[str, Any]          # Serialized optimizer config for reproducibility
```

### PromptOptimizer (ABC)

```python
class PromptOptimizer(ABC):
    """Abstract base for prompt optimizers."""

    @abstractmethod
    def optimize(
        self,
        current_prompt: str,
        samples: list[HealthBenchSample] | None,
        metric: EndToEndMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt against a scoring metric.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset. Required for DSPy/TextGrad,
                optional for critique-refine (mutation-only mode).
            metric: Callable that scores a prompt end-to-end.
                Required for DSPy/TextGrad, optional for critique-refine.

        Returns:
            OptimizationResult with the best prompt and trial history.
        """
        ...
```

### EndToEndMetric

```python
class EndToEndMetric:
    """Scores a prompt by running agent generation + LLM judge grading."""

    def __init__(
        self,
        agent_config: RootAgentPipelineConfig,
        judge: JudgeGrader,
        samples: list[HealthBenchSample],
    ) -> None:
        self.agent_config = agent_config
        self.judge = judge
        self.samples = samples

    def __call__(self, prompt: str) -> float:
        """Evaluate a candidate prompt end-to-end.

        1. Copies agent_config with the candidate prompt (no mutation)
        2. Builds a fresh AgentPipeline
        3. Generates responses for all samples
        4. Grades via judge
        5. Returns aggregate_scores()
        """
        ...
```

## Configuration

### BaseOptimizationConfig

```python
class BaseOptimizationConfig(BaseSettings):
    """Shared settings for all optimizers."""
    model_config = SettingsConfigDict(env_prefix="OPTIM_", env_file=".env")

    optimizer: str                           # Discriminator
    max_trials: int = Field(50, ge=1)
    sample_size: int = Field(20, ge=1)
    seed: int = 42
    meta_model: str = "gemini-2.5-flash"
    meta_provider: str = "gemini"

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
```

### DSPyConfig

```python
class DSPyConfig(BaseOptimizationConfig):
    optimizer: str = "dspy"
    dspy_optimizer: str = "copro"            # "copro" or "miprov2"
    max_bootstrapped_demos: int = 0          # 0 = instruction-only
```

### TextGradConfig

```python
class TextGradConfig(BaseOptimizationConfig):
    optimizer: str = "textgrad"
    steps: int = Field(10, ge=1)
```

### CritiqueRefineConfig

```python
class CritiqueRefineConfig(BaseOptimizationConfig):
    optimizer: str = "critique_refine"
    mutation_rounds: int = Field(3, ge=1)
    refine_iterations: int = Field(3, ge=1)
    style_variations: int = Field(5, ge=1)
```

## Optimizer Registry

```python
_PROMPT_OPTIMIZER_REGISTRY: dict[str, tuple[type[BaseOptimizationConfig], type[PromptOptimizer]]] = {}

def register_prompt_optimizer(name, config_class):
    """Class decorator that registers an optimizer with its config type."""
    def decorator(cls):
        _PROMPT_OPTIMIZER_REGISTRY[name] = (config_class, cls)
        return cls
    return decorator

def create_prompt_optimizer(config: BaseOptimizationConfig) -> PromptOptimizer:
    """Create an optimizer from config, dispatching via registry."""
    _, optimizer_class = _PROMPT_OPTIMIZER_REGISTRY[config.optimizer]
    return optimizer_class(config)
```

## Adapter Responsibilities

### DSPyOptimizer
- Wraps `current_prompt` as a `dspy.Signature` with instruction.
- Configures `dspy.LM` from `config.meta_model`.
- Runs COPRO or MIPROv2 with `max_bootstrapped_demos=0` (instruction-only).
- Extracts optimized instruction from compiled module.
- Collects trial history from DSPy's internal logging.
- **Requires** `samples` and `metric`.

### TextGradOptimizer
- Wraps `current_prompt` as `textgrad.Variable(requires_grad=True)`.
- Defines loss as `1.0 - metric(prompt)` (TextGrad minimizes).
- Runs `optimizer.step()` for `config.steps` iterations.
- Extracts best prompt from variable history.
- **Requires** `samples` and `metric`.

### CritiqueRefineOptimizer
- No external dependency -- uses `SamplerBase` for LLM calls.
- **Mutation phase**: generates `mutation_rounds x style_variations` prompt variants
  by asking meta-LLM to rewrite the prompt mixing predefined thinking styles.
- **Critique-refine phase** (when `metric` provided): evaluates candidates, asks
  meta-LLM to critique failures, refines prompt based on critique. Repeats for
  `refine_iterations` cycles.
- **Mutation-only mode** (when `metric=None`): skips scoring, returns best mutation
  by meta-LLM self-ranking.
- Thinking styles kept as a module-level constant list.

## Experiment Tracking Integration

- Each optimization run logs to MLflow via existing `log_evaluation_run()`.
- Additional params: `optimizer_name`, `max_trials`, `num_trials`, `baseline_score`,
  `optimized_score`, `improvement`.
- Trial history saved as JSON artifact (`optimization_trials.json`).

## Prompt Saving

- Optimized prompt saved as new versioned YAML:
  `prompts/baseline_agent/v1_baseline.yaml` -> `prompts/baseline_agent/v2_optimized.yaml`.
- Version bump + metadata (optimizer, score delta, parent version) in YAML header.
- Agent config YAML is NOT modified -- user manually points to new prompt.

## CLI Entry Point

```bash
uv run optimize-prompt \
    --agent-config config/agents/baseline_agent.yaml \
    --optimizer dspy \
    --sample-size 20 \
    --max-trials 50 \
    --subset consensus \
    --seed 42
```

## New Dependencies

```
dspy       # DSPy optimizer adapter (lazy import)
textgrad   # TextGrad optimizer adapter (lazy import)
```

Both lazy-imported -- only loaded when the corresponding adapter is requested.
`CritiqueRefineOptimizer` has no external dependencies.

## Testing Strategy

- **optimizer.py**: `OptimizationResult` construction, frozen invariants.
- **config.py**: Each config validates its fields, env var override.
- **metric.py**: `EndToEndMetric` with mocked `AgentPipeline` and `JudgeGrader`.
- **optimizer_registry.py**: Registration, lookup, unknown optimizer error.
- **dspy_adapter.py**: Mocked DSPy calls, trial collection.
- **textgrad_adapter.py**: Mocked TextGrad calls, gradient steps.
- **critique_refine_adapter.py**: Mutation generation, critique-refine loop with
  mocked sampler, mutation-only mode (no metric).
