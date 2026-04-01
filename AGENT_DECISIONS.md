# AGENT_DECISIONS.md

Exhaustive design decisions for Phase 2 (Agent Development) and Phase 3 (Evaluation Framework).
Each decision documents all options considered, pros/cons, data-driven rationale, and the final choice with an ADK code snippet.

---

## Phase 2 — Agent Development

---

### Decision 1: LLM Model Selection

**Context:** All three agent architectures need a model. The SPEC specifies `gemini-2.0-flash` as the default, with `gemini-2.5-pro` as an option for the reviewer.

#### Option A — `gemini-2.0-flash` for all agents

| | |
|---|---|
| **Pros** | Lowest latency (~200ms/turn); lowest cost per token; consistent baseline across architectures (fair comparison); well-tested in ADK ecosystem; sufficient for straightforward health Q&A |
| **Cons** | Lower reasoning capability on complex rubrics; may underperform on `hard` subset (penalty_mass_ratio 0.642); less reliable tool-calling on multi-step tasks; `communication_quality` axis may suffer from less nuanced output |

#### Option B — `gemini-2.5-flash` for all agents

| | |
|---|---|
| **Pros** | Better reasoning than 2.0-flash; native thinking support via `BuiltInPlanner`; improved instruction following (addresses Insight 6 — 36.5% penalty on `instruction_following`); better multi-turn context handling; still fast enough for interactive use |
| **Cons** | Higher cost than 2.0-flash; slightly higher latency; may not be available in all regions; newer model with potentially less stable API |

#### Option C — `gemini-2.0-flash` for baseline/tool, `gemini-2.5-pro` for reviewer

| | |
|---|---|
| **Pros** | Puts strongest model where it matters most (Insight 4 — reviewer is critical for hard samples); keeps inference cost low for bulk agents; reviewer checks are infrequent relative to main generation; optimal cost/quality tradeoff |
| **Cons** | Mixed models complicate latency analysis; harder to attribute improvements to architecture vs model quality; `2.5-pro` is significantly more expensive; comparison is no longer purely architecture-driven |

**Decision:** Use **`gemini-2.0-flash`** for baseline and tool agents, and **`gemini-2.0-flash`** for the multi-agent pipeline initially. This keeps comparisons fair — the improvement signal comes from architecture, not model quality. The reviewer can be upgraded to `gemini-2.5-flash` or `gemini-2.5-pro` in Phase 4 iteration runs, with the model logged as an MLflow parameter.

**ADK snippet:**
```python
from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="baseline_agent",
    model="gemini-2.0-flash",
    instruction="...",
)
```

---

### Decision 2: Prompt Loading Strategy

**Context:** Prompts are versioned YAML files in `prompts/`. Need to decide how agents load them at startup.

#### Option A — Inline instruction string in `agent.py`

| | |
|---|---|
| **Pros** | Simplest implementation; zero dependencies; ADK quickstart pattern; easy to read agent definition at a glance |
| **Cons** | Prompt changes require code changes; no version metadata (version, rationale, source); violates SPEC requirement for YAML-versioned prompts; harder to diff prompt evolution in Git |

#### Option B — Load YAML at module level in `agent.py`

| | |
|---|---|
| **Pros** | Prompts live in `prompts/*.yaml` with version/rationale metadata; Git diff shows prompt changes cleanly; same prompt file can be referenced by multiple agents; aligns with SPEC §9.3; prompt version trackable in MLflow |
| **Cons** | Adds `pyyaml` dependency (already installed); file I/O at import time; slightly more complex agent definition; YAML parsing overhead (negligible) |

#### Option C — ADK Agent Config YAML (no Python `agent.py`)

| | |
|---|---|
| **Pros** | Declarative agent definition; ADK handles parsing; supports `config_path` sub-agent references; no code needed for simple agents |
| **Cons** | Limited to Gemini models only; custom tool functions harder to reference (dot notation); less flexible than Python; cannot dynamically compose instructions; not compatible with existing SPEC file structure (`agent.py` files expected); harder to unit test |

**Decision:** **Option B — Load YAML at module level.** Each `agent.py` loads its prompt from `prompts/v{n}_{name}.yaml` at import time, reading the `instruction` field. This satisfies SPEC §9.3 (YAML-versioned prompts) while keeping the ADK `Agent()` constructor pattern.

**ADK snippet:**
```python
import yaml
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v1_baseline.yaml"

def _load_instruction() -> str:
    """Load the instruction text from the versioned YAML prompt file."""
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["instruction"]

root_agent = LlmAgent(
    name="baseline_agent",
    model="gemini-2.0-flash",
    instruction=_load_instruction(),
)
```

---

### Decision 3: Baseline Agent Design (Architecture A)

**Context:** Simplest agent — no tools, no sub-agents. Establishes performance floor.

#### Option A — Minimal prompt, no structured output

| | |
|---|---|
| **Pros** | True baseline — isolates model-only performance; fast to implement; clear performance floor; any improvement in Architecture B/C is attributable to tools/multi-LlmAgent |
| **Cons** | Will score poorly on `accuracy` (Insight 1 — 7.0-7.8 penalty pts/item); no emergency detection capability; no context seeking guidance; expected to fail ~34% of emergency rubric items |

#### Option B — Baseline with structured output_schema

| | |
|---|---|
| **Pros** | Consistent response format; easier to parse for evaluation; demonstrates ADK `output_schema` feature |
| **Cons** | Structured output constrains natural response generation; model may lose quality when forced into schema; not a "true" baseline; `output_schema` + no tools requires specific model support |

**Decision:** **Option A — Minimal prompt, no structured output.** The baseline must be simple to provide a fair comparison floor. The prompt includes only essential health response guidance.

**ADK snippet:**
```python
root_agent = LlmAgent(
    name="baseline_agent",
    model="gemini-2.0-flash",
    description="Baseline health assistant with no tools or sub-agents.",
    instruction=_load_instruction(),  # from prompts/v1_baseline.yaml
)
```

---

### Decision 4: Tool Design — Return Types & Error Handling

**Context:** Three tools: `drug_reference()`, `symptom_checker()`, `emergency_flag()`. ADK expects dict returns.

#### Option A — Plain dict returns with status field

| | |
|---|---|
| **Pros** | Follows ADK best practice (dict with "status" key); LLM can interpret structured results; simple to implement; no custom classes needed |
| **Cons** | No type safety; schema not enforced; error handling is convention-based; tool response format may vary |

#### Option B — Pydantic model returns, serialized to dict

| | |
|---|---|
| **Pros** | Type-safe response schema; validation at construction time; self-documenting; consistent across tools; IDE autocompletion |
| **Cons** | ADK expects dict, not Pydantic models (needs `.model_dump()`); over-engineering for mock tools; adds complexity without benefit since tools return static data; Pydantic models are in `domain/` but tools are in `agents/` (dependency inversion violation) |

#### Option C — String returns (simplest)

| | |
|---|---|
| **Pros** | Maximum simplicity; LLM processes natural language directly; no parsing needed |
| **Cons** | Harder for LLM to extract structured info; inconsistent with ADK patterns; loses structured data capabilities; `emergency_flag()` urgency level harder to interpret |

**Decision:** **Option A — Plain dict returns with status field.** This follows ADK conventions. Each tool returns `{"status": "success"|"error", ...tool-specific-fields}`. The tools are mock implementations with curated medical reference data.

**ADK snippet:**
```python
def drug_reference(drug_name: str) -> dict:
    """Look up drug information including dosage, interactions, and contraindications.

    Args:
        drug_name: The name of the drug to look up (generic or brand name).

    Returns:
        dict with drug information or error if drug not found.
    """
    # Mock implementation with curated data
    drugs = {
        "metformin": {
            "status": "success",
            "generic_name": "metformin",
            "drug_class": "biguanide",
            "common_dosage": "500-2000mg daily",
            "contraindications": ["severe renal impairment", "metabolic acidosis"],
            "interactions": ["alcohol", "iodinated contrast agents"],
        },
    }
    key = drug_name.lower().strip()
    return drugs.get(key, {"status": "error", "error_message": f"Drug '{drug_name}' not found."})
```

---

### Decision 5: `emergency_flag()` Tool — Sensitivity vs Specificity

**Context:** Insight 3 — 34% of samples have emergency criteria across diverse themes. False negatives are clinically dangerous.

#### Option A — Keyword-based classifier (high sensitivity, lower specificity)

| | |
|---|---|
| **Pros** | Deterministic; zero latency; no API call needed; easy to test; broad keyword list catches diverse emergencies (chest pain, suicide, overdose, anaphylaxis); aligns with Insight 3 (must be broad) |
| **Cons** | False positives on benign mentions ("I read about chest pain"); no semantic understanding; keyword list requires maintenance; cannot detect implicit emergencies ("my baby won't stop crying and is turning blue") |

#### Option B — LLM-based classification (calls the model)

| | |
|---|---|
| **Pros** | Semantic understanding of context; catches implicit emergencies; adapts to nuanced language; handles multilingual descriptions |
| **Cons** | Adds latency (API call within a tool); costs money; recursive LLM call is fragile; harder to test deterministically; may disagree with the outer agent's model |

#### Option C — Rule-based with confidence tiers

| | |
|---|---|
| **Pros** | Combines keyword matching with severity tiers; returns `{"urgency": "emergency"|"urgent"|"routine", "confidence": float, "matched_keywords": list}`; deterministic; testable; the agent can interpret confidence and act accordingly |
| **Cons** | More complex implementation than pure keyword; still misses purely semantic emergencies; confidence scoring is heuristic |

**Decision:** **Option C — Rule-based with confidence tiers.** This balances Insight 3's requirement for broad sensitivity with deterministic testability. Keywords are grouped by severity, and the tool returns urgency level + matched keywords so the agent can incorporate them into its response.

**ADK snippet:**
```python
def emergency_flag(description: str) -> dict:
    """Classify a clinical description for emergency urgency.

    Args:
        description: Patient's description of symptoms or situation.

    Returns:
        dict with urgency level, confidence, and matched keywords.
    """
    description_lower = description.lower()
    EMERGENCY_KEYWORDS = ["chest pain", "difficulty breathing", "unconscious",
                          "suicide", "overdose", "severe bleeding", ...]
    URGENT_KEYWORDS = ["high fever", "persistent vomiting", "severe headache", ...]

    emergency_matches = [k for k in EMERGENCY_KEYWORDS if k in description_lower]
    urgent_matches = [k for k in URGENT_KEYWORDS if k in description_lower]

    if emergency_matches:
        return {"status": "success", "urgency": "emergency",
                "confidence": min(1.0, len(emergency_matches) * 0.3 + 0.4),
                "matched_keywords": emergency_matches,
                "recommendation": "Seek immediate emergency medical care (call 911)."}
    if urgent_matches:
        return {"status": "success", "urgency": "urgent", ...}
    return {"status": "success", "urgency": "routine", ...}
```

---

### Decision 6: `symptom_checker()` Tool Design

**Context:** Insight 2 — `health_data_tasks` has highest penalty ratio (0.754-0.896). Tool must hedge on interpretations.

#### Option A — Symptom → possible conditions mapping (dict lookup)

| | |
|---|---|
| **Pros** | Deterministic; fast; testable; returns curated condition list with urgency; no API dependency; safe (always includes "consult a doctor" qualifier) |
| **Cons** | Limited symptom vocabulary; cannot combine multiple symptoms intelligently; static knowledge base; may miss rare conditions |

#### Option B — Embedding-based symptom similarity search

| | |
|---|---|
| **Pros** | Handles fuzzy symptom descriptions; finds similar symptoms; supports multilingual input; more comprehensive coverage |
| **Cons** | Requires embedding model (adds dependency); latency for embedding computation; harder to test deterministically; over-engineered for mock tool; adds vector DB dependency |

**Decision:** **Option A — Dict lookup with curated symptom data.** The tool provides structured possible conditions with urgency levels and always includes a "consult healthcare provider" qualifier. This aligns with Insight 2's requirement for conservative data interpretation.

**ADK snippet:**
```python
def symptom_checker(symptoms: str) -> dict:
    """Check symptoms against a medical knowledge base for possible conditions.

    Args:
        symptoms: Comma-separated list of symptoms the patient is experiencing.

    Returns:
        dict with possible conditions, urgency, and recommendation to seek care.
    """
    symptom_list = [s.strip().lower() for s in symptoms.split(",")]
    # Mock curated database
    conditions = _match_conditions(symptom_list)
    return {
        "status": "success",
        "symptoms_analyzed": symptom_list,
        "possible_conditions": conditions,
        "disclaimer": "This is not a diagnosis. Please consult a healthcare provider.",
    }
```

---

### Decision 7: Multi-Agent Orchestration Pattern (Architecture C)

**Context:** SPEC requires Triage → Specialist → Reviewer pipeline.

#### Option A — `SequentialAgent` wrapping three LLM agents

| | |
|---|---|
| **Pros** | Deterministic execution order; every query goes through all three stages; consistent latency (always 3 LLM calls); simple to implement; aligns with SPEC description; output_key enables data passing between stages |
| **Cons** | No routing — every query hits all three agents even when unnecessary; higher latency (3 serial LLM calls); emergency queries are delayed by sequential processing; no way to skip reviewer for simple queries |

#### Option B — Root LLM agent with `sub_agents` for delegation

| | |
|---|---|
| **Pros** | LLM-driven routing based on intent; can skip unnecessary stages; more natural delegation; ADK handles transfer automatically; lower latency for simple queries (may only use 1-2 agents) |
| **Cons** | Non-deterministic routing — triage may misroute; harder to guarantee reviewer always runs; delegation errors are hard to debug; Insight 4 says reviewer is critical for hard samples — skipping it loses the main benefit |

#### Option C — Hybrid: `SequentialAgent` with LLM triage as first step

| | |
|---|---|
| **Pros** | Deterministic pipeline (all stages always run) while triage classifies the query for downstream agents; reviewer always checks; combines structured pipeline with intelligent routing; output_key from triage stage informs specialist behavior; aligns with SPEC architecture diagram |
| **Cons** | Most complex implementation; three mandatory LLM calls; triage classification stored in state may not be perfectly consumed by specialist; requires careful prompt engineering for state reading |

**Decision:** **Option C — Hybrid `SequentialAgent` with LLM triage.** The triage agent classifies urgency/topic/expertise into session state via `output_key`. The specialist agent reads this classification. The reviewer always runs (Insight 4 — critical for hard samples). The sequential structure guarantees all stages execute.

**ADK snippet:**
```python
from google.adk.agents import LlmAgent, SequentialAgent

triage_agent = LlmAgent(
    name="triage_agent",
    model="gemini-2.0-flash",
    description="Classifies query urgency, topic, and user expertise level.",
    instruction=_load_triage_instruction(),
    output_key="triage_classification",
)

emergency_agent = LlmAgent(
    name="emergency_agent",
    model="gemini-2.0-flash",
    description="Handles high-urgency medical queries requiring immediate referral.",
    instruction=_load_emergency_instruction(),
    tools=[emergency_flag],
    output_key="specialist_response",
)

general_health_agent = LlmAgent(
    name="general_health_agent",
    model="gemini-2.0-flash",
    description="Handles general health queries with medical reference tools.",
    instruction=_load_general_instruction(),
    tools=[drug_reference, symptom_checker, emergency_flag],
    output_key="specialist_response",
)

reviewer_agent = LlmAgent(
    name="reviewer_agent",
    model="gemini-2.0-flash",
    description="Reviews responses for completeness, safety, and communication quality.",
    instruction=_load_reviewer_instruction(),
    output_key="final_response",
)

# Root agent delegates to specialists, then always reviews
root_agent = LlmAgent(
    name="multi_agent",
    model="gemini-2.0-flash",
    description="Multi-agent health pipeline: triage → specialist → reviewer.",
    instruction=_load_coordinator_instruction(),
    sub_agents=[emergency_agent, general_health_agent],
    output_key="specialist_response",
)
```

---

### Decision 8: Specialist Routing — Triage → Emergency vs General

**Context:** Triage agent must route to EmergencyAgent or GeneralHealthAgent.

#### Option A — LLM-driven delegation via `sub_agents`

| | |
|---|---|
| **Pros** | ADK handles delegation automatically based on sub-agent descriptions; most natural routing; model understands intent nuances; handles ambiguous cases; aligns with ADK `sub_agents` pattern |
| **Cons** | Non-deterministic; may misroute edge cases; delegation depends on quality of sub-agent `description` fields; harder to test routing logic in isolation |

#### Option B — Rule-based routing via `emergency_flag()` tool in triage

| | |
|---|---|
| **Pros** | Deterministic routing; triage calls `emergency_flag()` and routes based on result; testable; fast; aligns with Insight 3 (broad emergency detection) |
| **Cons** | Requires the triage agent to call a tool and interpret results programmatically; ADK doesn't natively support conditional routing based on tool output; would need custom orchestration code |

#### Option C — Root LLM agent with both specialists as `sub_agents`

| | |
|---|---|
| **Pros** | Root agent reads triage classification from state and delegates to appropriate specialist; uses ADK's delegation mechanism; description-based routing; all tools available to general health LlmAgent |
| **Cons** | Relies on LLM correctly reading state and delegating; adds one more LLM call; delegation may not always work as expected |

**Decision:** **Option A — LLM-driven delegation via `sub_agents`.** The root coordinator agent has both `emergency_agent` and `general_health_agent` as `sub_agents`. Its instruction tells it to read the triage classification from session state and delegate accordingly. This uses ADK's native delegation pattern.

---

### Decision 9: Reviewer Agent — What to Check

**Context:** Insight 4 — hard samples are driven by penalty mass, not complexity. Reviewer is the most critical component.

#### Option A — Completeness-focused review

| | |
|---|---|
| **Pros** | Completeness is the largest axis (39% of criteria, 22,285 items); catches missing information; ensures responses address all aspects of the query |
| **Cons** | Insight 4 shows penalty mass drives hard scores, not missing info; completeness checking alone misses accuracy errors (the biggest penalty source); doesn't address communication quality |

#### Option B — Penalty-focused review (safety net)

| | |
|---|---|
| **Pros** | Directly addresses Insight 1 (accuracy penalties 7.0-7.8 pts/item) and Insight 4 (penalty mass ratio); catches unsupported diagnoses, missing safety caveats, overconfident language; highest impact on hard subset; addresses Insight 6 (communication_quality 37.5% penalty frequency) |
| **Cons** | May be overly conservative; could flag correct but assertive statements; reviewer may not have medical knowledge to judge accuracy; risk of reviewer degrading good responses |

#### Option C — Multi-axis review checklist

| | |
|---|---|
| **Pros** | Comprehensive — checks accuracy, completeness, communication quality, safety, and instruction following; structured checklist in prompt ensures consistent review; addresses all insights simultaneously |
| **Cons** | Most complex prompt; reviewer may not deeply evaluate any single axis; longer processing time; prompt engineering must be very precise; risk of superficial checking |

**Decision:** **Option B — Penalty-focused review.** The reviewer prompt explicitly checks for: (1) unsupported factual claims, (2) missing emergency/safety caveats, (3) overconfident language on uncertain topics, (4) tone mismatch with user expertise. This targets the highest-impact penalty criteria per Insights 1, 4, and 6.

---

### Decision 10: State Passing Between Agents

**Context:** In the multi-agent pipeline, agents need to share data (triage classification, specialist response).

#### Option A — `output_key` on each agent

| | |
|---|---|
| **Pros** | Built-in ADK mechanism; automatic state storage; downstream agents read from session state; clean separation; no custom code needed |
| **Cons** | Only stores final text response, not structured data; downstream agents must parse text; no type safety; state keys are string-based |

#### Option B — `ToolContext.state` for structured data passing

| | |
|---|---|
| **Pros** | Can store structured dicts/lists; tools can read/write state; more flexible than output_key; supports typed data |
| **Cons** | Only available inside tool functions, not in agent instructions; adds complexity; requires tools specifically for state management |

#### Option C — Combined: `output_key` for text, structured prompt for parsing

| | |
|---|---|
| **Pros** | Uses ADK's native `output_key`; triage agent writes a structured text format (e.g., "URGENCY: emergency\nTOPIC: chest_pain\nEXPERTISE: patient"); downstream agent instructions reference `{triage_classification}` template variable; no custom tools needed |
| **Cons** | Relies on triage agent producing consistent format; parsing is LLM-dependent; fragile if format varies |

**Decision:** **Option A — `output_key`.** Each stage stores its output via `output_key` into session state. The downstream agent's instruction references the key via ADK's template variable syntax `{key_name}`. This is the simplest ADK-native approach.

---

## Phase 3 — Evaluation Framework

---

### Decision 11: Grader Implementation — Verbatim vs Modified Prompt

**Context:** SPEC §5.6 requires the grader prompt to be verbatim from simple-evals for comparability.

#### Option A — Verbatim reproduction of simple-evals prompt

| | |
|---|---|
| **Pros** | Results directly comparable to published HealthBench scores; reproducible; validated against physician judgments (MF1 ≈ 0.71); no prompt engineering risk; matches meta-evaluation data |
| **Cons** | Cannot optimize for specific failure modes; prompt may not be optimal for non-OpenAI grader models; locked to original format |

#### Option B — Modified prompt optimized for Gemini grader

| | |
|---|---|
| **Pros** | Could improve grading accuracy on Gemini; tailored for specific model strengths; may reduce grader variance |
| **Cons** | Breaks comparability with published scores; no meta-evaluation data for modified prompt; introduces confounding variable; requires new validation study |

**Decision:** **Option A — Verbatim.** SPEC §5.9.3 rule 4: "Never edit a prompt file in place." The grader prompt is reproduced exactly from `simple-evals/healthbench_eval.py` in `prompts/llm_v1_llm_grader.yaml`. Any modifications create `grader_v2.yaml`.

---

### Decision 12: Sampler Architecture — Provider Abstraction

**Context:** Need `OpenAIChatSampler` and `GeminiChatSampler` implementing `SamplerBase`.

#### Option A — Direct SDK calls in each sampler

| | |
|---|---|
| **Pros** | Simple; each sampler wraps its SDK directly; no shared abstraction beyond `SamplerBase`; easy to understand; follows simple-evals pattern |
| **Cons** | Duplicated retry/timeout logic; no shared error handling; each sampler implements rate limiting independently |

#### Option B — Shared base with provider-specific adapters

| | |
|---|---|
| **Pros** | Shared retry/timeout/rate-limit logic in base; DRY; consistent error handling; easier to add new providers |
| **Cons** | Over-engineering for two providers; abstraction may not fit both SDKs well; adds complexity; violates YAGNI |

**Decision:** **Option A — Direct SDK calls.** Two providers is not enough to justify a shared base beyond `SamplerBase`. Each sampler directly wraps its SDK with provider-specific retry logic.

---

### Decision 13: EvalRunner Execution Modes

**Context:** SPEC §5.7 defines async (ThreadPool) and batch (OpenAI Batch API) modes.

#### Option A — ThreadPoolExecutor for async mode

| | |
|---|---|
| **Pros** | Works with both OpenAI and Gemini SDKs (both are sync); simple concurrency model; `max_workers` controls parallelism; good for ≤500 samples; immediate results |
| **Cons** | Hits rate limits under high concurrency; memory scales with inflight requests; non-deterministic ordering |

#### Option B — asyncio.gather for async mode

| | |
|---|---|
| **Pros** | Native Python async; lower memory overhead; better for I/O-bound tasks; asyncio semaphore for rate limiting |
| **Cons** | OpenAI SDK sync client doesn't work in asyncio (needs async client); mixing sync/async is error-prone; more complex error handling |

#### Option C — ThreadPoolExecutor + OpenAI Batch API (dual mode)

| | |
|---|---|
| **Pros** | Best of both: ThreadPool for development iterations, Batch API for full benchmark runs (50% cost reduction); aligns with SPEC §5.7 decision guide; batch mode produces auditable JSONL files |
| **Cons** | Two code paths to maintain; batch mode requires polling for completion; 24-hour latency for batch; different result formats |

**Decision:** **Option C — Dual mode.** `EvalRunner` supports both modes: `mode="async"` uses ThreadPoolExecutor (default for dev), `mode="batch"` uses OpenAI Batch API (for full benchmark runs). Mode is logged to MLflow.

---

### Decision 14: ADK Evaluation Integration Strategy

**Context:** SPEC §5.8 evaluates four options (A-D) for integrating LLM-as-judge with ADK.

#### Option A — ADK `rubrics_based_criterion` native

| | |
|---|---|
| **Pros** | Zero eval infrastructure code; `adk eval` works out of the box; integrated HTML report; automatic tool trajectory scoring; CI-ready with pytest |
| **Cons** | No grader prompt control; not reproducible against simple-evals; no OpenAI grader support; aggregation differs from HealthBench formula; no tag stratification |

#### Option B — `llm_eval` as sole scorer (ADK-independent)

| | |
|---|---|
| **Pros** | 100% reproducible against HealthBench paper; grader model swappable; full tag stratification; independent of ADK internals; results logged to MLflow |
| **Cons** | Two pipelines (ADK for inference, llm_eval for scoring); no tool trajectory scoring; more boilerplate |

**Decision:** **Option B primary, Option A for CI gate.** Per SPEC §5.8 recommendation: `llm_eval` is the primary scorer for benchmark comparisons. ADK `rubrics_based_criterion` is used as a lightweight CI gate on golden examples during development.

---

### Decision 15: JudgeConfig — Settings Management

**Context:** SPEC §5.9.1 specifies `pydantic-settings` for judge configuration.

| | |
|---|---|
| **Pros of pydantic-settings** | Type-validated on construction; env var override (`JUDGE_MODEL`, `JUDGE_TEMPERATURE`); `model_dump()` serializes to MLflow params; already a transitive dependency; separates secrets from config |
| **Cons of pydantic-settings** | Slightly more complex than plain dataclass; env prefix convention must be communicated; `.env` file interaction can be surprising |

**Decision:** Use `pydantic-settings` `BaseSettings` as specified in SPEC §5.9.1. `JudgeConfig` lives in `src/healthbench_agent/llm_eval/config.py`.

---

### Decision 16: Prompt Templating — Jinja2 vs f-strings

**Context:** SPEC §5.9.2 specifies Jinja2 for grader prompt templating.

#### Option A — Python f-strings

| | |
|---|---|
| **Pros** | Zero dependency; familiar; simple for basic substitution |
| **Cons** | No conditional blocks; no template inheritance; no auto-escaping; harder to version template separately from code; can't hash template independently of values |

#### Option B — Jinja2

| | |
|---|---|
| **Pros** | Conditional blocks for future prompt variants; template hashing before rendering (stable fingerprint); already a project dependency; standard templating; SPEC mandates it |
| **Cons** | Slightly more complex syntax; overkill for simple substitution; adds learning curve |

**Decision:** **Option B — Jinja2.** As mandated by SPEC §5.9.2. Template is loaded from YAML, hashed before rendering for MLflow fingerprint, then rendered with conversation + rubric_item variables.

---

### Decision 17: Statistical Comparison Methods

**Context:** SPEC §5.5 defines four statistical methods for agent comparison.

All four methods are implemented as specified:
1. **Paired bootstrap CI (n=10,000)** — Primary comparison (determines if CI excludes zero)
2. **Paired t-test** — Quick significance check
3. **Cohen's d** — Effect size (practical significance)
4. **Bonferroni correction** — Multiple testing correction

**Decision rule (from SPEC):** An improvement is reported only when 95% bootstrap CI excludes zero AND p < 0.05 after Bonferroni correction.

---

### Decision 18: MLflow Experiment Tracking Schema

**Context:** SPEC §5.4 defines what to log per evaluation run.

**Decision:** Follow SPEC exactly. Each run logs:
- **Parameters:** `agent_name`, `prompt_version`, `model`, `sample_size`, `timestamp`, `grader_provider`, `grader_model`, `grader_temperature`, `grader_prompt_version`, `grader_prompt_sha256`, `eval_mode`
- **Metrics:** `overall_score`, per-theme means, per-axis means
- **Artifacts:** Full results JSON, prompt YAML

---

## Summary: Build Priority Order (Data-Driven)

From SPEC §Phase 1A insights:

| Priority | Component | Subtask | Rationale |
|----------|-----------|---------|-----------|
| **P0** | `prompts/v1_baseline.yaml` | 2.1 | Foundation for all agents; addresses Insight 6 (communication/instruction penalties) |
| **P0** | `agents/baseline_agent/` | 2.1 | Performance floor; required for all comparisons |
| **P0** | `drug_reference()` tool | 2.2 | Insight 1: accuracy penalties dominate (7.0-7.8 pts/item) |
| **P0** | `emergency_flag()` tool | 2.2 | Insight 3: 34% of samples have emergency criteria |
| **P1** | `prompts/v1_clinical.yaml` | 2.2 | Insight 2: data tasks need conservative strategy |
| **P1** | `agents/tool_agent/` | 2.3 | Validates tool-augmented improvement hypothesis |
| **P1** | Reviewer LlmAgent | 2.4-2.5 | Insight 4: critical for hard samples (penalty mass) |
| **P2** | `symptom_checker()` tool | 2.2 | Completeness axis (39%) but lower penalty density |
| **P2** | Triage + Specialist routing | 2.4-2.5 | Insight 5: context pass-through is low-risk |
| **P3** | Golden datasets | 2.6 | Needed for CI gate, but after agents are stable |
