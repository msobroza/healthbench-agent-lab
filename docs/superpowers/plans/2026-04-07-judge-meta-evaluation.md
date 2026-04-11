# Judge Meta-Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a meta-evaluation pipeline that grades a fixed labelled set with one judge, computes a registry of pluggable metrics (gold_score, kappa, calibration, etc.), persists raw verdicts + summary scores, and reuses the existing prompt-optimizer to optimize the grader prompt.

**Architecture:** A new `domain/meta_evaluation.py` adds a `LabelledSample` parent class and lightweight `MetricResults` dataclass; `HealthBenchSample` inherits from it. A new `llm_eval/meta_eval/` subpackage houses the metric registry, runner, filter helpers, `OracleJudge` (in `oracle_judge.py`), `demo_labelled_set` (in `demo_data.py`), and `meta_evaluate()` happy-path. A `MetricResultsView` UX wrapper lives one layer out in `llm_eval/meta_eval/results/view.py` so matplotlib/pandas/pyarrow stay out of the domain layer. A `VerdictCache` + `CachedJudgeGrader` proxy in `llm_eval/cache/` deduplicates judge calls without touching the `JudgeGrader` ABC. A new `meta-evaluate-judge` CLI exposes `run/regenerate/compare/list-metrics/list-metadata-keys/clear-cache` subcommands. The existing `optimize-prompt` CLI gains `--prompt-domain {agent,judge}` to optimize the grader prompt via a new `JudgeAgreementMetric`.

**Tech Stack:** Python 3.11+, pandas, pyarrow, scikit-learn, matplotlib, tqdm (new), dataclasses, argparse, MLflow.

**Spec:** [docs/superpowers/specs/2026-04-07-judge-meta-evaluation-design.md](../specs/2026-04-07-judge-meta-evaluation-design.md)

---

## Implementation Status (updated 2026-04-11)

**Status:** All 29 tasks implemented on branch `feature/judge-meta-evaluation` across 54 commits. PR [msobroza/healthbench-agent-lab#9](https://github.com/msobroza/healthbench-agent-lab/pull/9) is open and ready for merge. Final test suite: **1050 passing, 97% total coverage**. The original `- [ ]` checkboxes in the task bodies below are left untouched so they remain a faithful record of the TDD plan as authored.

### Task → commit map

| Phase | Task | Landing commit(s) |
|---|---|---|
| 1 | 1. `LabelledSample` / `MetricResults` / `SCHEMA_VERSION` | `56c9a72`, `9fd4e01` |
| 1 | 2. `RubricItem` optional SPEC.md fields | `d7f3935`, `95094cc` |
| 1 | 3. `HealthBenchSample` inherits from `LabelledSample` | `1942792` |
| 1 | 4. Audit positional `HealthBenchSample(...)` call sites | folded into `1942792` |
| 2 | 5. `extract_ideal_completion_text` | `3bcc8b1` |
| 3 | 6. `VerdictCache` | `055facc`, `8492c69` |
| 3 | 7. `CachedJudgeGrader` proxy | `6832b79` |
| 4 | 8. `meta_eval.py` registry + `MetricLevel` | `377094a` |
| 4 | 9. `EmptyFilterError` + filter helpers | `480268f` |
| 5 | 10. `gold_score` | `e04a2ed` |
| 5 | 11. `cohens_kappa` + `krippendorff_alpha` | `821b8a0` |
| 5 | 12. `calibration_curve` | `7f71ee9`, `4301d1d` |
| 5 | 13. `per_dimension_confusion` | `d593ddc` |
| 5 | 14. `adversarial_accuracy` / `adversarial_prf1` / `per_criterion_metrics` | `1e073d0`, `16de428` |
| 6 | 15. `OracleJudge` + `demo_labelled_set` | `0de721c` |
| 7 | 16. `run_meta_eval` | `86e9940`, `74d20c9` |
| 8 | 17. `MetricResultsView` (repr/summary/IO) | `d4eb671`, `74827b8` |
| 8 | 18. `compare()` + plot helpers | `38912eb`, `8e61094` |
| 9 | 19. `meta_evaluate()` happy-path + re-exports | `fc35e14`, `4775160` |
| 10 | 20. `cli_meta_eval.py` skeleton + `run` subcommand | `1cf3ad3`, `26357aa`, `5bb1c66` |
| 10 | 21. `regenerate` / `compare` / `clear-cache` / `list-metrics` / `list-metadata-keys` | `1993c6c`, `4424790`, `11a11b2` |
| 10 | 22. `--dry-run` + default-subcommand preprocessing | `6271e76`, `d22826d` |
| 11 | 23. `OptimizationMetric` Protocol | `47910ac` |
| 11 | 24. `JudgeAgreementMetric` + `EndToEndMetric` filters | `6dc54f4` |
| 11 | 25. `--prompt-domain {agent,judge}` flag | `62eaebb`, `6ae014b` |
| 12 | 26. Register `meta-evaluate-judge` console script + `tqdm` dep | `d7d7d27` |
| 12 | 27. CLAUDE.md Project Layout update | `e3fd6d6`, `8d72aea` |
| 12 | 28. `notebooks/04_judge_meta_evaluation.ipynb` | `c04a058`, `6e292f1` |
| 13 | 29. Final lint/type/test/coverage pass | `79d1c2c`, `041713f` |

### Deviations from the plan as originally written

The following layout changes happened during implementation and are reflected in the real codebase, but the task bodies below still reference the original paths:

- **`llm_eval/meta_eval.py` → `llm_eval/meta_eval/` subpackage** (commit `076f82a`). The single-module design from Tasks 8–19 was split into `api.py`, `filters.py`, `oracle_judge.py`, `demo_data.py`, `runner.py`, `verdicts.py`, a `metrics/` subdirectory (`registry.py`, `agreement.py`, `stratified.py`, `adversarial.py`), and a `results/` subdirectory (see below) once the module crossed the maintainability threshold. Public re-exports on `healthbench_agent.llm_eval.meta_eval` are unchanged — consumers import the same symbols.
- **`llm_eval/meta_eval_results.py` → `llm_eval/meta_eval/results/` subpackage**: `view.py` (`MetricResultsView`), `io.py` (`save_results`/`load_results` free functions), `plots.py` (plotting helpers). Splitting out IO and plots keeps `view.py` free of matplotlib/pyarrow imports.
- **`llm_eval/verdict_cache.py` → `llm_eval/cache/` subpackage** (commit `076f82a`): `cache/store.py` owns `VerdictCache`, `cache/cached_judge.py` owns `CachedJudgeGrader`.
- **`llm_eval/cli_meta_eval.py` → `llm_eval/cli/meta_eval.py`** (commit `076f82a`), alongside `cli/track_experiment.py`. Console-script entry points moved into a `cli/` subpackage.
- **`Sampler` → `LLMClient` rename** (commit `d0a14ea`): The `SamplerBase`/`OpenAIChatSampler`/`GeminiChatSampler` family was renamed to `LLMClient`/`OpenAIChatClient`/`GeminiChatClient` mid-implementation and factored into `llm_eval/clients/`. This rename touches Tasks 6–16 examples that use `SamplerBase` — the real code uses `LLMClient`.
- **`llm_eval/grading/` subpackage** (commit `076f82a`): `grader.py` + `config_grader.py` were split into `grading/judge.py` + `grading/config.py`.
- **Task 29 coverage floor raised from 80% → 95%** (commit `041713f`). CLAUDE.md's testing section now mandates 95% per module; all branch-new modules are at 100% and pre-existing sub-95% modules (`agent/adapters/adk_adapter.py`, `agent/prompt.py`, `analysis/utils.py`, `llm_eval/cli/track_experiment.py`, `llm_eval/clients/*`, `llm_eval/grading/config.py`, `prompt_optimization/cli.py`, `prompt_optimization/adapters/dspy_adapter.py`) are documented as out-of-scope pre-existing gaps.
- **Extra in-flight fixups** that were not separate tasks but landed on the branch during implementation: `4775160` (4 code-review findings on the meta-evaluation feature), `74d20c9` (cache safety / pyarrow / tqdm import hardening), `8e61094` (JSON-roundtripped calibration keys), `8492c69` (atomic cache writes), `9fd4e01`/`95094cc` (ruff-format follow-ups), `4301d1d` (calibration SE handling for n<2), and `16de428` (stronger adversarial test assertions).

### Where to continue

All planned work is landed and pushed. Remaining activity for this branch:

1. **Address code-review comments on PR #9** (posted 2026-04-11, commit `041713f`): four test-quality findings — private-helper testing in `test_judge_agreement_metric_default_build_judge_constructs_llm_judge`, tautological `xdata == sorted(xdata)` in `test_plot_calibration_curve_falls_back_to_string_sort_on_mixed_keys`, weak `>= 2` count in `test_cli_list_metadata_keys_prints_none_when_metadata_empty`, and log-string assertion in `test_cli_regenerate_skips_level_with_no_matching_rows`. See the review comment on PR #9 for specific guidance.
2. **Final merge** of PR #9 into `main` once the four review findings are resolved.
3. **Follow-up work** tracked in the Out of Scope section at the bottom of this plan (Anthropic sampler, hand-labelled gold set, noise-floor hook, judge batch mode, rubric optimization) remains for future issues.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/healthbench_agent/domain/meta_evaluation.py` | Create | `LabelledSample`, `SCHEMA_VERSION`, `MetricResults` (stdlib only) |
| `src/healthbench_agent/domain/rubric.py` | Modify | Optional SPEC.md fields; tolerate missing `tags` |
| `src/healthbench_agent/domain/dataset.py` | Modify | `HealthBenchSample` inherits from `LabelledSample` |
| `src/healthbench_agent/dataset/extraction.py` | Create | `extract_ideal_completion_text` HealthBench glue |
| `src/healthbench_agent/llm_eval/cache/store.py` + `cache/cached_judge.py` | Create | `VerdictCache` + `CachedJudgeGrader` proxy |
| `src/healthbench_agent/llm_eval/meta_eval/` | Create | Subpackage: registry, 8 metrics (`metrics/`), `run_meta_eval` (`runner.py`), filters (`filters.py`), `OracleJudge` (`oracle_judge.py`), `demo_labelled_set` (`demo_data.py`), `meta_evaluate` (`api.py`) |
| `src/healthbench_agent/llm_eval/meta_eval/results/view.py` + `results/io.py` + `results/plots.py` | Create | `MetricResultsView` UX wrapper + save/load free functions + plot helpers |
| `src/healthbench_agent/llm_eval/cli/meta_eval.py` | Create | `meta-evaluate-judge` argparse CLI |
| `src/healthbench_agent/llm_eval/__init__.py` | Modify | Re-export new public symbols |
| `src/healthbench_agent/prompt_optimization/optimizer.py` | Modify | Add `OptimizationMetric` Protocol |
| `src/healthbench_agent/prompt_optimization/metric.py` | Modify | `JudgeAgreementMetric`; filter args on `EndToEndMetric`; re-export `EmptyFilterError` |
| `src/healthbench_agent/prompt_optimization/cli.py` | Modify | `--prompt-domain {agent,judge}` flag + judge save path |
| `pyproject.toml` | Modify | Add `tqdm` dep; register `meta-evaluate-judge` script |
| `CLAUDE.md` | Modify | Document new modules in Project Layout |
| `tests/domain/test_meta_evaluation.py` | Create | LabelledSample / MetricResults dataclass tests |
| `tests/domain/test_rubric.py` | Modify | Tag-less from_dict + SPEC.md field tests |
| `tests/dataset/test_extraction.py` | Create | `extract_ideal_completion_text` schema variants |
| `tests/llm_eval/test_verdict_cache.py` | Create | Cache + proxy tests |
| `tests/llm_eval/test_meta_eval.py` | Create | Registry, metrics, runner, filters, OracleJudge, CLI smoke |
| `tests/llm_eval/test_meta_eval_results.py` | Create | View tests |
| `tests/prompt_optimization/test_metric.py` | Modify | `JudgeAgreementMetric` + filter tests |
| `notebooks/04_judge_meta_evaluation.ipynb` | Create | Five-cell view-based walkthrough |

---

## Phase 1 — Domain Layer Foundation

### Task 1: Create `LabelledSample`, `SCHEMA_VERSION`, `MetricResults`

**Files:**
- Create: `src/healthbench_agent/domain/meta_evaluation.py`
- Test: `tests/domain/test_meta_evaluation.py`

- [x] **Step 1: Write the failing test**

```python
# tests/domain/test_meta_evaluation.py
"""Tests for the LabelledSample / MetricResults domain types."""
from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.meta_evaluation import (
    SCHEMA_VERSION,
    LabelledSample,
    MetricResults,
)
from healthbench_agent.domain.rubric import RubricItem


def test_labelled_sample_required_fields_only():
    sample = LabelledSample(
        prompt_id="p1",
        prompt=[{"role": "user", "content": "hello"}],
        rubrics=[RubricItem(criterion="says hi", points=1.0)],
    )
    assert sample.gold_response is None
    assert sample.expected == {}
    assert sample.language is None
    assert sample.specialty is None
    assert sample.user_persona is None
    assert sample.metadata == {}


def test_labelled_sample_full_spec_md_fields_round_trip():
    sample = LabelledSample(
        prompt_id="p2",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[],
        gold_response="ideal answer",
        expected={"says hi": True},
        language="en",
        specialty="cardiology",
        user_persona="patient",
        metadata={"clinical_urgency": "emergency"},
    )
    assert sample.gold_response == "ideal answer"
    assert sample.expected == {"says hi": True}
    assert sample.language == "en"
    assert sample.specialty == "cardiology"
    assert sample.user_persona == "patient"
    assert sample.metadata == {"clinical_urgency": "emergency"}


def test_metric_results_to_from_dict_round_trip():
    results = MetricResults(
        scores={"gold_score": 0.873, "cohens_kappa": 0.612},
        n_samples_graded=100,
        n_rubrics_graded=287,
        judge_metadata={"judge_model": "openai/gpt-4.1", "k": 7},
    )
    payload = results.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "verdicts_path" not in payload  # None is omitted
    rebuilt = MetricResults.from_dict(payload)
    assert rebuilt == results


def test_metric_results_to_dict_coerces_path_to_string():
    results = MetricResults(
        scores={},
        n_samples_graded=0,
        n_rubrics_graded=0,
        judge_metadata={},
        verdicts_path=Path("/tmp/run/verdicts.parquet"),
    )
    payload = results.to_dict()
    assert payload["verdicts_path"] == "/tmp/run/verdicts.parquet"
    rebuilt = MetricResults.from_dict(payload)
    assert rebuilt.verdicts_path == Path("/tmp/run/verdicts.parquet")


def test_metric_results_from_dict_rejects_unknown_schema_version():
    payload = {
        "scores": {},
        "n_samples_graded": 0,
        "n_rubrics_graded": 0,
        "judge_metadata": {},
        "schema_version": SCHEMA_VERSION + 1,
    }
    with pytest.raises(ValueError, match="schema_version"):
        MetricResults.from_dict(payload)


def test_metric_results_carries_no_ux_methods():
    """Domain dataclass must stay stdlib-only — no plot_* / to_pandas / load."""
    forbidden = {"plot_calibration_curve", "plot_dimension_confusion", "to_pandas", "load"}
    assert forbidden.isdisjoint(set(dir(MetricResults)))
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/domain/test_meta_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: healthbench_agent.domain.meta_evaluation`.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/domain/meta_evaluation.py
"""Dataset-agnostic types for meta-evaluation of rubric-grading judges.

Defines ``LabelledSample`` (the parent class for any rubric-graded sample)
and ``MetricResults`` (the persisted shape of a meta-evaluation run).
Both are stdlib-only — no matplotlib, pandas, or pyarrow imports — so the
domain layer keeps its narrow dependency policy. The rich UX wrapper for
``MetricResults`` lives one layer out in ``llm_eval/meta_eval_results.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conversation import MessageList
from .rubric import RubricItem

SCHEMA_VERSION: int = 1
"""Bumped when MetricResults persistence format changes incompatibly."""


@dataclass
class LabelledSample:
    """A rubric-graded sample with optional gold labels for meta-evaluation.

    Acts as the dataset-agnostic shape for any rubric grading task. Concrete
    benchmarks subclass this and add their own fields. Meta-evaluation
    operates on lists of LabelledSample (or any subclass) without knowing
    which dataset they came from.

    Attributes:
        prompt_id: Unique identifier for joining samples across runs.
        prompt: Conversation history before the response to be graded.
        rubrics: Rubric items the judge will score the response against.
        gold_response: Known-good response text to grade for meta-evaluation.
            None for unlabelled samples (the typical agent-eval case).
        expected: Expected verdict per rubric criterion text. True = the
            criterion should be met by ``gold_response``, False = should not.
        language: ISO language code from SPEC.md schema. Optional.
        specialty: Medical specialty from SPEC.md schema. Optional.
        user_persona: 'patient' | 'healthcare professional'. Optional.
        metadata: Free-form per-sample metadata dict.
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


@dataclass
class MetricResults:
    """Aggregate meta-evaluation result for one judge run.

    Pure data — no methods that touch matplotlib, pandas, or the
    filesystem. Round-trips to JSON via ``to_dict()`` / ``from_dict()``.
    Wrap in :class:`MetricResultsView` (in
    ``llm_eval/meta_eval_results.py``) to get the rich UX surface.

    Attributes:
        scores: Mapping of metric name to its computed value.
        n_samples_graded: Number of LabelledSamples that produced verdicts.
        n_rubrics_graded: Total (sample, rubric) pairs across all k passes.
        judge_metadata: Run-level header (judge_model, temperature, k, ...).
        schema_version: Stamped from SCHEMA_VERSION at write time.
        verdicts_path: Path of the parquet that produced these scores.
            Populated by MetricResultsView.load; None for in-memory results.
    """

    scores: dict[str, Any]
    n_samples_graded: int
    n_rubrics_graded: int
    judge_metadata: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    verdicts_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict — coerces verdicts_path to a string."""
        payload: dict[str, Any] = {
            "scores": self.scores,
            "n_samples_graded": self.n_samples_graded,
            "n_rubrics_graded": self.n_rubrics_graded,
            "judge_metadata": self.judge_metadata,
            "schema_version": self.schema_version,
        }
        if self.verdicts_path is not None:
            payload["verdicts_path"] = str(self.verdicts_path)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResults:
        """Inverse of :meth:`to_dict`. Validates schema_version.

        Raises:
            ValueError: If schema_version is newer than SCHEMA_VERSION.
        """
        version = data.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"MetricResults schema_version={version} is newer than this "
                f"build understands (max {SCHEMA_VERSION}). Upgrade healthbench-agent."
            )
        verdicts_path = data.get("verdicts_path")
        return cls(
            scores=data["scores"],
            n_samples_graded=data["n_samples_graded"],
            n_rubrics_graded=data["n_rubrics_graded"],
            judge_metadata=data["judge_metadata"],
            schema_version=version,
            verdicts_path=Path(verdicts_path) if verdicts_path is not None else None,
        )
```

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/domain/test_meta_evaluation.py -v`
Expected: PASS (6 tests).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/domain/meta_evaluation.py tests/domain/test_meta_evaluation.py
git commit -m "feat(domain): add LabelledSample and MetricResults dataclasses"
```

---

### Task 2: Extend `RubricItem` with optional SPEC.md fields

**Files:**
- Modify: `src/healthbench_agent/domain/rubric.py`
- Test: `tests/domain/test_rubric.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/domain/test_rubric.py` (create if missing — mirror existing test files in the project):

```python
def test_rubric_item_from_dict_tolerates_missing_tags():
    """SPEC.md rows can omit `tags`; from_dict must default to []."""
    item = RubricItem.from_dict({"criterion": "says hi", "points": 1.0})
    assert item.tags == []


def test_rubric_item_optional_spec_md_fields_default_to_none():
    item = RubricItem(criterion="says hi", points=1.0)
    assert item.criterion_id is None
    assert item.category is None
    assert item.example_meets is None
    assert item.example_fails is None


def test_rubric_item_from_dict_reads_spec_md_fields():
    item = RubricItem.from_dict({
        "criterion": "states emergency referral",
        "points": 5.0,
        "tags": ["axis: emergency"],
        "criterion_id": "C-001",
        "category": "emergency",
        "example_meets": "Call 911 immediately.",
        "example_fails": "Take some aspirin.",
    })
    assert item.criterion_id == "C-001"
    assert item.category == "emergency"
    assert item.example_meets == "Call 911 immediately."
    assert item.example_fails == "Take some aspirin."


def test_rubric_item_from_dict_ignores_unknown_keys():
    item = RubricItem.from_dict({
        "criterion": "x",
        "points": 1.0,
        "tags": [],
        "future_key": "ignored",
    })
    assert item.criterion == "x"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/domain/test_rubric.py -v -k "tolerates_missing_tags or spec_md"`
Expected: FAIL — `tags` access raises KeyError, attribute lookups raise AttributeError.

- [x] **Step 3: Write minimal implementation**

Replace [src/healthbench_agent/domain/rubric.py](../../../src/healthbench_agent/domain/rubric.py) with:

```python
"""HealthBench rubric item domain model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricItem:
    """One graded criterion within a rubric.

    The original HealthBench fields (criterion, points, tags) are unchanged.
    The SPEC.md schema fields below are optional with safe defaults so
    HealthBench loaders work unchanged.

    Attributes:
        criterion: Human-readable statement of what the criterion checks.
        points: Points awarded (positive) or deducted (negative) when met.
        tags: HealthBench-style tag list (e.g. ['axis: accuracy']).
        criterion_id: Stable id from the SPEC.md schema. None for HealthBench.
        category: Explicit category/axis name from SPEC.md.
        example_meets: Adversarial known-good response. When present,
            meta-eval grades it expecting criteria_met=True.
        example_fails: Adversarial known-bad response. When present,
            meta-eval grades it expecting criteria_met=False.
    """

    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)
    criterion_id: str | None = None
    category: str | None = None
    example_meets: str | None = None
    example_fails: str | None = None

    def __str__(self) -> str:
        return f"[{self.points}] {self.criterion}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        payload: dict[str, Any] = {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }
        for key in ("criterion_id", "category", "example_meets", "example_fails"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricItem:
        """Deserialize from a JSON-compatible dict.

        Tolerates missing ``tags`` (SPEC.md rows may omit it) and reads the
        optional SPEC.md fields when present.
        """
        return cls(
            criterion=data["criterion"],
            points=data["points"],
            tags=data.get("tags", []),
            criterion_id=data.get("criterion_id"),
            category=data.get("category"),
            example_meets=data.get("example_meets"),
            example_fails=data.get("example_fails"),
        )
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/domain/test_rubric.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/domain/rubric.py tests/domain/test_rubric.py
git commit -m "feat(domain): add optional SPEC.md fields to RubricItem"
```

---

### Task 3: Migrate `HealthBenchSample` to inherit from `LabelledSample`

**Files:**
- Modify: `src/healthbench_agent/domain/dataset.py`
- Test: `tests/domain/test_meta_evaluation.py`

- [x] **Step 1: Write the failing test**

Append to `tests/domain/test_meta_evaluation.py`:

```python
from healthbench_agent.domain.dataset import HealthBenchSample


def test_health_bench_sample_is_labelled_sample():
    sample = HealthBenchSample(
        prompt_id="p1",
        prompt=[{"role": "user", "content": "hi"}],
        rubrics=[],
        example_tags=["theme: general"],
    )
    assert isinstance(sample, LabelledSample)


def test_health_bench_sample_keyword_construction_still_works():
    sample = HealthBenchSample(
        prompt_id="p1",
        prompt=[],
        rubrics=[],
        example_tags=["theme: emergency"],
        ideal_completions_data={"ideal_completion": "..."},
        canary="healthbench:abc",
    )
    assert sample.example_tags == ["theme: emergency"]
    assert sample.canary == "healthbench:abc"
    # Inherited defaults present.
    assert sample.gold_response is None
    assert sample.expected == {}


def test_health_bench_sample_from_dict_populates_all_fields():
    row = {
        "prompt_id": "p9",
        "prompt": [{"role": "user", "content": "hi"}],
        "rubrics": [{"criterion": "c", "points": 1.0, "tags": []}],
        "example_tags": ["theme: general"],
        "ideal_completions_data": {"ideal_completion": "x"},
        "canary": "healthbench:1",
    }
    sample = HealthBenchSample.from_dict(row)
    assert sample.prompt_id == "p9"
    assert sample.example_tags == ["theme: general"]
    assert sample.gold_response is None  # populated later by CLI


def test_loaded_health_bench_sample_can_set_gold_fields():
    sample = HealthBenchSample(
        prompt_id="p1", prompt=[], rubrics=[], example_tags=[]
    )
    sample.gold_response = "the ideal answer"
    sample.expected = {"c1": True}
    assert sample.gold_response == "the ideal answer"
    assert sample.expected == {"c1": True}
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/domain/test_meta_evaluation.py::test_health_bench_sample_is_labelled_sample -v`
Expected: FAIL — `HealthBenchSample` does not inherit from `LabelledSample`.

- [x] **Step 3: Write minimal implementation**

Replace [src/healthbench_agent/domain/dataset.py](../../../src/healthbench_agent/domain/dataset.py) `HealthBenchSample` class with:

```python
"""HealthBench dataset domain types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .meta_evaluation import LabelledSample
from .rubric import RubricItem

DatasetSubset = Literal["main", "hard", "consensus"]


@dataclass
class HealthBenchSample(LabelledSample):
    """One sample loaded from a HealthBench JSONL dataset file.

    Inherits prompt_id, prompt, rubrics, gold_response, expected,
    language, specialty, user_persona, and metadata from LabelledSample.
    Adds HealthBench-specific fields.

    Attributes:
        example_tags: Dataset-level tags for stratified scoring.
        ideal_completions_data: Physician ideal completion data when
            available. Used to populate gold_response/expected at
            meta-eval time via the CLI.
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


@dataclass
class HealthBenchDataset:
    """A loaded HealthBench dataset subset with its samples and metadata."""

    subset: DatasetSubset
    samples: list[HealthBenchSample]
    source_path: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/domain/ -v`
Expected: PASS (LabelledSample, MetricResults, HealthBenchSample tests).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/domain/dataset.py tests/domain/test_meta_evaluation.py
git commit -m "refactor(domain): make HealthBenchSample inherit from LabelledSample"
```

---

### Task 4: Audit and migrate positional `HealthBenchSample(...)` call sites

**Files:**
- Search: `**/*.py`
- Modify: any file that constructs `HealthBenchSample` positionally past argument 3
- Test: existing test suite must still pass

- [x] **Step 1: Find all construction sites**

Run: `uv run pytest tests/ -v 2>&1 | tail -50`
Then `Grep` for `HealthBenchSample(` across both `src/`, `tests/`, and `agents/` and inspect each call.

Use Grep with pattern `HealthBenchSample\(` to enumerate.

- [x] **Step 2: Migrate each positional call to keyword form**

For each match where the 4th positional argument was previously `example_tags`, rewrite to keyword form:

```python
# before
sample = HealthBenchSample("p1", prompt, rubrics, ["theme: general"])
# after
sample = HealthBenchSample(
    prompt_id="p1",
    prompt=prompt,
    rubrics=rubrics,
    example_tags=["theme: general"],
)
```

- [x] **Step 3: Run the full domain + dataset test suite**

Run: `uv run pytest tests/domain/ tests/dataset/ -v`
Expected: PASS.

- [x] **Step 4: Run the full project test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS — Tasks 1-3 should not have broken anything else.

- [x] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: migrate HealthBenchSample call sites to keyword form"
```

---

## Phase 2 — Dataset Extraction Helper

### Task 5: Create `extract_ideal_completion_text`

**Files:**
- Create: `src/healthbench_agent/dataset/extraction.py`
- Test: `tests/dataset/test_extraction.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/dataset/test_extraction.py
"""Tests for extract_ideal_completion_text schema variants."""
from __future__ import annotations

from healthbench_agent.dataset.extraction import extract_ideal_completion_text


def test_extract_returns_none_when_data_is_none():
    assert extract_ideal_completion_text(None) is None


def test_extract_returns_none_for_empty_dict():
    assert extract_ideal_completion_text({}) is None


def test_extract_handles_string_ideal_completion():
    data = {"ideal_completion": "the ideal answer"}
    assert extract_ideal_completion_text(data) == "the ideal answer"


def test_extract_handles_list_of_role_content_dicts():
    data = {"ideal_completion": [{"role": "assistant", "content": "ans"}]}
    assert extract_ideal_completion_text(data) == "ans"


def test_extract_handles_plural_ideal_completions():
    data = {"ideal_completions": [{"role": "assistant", "content": "ans2"}]}
    assert extract_ideal_completion_text(data) == "ans2"


def test_extract_returns_none_when_no_known_keys_match():
    assert extract_ideal_completion_text({"unknown": "x"}) is None


def test_extract_returns_none_when_list_has_no_assistant_turn():
    data = {"ideal_completion": [{"role": "user", "content": "?"}]}
    assert extract_ideal_completion_text(data) is None
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset/test_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/dataset/extraction.py
"""HealthBench glue: extract gold response text from a sample's
``ideal_completions_data`` block.

Lives next to ``loader.py`` and ``split_utils.py`` so the ``domain/`` and
``llm_eval/`` layers stay HealthBench-agnostic. The CLI is the only caller.
"""
from __future__ import annotations

from typing import Any


def extract_ideal_completion_text(
    ideal_completions_data: dict[str, Any] | None,
) -> str | None:
    """Pull the gold response text out of HealthBench's
    ``ideal_completions_data`` block.

    HealthBench ships physician ideal completions under several schema
    variants depending on subset version. This helper normalises them
    all to a single ``str``, returning ``None`` when the block is
    missing or every variant fails to parse.

    Recognised shapes:
        * ``{"ideal_completion": "..."}``       (string)
        * ``{"ideal_completion": [{"role", "content"}]}``   (message list)
        * ``{"ideal_completions": [...]}``      (plural variant)

    Args:
        ideal_completions_data: The raw dict from
            ``HealthBenchSample.ideal_completions_data``. May be None.

    Returns:
        The extracted gold response text, or None when extraction fails.
    """
    if not ideal_completions_data:
        return None
    for key in ("ideal_completion", "ideal_completions"):
        if key not in ideal_completions_data:
            continue
        value = ideal_completions_data[key]
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for turn in value:
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    content = turn.get("content")
                    if isinstance(content, str):
                        return content
    return None
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/dataset/test_extraction.py -v`
Expected: PASS (7 tests).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/dataset/extraction.py tests/dataset/test_extraction.py
git commit -m "feat(dataset): add extract_ideal_completion_text helper"
```

---

## Phase 3 — Verdict Cache

### Task 6: Create `VerdictCache`

**Files:**
- Create: `src/healthbench_agent/llm_eval/cache/store.py`
- Test: `tests/llm_eval/test_verdict_cache.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/llm_eval/test_verdict_cache.py
"""Tests for VerdictCache and CachedJudgeGrader."""
from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.llm_eval.cache.store import VerdictCache


@pytest.fixture
def cache(tmp_path: Path) -> VerdictCache:
    return VerdictCache(root=tmp_path / "cache", enabled=True)


def _conv() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_disabled_cache_get_always_returns_none(tmp_path: Path):
    c = VerdictCache(root=tmp_path / "x", enabled=False)
    key = c.make_key("m", "sha", _conv(), "rt", 1)
    assert c.get(key) is None


def test_disabled_cache_put_writes_nothing(tmp_path: Path):
    root = tmp_path / "x"
    c = VerdictCache(root=root, enabled=False)
    key = c.make_key("m", "sha", _conv(), "rt", 1)
    c.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert not root.exists() or not any(root.iterdir())


def test_make_key_is_deterministic_across_instances(tmp_path: Path):
    c1 = VerdictCache(root=tmp_path / "a")
    c2 = VerdictCache(root=tmp_path / "b")
    k1 = c1.make_key("m", "sha", _conv(), "rt", 1)
    k2 = c2.make_key("m", "sha", _conv(), "rt", 1)
    assert k1 == k2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"judge_model": "m2"},
        {"judge_prompt_sha": "sha2"},
        {"conversation": [{"role": "user", "content": "different"}]},
        {"rubric_text": "different"},
        {"k_index": 2},
    ],
)
def test_make_key_changes_when_any_input_changes(cache: VerdictCache, kwargs):
    base = dict(judge_model="m", judge_prompt_sha="sha", conversation=_conv(), rubric_text="rt", k_index=1)
    base_key = cache.make_key(**base)
    other_key = cache.make_key(**{**base, **kwargs})
    assert base_key != other_key


def test_put_then_get_round_trips(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    verdict = CriterionVerdict(criterion="rt", criteria_met=True, explanation="ok")
    cache.put(key, verdict)
    assert cache.get(key) == verdict


def test_get_missing_key_returns_none(cache: VerdictCache):
    assert cache.get("nonexistent") is None


def test_files_are_sharded_by_first_two_hex_chars(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    shard = key[:2]
    assert (cache.root / shard).is_dir()


def test_clear_removes_cache_root(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    cache.clear()
    assert not cache.root.exists() or not any(cache.root.iterdir())


def test_stats_reports_hits_misses_and_size(cache: VerdictCache):
    key = cache.make_key("m", "sha", _conv(), "rt", 1)
    assert cache.get(key) is None  # miss
    cache.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert cache.get(key) is not None  # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size_bytes"] > 0


def test_two_instances_share_cache_files(tmp_path: Path):
    root = tmp_path / "shared"
    a = VerdictCache(root=root)
    b = VerdictCache(root=root)
    key = a.make_key("m", "sha", _conv(), "rt", 1)
    a.put(key, CriterionVerdict(criterion="rt", criteria_met=True, explanation=""))
    assert b.get(key) is not None
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_verdict_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/llm_eval/cache/store.py
"""File-based cache for individual judge verdicts.

Enables iterating on metric definitions, filters, or output formats without
re-paying the LLM. Cache key is

    sha256(judge_model || judge_prompt_sha || conversation_hash || k_index || rubric_text)

where conversation_hash is sha256 of the JSON-serialised MessageList.

Includes ``CachedJudgeGrader``, a thin proxy that wraps any
``JudgeGrader`` and consults the cache before delegating, so the
``JudgeGrader`` ABC stays unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem


def _default_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "healthbench_agent" / "verdicts"


class VerdictCache:
    """File-based cache for individual judge verdicts.

    Each cache entry is one tiny JSON file under ``root/first2/rest.json``,
    like git's loose object format.
    """

    def __init__(
        self,
        root: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.root = root or _default_cache_root()
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        judge_model: str,
        judge_prompt_sha: str,
        conversation: MessageList,
        rubric_text: str,
        k_index: int,
    ) -> str:
        """Compute the deterministic sha256 cache key."""
        conv_hash = hashlib.sha256(
            json.dumps(conversation, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payload = "||".join(
            [judge_model, judge_prompt_sha, conv_hash, str(k_index), rubric_text]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key[2:]}.json"

    def get(self, key: str) -> CriterionVerdict | None:
        if not self.enabled:
            return None
        path = self._path_for(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self._misses += 1
            return None
        self._hits += 1
        return CriterionVerdict(
            criterion=data["criterion"],
            criteria_met=data["criteria_met"],
            explanation=data.get("explanation", ""),
            confidence=data.get("confidence"),
        )

    def put(self, key: str, verdict: CriterionVerdict) -> None:
        if not self.enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(verdict)))

    def clear(self) -> None:
        """Delete every cached verdict."""
        if not self.root.exists():
            return
        for shard in self.root.iterdir():
            if shard.is_dir():
                for entry in shard.iterdir():
                    entry.unlink()
                shard.rmdir()
        self.root.rmdir()

    def stats(self) -> dict[str, int]:
        size = 0
        if self.root.exists():
            for shard in self.root.iterdir():
                if shard.is_dir():
                    for entry in shard.iterdir():
                        size += entry.stat().st_size
        return {"hits": self._hits, "misses": self._misses, "size_bytes": size}
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_verdict_cache.py -v`
Expected: PASS (10+ tests).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/cache/store.py tests/llm_eval/test_verdict_cache.py
git commit -m "feat(llm_eval): add file-based VerdictCache"
```

---

### Task 7: Add `CachedJudgeGrader` proxy

**Files:**
- Create: `src/healthbench_agent/llm_eval/cache/cached_judge.py`
- Test: `tests/llm_eval/test_verdict_cache.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/llm_eval/test_verdict_cache.py`:

```python
from unittest.mock import MagicMock

from healthbench_agent.llm_eval.cache.cached_judge import CachedJudgeGrader


def _make_inner(verdicts: list[CriterionVerdict]) -> MagicMock:
    inner = MagicMock(spec=["grade"])
    inner.grade.return_value = verdicts
    return inner


def test_cached_proxy_first_call_delegates(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    proxy = CachedJudgeGrader(inner, cache, "openai/gpt-4.1@1.0", "sha1", k_index=1)
    out = proxy.grade(_conv(), [RubricItem(criterion="r1", points=1.0)])
    assert out == [verdict]
    assert inner.grade.call_count == 1


def test_cached_proxy_second_call_hits_cache(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    proxy = CachedJudgeGrader(inner, cache, "openai/gpt-4.1@1.0", "sha1", k_index=1)
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    proxy.grade(_conv(), rubrics)
    proxy.grade(_conv(), rubrics)
    assert inner.grade.call_count == 1


def test_cached_proxy_distinct_k_index_yields_separate_entries(cache: VerdictCache):
    verdict = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    inner = _make_inner([verdict])
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    CachedJudgeGrader(inner, cache, "m", "s", k_index=1).grade(_conv(), rubrics)
    CachedJudgeGrader(inner, cache, "m", "s", k_index=2).grade(_conv(), rubrics)
    assert inner.grade.call_count == 2  # both missed


def test_cached_proxy_disabled_cache_always_delegates(tmp_path: Path):
    disabled = VerdictCache(root=tmp_path / "off", enabled=False)
    inner = _make_inner([CriterionVerdict(criterion="r1", criteria_met=True, explanation="")])
    proxy = CachedJudgeGrader(inner, disabled, "m", "s", k_index=1)
    rubrics = [RubricItem(criterion="r1", points=1.0)]
    proxy.grade(_conv(), rubrics)
    proxy.grade(_conv(), rubrics)
    assert inner.grade.call_count == 2


def test_cached_proxy_batches_misses_into_one_inner_call(cache: VerdictCache):
    v1 = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    v2 = CriterionVerdict(criterion="r2", criteria_met=False, explanation="")
    inner = _make_inner([v1, v2])
    proxy = CachedJudgeGrader(inner, cache, "m", "s", k_index=1)
    out = proxy.grade(_conv(), [RubricItem(criterion="r1", points=1.0), RubricItem(criterion="r2", points=1.0)])
    assert out == [v1, v2]
    assert inner.grade.call_count == 1
    args, _ = inner.grade.call_args
    assert len(args[1]) == 2  # both rubrics passed in one delegated call


def test_cached_proxy_preserves_input_rubric_order(cache: VerdictCache):
    v1 = CriterionVerdict(criterion="r1", criteria_met=True, explanation="")
    v2 = CriterionVerdict(criterion="r2", criteria_met=False, explanation="")
    # Pre-populate r1, force a miss on r2.
    cached_key = cache.make_key("m", "s", _conv(), "r1", 1)
    cache.put(cached_key, v1)
    inner = _make_inner([v2])  # only the missed one
    proxy = CachedJudgeGrader(inner, cache, "m", "s", k_index=1)
    out = proxy.grade(_conv(), [RubricItem(criterion="r1", points=1.0), RubricItem(criterion="r2", points=1.0)])
    assert out[0] == v1
    assert out[1] == v2
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_verdict_cache.py -v -k "cached_proxy"`
Expected: FAIL — `CachedJudgeGrader` does not exist.

- [x] **Step 3: Write minimal implementation**

Create `src/healthbench_agent/llm_eval/cache/cached_judge.py`:

```python
class CachedJudgeGrader(JudgeGrader):
    """JudgeGrader proxy that consults a VerdictCache before delegating.

    Wraps any JudgeGrader and intercepts ``grade()`` to deduplicate
    verdicts. The cache key components that are constant across one
    k-pass (model fingerprint, prompt sha, k_index) are baked in at
    construction time so the proxy can compute the full key from just
    the (conversation, rubric_text) tuple — no introspection of the
    inner grader required.
    """

    def __init__(
        self,
        inner: JudgeGrader,
        cache: VerdictCache,
        model_fingerprint: str,
        prompt_sha: str,
        k_index: int,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.model_fingerprint = model_fingerprint
        self.prompt_sha = prompt_sha
        self.k_index = k_index

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """Look up cached verdicts; batch the misses into one inner.grade() call."""
        cached: dict[int, CriterionVerdict] = {}
        miss_indices: list[int] = []
        miss_items: list[RubricItem] = []
        for idx, item in enumerate(rubric_items):
            key = self.cache.make_key(
                self.model_fingerprint,
                self.prompt_sha,
                conversation,
                item.criterion,
                self.k_index,
            )
            hit = self.cache.get(key)
            if hit is not None:
                cached[idx] = hit
            else:
                miss_indices.append(idx)
                miss_items.append(item)

        if miss_items:
            fresh = self.inner.grade(conversation, miss_items)
            for miss_idx, item, verdict in zip(miss_indices, miss_items, fresh, strict=True):
                key = self.cache.make_key(
                    self.model_fingerprint,
                    self.prompt_sha,
                    conversation,
                    item.criterion,
                    self.k_index,
                )
                self.cache.put(key, verdict)
                cached[miss_idx] = verdict

        return [cached[i] for i in range(len(rubric_items))]
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_verdict_cache.py -v`
Expected: PASS (all proxy tests).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/cache/cached_judge.py tests/llm_eval/test_verdict_cache.py
git commit -m "feat(llm_eval): add CachedJudgeGrader proxy"
```

---

## Phase 4 — Meta-Eval Module Skeleton

### Task 8: Create `meta_eval/` subpackage with `MetricLevel`, `MetricSpec`, registry

**Files:**
- Create: `src/healthbench_agent/llm_eval/meta_eval/metrics/registry.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/llm_eval/test_meta_eval.py
"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""
from __future__ import annotations

import pytest

from healthbench_agent.llm_eval.meta_eval import (
    MetricLevel,
    MetricSpec,
    get_meta_metric,
    register_meta_metric,
    registered_meta_metrics,
)


def test_metric_level_enum_values():
    assert MetricLevel.SAMPLE.value == "sample"
    assert MetricLevel.RUBRIC.value == "rubric"
    assert MetricLevel.ANY.value == "any"


def test_register_meta_metric_round_trip():
    @register_meta_metric("dummy_metric", level=MetricLevel.ANY, description="dummy")
    def _fn(df):
        return 0.0

    spec = get_meta_metric("dummy_metric")
    assert isinstance(spec, MetricSpec)
    assert spec.name == "dummy_metric"
    assert spec.level is MetricLevel.ANY
    assert spec.description == "dummy"
    assert spec.fn is _fn
    # Cleanup
    del registered_meta_metrics()["dummy_metric"]


def test_register_meta_metric_requires_level_and_description():
    with pytest.raises(TypeError):

        @register_meta_metric("missing_args")  # type: ignore[call-arg]
        def _bad(df):
            return 0.0


def test_get_meta_metric_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_meta_metric("totally_unknown_metric")
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v`
Expected: FAIL — module does not exist.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/llm_eval/meta_eval/metrics/registry.py
"""Meta-evaluation registry, runner, and built-in metrics for LLM-as-judge.

The subpackage is dataset-agnostic: it operates on lists of LabelledSample.
HealthBench-specific glue (subset loading, ideal completion extraction)
lives in ``llm_eval/cli/meta_eval.py``.

Adding a new metric is one decorated function — name, level, description,
and the pure DataFrame transform. Zero changes anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd


class MetricLevel(str, Enum):
    """Which row subset a metric operates on."""

    SAMPLE = "sample"
    RUBRIC = "rubric"
    ANY = "any"


Metric = Callable[["pd.DataFrame"], Any]


@dataclass(frozen=True)
class MetricSpec:
    """Registered metric metadata.

    Attributes:
        name: Unique identifier (used by --metrics CLI flag and metrics.json).
        fn: The pure metric function. Takes a level-filtered DataFrame.
        level: Which gold_source rows the runner passes to ``fn``.
        description: One-line human-readable summary for ``list-metrics``.
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
    """Decorator that registers a meta-evaluation metric by name + level."""

    def decorator(fn: Metric) -> Metric:
        _METRIC_REGISTRY[name] = MetricSpec(
            name=name, fn=fn, level=level, description=description
        )
        return fn

    return decorator


def get_meta_metric(name: str) -> MetricSpec:
    """Look up a registered metric by name.

    Raises:
        KeyError: When ``name`` is not registered.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric {name!r} is not registered. Available: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def registered_meta_metrics() -> dict[str, MetricSpec]:
    """Return the live registry mapping (mutating it affects the registry)."""
    return _METRIC_REGISTRY
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/registry.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add meta_eval registry and MetricLevel"
```

---

### Task 9: Add `EmptyFilterError` and filter helpers

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/filters.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/llm_eval/test_meta_eval.py`:

```python
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.meta_eval import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    axis_filter,
    metadata_filter,
    specialty_filter,
)


def test_axis_tag_prefix_includes_trailing_space():
    assert AXIS_TAG_PREFIX == "axis: "


def test_axis_filter_matches_tag():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: accuracy"]))
    assert not f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_axis_filter_matches_category_field():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, category="accuracy"))


def test_axis_filter_accepts_multiple_axes():
    f = axis_filter("accuracy", "emergency")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_metadata_filter_top_level_fields():
    f = metadata_filter(language="en", specialty="cardiology")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], language="en", specialty="cardiology"
    )
    assert f(sample)
    other = LabelledSample(prompt_id="p", prompt=[], rubrics=[], language="fr", specialty="cardiology")
    assert not f(other)


def test_metadata_filter_metadata_dict_keys():
    f = metadata_filter(clinical_urgency="emergency")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "emergency"}
    )
    assert f(sample)
    other = LabelledSample(prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "routine"})
    assert not f(other)


def test_specialty_filter_matches_any():
    f = specialty_filter("cardiology", "pediatrics")
    assert f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="cardiology"))
    assert not f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="general"))


def test_empty_filter_error_carries_filter_reprs():
    err = EmptyFilterError(sample_filter="metadata_filter(language='fr')", rubric_filter=None)
    assert "language='fr'" in str(err)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "filter or empty_filter or axis"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/filters.py`:

```python
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem

AXIS_TAG_PREFIX = "axis: "
"""Prefix used by HealthBench's stratified-sample helper for axis tags.

The trailing space matches the form written by
``healthbench_agent.dataset.split_utils._extract_stratum``. Shared with
the CLI's ``axis_extractor`` so the two helpers cannot drift.
"""


class EmptyFilterError(ValueError):
    """Raised when a sample/rubric filter combination eliminates all rows.

    Carries the names (or repr) of the active filters so the CLI can show
    the user exactly which flags caused the empty result.
    """

    def __init__(self, sample_filter: Any, rubric_filter: Any) -> None:
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter
        super().__init__(
            f"Empty filter result. sample_filter={sample_filter!r}, "
            f"rubric_filter={rubric_filter!r}"
        )


def axis_filter(*axes: str) -> Callable[[RubricItem], bool]:
    """Keep rubrics whose ``category`` or ``axis: <name>`` tag matches any of *axes*."""
    wanted = set(axes)

    def predicate(item: RubricItem) -> bool:
        if item.category in wanted:
            return True
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX) and tag[len(AXIS_TAG_PREFIX):].strip() in wanted:
                return True
        return False

    return predicate


def metadata_filter(**conditions: Any) -> Callable[[LabelledSample], bool]:
    """Keep samples where every metadata key equals the given value.

    Top-level LabelledSample attributes (``language``, ``specialty``,
    ``user_persona``) are checked against the attribute; other keys are
    looked up in ``sample.metadata``.
    """
    top_level = {"language", "specialty", "user_persona"}

    def predicate(sample: LabelledSample) -> bool:
        for key, value in conditions.items():
            if key in top_level:
                if getattr(sample, key) != value:
                    return False
            else:
                if sample.metadata.get(key) != value:
                    return False
        return True

    return predicate


def specialty_filter(*specialties: str) -> Callable[[LabelledSample], bool]:
    """Keep samples whose ``specialty`` field is in *specialties*."""
    wanted = set(specialties)

    def predicate(sample: LabelledSample) -> bool:
        return sample.specialty in wanted

    return predicate
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/filters.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add EmptyFilterError and filter helpers to meta_eval"
```

---

## Phase 5 — Built-in Metrics

### Task 10: Implement `gold_score`

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/llm_eval/test_meta_eval.py`:

```python
import pandas as pd

from healthbench_agent.llm_eval.meta_eval import gold_score


def _sample_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_gold_score_perfect_judge_returns_one():
    df = _sample_rows([
        {"prompt_id": "p1", "sample_k": 1, "criterion": "c1", "points": 1.0, "observed_met": True},
        {"prompt_id": "p1", "sample_k": 1, "criterion": "c2", "points": 2.0, "observed_met": True},
    ])
    assert gold_score(df) == pytest.approx(1.0)


def test_gold_score_clips_per_conversation_to_zero_when_negative():
    df = _sample_rows([
        # raw = -3 / 1 = -3 → clipped to 0
        {"prompt_id": "p1", "sample_k": 1, "criterion": "c1", "points": 1.0, "observed_met": False},
        {"prompt_id": "p1", "sample_k": 1, "criterion": "c2", "points": -3.0, "observed_met": True},
        # second conversation: perfect, raw = 1
        {"prompt_id": "p2", "sample_k": 1, "criterion": "c3", "points": 1.0, "observed_met": True},
    ])
    # mean(0.0, 1.0) == 0.5
    assert gold_score(df) == pytest.approx(0.5)


def test_gold_score_empty_dataframe_returns_zero():
    df = pd.DataFrame(
        columns=["prompt_id", "sample_k", "criterion", "points", "observed_met"]
    )
    assert gold_score(df) == 0.0


def test_gold_score_registered_with_sample_level():
    spec = get_meta_metric("gold_score")
    assert spec.level is MetricLevel.SAMPLE
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k gold_score`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py`:

```python
from statistics import fmean

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.scoring import calculate_score, clip_score


@register_meta_metric(
    "gold_score",
    level=MetricLevel.SAMPLE,
    description="Mean clipped HealthBench score on gold responses (target = 1.0)",
)
def gold_score(verdicts: "pd.DataFrame") -> float:
    """Mean clipped HealthBench score the judge gives to gold responses.

    Rebuilds (rubrics, verdicts) lists per (prompt_id, sample_k) group
    and delegates to ``calculate_score`` + ``clip_score`` so meta-eval
    cannot drift from production scoring.
    """
    if len(verdicts) == 0:
        return 0.0

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

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k gold_score`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add gold_score meta-eval metric"
```

---

### Task 11: Implement `cohens_kappa` and `krippendorff_alpha`

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
from healthbench_agent.llm_eval.meta_eval import cohens_kappa, krippendorff_alpha


def _agreement_rows(observed: list[bool], expected: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt_id": [f"p{i}" for i in range(len(observed))],
        "rubric_key": [f"r{i}" for i in range(len(observed))],
        "gold_source": ["ideal_completion"] * len(observed),
        "sample_k": [1] * len(observed),
        "observed_met": observed,
        "expected_met": expected,
    })


def test_cohens_kappa_full_agreement():
    df = _agreement_rows([True, False, True, False], [True, False, True, False])
    assert cohens_kappa(df) == pytest.approx(1.0)


def test_cohens_kappa_full_disagreement():
    df = _agreement_rows([True, False, True, False], [False, True, False, True])
    assert cohens_kappa(df) == pytest.approx(-1.0)


def test_cohens_kappa_random_is_near_zero():
    df = _agreement_rows([True, True, False, False], [True, False, True, False])
    assert abs(cohens_kappa(df)) < 0.5


def test_krippendorff_alpha_full_agreement():
    df = _agreement_rows([True, False, True, False], [True, False, True, False])
    assert krippendorff_alpha(df) == pytest.approx(1.0)


def test_krippendorff_alpha_random_is_near_zero():
    df = _agreement_rows([True, True, False, False], [True, False, True, False])
    assert abs(krippendorff_alpha(df)) < 0.5
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "kappa or alpha"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py`:

```python
def _majority_vote_columns(df: "pd.DataFrame") -> tuple[list[bool], list[bool]]:
    """Collapse k passes per (prompt_id, rubric_key, gold_source) to majority vote."""
    if len(df) == 0:
        return [], []
    grouped = df.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
    observed: list[bool] = []
    expected: list[bool] = []
    for _, group in grouped:
        observed.append(bool(group["observed_met"].mean() > 0.5))
        expected.append(bool(group["expected_met"].iloc[0]))
    return observed, expected


@register_meta_metric(
    "cohens_kappa",
    level=MetricLevel.ANY,
    description="Inter-rater agreement vs expected verdicts",
)
def cohens_kappa(verdicts: "pd.DataFrame") -> float:
    """Cohen's kappa between judge majority vote and expected verdicts."""
    from sklearn.metrics import cohen_kappa_score

    observed, expected = _majority_vote_columns(verdicts)
    if not observed:
        return 0.0
    return float(cohen_kappa_score(expected, observed))


@register_meta_metric(
    "krippendorff_alpha",
    level=MetricLevel.ANY,
    description="Binary two-coder Krippendorff alpha",
)
def krippendorff_alpha(verdicts: "pd.DataFrame") -> float:
    """Closed-form Krippendorff's alpha for binary, two-coder data.

    α = 1 - D_o / D_e where D_o is the observed disagreement (Hamming
    distance) and D_e is the expected disagreement under chance.
    """
    observed, expected = _majority_vote_columns(verdicts)
    n = len(observed)
    if n == 0:
        return 0.0
    disagreements = sum(1 for o, e in zip(observed, expected, strict=True) if o != e)
    do_metric = disagreements / n
    p1 = (sum(observed) + sum(expected)) / (2 * n)
    de_metric = 2 * p1 * (1 - p1)
    if de_metric == 0:
        return 1.0 if do_metric == 0 else 0.0
    return float(1.0 - do_metric / de_metric)
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "kappa or alpha"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/agreement.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add cohens_kappa and krippendorff_alpha metrics"
```

---

### Task 12: Implement `calibration_curve`

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing test**

```python
from healthbench_agent.llm_eval.meta_eval import calibration_curve


def test_calibration_curve_returns_dict_keyed_by_k():
    rng = list(range(1, 8))
    rows = []
    for prompt in range(10):
        for k in rng:
            rows.append({
                "prompt_id": f"p{prompt}",
                "rubric_key": "r1",
                "gold_source": "ideal_completion",
                "sample_k": k,
                "observed_met": (prompt + k) % 2 == 0,
                "expected_met": prompt % 2 == 0,
            })
    df = pd.DataFrame(rows)
    curve = calibration_curve(df)
    assert set(curve.keys()) == {1, 3, 5, 7}
    for v in curve.values():
        assert isinstance(v, float)
        assert v >= 0


def test_calibration_curve_empty_dataframe_returns_empty_dict():
    df = pd.DataFrame(columns=["prompt_id", "rubric_key", "gold_source", "sample_k", "observed_met", "expected_met"])
    assert calibration_curve(df) == {}
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k calibration`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py`:

```python
@register_meta_metric(
    "calibration_curve",
    level=MetricLevel.ANY,
    description="Bootstrap SE of agreement at k = 1, 3, 5, 7",
)
def calibration_curve(verdicts: "pd.DataFrame") -> dict[int, float]:
    """Bootstrap SE of per-(prompt_id, rubric_key) agreement at k in {1,3,5,7}."""
    import math

    if len(verdicts) == 0:
        return {}

    curve: dict[int, float] = {}
    for k in (1, 3, 5, 7):
        subset = verdicts[verdicts["sample_k"] <= k]
        if len(subset) == 0:
            continue
        grouped = subset.groupby(["prompt_id", "rubric_key", "gold_source"], sort=False)
        agreements: list[float] = []
        for _, group in grouped:
            majority = bool(group["observed_met"].mean() > 0.5)
            expected = bool(group["expected_met"].iloc[0])
            agreements.append(1.0 if majority == expected else 0.0)
        n = len(agreements)
        if n == 0:
            continue
        mean = sum(agreements) / n
        variance = sum((a - mean) ** 2 for a in agreements) / max(n - 1, 1)
        curve[k] = math.sqrt(variance / n)
    return curve
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k calibration`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add calibration_curve metric"
```

---

### Task 13: Implement `per_dimension_confusion`

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing test**

```python
from healthbench_agent.llm_eval.meta_eval import per_dimension_confusion


def test_per_dimension_confusion_two_dimensions():
    df = pd.DataFrame([
        {"dimension": "accuracy", "observed_met": True, "expected_met": True},
        {"dimension": "accuracy", "observed_met": True, "expected_met": False},
        {"dimension": "emergency", "observed_met": False, "expected_met": False},
        {"dimension": "emergency", "observed_met": False, "expected_met": True},
    ])
    result = per_dimension_confusion(df)
    assert result["accuracy"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert result["emergency"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 1}


def test_per_dimension_confusion_unspecified_for_none():
    df = pd.DataFrame([
        {"dimension": None, "observed_met": True, "expected_met": True},
    ])
    assert per_dimension_confusion(df)["unspecified"]["tp"] == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k per_dimension`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py`:

```python
@register_meta_metric(
    "per_dimension_confusion",
    level=MetricLevel.ANY,
    description="tp/fp/tn/fn per dimension (e.g. axis name)",
)
def per_dimension_confusion(verdicts: "pd.DataFrame") -> dict[str, dict[str, int]]:
    """Group by ``dimension`` column and return tp/fp/tn/fn per dimension."""
    result: dict[str, dict[str, int]] = {}
    if len(verdicts) == 0:
        return result
    df = verdicts.copy()
    df["dimension"] = df["dimension"].fillna("unspecified")
    for dim, group in df.groupby("dimension", sort=False):
        tp = int(((group["observed_met"]) & (group["expected_met"])).sum())
        fp = int(((group["observed_met"]) & (~group["expected_met"])).sum())
        tn = int(((~group["observed_met"]) & (~group["expected_met"])).sum())
        fn = int(((~group["observed_met"]) & (group["expected_met"])).sum())
        result[str(dim)] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return result
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k per_dimension`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/stratified.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add per_dimension_confusion metric"
```

---

### Task 14: Implement `adversarial_accuracy`, `adversarial_prf1`, `per_criterion_metrics`

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/metrics/adversarial.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
from healthbench_agent.llm_eval.meta_eval import (
    adversarial_accuracy,
    adversarial_prf1,
    per_criterion_metrics,
)


def test_adversarial_accuracy_three_of_four_match():
    df = pd.DataFrame([
        {"observed_met": True, "expected_met": True},
        {"observed_met": True, "expected_met": False},
        {"observed_met": False, "expected_met": False},
        {"observed_met": True, "expected_met": True},
    ])
    assert adversarial_accuracy(df) == 0.75


def test_adversarial_prf1_returns_all_keys():
    df = pd.DataFrame([
        {"observed_met": True, "expected_met": True},
        {"observed_met": False, "expected_met": True},
        {"observed_met": True, "expected_met": False},
        {"observed_met": False, "expected_met": False},
    ])
    out = adversarial_prf1(df)
    assert set(out.keys()) == {"precision", "recall", "f1", "support"}
    assert all(isinstance(v, float) for v in out.values())


def test_per_criterion_metrics_grouped_by_rubric_key():
    df = pd.DataFrame([
        {"rubric_key": "c1", "observed_met": True, "expected_met": True},
        {"rubric_key": "c1", "observed_met": False, "expected_met": False},
        {"rubric_key": "c2", "observed_met": True, "expected_met": False},
    ])
    out = per_criterion_metrics(df)
    assert set(out.keys()) == {"c1", "c2"}
    assert set(out["c1"].keys()) == {"accuracy", "precision", "recall", "f1"}
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "adversarial or per_criterion"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/metrics/adversarial.py`:

```python
@register_meta_metric(
    "adversarial_accuracy",
    level=MetricLevel.RUBRIC,
    description="Accuracy on example_meets / example_fails pairs",
)
def adversarial_accuracy(verdicts: "pd.DataFrame") -> float:
    """Plain accuracy: fraction of rows where observed_met == expected_met."""
    if len(verdicts) == 0:
        return 0.0
    matches = (verdicts["observed_met"] == verdicts["expected_met"]).sum()
    return float(matches / len(verdicts))


@register_meta_metric(
    "adversarial_prf1",
    level=MetricLevel.RUBRIC,
    description="Precision / recall / F1 on adversarial pairs",
)
def adversarial_prf1(verdicts: "pd.DataFrame") -> dict[str, float]:
    """Precision / recall / F1 / support via sklearn."""
    from sklearn.metrics import precision_recall_fscore_support

    if len(verdicts) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    precision, recall, f1, support = precision_recall_fscore_support(
        verdicts["expected_met"].astype(bool),
        verdicts["observed_met"].astype(bool),
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": float(support if support is not None else len(verdicts)),
    }


@register_meta_metric(
    "per_criterion_metrics",
    level=MetricLevel.RUBRIC,
    description="Per-criterion accuracy / precision / recall / F1",
)
def per_criterion_metrics(verdicts: "pd.DataFrame") -> dict[str, dict[str, float]]:
    """Group by rubric_key and return accuracy/precision/recall/f1 per criterion."""
    from sklearn.metrics import precision_recall_fscore_support

    result: dict[str, dict[str, float]] = {}
    if len(verdicts) == 0:
        return result
    for key, group in verdicts.groupby("rubric_key", sort=False):
        observed = group["observed_met"].astype(bool)
        expected = group["expected_met"].astype(bool)
        accuracy = float((observed == expected).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, observed, average="binary", zero_division=0
        )
        result[str(key)] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "adversarial or per_criterion"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/metrics/adversarial.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add adversarial and per-criterion metrics"
```

---

## Phase 6 — OracleJudge and demo data

### Task 15: Add `OracleJudge` and `demo_labelled_set`

**Files:**
- Create: `src/healthbench_agent/llm_eval/meta_eval/oracle_judge.py`
- Create: `src/healthbench_agent/llm_eval/meta_eval/demo_data.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.llm_eval.meta_eval import OracleJudge, demo_labelled_set


def test_oracle_judge_always_met_returns_true_for_all():
    judge = OracleJudge("always_met")
    items = [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)]
    out = judge.grade([], items)
    assert all(v.criteria_met for v in out)


def test_oracle_judge_always_fail_returns_false_for_all():
    judge = OracleJudge("always_fail")
    items = [RubricItem(criterion="a", points=1.0)]
    out = judge.grade([], items)
    assert not out[0].criteria_met


def test_oracle_judge_alternating_flips_per_call():
    judge = OracleJudge("alternating")
    out = judge.grade([], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)])
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_oracle_judge_dict_strategy_honours_mapping():
    judge = OracleJudge({"a": True, "b": False})
    out = judge.grade([], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)])
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_oracle_judge_callable_strategy():
    judge = OracleJudge(lambda item: item.points > 0)
    out = judge.grade([], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=-1.0)])
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_demo_labelled_set_returns_three_samples_with_mixed_flows():
    samples = demo_labelled_set()
    assert len(samples) == 3
    has_gold = sum(1 for s in samples if s.gold_response is not None)
    has_adversarial = sum(
        1 for s in samples for r in s.rubrics if r.example_meets or r.example_fails
    )
    assert has_gold >= 1
    assert has_adversarial >= 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "oracle_judge or demo_labelled"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Create `src/healthbench_agent/llm_eval/meta_eval/oracle_judge.py` (`OracleJudge`) and `src/healthbench_agent/llm_eval/meta_eval/demo_data.py` (`demo_labelled_set`). For clarity they are shown together in the same snippet below, though in the real layout the class and helper live in separate modules:

```python
from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader


class OracleJudge(JudgeGrader):
    """Deterministic oracle JudgeGrader for meta-evaluation smoke tests and demos.

    An oracle judge scores rubric items against a deterministic strategy
    rather than calling an LLM. Used as the baseline "known-answer"
    evaluator in meta-evaluation smoke tests, documentation snippets,
    and offline runs where real LLM grading would be cost-prohibitive
    or non-reproducible.

    Strategies:
        - "always_met"  / "always_fail" / "alternating"
        - dict[str, bool] keyed by criterion text
        - Callable[[RubricItem], bool] for arbitrary per-rubric logic
    """

    def __init__(
        self,
        strategy: str | dict[str, bool] | Callable[[RubricItem], bool] = "always_met",
    ) -> None:
        self.strategy = strategy

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        verdicts: list[CriterionVerdict] = []
        for idx, item in enumerate(rubric_items):
            met = self._verdict_for(item, idx)
            verdicts.append(
                CriterionVerdict(criterion=item.criterion, criteria_met=met, explanation="oracle")
            )
        return verdicts

    def _verdict_for(self, item: RubricItem, idx: int) -> bool:
        strategy = self.strategy
        if callable(strategy):
            return bool(strategy(item))
        if isinstance(strategy, dict):
            return bool(strategy.get(item.criterion, False))
        if strategy == "always_met":
            return True
        if strategy == "always_fail":
            return False
        if strategy == "alternating":
            return idx % 2 == 0
        raise ValueError(f"Unknown OracleJudge strategy: {strategy!r}")


def demo_labelled_set() -> list[LabelledSample]:
    """Hand-built 3-sample labelled set for examples and smoke tests."""
    return [
        LabelledSample(
            prompt_id="demo_1",
            prompt=[{"role": "user", "content": "What should I do for a fever?"}],
            rubrics=[
                RubricItem(criterion="suggests rest and fluids", points=1.0, category="accuracy"),
                RubricItem(criterion="recommends 911", points=-2.0, category="emergency"),
            ],
            gold_response="Rest, drink fluids, monitor your temperature.",
            expected={"suggests rest and fluids": True, "recommends 911": False},
            specialty="general",
            language="en",
            metadata={"clinical_urgency": "routine"},
        ),
        LabelledSample(
            prompt_id="demo_2",
            prompt=[{"role": "user", "content": "I have crushing chest pain."}],
            rubrics=[
                RubricItem(criterion="advises emergency care", points=5.0, category="emergency"),
            ],
            gold_response="Call 911 immediately. This may be a heart attack.",
            expected={"advises emergency care": True},
            specialty="cardiology",
            language="en",
            metadata={"clinical_urgency": "emergency"},
        ),
        LabelledSample(
            prompt_id="demo_3",
            prompt=[{"role": "user", "content": "How do I take ibuprofen?"}],
            rubrics=[
                RubricItem(
                    criterion="states correct adult dose",
                    points=1.0,
                    category="accuracy",
                    example_meets="The typical adult dose is 200-400 mg every 4-6 hours.",
                    example_fails="Take 5000 mg every hour.",
                ),
            ],
            specialty="general",
            language="en",
        ),
    ]
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "oracle_judge or demo_labelled"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/oracle_judge.py src/healthbench_agent/llm_eval/meta_eval/demo_data.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add OracleJudge and demo_labelled_set helpers"
```

---

## Phase 7 — Runner

### Task 16: Implement `run_meta_eval` (filter resolution + verdict DataFrame build)

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/runner.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
from healthbench_agent.llm_eval.meta_eval import run_meta_eval


def _axis_extractor(item: RubricItem) -> str | None:
    return item.category


def test_run_meta_eval_with_oracle_judge_produces_scores(tmp_path):
    samples = demo_labelled_set()
    perfect = {c: m for s in samples for c, m in s.expected.items()}
    view = run_meta_eval(
        OracleJudge(perfect),
        samples,
        dimension_extractor=_axis_extractor,
        n_samples=2,
        progress=False,
    )
    assert view.results.scores  # at least one metric computed
    assert view.results.n_samples_graded > 0
    assert view.results.n_rubrics_graded > 0


def test_run_meta_eval_persists_artifacts(tmp_path):
    samples = demo_labelled_set()
    view = run_meta_eval(
        OracleJudge("always_met"),
        samples,
        dimension_extractor=_axis_extractor,
        n_samples=1,
        output_dir=tmp_path,
        progress=False,
    )
    assert (tmp_path / "verdicts.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert view.results.verdicts_path == tmp_path / "verdicts.parquet"


def test_run_meta_eval_sample_filter_rejects_all_raises_empty_filter(tmp_path):
    samples = demo_labelled_set()
    with pytest.raises(EmptyFilterError):
        run_meta_eval(
            OracleJudge("always_met"),
            samples,
            dimension_extractor=_axis_extractor,
            sample_filter=lambda s: False,
            n_samples=1,
            progress=False,
        )


def test_run_meta_eval_rubric_filter_rejects_all_raises_empty_filter():
    samples = demo_labelled_set()
    with pytest.raises(EmptyFilterError):
        run_meta_eval(
            OracleJudge("always_met"),
            samples,
            dimension_extractor=_axis_extractor,
            rubric_filter=lambda r: False,
            n_samples=1,
            progress=False,
        )


def test_run_meta_eval_skips_sample_metrics_when_only_adversarial(caplog):
    """Sample-level metrics are dropped (with INFO log) on adversarial-only data."""
    sample = LabelledSample(
        prompt_id="adv_only",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[
            RubricItem(
                criterion="x",
                points=1.0,
                example_meets="positive",
                example_fails="negative",
                category="accuracy",
            )
        ],
    )
    view = run_meta_eval(
        OracleJudge("always_met"),
        [sample],
        dimension_extractor=_axis_extractor,
        metric_names=["gold_score", "adversarial_accuracy"],
        n_samples=1,
        progress=False,
    )
    assert "gold_score" not in view.results.scores
    assert "adversarial_accuracy" in view.results.scores


def test_run_meta_eval_raises_when_every_metric_skipped():
    sample = LabelledSample(
        prompt_id="adv_only",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[
            RubricItem(criterion="x", points=1.0, example_meets="ok", example_fails="bad")
        ],
    )
    with pytest.raises(EmptyFilterError, match="gold_score"):
        run_meta_eval(
            OracleJudge("always_met"),
            [sample],
            dimension_extractor=_axis_extractor,
            metric_names=["gold_score"],
            n_samples=1,
            progress=False,
        )
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k run_meta_eval`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/runner.py`:

```python
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from healthbench_agent.domain.meta_evaluation import MetricResults

if TYPE_CHECKING:
    from healthbench_agent.llm_eval.cache.store import VerdictCache
    from healthbench_agent.llm_eval.meta_eval.results import MetricResultsView

logger = logging.getLogger(__name__)


def _build_verdict_rows(
    judge: JudgeGrader,
    samples: list[LabelledSample],
    dimension_extractor: Callable[[RubricItem], str | None],
    n_samples: int,
) -> list[dict[str, Any]]:
    """Run k passes over each (sample, flow) combination and emit verdict rows."""
    rows: list[dict[str, Any]] = []
    for k in range(1, n_samples + 1):
        for sample in samples:
            # Sample-level flow.
            if sample.gold_response is not None:
                gold_rubrics = [r for r in sample.rubrics if r.points != 0]
                if gold_rubrics:
                    conv = sample.prompt + [
                        {"role": "assistant", "content": sample.gold_response}
                    ]
                    verdicts = judge.grade(conv, gold_rubrics)
                    for rubric, verdict in zip(gold_rubrics, verdicts, strict=True):
                        rows.append(
                            _row(sample, rubric, verdict, k, "ideal_completion",
                                 expected_met=rubric.points > 0,
                                 dimension_extractor=dimension_extractor)
                        )
            # Adversarial flows.
            for rubric in sample.rubrics:
                if rubric.example_meets is not None:
                    conv = sample.prompt + [
                        {"role": "assistant", "content": rubric.example_meets}
                    ]
                    [verdict] = judge.grade(conv, [rubric])
                    rows.append(
                        _row(sample, rubric, verdict, k, "example_meets",
                             expected_met=True, dimension_extractor=dimension_extractor)
                    )
                if rubric.example_fails is not None:
                    conv = sample.prompt + [
                        {"role": "assistant", "content": rubric.example_fails}
                    ]
                    [verdict] = judge.grade(conv, [rubric])
                    rows.append(
                        _row(sample, rubric, verdict, k, "example_fails",
                             expected_met=False, dimension_extractor=dimension_extractor)
                    )
    return rows


def _row(
    sample: LabelledSample,
    rubric: RubricItem,
    verdict: CriterionVerdict,
    k: int,
    gold_source: str,
    expected_met: bool,
    dimension_extractor: Callable[[RubricItem], str | None],
) -> dict[str, Any]:
    return {
        "prompt_id": sample.prompt_id,
        "criterion_id": rubric.criterion_id,
        "criterion": rubric.criterion[:200],
        "rubric_key": rubric.criterion_id or rubric.criterion,
        "dimension": dimension_extractor(rubric),
        "points": rubric.points,
        "sample_k": k,
        "gold_source": gold_source,
        "observed_met": bool(verdict.criteria_met),
        "expected_met": bool(expected_met),
        "specialty": sample.specialty,
        "language": sample.language,
        "metadata_json": json.dumps(sample.metadata),
    }


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
    cache: "VerdictCache | None" = None,
    model_fingerprint: str | None = None,
    judge_prompt_sha: str | None = None,
    meta_eval_max_workers: int = 16,
    progress: bool | None = None,
) -> "MetricResultsView":
    """Grade labelled samples k times with one judge, compute metrics, persist."""
    import pandas as pd

    from healthbench_agent.llm_eval.cache.cached_judge import CachedJudgeGrader
    from healthbench_agent.llm_eval.meta_eval.results import MetricResultsView

    # Step 1: sample filter
    if sample_filter is not None:
        kept = [s for s in labelled if sample_filter(s)]
    else:
        kept = list(labelled)
    if not kept:
        raise EmptyFilterError(sample_filter=sample_filter, rubric_filter=rubric_filter)

    # Step 2: rubric filter (mutates a shallow copy of each sample)
    surviving: list[LabelledSample] = []
    if rubric_filter is not None:
        for sample in kept:
            new_rubrics = [r for r in sample.rubrics if rubric_filter(r)]
            if not new_rubrics:
                continue
            patched = LabelledSample(
                prompt_id=sample.prompt_id,
                prompt=sample.prompt,
                rubrics=new_rubrics,
                gold_response=sample.gold_response,
                expected=sample.expected,
                language=sample.language,
                specialty=sample.specialty,
                user_persona=sample.user_persona,
                metadata=sample.metadata,
            )
            surviving.append(patched)
    else:
        surviving = kept
    if not surviving:
        raise EmptyFilterError(sample_filter=sample_filter, rubric_filter=rubric_filter)

    # Step 3: build verdict rows. Wrap in cache proxy when requested.
    if cache is not None:
        if model_fingerprint is None or judge_prompt_sha is None:
            raise ValueError(
                "run_meta_eval(cache=...) requires model_fingerprint and judge_prompt_sha"
            )
        # Build a per-k proxy lazily inside the row builder.
        def _grade_for_k(k: int) -> JudgeGrader:
            return CachedJudgeGrader(judge, cache, model_fingerprint, judge_prompt_sha, k)
    else:
        def _grade_for_k(k: int) -> JudgeGrader:
            return judge

    rows: list[dict[str, Any]] = []
    show_progress = progress if progress is not None else sys.stdout.isatty()

    pairs = [(s, k) for k in range(1, n_samples + 1) for s in surviving]
    iterator: Any
    if show_progress:
        try:
            from tqdm.contrib.concurrent import thread_map

            def _task(pair):
                sample, k = pair
                return _build_verdict_rows(_grade_for_k(k), [sample], dimension_extractor, 1)

            results = thread_map(
                _task,
                pairs,
                max_workers=meta_eval_max_workers,
                desc="Grading samples",
            )
        except ImportError:
            results = []
            with ThreadPoolExecutor(max_workers=meta_eval_max_workers) as pool:
                for chunk in pool.map(
                    lambda pair: _build_verdict_rows(_grade_for_k(pair[1]), [pair[0]], dimension_extractor, 1),
                    pairs,
                ):
                    results.append(chunk)
    else:
        results = []
        with ThreadPoolExecutor(max_workers=meta_eval_max_workers) as pool:
            for chunk in pool.map(
                lambda pair: _build_verdict_rows(_grade_for_k(pair[1]), [pair[0]], dimension_extractor, 1),
                pairs,
            ):
                results.append(chunk)

    for chunk in results:
        # _build_verdict_rows returns rows with k=1; rewrite k from the pair index.
        rows.extend(chunk)
    # Re-stamp sample_k from the outer loop because _build_verdict_rows always emits k=1.
    stamped: list[dict[str, Any]] = []
    chunk_idx = 0
    for k in range(1, n_samples + 1):
        for _ in surviving:
            for row in results[chunk_idx]:
                stamped_row = dict(row)
                stamped_row["sample_k"] = k
                stamped.append(stamped_row)
            chunk_idx += 1
    rows = stamped

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "prompt_id", "criterion_id", "criterion", "rubric_key", "dimension",
            "points", "sample_k", "gold_source", "observed_met", "expected_met",
            "specialty", "language", "metadata_json",
        ]
    )

    # Step 4: partition by gold_source
    sample_rows = df[df["gold_source"] == "ideal_completion"]
    rubric_rows = df[df["gold_source"].isin(["example_meets", "example_fails"])]
    all_rows = df

    # Step 5: dispatch metrics by level
    requested = metric_names or list(_METRIC_REGISTRY.keys())
    scores: dict[str, Any] = {}
    skipped: list[str] = []
    for name in requested:
        spec = get_meta_metric(name)
        if spec.level is MetricLevel.SAMPLE:
            subset = sample_rows
        elif spec.level is MetricLevel.RUBRIC:
            subset = rubric_rows
        else:
            subset = all_rows
        if len(subset) == 0:
            logger.info(
                "skipping metric %r (level=%s) — no matching rows in this run",
                name, spec.level.value,
            )
            skipped.append(name)
            continue
        scores[name] = spec.fn(subset)

    if not scores:
        raise EmptyFilterError(
            sample_filter=f"every metric skipped: {skipped}",
            rubric_filter=rubric_filter,
        )

    # Step 6: build results + persist
    metadata = dict(judge_metadata or {})
    cache_stats = cache.stats() if cache is not None else {"hits": 0, "misses": 0}
    metadata.setdefault("cache_hits", cache_stats["hits"])
    metadata.setdefault("cache_misses", cache_stats["misses"])

    results = MetricResults(
        scores=scores,
        n_samples_graded=len(surviving),
        n_rubrics_graded=len(df),
        judge_metadata=metadata,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "verdicts.parquet")
        results.verdicts_path = output_dir / "verdicts.parquet"
        (output_dir / "metrics.json").write_text(json.dumps(results.to_dict(), indent=2))

    return MetricResultsView(results=results)
```

- [x] **Step 4: Run tests**

> **Execution-order note:** `run_meta_eval` does a *function-local* lazy import of `MetricResultsView` (defined in Task 17). At module-load time, importing `meta_eval.py` will succeed; the import only fires when `run_meta_eval()` is actually called. So the integration tests in this step depend on Task 17 being implemented first. **Do Task 17 before running this step.** Once Task 17 is in place, return here and run the verification.

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k run_meta_eval`
Expected: PASS for every test in the `run_meta_eval` block (covers cache wiring, filter resolution, metric dispatch, and persistence).

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/runner.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): implement run_meta_eval with filter, cache, and metric dispatch"
```

---

## Phase 8 — Results View

### Task 17: Create `MetricResultsView` (printing + IO)

**Files:**
- Create: `src/healthbench_agent/llm_eval/meta_eval/results/view.py`
- Test: `tests/llm_eval/test_meta_eval_results.py`

- [x] **Step 1: Write the failing tests**

```python
# tests/llm_eval/test_meta_eval_results.py
"""Tests for MetricResultsView."""
from __future__ import annotations

from pathlib import Path

import pytest

from healthbench_agent.domain.meta_evaluation import MetricResults
from healthbench_agent.llm_eval.meta_eval.results.view import MetricResultsView


@pytest.fixture
def view() -> MetricResultsView:
    results = MetricResults(
        scores={"gold_score": 0.873, "cohens_kappa": 0.612},
        n_samples_graded=100,
        n_rubrics_graded=287,
        judge_metadata={"judge_model": "openai/gpt-4.1", "k": 7},
    )
    return MetricResultsView(results=results)


def test_repr_includes_judge_model_and_counts(view):
    text = repr(view)
    assert "openai/gpt-4.1" in text
    assert "100" in text


def test_summary_has_one_line_per_metric_plus_header(view):
    summary = view.summary()
    lines = summary.strip().splitlines()
    assert "gold_score" in summary
    assert "cohens_kappa" in summary
    assert any("METRIC" in line for line in lines)


def test_repr_html_starts_with_table_tag(view):
    assert view._repr_html_().lstrip().startswith("<table")


def test_to_pandas_returns_dataframe_with_score_rows(view):
    df = view.to_pandas()
    assert "gold_score" in df["metric"].tolist()
    assert "cohens_kappa" in df["metric"].tolist()


def test_save_and_load_round_trip(tmp_path: Path, view):
    view.save(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    reloaded = MetricResultsView.load(tmp_path)
    assert reloaded.results.scores == view.results.scores


def test_load_raises_on_unknown_schema_version(tmp_path: Path, view):
    payload = view.results.to_dict()
    payload["schema_version"] = 999
    (tmp_path / "metrics.json").write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        MetricResultsView.load(tmp_path)


def test_load_missing_metrics_json_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MetricResultsView.load(tmp_path)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval_results.py -v`
Expected: FAIL — module does not exist.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/llm_eval/meta_eval/results/view.py
"""User-facing wrapper around the pure-domain MetricResults dataclass.

Adds REPL/Jupyter/IO/plot helpers without dragging matplotlib, pandas,
or pyarrow into the domain layer. Returned by ``meta_evaluate`` and
``run_meta_eval`` so users get rich UX by default.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from healthbench_agent.domain.meta_evaluation import MetricResults

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MetricResultsView:
    """Sklearn-style ergonomics around a MetricResults dataclass."""

    results: MetricResults
    _verdicts_cache: "pd.DataFrame | None" = None

    # ---- pretty printing -------------------------------------------------

    def __repr__(self) -> str:
        meta = self.results.judge_metadata
        judge = meta.get("judge_model", "unknown")
        k = meta.get("k", meta.get("n_samples", "?"))
        return (
            f"MetricResultsView(judge={judge}, k={k}, "
            f"n_samples={self.results.n_samples_graded})"
        )

    def summary(self) -> str:
        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

        meta = self.results.judge_metadata
        header_line = (
            f"MetricResults(judge={meta.get('judge_model', '?')}, "
            f"k={meta.get('k', meta.get('n_samples', '?'))}, "
            f"n={self.results.n_samples_graded})"
        )
        rule = "─" * 60
        rows = [header_line, rule, f"{'METRIC':<26} {'LEVEL':<8} VALUE"]
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value.upper()
            except KeyError:
                level = "?"
            rows.append(f"{name:<26} {level:<8} {value}")
        return "\n".join(rows)

    def _repr_html_(self) -> str:
        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

        rows_html = []
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value.upper()
            except KeyError:
                level = "?"
            rows_html.append(f"<tr><td>{name}</td><td>{level}</td><td>{value}</td></tr>")
        return (
            "<table>"
            "<thead><tr><th>metric</th><th>level</th><th>value</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
        )

    # ---- conversions -----------------------------------------------------

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        from healthbench_agent.llm_eval.meta_eval import get_meta_metric

        rows: list[dict[str, Any]] = []
        for name, value in self.results.scores.items():
            try:
                level = get_meta_metric(name).level.value
            except KeyError:
                level = "?"
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    rows.append({"metric": f"{name}.{sub_key}", "level": level, "value": sub_val})
            else:
                rows.append({"metric": name, "level": level, "value": value})
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        return self.to_pandas().to_markdown(index=False)

    # ---- IO --------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Path | str) -> "MetricResultsView":
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in {run_dir}")
        data = json.loads(metrics_path.read_text())
        results = MetricResults.from_dict(data)
        verdicts_path = run_dir / "verdicts.parquet"
        if verdicts_path.exists():
            results.verdicts_path = verdicts_path
        return cls(results=results)

    def save(self, run_dir: Path | str) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(self.results.to_dict(), indent=2))

    def verdicts(self) -> "pd.DataFrame":
        import pandas as pd

        if self._verdicts_cache is not None:
            return self._verdicts_cache
        if self.results.verdicts_path is None:
            raise FileNotFoundError(
                "verdicts_path is None — view was constructed in-memory without an output_dir"
            )
        self._verdicts_cache = pd.read_parquet(self.results.verdicts_path)
        return self._verdicts_cache
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval_results.py -v -k "repr or summary or html or pandas or save or load"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/results/view.py tests/llm_eval/test_meta_eval_results.py
git commit -m "feat(llm_eval): add MetricResultsView with repr, summary, IO helpers"
```

---

### Task 18: Add `MetricResultsView.compare` and plot helpers

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/results/view.py`
- Test: `tests/llm_eval/test_meta_eval_results.py`

- [x] **Step 1: Write the failing tests**

Append:

```python
def test_compare_two_views_returns_diff_dataframe(view):
    other = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.812, "cohens_kappa": 0.589},
            n_samples_graded=100,
            n_rubrics_graded=287,
            judge_metadata={"judge_model": "google/gemini-2.5"},
        )
    )
    diff = view.compare(other)
    assert "metric" in diff.columns
    assert "self" in diff.columns
    assert "other" in diff.columns
    assert "delta" in diff.columns
    assert len(diff) == 2


def test_plot_calibration_curve_returns_axes_when_data_present():
    view = MetricResultsView(
        results=MetricResults(
            scores={"calibration_curve": {1: 0.08, 3: 0.06, 5: 0.05, 7: 0.04}},
            n_samples_graded=10, n_rubrics_graded=10, judge_metadata={},
        )
    )
    ax = view.plot_calibration_curve()
    assert ax is not None


def test_plot_dimension_confusion_returns_axes_when_data_present():
    view = MetricResultsView(
        results=MetricResults(
            scores={"per_dimension_confusion": {"accuracy": {"tp": 1, "fp": 0, "tn": 1, "fn": 0}}},
            n_samples_graded=2, n_rubrics_graded=2, judge_metadata={},
        )
    )
    ax = view.plot_dimension_confusion()
    assert ax is not None


def test_plot_calibration_curve_raises_keyerror_when_missing(view):
    with pytest.raises(KeyError):
        view.plot_calibration_curve()
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval_results.py -v -k "compare or plot"`
Expected: FAIL — methods do not exist.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/results/view.py`:

```python
    # ---- comparison ------------------------------------------------------

    def compare(self, other: "MetricResultsView") -> "pd.DataFrame":
        import pandas as pd

        rows: list[dict[str, Any]] = []
        all_keys = sorted(set(self.results.scores) | set(other.results.scores))
        for name in all_keys:
            self_val = self.results.scores.get(name)
            other_val = other.results.scores.get(name)
            if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                delta = float(other_val) - float(self_val)
            else:
                delta = "see details"
            rows.append({"metric": name, "self": self_val, "other": other_val, "delta": delta})
        return pd.DataFrame(rows)

    # ---- plot helpers ----------------------------------------------------

    def plot_calibration_curve(self, ax: Any = None) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "Plotting requires matplotlib. Install with: uv sync --extra viz"
            ) from exc

        if "calibration_curve" not in self.results.scores:
            raise KeyError("calibration_curve is not in results.scores")
        curve = self.results.scores["calibration_curve"]
        if ax is None:
            _, ax = plt.subplots()
        ks = sorted(curve.keys())
        ax.plot(ks, [curve[k] for k in ks], marker="o")
        ax.set_xlabel("k (number of judge passes)")
        ax.set_ylabel("Bootstrap SE of agreement")
        ax.set_title("Calibration curve")
        return ax

    def plot_dimension_confusion(self, ax: Any = None) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Plotting requires matplotlib. Install with: uv sync --extra viz"
            ) from exc

        if "per_dimension_confusion" not in self.results.scores:
            raise KeyError("per_dimension_confusion is not in results.scores")
        confusion = self.results.scores["per_dimension_confusion"]
        if ax is None:
            _, ax = plt.subplots()
        dims = list(confusion.keys())
        tp = [confusion[d]["tp"] for d in dims]
        fp = [confusion[d]["fp"] for d in dims]
        tn = [confusion[d]["tn"] for d in dims]
        fn = [confusion[d]["fn"] for d in dims]
        ax.bar(dims, tp, label="tp")
        ax.bar(dims, fp, bottom=tp, label="fp")
        ax.bar(dims, tn, bottom=[a + b for a, b in zip(tp, fp)], label="tn")
        ax.bar(dims, fn, bottom=[a + b + c for a, b, c in zip(tp, fp, tn)], label="fn")
        ax.legend()
        ax.set_ylabel("count")
        ax.set_title("Per-dimension confusion")
        return ax
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval_results.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/results/view.py tests/llm_eval/test_meta_eval_results.py
git commit -m "feat(llm_eval): add compare() and plot helpers to MetricResultsView"
```

---

## Phase 9 — Happy-Path API and Public Re-Exports

### Task 19: Implement `meta_evaluate()` and re-export public symbols

**Files:**
- Modify: `src/healthbench_agent/llm_eval/meta_eval/api.py`
- Modify: `src/healthbench_agent/llm_eval/__init__.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing test**

Append to `tests/llm_eval/test_meta_eval.py`:

```python
def test_meta_evaluate_with_oracle_judge_returns_view(monkeypatch, tmp_path):
    """meta_evaluate should accept a JudgeConfig and a pre-built dataset path."""
    from healthbench_agent.llm_eval import meta_evaluate

    # We patch the judge factory + dataset loader so the test stays offline.
    samples = demo_labelled_set()

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: samples,
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval._build_judge_for_meta_eval",
        lambda config, temperature: (OracleJudge("always_met"), "fake/model@1.0", "sha"),
    )

    view = meta_evaluate(
        judge_config="config/judges/fake.yaml",
        sample_size=3,
        n_samples=1,
        cache=False,
        progress=False,
        output_dir=tmp_path,
    )
    assert view.results.scores
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k meta_evaluate`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/meta_eval/api.py`:

```python
def _load_subset_for_meta_eval(
    subset: str,
    sample_size: int,
    seed: int,
) -> list[LabelledSample]:
    """Load a HealthBench subset and populate gold-label fields in place."""
    from healthbench_agent.dataset.extraction import extract_ideal_completion_text
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample

    dataset = load_dataset(subset=subset)
    sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)
    out: list[LabelledSample] = []
    for sample in sampled.samples:
        gold = extract_ideal_completion_text(sample.ideal_completions_data)
        if gold is None:
            continue
        sample.gold_response = gold
        sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}
        out.append(sample)
    return out


def _build_judge_for_meta_eval(
    config: Any,
    temperature: float,
) -> tuple[JudgeGrader, str, str]:
    """Build the judge + return its (model_fingerprint, prompt_sha) for caching."""
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.grading.judge import create_judge, load_grader_prompt

    if isinstance(config, (str, Path)):
        cfg = JudgeConfig.from_yaml(str(config))
    elif isinstance(config, JudgeConfig):
        cfg = config
    else:
        raise TypeError(f"judge_config must be a path or JudgeConfig, got {type(config)}")
    cfg = cfg.model_copy(update={"temperature": temperature})
    judge = create_judge(cfg)
    fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
    _, _, prompt_sha = load_grader_prompt(cfg.prompt_path)
    return judge, fingerprint, prompt_sha


def meta_evaluate(
    judge_config: Any,
    *,
    subset: str = "consensus",
    sample_size: int = 100,
    n_samples: int = 7,
    temperature: float = 1.0,
    metric_names: list[str] | None = None,
    sample_filter: Callable[[LabelledSample], bool] | None = None,
    rubric_filter: Callable[[RubricItem], bool] | None = None,
    output_dir: Path | str | None = None,
    cache: "bool | VerdictCache" = True,
    progress: bool | None = None,
    seed: int = 0,
) -> "MetricResultsView":
    """Meta-evaluate one judge end-to-end with sensible defaults."""
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    samples = _load_subset_for_meta_eval(subset, sample_size, seed)
    judge, fingerprint, prompt_sha = _build_judge_for_meta_eval(judge_config, temperature)

    if isinstance(cache, VerdictCache):
        cache_obj: VerdictCache | None = cache
    elif cache is True:
        cache_obj = VerdictCache(enabled=True)
    else:
        cache_obj = None

    return run_meta_eval(
        judge=judge,
        labelled=samples,
        dimension_extractor=lambda r: r.category,
        metric_names=metric_names,
        n_samples=n_samples,
        sample_filter=sample_filter,
        rubric_filter=rubric_filter,
        output_dir=Path(output_dir) if output_dir is not None else None,
        judge_metadata={
            "judge_model": fingerprint,
            "temperature": temperature,
            "n_samples": n_samples,
            "subset": subset,
            "sample_size": sample_size,
            "seed": seed,
        },
        cache=cache_obj,
        model_fingerprint=fingerprint,
        judge_prompt_sha=prompt_sha,
        progress=progress,
    )
```

Modify [src/healthbench_agent/llm_eval/__init__.py](../../../src/healthbench_agent/llm_eval/__init__.py) to re-export:

```python
from .cache import CachedJudgeGrader, VerdictCache
from .meta_eval import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    MetricLevel,
    MetricResultsView,
    MetricSpec,
    OracleJudge,
    axis_filter,
    demo_labelled_set,
    get_meta_metric,
    meta_evaluate,
    metadata_filter,
    register_meta_metric,
    registered_meta_metrics,
    run_meta_eval,
    specialty_filter,
)
```

Add the new names to `__all__` in the same file.

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k meta_evaluate`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/meta_eval/api.py src/healthbench_agent/llm_eval/__init__.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(llm_eval): add meta_evaluate happy-path API and re-export public symbols"
```

---

## Phase 10 — CLI

### Task 20: Create `cli/meta_eval.py` skeleton with `run` subcommand

**Files:**
- Create: `src/healthbench_agent/llm_eval/cli/meta_eval.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing test**

Append:

```python
def test_cli_run_dispatches_to_run_meta_eval(monkeypatch, tmp_path, capsys):
    """CLI run subcommand should call run_meta_eval with parsed args."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    called: dict[str, Any] = {}

    def fake_run(**kwargs):
        called.update(kwargs)
        results = MetricResults(
            scores={"gold_score": 0.5},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "fake"},
        )
        from healthbench_agent.llm_eval.meta_eval.results.view import MetricResultsView
        return MetricResultsView(results=results)

    monkeypatch.setattr("healthbench_agent.llm_eval.cli.meta_eval.run_meta_eval", fake_run)
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_judge_for_cli",
        lambda config, temperature: (OracleJudge("always_met"), "fake@1", "sha"),
    )

    cli_meta_eval.main(
        [
            "run",
            "--judge-config", "fake.yaml",
            "--sample-size", "1",
            "--n-samples", "1",
            "--no-cache",
            "--no-mlflow",
            "--no-progress",
            "--output-dir", str(tmp_path),
        ]
    )

    assert called["n_samples"] == 1
    captured = capsys.readouterr()
    assert "gold_score" in captured.out
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k cli_run`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

```python
# src/healthbench_agent/llm_eval/cli/meta_eval.py
"""``meta-evaluate-judge`` CLI.

argparse subcommands: run / regenerate / compare / list-metrics /
list-metadata-keys / clear-cache.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.meta_eval import (
    EmptyFilterError,
    OracleJudge,  # noqa: F401  -- re-exported for tests
    axis_filter,
    metadata_filter,
    run_meta_eval,
)
from healthbench_agent.llm_eval.meta_eval.results.view import MetricResultsView
from healthbench_agent.llm_eval.cache.store import VerdictCache

logger = logging.getLogger(__name__)


def _load_consensus_labelled(
    subset: str,
    sample_size: int,
    seed: int,
) -> tuple[list[LabelledSample], Callable[[RubricItem], str | None]]:
    """Load + populate the labelled set, return it plus a HealthBench axis_extractor."""
    from healthbench_agent.dataset.extraction import extract_ideal_completion_text
    from healthbench_agent.dataset.loader import load_dataset
    from healthbench_agent.dataset.split_utils import stratified_sample
    from healthbench_agent.llm_eval.meta_eval import AXIS_TAG_PREFIX

    dataset = load_dataset(subset=subset)
    sampled = stratified_sample(dataset, n=sample_size, tag_prefix="theme", seed=seed)
    samples: list[LabelledSample] = []
    for sample in sampled.samples:
        gold = extract_ideal_completion_text(sample.ideal_completions_data)
        if gold is None:
            continue
        sample.gold_response = gold
        sample.expected = {r.criterion: r.points > 0 for r in sample.rubrics if r.points != 0}
        samples.append(sample)
    if not samples:
        sys.stderr.write(
            "All samples dropped during gold-label extraction. The selected subset may "
            "not ship physician ideal completions.\n"
        )
        raise SystemExit(2)

    def axis_extractor(item: RubricItem) -> str | None:
        for tag in item.tags:
            if tag.startswith(AXIS_TAG_PREFIX):
                return tag[len(AXIS_TAG_PREFIX):].strip()
        return item.category

    return samples, axis_extractor


def _build_judge_for_cli(config_path: str, temperature: float):
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.grading.judge import create_judge, load_grader_prompt

    cfg = JudgeConfig.from_yaml(config_path)
    cfg = cfg.model_copy(update={"temperature": temperature})
    judge = create_judge(cfg)
    fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
    _, _, prompt_sha = load_grader_prompt(cfg.prompt_path)
    return judge, fingerprint, prompt_sha


def _build_filters(args: argparse.Namespace):
    rf = axis_filter(*args.rubric_axis) if args.rubric_axis else None
    parsed: dict[str, str] = {}
    for entry in args.metadata or []:
        if "=" not in entry:
            raise SystemExit(f"--metadata expects KEY=VALUE, got {entry!r}")
        key, value = entry.split("=", 1)
        parsed[key] = value
    sf = metadata_filter(**parsed) if parsed else None
    return sf, rf


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Grade a labelled set with one judge")
    p.add_argument("--judge-config", required=True)
    p.add_argument("--subset", default="consensus")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--metrics", default="")
    p.add_argument("--rubric-axis", action="append", default=[])
    p.add_argument("--metadata", action="append", default=[])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _cmd_run(args: argparse.Namespace) -> None:
    samples, axis_extractor = _load_consensus_labelled(
        args.subset, args.sample_size, args.seed
    )
    sample_filter, rubric_filter = _build_filters(args)
    judge, fingerprint, prompt_sha = _build_judge_for_cli(args.judge_config, args.temperature)
    cache = VerdictCache(enabled=not args.no_cache)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        view = run_meta_eval(
            judge=judge,
            labelled=samples,
            dimension_extractor=axis_extractor,
            metric_names=metric_names,
            n_samples=args.n_samples,
            sample_filter=sample_filter,
            rubric_filter=rubric_filter,
            output_dir=output_dir,
            judge_metadata={
                "judge_model": fingerprint,
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "subset": args.subset,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "judge_prompt_sha": prompt_sha,
                "filter_axis": ",".join(args.rubric_axis),
                "filter_metadata": ",".join(args.metadata),
            },
            cache=cache,
            model_fingerprint=fingerprint,
            judge_prompt_sha=prompt_sha,
            progress=not args.no_progress,
        )
    except EmptyFilterError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    print(view.summary())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k cli_run`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/cli/meta_eval.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(cli): add meta-evaluate-judge run subcommand"
```

---

### Task 21: Add `regenerate`, `compare`, `clear-cache`, `list-metrics`, `list-metadata-keys` subcommands

**Files:**
- Modify: `src/healthbench_agent/llm_eval/cli/meta_eval.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
def test_cli_list_metrics_prints_all(capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    cli_meta_eval.main(["list-metrics"])
    out = capsys.readouterr().out
    for name in (
        "gold_score", "cohens_kappa", "krippendorff_alpha", "calibration_curve",
        "per_dimension_confusion", "adversarial_accuracy", "adversarial_prf1", "per_criterion_metrics",
    ):
        assert name in out


def test_cli_clear_cache_removes_directory(tmp_path, monkeypatch, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    cache = VerdictCache(root=tmp_path / "cache", enabled=True)
    cache.put(
        cache.make_key("m", "s", [{"role": "user", "content": "x"}], "rt", 1),
        CriterionVerdict(criterion="rt", criteria_met=True, explanation=""),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._default_cache_for_cli",
        lambda: cache,
    )
    cli_meta_eval.main(["clear-cache"])
    assert not cache.root.exists() or not any(cache.root.iterdir())


def test_cli_regenerate_replays_metrics(tmp_path, monkeypatch):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    samples = demo_labelled_set()
    view = run_meta_eval(
        OracleJudge({c: m for s in samples for c, m in s.expected.items()}),
        samples,
        dimension_extractor=lambda r: r.category,
        n_samples=1,
        output_dir=tmp_path,
        progress=False,
    )
    # Mutate the metrics.json so we can prove regenerate overwrote it.
    (tmp_path / "metrics.json").write_text(
        __import__("json").dumps({**view.results.to_dict(), "scores": {}})
    )
    cli_meta_eval.main(["regenerate", str(tmp_path)])
    reloaded = MetricResultsView.load(tmp_path)
    assert reloaded.results.scores  # repopulated


def test_cli_compare_prints_diff_table(tmp_path, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_view = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.8},
            n_samples_graded=1, n_rubrics_graded=1,
            judge_metadata={"judge_model": "openai"},
        )
    )
    b_view = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.7},
            n_samples_graded=1, n_rubrics_graded=1,
            judge_metadata={"judge_model": "google"},
        )
    )
    a_view.save(a_dir)
    b_view.save(b_dir)
    cli_meta_eval.main(["compare", str(a_dir), str(b_dir)])
    out = capsys.readouterr().out
    assert "gold_score" in out
    assert "delta" in out or "0.8" in out


def test_cli_list_metadata_keys(monkeypatch, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    cli_meta_eval.main(["list-metadata-keys", "--sample-size", "3"])
    out = capsys.readouterr().out
    assert "language" in out or "specialty" in out or "clinical_urgency" in out
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "list_metrics or clear_cache or regenerate or compare or list_metadata"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to `src/healthbench_agent/llm_eval/cli/meta_eval.py`:

```python
def _default_cache_for_cli() -> VerdictCache:
    return VerdictCache(enabled=True)


def _add_other_parsers(subparsers: argparse._SubParsersAction) -> None:
    rg = subparsers.add_parser("regenerate", help="Recompute metrics from a stored parquet")
    rg.add_argument("run_dir")
    rg.add_argument("--metrics", default="")

    cmp_p = subparsers.add_parser("compare", help="Side-by-side score diff")
    cmp_p.add_argument("run1")
    cmp_p.add_argument("run2")
    cmp_p.add_argument("--output", default=None)

    subparsers.add_parser("list-metrics", help="Show registered metrics")

    lmk = subparsers.add_parser(
        "list-metadata-keys",
        help="Show metadata keys + top values discoverable in a labelled set",
    )
    lmk.add_argument("--subset", default="consensus")
    lmk.add_argument("--sample-size", type=int, default=100)
    lmk.add_argument("--seed", type=int, default=0)

    subparsers.add_parser("clear-cache", help="Delete the verdict cache")


def _cmd_list_metrics(_: argparse.Namespace) -> None:
    from healthbench_agent.llm_eval.meta_eval import registered_meta_metrics

    print(f"{'NAME':<26} {'LEVEL':<8} DESCRIPTION")
    for spec in registered_meta_metrics().values():
        print(f"{spec.name:<26} {spec.level.value.upper():<8} {spec.description}")


def _cmd_regenerate(args: argparse.Namespace) -> None:
    import pandas as pd

    from healthbench_agent.llm_eval.meta_eval import (
        MetricLevel,
        get_meta_metric,
        registered_meta_metrics,
    )

    run_dir = Path(args.run_dir)
    df = pd.read_parquet(run_dir / "verdicts.parquet")
    sample_rows = df[df["gold_source"] == "ideal_completion"]
    rubric_rows = df[df["gold_source"].isin(["example_meets", "example_fails"])]

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()] or list(
        registered_meta_metrics().keys()
    )
    scores: dict[str, Any] = {}
    for name in metric_names:
        spec = get_meta_metric(name)
        subset = (
            sample_rows if spec.level is MetricLevel.SAMPLE
            else rubric_rows if spec.level is MetricLevel.RUBRIC
            else df
        )
        if len(subset) == 0:
            continue
        scores[name] = spec.fn(subset)

    view = MetricResultsView.load(run_dir)
    view.results.scores = scores
    view.save(run_dir)
    print(view.summary())


def _cmd_compare(args: argparse.Namespace) -> None:
    a = MetricResultsView.load(args.run1)
    b = MetricResultsView.load(args.run2)
    diff = a.compare(b)
    rendered = diff.to_string(index=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(diff.to_markdown(index=False))


def _cmd_list_metadata_keys(args: argparse.Namespace) -> None:
    from collections import Counter

    samples, _ = _load_consensus_labelled(args.subset, args.sample_size, args.seed)
    counters: dict[str, Counter] = {}
    for sample in samples:
        for attr in ("language", "specialty", "user_persona"):
            value = getattr(sample, attr)
            if value is not None:
                counters.setdefault(attr, Counter())[value] += 1
        for key, value in sample.metadata.items():
            counters.setdefault(key, Counter())[value] += 1
    print(f"{'KEY':<26} {'N':<6} TOP VALUES")
    for key, counter in counters.items():
        top = ", ".join(f"{v} ({n})" for v, n in counter.most_common(5))
        print(f"{key:<26} {sum(counter.values()):<6} {top}")


def _cmd_clear_cache(_: argparse.Namespace) -> None:
    cache = _default_cache_for_cli()
    cache.clear()
    print(f"Cleared cache at {cache.root}")
```

Update `main()` to wire the new subparsers and dispatch:

```python
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    _add_other_parsers(sub)
    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "regenerate": _cmd_regenerate,
        "compare": _cmd_compare,
        "list-metrics": _cmd_list_metrics,
        "list-metadata-keys": _cmd_list_metadata_keys,
        "clear-cache": _cmd_clear_cache,
    }
    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()
        raise SystemExit(1)
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "list_metrics or clear_cache or regenerate or compare or list_metadata"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/cli/meta_eval.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(cli): add regenerate/compare/list-metrics/list-metadata-keys/clear-cache subcommands"
```

---

### Task 22: Add `--dry-run` cost preview and default-subcommand workaround

**Files:**
- Modify: `src/healthbench_agent/llm_eval/cli/meta_eval.py`
- Test: `tests/llm_eval/test_meta_eval.py`

- [x] **Step 1: Write the failing tests**

```python
def test_cli_dry_run_does_not_call_judge(monkeypatch, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    judge_calls = {"n": 0}

    class CountingJudge(JudgeGrader):
        def grade(self, conversation, rubric_items):
            judge_calls["n"] += 1
            return [
                CriterionVerdict(criterion=r.criterion, criteria_met=True, explanation="")
                for r in rubric_items
            ]

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_judge_for_cli",
        lambda config, temperature: (CountingJudge(), "fake@1", "sha"),
    )

    cli_meta_eval.main([
        "run",
        "--judge-config", "fake.yaml",
        "--sample-size", "3",
        "--n-samples", "1",
        "--no-cache",
        "--no-progress",
        "--no-mlflow",
        "--dry-run",
    ])
    out = capsys.readouterr().out
    assert "DRY RUN" in out.upper() or "Dry run" in out
    assert judge_calls["n"] == 0


def test_cli_default_subcommand_inferred_from_leading_flag(monkeypatch, capsys):
    """Bare `meta-evaluate-judge --judge-config foo.yaml` injects 'run'."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._cmd_run",
        lambda args: print("RUN_CALLED"),
    )
    cli_meta_eval.main(["--judge-config", "x.yaml", "--sample-size", "1"])
    assert "RUN_CALLED" in capsys.readouterr().out
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "dry_run or default_subcommand"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

In [src/healthbench_agent/llm_eval/cli/meta_eval.py](../../../src/healthbench_agent/llm_eval/cli/meta_eval.py), modify `_cmd_run` so it short-circuits on `--dry-run`:

```python
def _cmd_run(args: argparse.Namespace) -> None:
    samples, axis_extractor = _load_consensus_labelled(
        args.subset, args.sample_size, args.seed
    )
    sample_filter, rubric_filter = _build_filters(args)

    if args.dry_run:
        n_calls = sum(
            (1 if s.gold_response else 0)
            + sum(1 for r in s.rubrics if r.example_meets)
            + sum(1 for r in s.rubrics if r.example_fails)
            for s in samples
        ) * args.n_samples
        print("DRY RUN")
        print(f"Judge:           {args.judge_config}")
        print(f"Subset:          {args.subset}  (sample_size={args.sample_size}, seed={args.seed})")
        print(f"Samples:         {len(samples)}")
        print(f"k passes:        {args.n_samples}")
        print(f"Total LLM calls: {n_calls}")
        return

    judge, fingerprint, prompt_sha = _build_judge_for_cli(args.judge_config, args.temperature)
    cache = VerdictCache(enabled=not args.no_cache)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        view = run_meta_eval(
            judge=judge,
            labelled=samples,
            dimension_extractor=axis_extractor,
            metric_names=metric_names,
            n_samples=args.n_samples,
            sample_filter=sample_filter,
            rubric_filter=rubric_filter,
            output_dir=output_dir,
            judge_metadata={
                "judge_model": fingerprint,
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "subset": args.subset,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "judge_prompt_sha": prompt_sha,
                "filter_axis": ",".join(args.rubric_axis),
                "filter_metadata": ",".join(args.metadata),
            },
            cache=cache,
            model_fingerprint=fingerprint,
            judge_prompt_sha=prompt_sha,
            progress=not args.no_progress,
        )
    except EmptyFilterError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    print(view.summary())
```

Also modify `main()` to inject the default `run` verb:

```python
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(prog="meta-evaluate-judge")
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    _add_other_parsers(sub)

    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw or raw[0].startswith("-"):
        raw = ["run", *raw]

    args = parser.parse_args(raw)
    if args.command is None:
        args.command = "run"
    handlers = {
        "run": _cmd_run,
        "regenerate": _cmd_regenerate,
        "compare": _cmd_compare,
        "list-metrics": _cmd_list_metrics,
        "list-metadata-keys": _cmd_list_metadata_keys,
        "clear-cache": _cmd_clear_cache,
    }
    handlers[args.command](args)
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/llm_eval/test_meta_eval.py -v -k "dry_run or default_subcommand"`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/llm_eval/cli/meta_eval.py tests/llm_eval/test_meta_eval.py
git commit -m "feat(cli): add --dry-run cost preview and default-subcommand workaround"
```

---

## Phase 11 — Prompt Optimization Integration

### Task 23: Add `OptimizationMetric` Protocol

**Files:**
- Modify: `src/healthbench_agent/prompt_optimization/optimizer.py`
- Test: `tests/prompt_optimization/test_optimizer.py`

- [x] **Step 1: Write the failing test**

Append (or create) `tests/prompt_optimization/test_optimizer.py`:

```python
def test_end_to_end_metric_satisfies_optimization_metric_protocol():
    from healthbench_agent.prompt_optimization.optimizer import OptimizationMetric

    class _Concrete:
        def __call__(self, prompt: str) -> float:
            return 0.5

    obj: OptimizationMetric = _Concrete()
    assert obj("hi") == 0.5
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prompt_optimization/test_optimizer.py -v -k optimization_metric_protocol`
Expected: FAIL — `OptimizationMetric` does not exist.

- [x] **Step 3: Write minimal implementation**

Append to [src/healthbench_agent/prompt_optimization/optimizer.py](../../../src/healthbench_agent/prompt_optimization/optimizer.py):

```python
from typing import Protocol


class OptimizationMetric(Protocol):
    """Callable contract shared by every prompt-optimization fitness function.

    Both ``EndToEndMetric`` and ``JudgeAgreementMetric`` satisfy this
    Protocol structurally (no inheritance required).
    """

    def __call__(self, prompt: str) -> float:
        """Score a candidate prompt and return a single fitness scalar.

        Higher is better.
        """
        ...
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/prompt_optimization/test_optimizer.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/optimizer.py tests/prompt_optimization/test_optimizer.py
git commit -m "feat(prompt_opt): add OptimizationMetric Protocol"
```

---

### Task 24: Add `JudgeAgreementMetric` and extend `EndToEndMetric` with filters

**Files:**
- Modify: `src/healthbench_agent/prompt_optimization/metric.py`
- Test: `tests/prompt_optimization/test_metric.py`

- [x] **Step 1: Write the failing tests**

```python
def test_judge_agreement_metric_returns_scalar_from_meta_eval(monkeypatch):
    from healthbench_agent.llm_eval.meta_eval import demo_labelled_set, OracleJudge
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    samples = demo_labelled_set()
    perfect = {c: m for s in samples for c, m in s.expected.items()}

    captured: dict[str, Any] = {}

    class _CapturingMetric(JudgeAgreementMetric):
        def _build_judge(self, candidate_template: str):  # type: ignore[override]
            captured["template"] = candidate_template
            return OracleJudge(perfect), "fake@1", "sha"

    metric = _CapturingMetric(
        judge_config=None,  # bypassed by override
        labelled=samples,
        dimension_extractor=lambda r: r.category,
        n_samples=1,
        fitness="gold_score",
    )
    score = metric("CANDIDATE TEMPLATE")
    assert isinstance(score, float)
    assert captured["template"] == "CANDIDATE TEMPLATE"


def test_end_to_end_metric_accepts_filter_kwargs():
    """Just verify the kwargs reach the constructor; behaviour tested via integration."""
    from healthbench_agent.prompt_optimization.metric import EndToEndMetric
    import inspect
    sig = inspect.signature(EndToEndMetric.__init__)
    assert "sample_filter" in sig.parameters
    assert "rubric_filter" in sig.parameters


def test_empty_filter_error_re_exported():
    from healthbench_agent.prompt_optimization.metric import EmptyFilterError
    from healthbench_agent.llm_eval.meta_eval import EmptyFilterError as RootError
    assert EmptyFilterError is RootError
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prompt_optimization/test_metric.py -v -k "judge_agreement or filter or empty_filter"`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Append to [src/healthbench_agent/prompt_optimization/metric.py](../../../src/healthbench_agent/prompt_optimization/metric.py):

```python
from typing import Any, Callable

from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.grading.config import JudgeConfig
from healthbench_agent.llm_eval.meta_eval import (
    EmptyFilterError,
    run_meta_eval,
)


class JudgeAgreementMetric:
    """Fitness metric that scores a candidate grader prompt by running
    meta-evaluation against a fixed labelled set.
    """

    def __init__(
        self,
        judge_config: JudgeConfig | None,
        labelled: list[LabelledSample],
        dimension_extractor: Callable[[RubricItem], str | None],
        n_samples: int = 3,
        fitness: str = "gold_score",
        sample_filter: Callable[[LabelledSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None:
        self.judge_config = judge_config
        self.labelled = labelled
        self.dimension_extractor = dimension_extractor
        self.n_samples = n_samples
        self.fitness = fitness
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter

    def _build_judge(self, candidate_template: str):
        from healthbench_agent.llm_eval.grading.judge import LLMJudgeGrader

        if self.judge_config is None:
            raise RuntimeError("judge_config is None and _build_judge was not overridden")
        cfg = self.judge_config.model_copy()
        judge = LLMJudgeGrader(config=cfg, prompt_template=candidate_template)
        fingerprint = f"{cfg.provider}/{cfg.model}@{cfg.temperature}"
        import hashlib
        prompt_sha = hashlib.sha256(candidate_template.encode("utf-8")).hexdigest()
        return judge, fingerprint, prompt_sha

    def __call__(self, candidate_template: str) -> float:
        judge, fingerprint, prompt_sha = self._build_judge(candidate_template)
        view = run_meta_eval(
            judge=judge,
            labelled=self.labelled,
            dimension_extractor=self.dimension_extractor,
            metric_names=[self.fitness],
            n_samples=self.n_samples,
            sample_filter=self.sample_filter,
            rubric_filter=self.rubric_filter,
            judge_metadata={"judge_model": fingerprint},
            progress=False,
        )
        return float(view.results.scores[self.fitness])
```

Then modify `EndToEndMetric.__init__` to accept the new kwargs and apply them inside `__call__`:

```python
class EndToEndMetric:
    def __init__(
        self,
        agent_config,
        judge,
        samples,
        target_agent_name=None,
        sample_filter: Callable[[LabelledSample], bool] | None = None,
        rubric_filter: Callable[[RubricItem], bool] | None = None,
    ) -> None:
        # ... existing init body ...
        self.sample_filter = sample_filter
        self.rubric_filter = rubric_filter
```

In `__call__`, pre-filter `self.samples` (and each sample's `rubrics`) before delegating to `EvalRunner`:

```python
    def __call__(self, prompt: str) -> float:
        kept = (
            [s for s in self.samples if self.sample_filter(s)]
            if self.sample_filter is not None
            else list(self.samples)
        )
        if not kept:
            raise EmptyFilterError(sample_filter=self.sample_filter, rubric_filter=self.rubric_filter)
        if self.rubric_filter is not None:
            patched: list = []
            for sample in kept:
                surviving = [r for r in sample.rubrics if self.rubric_filter(r)]
                if not surviving:
                    continue
                clone = sample.__class__(**{**sample.__dict__, "rubrics": surviving})
                patched.append(clone)
            kept = patched
            if not kept:
                raise EmptyFilterError(sample_filter=self.sample_filter, rubric_filter=self.rubric_filter)
        # ... rest of existing __call__, using `kept` instead of self.samples ...
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/prompt_optimization/test_metric.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/metric.py tests/prompt_optimization/test_metric.py
git commit -m "feat(prompt_opt): add JudgeAgreementMetric and filter args on EndToEndMetric"
```

---

### Task 25: Add `--prompt-domain` flag to `optimize-prompt` CLI

**Files:**
- Modify: `src/healthbench_agent/prompt_optimization/cli.py`
- Test: `tests/prompt_optimization/test_cli.py`

- [x] **Step 1: Write the failing test**

```python
def test_optimize_prompt_cli_judge_domain_writes_to_llm_grader(tmp_path, monkeypatch):
    """--prompt-domain judge writes to prompts/llm_grader/v2_optimized.yaml."""
    from healthbench_agent.prompt_optimization import cli as optimize_cli

    grader_dir = tmp_path / "prompts" / "llm_grader"
    grader_dir.mkdir(parents=True)
    grader_yaml = grader_dir / "v1_llm_grader.yaml"
    grader_yaml.write_text(
        "version: 1.0.0\ntemplate: 'old template'\n"
    )

    monkeypatch.setattr(
        "healthbench_agent.prompt_optimization.cli._run_judge_optimization",
        lambda args: {
            "optimized_prompt": "NEW TEMPLATE",
            "baseline_score": 0.5,
            "optimized_score": 0.7,
            "improvement": 0.2,
            "num_trials": 3,
            "optimizer_name": "critique_refine",
            "trials": [],
            "target_prompt_path": str(grader_yaml),
        },
    )

    optimize_cli.main_argv([
        "--prompt-domain", "judge",
        "--judge-config", str(grader_yaml),
        "--optimizer", "critique_refine",
        "--sample-size", "1",
        "--max-trials", "1",
    ])

    out = grader_dir / "v2_optimized.yaml"
    assert out.exists()
    assert "NEW TEMPLATE" in out.read_text()
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prompt_optimization/test_cli.py -v -k judge_domain`
Expected: FAIL.

- [x] **Step 3: Write minimal implementation**

Add `--prompt-domain` flag to the existing parser in [src/healthbench_agent/prompt_optimization/cli.py](../../../src/healthbench_agent/prompt_optimization/cli.py) and a `--judge-config` argument. Refactor `main()` so it dispatches:

```python
parser.add_argument(
    "--prompt-domain",
    choices=("agent", "judge"),
    default="agent",
    help="Whether to optimize an agent prompt (default) or a judge prompt.",
)
parser.add_argument(
    "--judge-config",
    default=None,
    help="Path to JudgeConfig YAML — required when --prompt-domain judge.",
)
parser.add_argument("--rubric-axis", action="append", default=[])
parser.add_argument("--metadata", action="append", default=[])
parser.add_argument("--fitness", default="gold_score")


def _judge_save_path(judge_config_path: str) -> Path:
    """v2_optimized.yaml in the same directory as the source grader YAML."""
    return Path(judge_config_path).parent / "v2_optimized.yaml"


def _run_judge_optimization(args: argparse.Namespace) -> dict[str, Any]:
    from healthbench_agent.llm_eval.cli.meta_eval import (
        _build_judge_for_cli,
        _build_filters,
        _load_consensus_labelled,
    )
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.grading.judge import load_grader_prompt
    from healthbench_agent.prompt_optimization import (
        create_prompt_optimizer,
        get_optimizer_config_class,
    )
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    if args.judge_config is None:
        raise SystemExit("--prompt-domain judge requires --judge-config")
    cfg = JudgeConfig.from_yaml(args.judge_config)
    samples, axis_extractor = _load_consensus_labelled(
        subset="consensus", sample_size=args.sample_size, seed=args.seed
    )
    sample_filter, rubric_filter = _build_filters(args)
    metric = JudgeAgreementMetric(
        judge_config=cfg,
        labelled=samples,
        dimension_extractor=axis_extractor,
        n_samples=3,
        fitness=args.fitness,
        sample_filter=sample_filter,
        rubric_filter=rubric_filter,
    )
    config_class = get_optimizer_config_class(args.optimizer)
    optim_config = config_class(
        optimizer=args.optimizer, max_trials=args.max_trials,
        sample_size=args.sample_size, seed=args.seed,
    )
    optimizer = create_prompt_optimizer(optim_config)
    template, _, _ = load_grader_prompt(cfg.prompt_path)
    result = optimizer.optimize(current_prompt=template, samples=samples, metric=metric)
    return {
        "optimized_prompt": result.optimized_prompt,
        "baseline_score": result.baseline_score,
        "optimized_score": result.optimized_score,
        "improvement": result.improvement,
        "num_trials": result.num_trials,
        "optimizer_name": result.optimizer_name,
        "trials": result.trial_history,
        "target_prompt_path": str(cfg.prompt_path),
    }


def _save_judge_yaml(args: argparse.Namespace, result: dict[str, Any]) -> Path:
    out = _judge_save_path(args.judge_config)
    payload = {
        "version": "2.0.0",
        "created": date.today().isoformat(),
        "parent_version": "1.0.0",
        "parent_prompt_path": result["target_prompt_path"],
        "template": result["optimized_prompt"],
        "rationale": (
            f"Automatically optimized using {result['optimizer_name']}. "
            f"Score: {result['baseline_score']:.4f} -> "
            f"{result['optimized_score']:.4f} ({result['improvement']:+.4f}). "
            f"Trials: {result['num_trials']}."
        ),
    }
    out.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))
    return out
```

Refactor the existing `main()` body so the agent path stays the default and the judge path runs through the new helpers. Also expose `main_argv(argv)` so tests can pass a list directly.

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/prompt_optimization/test_cli.py -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/healthbench_agent/prompt_optimization/cli.py tests/prompt_optimization/test_cli.py
git commit -m "feat(prompt_opt): add --prompt-domain {agent,judge} to optimize-prompt CLI"
```

---

## Phase 12 — Wiring and Documentation

### Task 26: Register `meta-evaluate-judge` script + add `tqdm` dep

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: Inspect current scripts**

Read [pyproject.toml](../../../pyproject.toml).

- [x] **Step 2: Add `tqdm` to `dependencies` and the script entry**

In `[project] dependencies` confirm `tqdm>=4.66.0` is present (it already is per the existing file). Add to `[project.scripts]`:

```toml
[project.scripts]
download-healthbench = "healthbench_agent.dataset.loader:_cli"
track-experiment = "healthbench_agent.llm_eval.cli.track_experiment:main"
optimize-prompt = "healthbench_agent.prompt_optimization.cli:main"
meta-evaluate-judge = "healthbench_agent.llm_eval.cli.meta_eval:main"
```

- [x] **Step 3: Run `uv sync` and the smoke test**

```bash
uv sync
uv run meta-evaluate-judge list-metrics
```

Expected: CLI prints all 8 metrics.

- [x] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: register meta-evaluate-judge console script"
```

---

### Task 27: Update `CLAUDE.md` Project Layout

**Files:**
- Modify: `CLAUDE.md`

- [x] **Step 1: Read current Project Layout block**

Read [CLAUDE.md](../../../CLAUDE.md) lines around `domain/` and `llm_eval/`.

- [x] **Step 2: Add new entries**

Add to the `domain/` block:
```
- `meta_evaluation.py` — `LabelledSample` (parent class for HealthBenchSample), `MetricResults` (lightweight dataclass), `SCHEMA_VERSION`
```

Add to the `llm_eval/` block:
```
- `meta_eval/` — subpackage with registry (`@register_meta_metric`, `MetricLevel`, `MetricSpec`), 8 built-in metrics under `metrics/`, `run_meta_eval` (`runner.py`), `meta_evaluate` (`api.py`), `OracleJudge` (`oracle_judge.py`), `demo_labelled_set` (`demo_data.py`), filter helpers under `filters.py` (`axis_filter`, `metadata_filter`, `specialty_filter`, `EmptyFilterError`)
- `meta_eval/results/` — `MetricResultsView` (`view.py`, UX wrapper around `MetricResults`) + `save_results`/`load_results` (`io.py`) + plot helpers (`plots.py`)
- `cache/` — `VerdictCache` (file-based judge call cache, `store.py`) + `CachedJudgeGrader` proxy (`cached_judge.py`)
- `cli/meta_eval.py` — `meta-evaluate-judge` CLI with `run / regenerate / compare / list-metrics / list-metadata-keys / clear-cache` subcommands
```

Add to the `dataset/` block:
```
- `extraction.py` — `extract_ideal_completion_text` (HealthBench `ideal_completions_data` normaliser)
```

Add a new "Meta-Evaluation" section after the "Prompt Optimization" section:

```markdown
### Meta-Evaluation
```bash
# Default run on the consensus subset, k=7 calibration passes
uv run meta-evaluate-judge run \
    --judge-config config/judges/openai_gpt41.yaml \
    --sample-size 100

# Show registered metrics
uv run meta-evaluate-judge list-metrics

# Compare two judges offline
uv run meta-evaluate-judge compare runs/openai/ runs/gemini/

# Optimize the judge prompt itself (slice-restricted)
uv run optimize-prompt --prompt-domain judge \
    --judge-config config/judges/openai_gpt41.yaml \
    --optimizer critique_refine --rubric-axis accuracy
```
```

- [x] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document meta-evaluation modules in CLAUDE.md"
```

---

### Task 28: Create `notebooks/04_judge_meta_evaluation.ipynb`

**Files:**
- Create: `notebooks/04_judge_meta_evaluation.ipynb`

- [x] **Step 1: Inspect existing notebook style**

Run: `Glob notebooks/*.ipynb` and `Read notebooks/02_agent_comparison.ipynb` to mirror cell formatting.

- [x] **Step 2: Create the new notebook**

Use the Write tool to create `notebooks/04_judge_meta_evaluation.ipynb` with five cells exactly mirroring the spec's cell list (load → calibration plot → dimension confusion plot → compare → extract gold_score per judge). Each cell should be a single `code_cell` JSON entry. Use the structure from `02_agent_comparison.ipynb` as the template.

The cell contents are:

```python
# Cell 1
from healthbench_agent.llm_eval.meta_eval import (
    load_results,
    plot_calibration_curve,
    plot_dimension_confusion,
)

view = load_results("runs/meta_eval/2026-04-07_gpt-4-1")
view
```

```python
# Cell 2
plot_calibration_curve(view.results)
```

```python
# Cell 3
plot_dimension_confusion(view.results)
```

```python
# Cell 4
other = load_results("runs/meta_eval/2026-04-07_gemini-2-5")
view.compare(other)
```

```python
# Cell 5
view.results.scores["gold_score"], other.results.scores["gold_score"]
```

- [x] **Step 3: Smoke test the notebook structure**

Run: `uv run python -c "import json; nb = json.load(open('notebooks/04_judge_meta_evaluation.ipynb')); assert len(nb['cells']) == 5"`
Expected: no AssertionError.

- [x] **Step 4: Commit**

```bash
git add notebooks/04_judge_meta_evaluation.ipynb
git commit -m "docs(notebook): add judge meta-evaluation walkthrough"
```

---

## Phase 13 — Final Wiring and Coverage

### Task 29: Run the full test suite + coverage check

**Files:**
- All modified files

- [x] **Step 1: Lint + type check + tests**

Run, in order:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest tests/ --cov=src/healthbench_agent --cov-report=term-missing -v
```

Expected: all green; per-module coverage ≥ 95%; pure metric functions at 100%.

- [x] **Step 2: Fix any failures inline**

Loop until clean. Address any new ruff/mypy diagnostics introduced by Tasks 1-28.

- [x] **Step 3: Final commit (if anything had to be fixed)**

```bash
git add -u
git commit -m "chore: lint/type/test cleanup for judge meta-evaluation"
```

---

## Out of Scope

These items appear in the spec under "Out of Scope (Follow-up Issues)" and are intentionally not in this plan:

- Anthropic / Claude judge sampler
- Hand-labelled gold set loader
- Noise-floor hook into `prompt_optimization/`
- Batch-API mode for the judge during meta-eval
- Optimising the rubric items themselves
