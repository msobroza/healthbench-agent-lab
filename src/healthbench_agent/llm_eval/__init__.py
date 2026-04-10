"""LLM-as-judge evaluation module (provider-agnostic).

Replicates the exact evaluation methodology from simple-evals
healthbench_eval.py. Supports any SamplerBase-compatible model.

Public API:
    - GRADER_TEMPLATE: The verbatim grader prompt from simple-evals.
    - LLMJudgeGrader: Concrete JudgeGrader that uses an LLM sampler.
    - create_judge: Factory to build a judge from JudgeConfig.
    - grade_sample: Convenience wrapper for one-off grading.
    - format_conversation: Format a MessageList for the grader prompt.
    - parse_grading_response: Parse JSON grading response.
    - load_grader_prompt: Load and hash a grader prompt YAML file.
    - EvalRunner: Orchestrates evaluation across multiple samples.
    - JudgeConfig: Type-safe judge configuration.
    - EvalMode: Execution mode enum (ASYNC, BATCH).
    - create_sampler: Factory to build a sampler from JudgeConfig.
    - meta_evaluate: Meta-evaluate one judge end-to-end.
    - run_meta_eval: Low-level meta-evaluation orchestrator.
    - MetricResultsView: Rich-UX view of meta-evaluation results.
    - VerdictCache: Thread-safe on-disk verdict cache.
"""

from .config_grader import EvalMode, JudgeConfig
from .grader import (
    GRADER_TEMPLATE,
    LLMJudgeGrader,
    create_judge,
    format_conversation,
    grade_sample,
    load_grader_prompt,
    parse_grading_response,
)
from .meta_eval import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    FakeJudge,
    MetricLevel,
    MetricSpec,
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
from .meta_eval_results import MetricResultsView
from .runner import EvalRunner
from .samplers import create_sampler
from .verdict_cache import CachedJudgeGrader, VerdictCache

__all__ = [
    "AXIS_TAG_PREFIX",
    "GRADER_TEMPLATE",
    "CachedJudgeGrader",
    "EmptyFilterError",
    "EvalMode",
    "EvalRunner",
    "FakeJudge",
    "JudgeConfig",
    "LLMJudgeGrader",
    "MetricLevel",
    "MetricResultsView",
    "MetricSpec",
    "VerdictCache",
    "axis_filter",
    "create_judge",
    "create_sampler",
    "demo_labelled_set",
    "format_conversation",
    "get_meta_metric",
    "grade_sample",
    "load_grader_prompt",
    "meta_evaluate",
    "metadata_filter",
    "parse_grading_response",
    "register_meta_metric",
    "registered_meta_metrics",
    "run_meta_eval",
    "specialty_filter",
]
