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
from .runner import EvalRunner
from .samplers import create_sampler

__all__ = [
    "GRADER_TEMPLATE",
    "LLMJudgeGrader",
    "create_judge",
    "format_conversation",
    "grade_sample",
    "load_grader_prompt",
    "parse_grading_response",
    "EvalRunner",
    "JudgeConfig",
    "EvalMode",
    "create_sampler",
]
