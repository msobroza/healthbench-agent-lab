"""LLM-as-judge grader config, template loader, and implementation.

Public surface:

* :class:`JudgeConfig`, :class:`EvalMode` from :mod:`.config`.
* :class:`LLMJudgeGrader`, :func:`create_judge`, :func:`grade_sample`,
  :func:`format_conversation`, :func:`parse_grading_response`,
  :func:`load_grader_prompt`, :func:`make_template` from :mod:`.judge`.

Grader templates themselves live under ``prompts/llm_grader/`` as YAML
files; this package does not embed any dataset-specific prompt string.
"""

from .config import EvalMode, JudgeConfig
from .judge import (
    LLMJudgeGrader,
    create_judge,
    format_conversation,
    grade_sample,
    load_grader_prompt,
    make_template,
    parse_grading_response,
)

__all__ = [
    "EvalMode",
    "JudgeConfig",
    "LLMJudgeGrader",
    "create_judge",
    "format_conversation",
    "grade_sample",
    "load_grader_prompt",
    "make_template",
    "parse_grading_response",
]
