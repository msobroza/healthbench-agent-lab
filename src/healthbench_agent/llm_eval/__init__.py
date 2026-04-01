"""LLM-as-judge evaluation module (provider-agnostic).

Replicates the exact evaluation methodology from simple-evals
healthbench_eval.py. Supports any SamplerBase-compatible model.

Public API:
    - GRADER_TEMPLATE: The verbatim grader prompt from simple-evals.
    - grade_sample: Grade a single sample against all rubric items.
    - format_conversation: Format a MessageList for the grader prompt.
    - parse_grading_response: Parse JSON grading response.
    - load_grader_prompt: Load and hash a grader prompt YAML file.
"""

from .grader import (
    GRADER_TEMPLATE,
    format_conversation,
    grade_sample,
    load_grader_prompt,
    parse_grading_response,
)

__all__ = [
    "GRADER_TEMPLATE",
    "format_conversation",
    "grade_sample",
    "load_grader_prompt",
    "parse_grading_response",
]
