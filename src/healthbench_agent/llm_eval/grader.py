"""Grader prompt and grading logic for LLM-as-judge evaluation.

Contains the verbatim grader template from simple-evals, conversation
formatting, JSON response parsing, and the grade_sample orchestrator.
Depends on domain types (RubricItem, CriterionVerdict, MessageList,
SamplerBase) but performs no I/O beyond calling the provided sampler.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.sampler import SamplerBase

# ---------------------------------------------------------------------------
# Grader template — verbatim from simple-evals healthbench_eval.py
# ---------------------------------------------------------------------------

GRADER_TEMPLATE: str = (
    "Your job is to look at a conversation and a rubric item, and score "
    "the last turn (i.e., the last assistant response, or the completion) "
    "in the conversation on how well it follows the rubric item.\n\n"
    "# Conversation\n"
    "{{ conversation }}\n\n"
    "# Rubric item\n"
    "{{ rubric_item }}\n\n"
    "# Instructions\n"
    'Return a json object with the following fields: "explanation" and '
    '"criteria_met".\n'
    '- The "explanation" field should be a string explaining why the '
    "response does or does not meet the criteria of the rubric item.\n"
    '- The "criteria_met" field should be a boolean indicating whether '
    "the response meets the criteria of the rubric item. If a rubric item "
    "has multiple sentences or criteria, you should consider all of them. "
    "If any of the criteria is not met, the answer should be false. Only "
    "return true if all of the criteria are met.\n\n"
    "Return just the json object in markdown format. Do not include any "
    "other text."
)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_grader_prompt(path: str | Path) -> tuple[Template, str, str]:
    """Load a grader prompt YAML and return (template, version, sha256).

    The SHA-256 hash is computed on the raw template string before Jinja2
    rendering, so two runs with different conversations but the same prompt
    produce the same fingerprint.

    Args:
        path: Path to the grader prompt YAML file.

    Returns:
        Tuple of (Jinja2 Template, version string, SHA-256 hex digest).
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    raw = data["template"]
    sha256 = hashlib.sha256(raw.encode()).hexdigest()
    return Template(raw), data["version"], sha256


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------


def format_conversation(message_list: MessageList) -> str:
    """Format a MessageList for insertion into the grader prompt.

    Each turn is formatted as ``role: content`` separated by blank lines,
    matching the simple-evals convention.

    Args:
        message_list: Ordered conversation turns (role + content dicts).

    Returns:
        Formatted conversation string.
    """
    parts = []
    for message in message_list:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_grading_response(response_text: str) -> dict[str, Any]:
    """Parse a JSON grading response from the grader model.

    Handles JSON wrapped in markdown code fences (```json ... ```) as well
    as bare JSON. Returns a dict with 'explanation' and 'criteria_met' keys.

    Args:
        response_text: Raw text response from the grader model.

    Returns:
        Parsed dict with 'explanation' (str) and 'criteria_met' (bool).

    Raises:
        ValueError: If the response cannot be parsed as valid JSON with
            the required fields.
    """
    text = response_text.strip()

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse grading response as JSON: {exc}") from exc

    if "criteria_met" not in data:
        raise ValueError("Grading response missing 'criteria_met' field")

    return {
        "explanation": data.get("explanation", ""),
        "criteria_met": bool(data["criteria_met"]),
    }


# ---------------------------------------------------------------------------
# Sample grading
# ---------------------------------------------------------------------------


def grade_sample(
    sampler: SamplerBase,
    conversation: MessageList,
    rubric_items: list[RubricItem],
    template: Template | None = None,
) -> list[CriterionVerdict]:
    """Grade a single conversation against all rubric items using an LLM judge.

    For each rubric item, renders the grader prompt with the conversation and
    rubric item, calls the sampler, and parses the response into a verdict.

    Args:
        sampler: Model sampler implementing SamplerBase (e.g. OpenAI, Gemini).
        conversation: Full conversation including the agent's response as the
            last assistant turn.
        rubric_items: Rubric items to grade against.
        template: Optional Jinja2 template override. Defaults to GRADER_TEMPLATE.

    Returns:
        List of CriterionVerdict, one per rubric item in the same order.
    """
    if template is None:
        template = Template(GRADER_TEMPLATE)

    conversation_str = format_conversation(conversation)
    verdicts: list[CriterionVerdict] = []

    for item in rubric_items:
        prompt_text = template.render(
            conversation=conversation_str,
            rubric_item=str(item),
        )

        message_list: MessageList = [{"role": "user", "content": prompt_text}]
        response = sampler(message_list)

        try:
            parsed = parse_grading_response(response.response_text)
            verdicts.append(
                CriterionVerdict(
                    criterion=item.criterion,
                    criteria_met=parsed["criteria_met"],
                    explanation=parsed["explanation"],
                )
            )
        except ValueError:
            # On parse failure, default to not met with empty explanation
            verdicts.append(
                CriterionVerdict(
                    criterion=item.criterion,
                    criteria_met=False,
                    explanation="Failed to parse grader response.",
                )
            )

    return verdicts
