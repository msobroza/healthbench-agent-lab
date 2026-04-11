"""Grader prompt loading, parsing utilities, and LLM-based judge implementation.

Grader templates are loaded from YAML via :func:`load_grader_prompt` — this
module no longer embeds any dataset-specific template strings. Default paths
live on :class:`JudgeConfig` so the judge can be pointed at any dataset's
grader YAML.

Depends on domain types (RubricItem, CriterionVerdict, MessageList,
LLMClient, JudgeGrader) but performs no I/O beyond calling the
provided LLM client and reading the configured prompt file.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, Template

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.llm_client import LLMClient
from healthbench_agent.domain.rubric import RubricItem

from .config import JudgeConfig

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


# Jinja2 environment matching simple-evals placeholder syntax (<<...>>).
_JINJA_ENV = Environment(
    variable_start_string="<<",
    variable_end_string=">>",
)


def make_template(raw: str) -> Template:
    """Create a Jinja2 Template using <<...>> delimiters.

    Args:
        raw: Raw template string with ``<<placeholder>>`` syntax.

    Returns:
        A compiled Jinja2 Template.
    """
    return _JINJA_ENV.from_string(raw)


def load_grader_prompt(path: str | Path) -> tuple[Any, str, str]:
    """Load a grader prompt YAML and return (template, version, sha256).

    The SHA-256 hash is computed on the raw template string before
    placeholder substitution, so two runs with different conversations
    but the same prompt produce the same fingerprint.

    Args:
        path: Path to the grader prompt YAML file.

    Returns:
        Tuple of (Jinja2 Template, version string, SHA-256 hex digest).
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    raw = data["template"].strip()
    sha256 = hashlib.sha256(raw.encode()).hexdigest()
    return make_template(raw), data["version"], sha256


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
# LLM-based judge grader
# ---------------------------------------------------------------------------


class LLMJudgeGrader(JudgeGrader):
    """Grades conversations against rubric items using an LLM as judge.

    Wraps an LLMClient implementation and a Jinja2 prompt template.
    For each rubric item, renders the prompt with the conversation and
    rubric text, calls the LLM client, and parses the response into a verdict.

    Uses ``<<conversation>>`` and ``<<rubric_item>>`` placeholder syntax
    (matching simple-evals) via a custom Jinja2 ``Environment``.

    Attributes:
        llm_client: LLM client implementing LLMClient.
        template: Jinja2 grader prompt template (``<<...>>`` delimiters).
        max_workers: Upper bound on concurrent threads for rubric grading.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        template: Template,
        max_workers: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.template = template
        self.max_workers = max_workers

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """Grade a conversation against all rubric items using the LLM judge.

        Rubric items are graded concurrently via a ThreadPoolExecutor to
        avoid sequential latency when a sample has many criteria.

        Args:
            conversation: Full conversation including the agent's response
                as the last assistant turn.
            rubric_items: Rubric items to grade against.

        Returns:
            List of CriterionVerdict, one per rubric item in the same order.
        """
        conversation_str = format_conversation(conversation)

        def _grade_item(item: RubricItem) -> CriterionVerdict:
            prompt_text = self.template.render(
                conversation=conversation_str,
                rubric_item=str(item),
            )
            message_list: MessageList = [{"role": "user", "content": prompt_text}]
            response = self.llm_client(message_list)
            try:
                parsed = parse_grading_response(response.response_text)
                return CriterionVerdict(
                    criterion=item.criterion,
                    criteria_met=parsed["criteria_met"],
                    explanation=parsed["explanation"],
                )
            except ValueError:
                return CriterionVerdict(
                    criterion=item.criterion,
                    criteria_met=False,
                    explanation="Failed to parse grader response.",
                )

        if not rubric_items:
            return []

        with ThreadPoolExecutor(max_workers=min(len(rubric_items), self.max_workers)) as executor:
            verdicts = list(executor.map(_grade_item, rubric_items))

        return verdicts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_judge(config: JudgeConfig) -> LLMJudgeGrader:
    """Create an LLMJudgeGrader from a JudgeConfig.

    Builds the appropriate LLM client from the config and loads the grader
    prompt template from ``config.prompt_path``. The default path targets
    HealthBench; override ``prompt_path`` to evaluate against a different
    dataset's grader YAML.

    Args:
        config: Judge configuration specifying provider, model, and
            prompt template path.

    Returns:
        A configured LLMJudgeGrader ready to grade samples.
    """
    from ..clients import create_llm_client

    llm_client = create_llm_client(config)
    template = load_grader_prompt(config.prompt_path)[0]
    return LLMJudgeGrader(
        llm_client=llm_client,
        template=template,
        max_workers=config.grader_max_workers,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def grade_sample(
    llm_client: LLMClient,
    conversation: MessageList,
    rubric_items: list[RubricItem],
    template: Template,
) -> list[CriterionVerdict]:
    """Grade a single conversation against all rubric items using an LLM judge.

    Convenience wrapper that creates a temporary LLMJudgeGrader and calls
    ``grade()``. Prefer :class:`LLMJudgeGrader` or :func:`create_judge`
    directly for repeated grading — they avoid re-creating the grader on
    every call.

    Args:
        llm_client: LLM client implementing LLMClient.
        conversation: Full conversation including the agent's response.
        rubric_items: Rubric items to grade against.
        template: Jinja2 grader template (load via :func:`load_grader_prompt`
            or build from a raw string with :func:`make_template`).

    Returns:
        List of CriterionVerdict, one per rubric item in the same order.
    """
    judge = LLMJudgeGrader(llm_client=llm_client, template=template)
    return judge.grade(conversation, rubric_items)
