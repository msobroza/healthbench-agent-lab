"""Grader prompt, parsing utilities, and LLM-based judge implementation.

Contains the verbatim grader template from simple-evals, conversation
formatting, JSON response parsing, the LLMJudgeGrader concrete class,
and a create_judge factory.

Depends on domain types (RubricItem, CriterionVerdict, MessageList,
SamplerBase, JudgeGrader) but performs no I/O beyond calling the
provided sampler.
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
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.sampler import SamplerBase

from .config_grader import JudgeConfig
from .samplers import create_sampler

# ---------------------------------------------------------------------------
# Grader template — verbatim from simple-evals healthbench_eval.py
# ---------------------------------------------------------------------------

GRADER_TEMPLATE: str = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


# Jinja2 environment matching simple-evals placeholder syntax (<<...>>).
_JINJA_ENV = Environment(
    variable_start_string="<<",
    variable_end_string=">>",
)


def _make_template(raw: str) -> Template:
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
    return _make_template(raw), data["version"], sha256


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

    Wraps a SamplerBase implementation and a Jinja2 prompt template.
    For each rubric item, renders the prompt with the conversation and
    rubric text, calls the sampler, and parses the response into a verdict.

    Uses ``<<conversation>>`` and ``<<rubric_item>>`` placeholder syntax
    (matching simple-evals) via a custom Jinja2 ``Environment``.

    Attributes:
        sampler: Model sampler implementing SamplerBase.
        template: Jinja2 grader prompt template (``<<...>>`` delimiters).
        max_workers: Upper bound on concurrent threads for rubric grading.
    """

    def __init__(
        self,
        sampler: SamplerBase,
        template: Any | None = None,
        max_workers: int = 8,
    ) -> None:
        self.sampler = sampler
        self.template = template or _make_template(GRADER_TEMPLATE)
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
            response = self.sampler(message_list)
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

    Builds the appropriate sampler from the config and loads the grader
    prompt template from the configured path.

    Args:
        config: Judge configuration specifying provider, model, and
            prompt template path.

    Returns:
        A configured LLMJudgeGrader ready to grade samples.
    """
    sampler = create_sampler(config)
    template = load_grader_prompt(config.prompt_path)[0]
    return LLMJudgeGrader(
        sampler=sampler,
        template=template,
        max_workers=config.grader_max_workers,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible module-level function
# ---------------------------------------------------------------------------


def grade_sample(
    sampler: SamplerBase,
    conversation: MessageList,
    rubric_items: list[RubricItem],
    template: Any | None = None,
) -> list[CriterionVerdict]:
    """Grade a single conversation against all rubric items using an LLM judge.

    Convenience wrapper that creates a temporary LLMJudgeGrader and calls
    grade(). Prefer using LLMJudgeGrader or create_judge() directly for
    repeated grading — they avoid re-creating the grader on every call.

    Args:
        sampler: Model sampler implementing SamplerBase.
        conversation: Full conversation including the agent's response.
        rubric_items: Rubric items to grade against.
        template: Optional Jinja2 template override. Defaults to GRADER_TEMPLATE.

    Returns:
        List of CriterionVerdict, one per rubric item in the same order.
    """
    judge = LLMJudgeGrader(sampler=sampler, template=template)
    return judge.grade(conversation, rubric_items)
