"""Prompt loading utilities for agent pipelines.

Loads instruction text from versioned YAML prompt files and renders
Jinja2 template variables with the provided context. Agent prompts
use standard ``{{ variable }}`` Jinja2 syntax.

The main template variable is ``{{ conversation }}``, which is formatted
from a ``MessageList`` using ``format_conversation()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from healthbench_agent.domain.conversation import MessageList


def format_conversation(message_list: MessageList) -> str:
    """Format a MessageList into a readable string for prompt insertion.

    Each turn is formatted as ``[role]: content`` separated by blank lines.

    Args:
        message_list: Ordered conversation turns (role + content dicts).

    Returns:
        Formatted conversation string.
    """
    parts = []
    for message in message_list:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def load_instruction(
    prompt_path: str | Path,
    key: str = "instruction",
    **context: Any,
) -> str:
    """Load and render an instruction from a prompt YAML file.

    Reads the YAML file, extracts the value at ``key``, treats it as a
    Jinja2 template, and renders it with the given context variables.
    If the instruction contains no template variables, it is returned
    unchanged.

    Args:
        prompt_path: Path to the prompt YAML file.
        key: YAML key containing the instruction text.
        **context: Template variables passed to Jinja2 rendering.
            Common variables: ``conversation`` (formatted MessageList).

    Returns:
        The rendered instruction string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If the key is not found in the YAML file.
    """
    with open(prompt_path) as f:
        data = yaml.safe_load(f)
    raw = data[key]
    template = Template(raw)
    return template.render(**context)
