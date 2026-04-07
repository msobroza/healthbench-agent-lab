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
    """Pull the gold response text out of HealthBench's ``ideal_completions_data`` block.

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
