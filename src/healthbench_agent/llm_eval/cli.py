"""CLI entry point for experiment tracking.

Thin wrapper that provides a ``track-experiment`` console script
entry point registered in ``pyproject.toml``. The actual CLI logic
lives in ``evaluation.experiment_tracker._cli`` (a working-directory
module). This wrapper adds the project root to ``sys.path`` so the
``evaluation`` package is importable from the console script context.

Usage::

    uv run track-experiment --agent-config config/agents/baseline_agent.yaml
    uv run track-experiment --agent-config config/agents/tool_agent.yaml \\
        --sample-size 50 --subset hard --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:  # pragma: no cover
    """Entry point for the ``track-experiment`` console script.

    Adds the project root to ``sys.path`` so that working-directory
    modules (``evaluation/``, ``agents/``, ``tools/``) are importable,
    then delegates to the experiment tracker CLI.
    """
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from evaluation.experiment_tracker import _cli

    _cli()
