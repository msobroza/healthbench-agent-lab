"""Baseline single agent — Architecture A.

Simplest possible agent: one LLM call with a minimal health instruction,
no tools, no sub-agents. Establishes the performance floor against which
Architecture B (tool-augmented) and C (multi-agent) are compared.

Run:
    uv run adk web agents/baseline_agent
    uv run adk run agents/baseline_agent
"""

from pathlib import Path

import yaml
from google.adk.agents import Agent

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v1_baseline.yaml"


def _load_instruction() -> str:
    """Load the instruction text from the versioned YAML prompt file.

    Returns:
        The instruction string from prompts/v1_baseline.yaml.
    """
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["instruction"]


root_agent = Agent(
    name="baseline_agent",
    model="gemini-2.0-flash",
    description="Baseline health assistant with no tools or sub-agents.",
    instruction=_load_instruction(),
)
