"""Tool-augmented agent — Architecture B.

Single agent with clinically-aware prompt and three medical reference tools:
drug_reference(), symptom_checker(), emergency_flag(). The agent is instructed
to use tools proactively to ground its responses in verified data.

Run:
    uv run adk web agents/tool_agent
    uv run adk run agents/tool_agent
"""

from pathlib import Path

import yaml
from google.adk.agents import Agent

from .tools import drug_reference, emergency_flag, symptom_checker

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v2_clinical.yaml"


def _load_instruction() -> str:
    """Load the instruction text from the versioned YAML prompt file.

    Returns:
        The instruction string from prompts/v2_clinical.yaml.
    """
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["instruction"]


root_agent = Agent(
    name="tool_agent",
    model="gemini-2.0-flash",
    description="Clinical health assistant with medical reference tools.",
    instruction=_load_instruction(),
    tools=[drug_reference, symptom_checker, emergency_flag],
)
