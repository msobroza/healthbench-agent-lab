"""Multi-agent pipeline — Architecture C.

Orchestrates a three-stage pipeline: triage → specialist → reviewer.
The coordinator agent delegates to either EmergencyAgent or
GeneralHealthAgent based on query urgency. The reviewer always runs
as a SequentialAgent wrapper ensures all stages execute.

Pipeline:
    1. Triage Agent → classifies urgency/topic/expertise → output_key: triage_classification
    2. Coordinator → delegates to specialist → output_key: specialist_response
    3. Reviewer Agent → checks safety/quality → output_key: final_response

Run:
    uv run adk web agents/multi_agent
    uv run adk run agents/multi_agent
"""

from pathlib import Path

import yaml
from google.adk.agents import Agent, SequentialAgent

from .sub_agents import (
    emergency_agent,
    general_health_agent,
    reviewer_agent,
    triage_agent,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v3_structured.yaml"


def _load_coordinator_instruction() -> str:
    """Load the coordinator instruction from the multi-agent prompt YAML.

    Returns:
        The coordinator instruction string.
    """
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["coordinator_instruction"]


coordinator_agent = Agent(
    name="coordinator_agent",
    model="gemini-2.0-flash",
    description="Routes queries to the appropriate specialist based on triage.",
    instruction=_load_coordinator_instruction(),
    sub_agents=[emergency_agent, general_health_agent],
    output_key="specialist_response",
)

root_agent = SequentialAgent(
    name="multi_agent",
    description="Multi-agent health pipeline: triage → specialist → reviewer.",
    sub_agents=[triage_agent, coordinator_agent, reviewer_agent],
)
