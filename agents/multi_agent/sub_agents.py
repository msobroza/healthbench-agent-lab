"""Specialist sub-agents for the multi-agent pipeline (Architecture C).

Defines four focused agents:
- triage_agent: classifies urgency, topic, and user expertise
- emergency_agent: handles high-urgency medical queries
- general_health_agent: handles routine health queries with tools
- reviewer_agent: reviews responses for safety and quality

Each agent stores its output via output_key for downstream consumption.

See AGENT_DECISIONS.md §7-10 for design rationale.
"""

from pathlib import Path

import yaml
from google.adk.agents import Agent

from agents.tool_agent.tools import drug_reference, emergency_flag, symptom_checker

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v3_structured.yaml"


def _load_prompt(key: str) -> str:
    """Load a specific instruction from the multi-agent prompt YAML.

    Args:
        key: The YAML key for the instruction (e.g. 'triage_instruction').

    Returns:
        The instruction string.
    """
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)[key]


triage_agent = Agent(
    name="triage_agent",
    model="gemini-2.0-flash",
    description="Classifies query urgency, topic, and user expertise level.",
    instruction=_load_prompt("triage_instruction"),
    output_key="triage_classification",
)

emergency_agent = Agent(
    name="emergency_agent",
    model="gemini-2.0-flash",
    description="Handles high-urgency medical queries requiring immediate referral.",
    instruction=_load_prompt("emergency_instruction"),
    tools=[emergency_flag],
    output_key="specialist_response",
)

general_health_agent = Agent(
    name="general_health_agent",
    model="gemini-2.0-flash",
    description="Handles general health queries with medical reference tools.",
    instruction=_load_prompt("general_health_instruction"),
    tools=[drug_reference, symptom_checker, emergency_flag],
    output_key="specialist_response",
)

reviewer_agent = Agent(
    name="reviewer_agent",
    model="gemini-2.0-flash",
    description="Reviews responses for completeness, safety, and communication quality.",
    instruction=_load_prompt("reviewer_instruction"),
    output_key="final_response",
)
