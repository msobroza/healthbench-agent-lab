"""Tests for agents/multi_agent/ — Architecture C.

Validates sub-agent configuration, orchestration pipeline, prompt loading,
and the root_agent contract. Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

import yaml

from google.adk.agents import SequentialAgent

from agents.multi_agent.agent import root_agent, coordinator_agent, _PROMPT_PATH
from agents.multi_agent.sub_agents import (
    triage_agent,
    emergency_agent,
    general_health_agent,
    reviewer_agent,
    _load_prompt,
)


# ---------------------------------------------------------------------------
# Prompt YAML tests
# ---------------------------------------------------------------------------


class TestMultiAgentPromptYaml:
    """Tests for prompts/v3_structured.yaml structure and content."""

    def test_prompt_file_exists(self):
        assert _PROMPT_PATH.exists()

    def test_prompt_yaml_has_all_instruction_keys(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        required_keys = [
            "triage_instruction",
            "emergency_instruction",
            "general_health_instruction",
            "reviewer_instruction",
            "coordinator_instruction",
        ]
        for key in required_keys:
            assert key in data, f"Missing prompt key: {key}"

    def test_prompt_yaml_has_metadata(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        assert "version" in data
        assert "rationale" in data

    def test_load_prompt_returns_string(self):
        instruction = _load_prompt("triage_instruction")
        assert isinstance(instruction, str)
        assert len(instruction) > 50


# ---------------------------------------------------------------------------
# Triage Agent tests
# ---------------------------------------------------------------------------


class TestTriageAgent:
    """Tests for the triage_agent sub-agent."""

    def test_name(self):
        assert triage_agent.name == "triage_agent"

    def test_model(self):
        assert triage_agent.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert triage_agent.output_key == "triage_classification"

    def test_has_no_tools(self):
        assert not triage_agent.tools

    def test_has_no_sub_agents(self):
        assert not triage_agent.sub_agents

    def test_instruction_contains_urgency_classification(self):
        instruction = triage_agent.instruction
        assert "URGENCY" in instruction
        assert "emergency" in instruction.lower()
        assert "urgent" in instruction.lower()
        assert "routine" in instruction.lower()

    def test_instruction_references_expertise_classification(self):
        instruction = triage_agent.instruction
        assert "EXPERTISE" in instruction

    def test_instruction_mentions_conversation_history(self):
        """Insight 5 — full conversation history must pass through triage."""
        instruction = triage_agent.instruction.lower()
        assert "previous turns" in instruction or "conversation" in instruction


# ---------------------------------------------------------------------------
# Emergency Agent tests
# ---------------------------------------------------------------------------


class TestEmergencyAgent:
    """Tests for the emergency_agent sub-agent."""

    def test_name(self):
        assert emergency_agent.name == "emergency_agent"

    def test_model(self):
        assert emergency_agent.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert emergency_agent.output_key == "specialist_response"

    def test_has_emergency_flag_tool(self):
        tool_names = [t.__name__ if callable(t) else str(t) for t in emergency_agent.tools]
        assert any("emergency_flag" in name for name in tool_names)

    def test_has_one_tool(self):
        assert len(emergency_agent.tools) == 1

    def test_instruction_mentions_emergency_care(self):
        instruction = emergency_agent.instruction.lower()
        assert "emergency" in instruction


# ---------------------------------------------------------------------------
# General Health Agent tests
# ---------------------------------------------------------------------------


class TestGeneralHealthAgent:
    """Tests for the general_health_agent sub-agent."""

    def test_name(self):
        assert general_health_agent.name == "general_health_agent"

    def test_model(self):
        assert general_health_agent.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert general_health_agent.output_key == "specialist_response"

    def test_has_three_tools(self):
        assert len(general_health_agent.tools) == 3

    def test_instruction_mentions_drug_reference(self):
        instruction = general_health_agent.instruction
        assert "drug_reference" in instruction

    def test_instruction_mentions_symptom_checker(self):
        instruction = general_health_agent.instruction
        assert "symptom_checker" in instruction

    def test_instruction_mentions_hedging(self):
        """Insight 2 — agents must hedge on data interpretation."""
        instruction = general_health_agent.instruction.lower()
        assert "hedge" in instruction or "qualify" in instruction or "typical ranges" in instruction


# ---------------------------------------------------------------------------
# Reviewer Agent tests
# ---------------------------------------------------------------------------


class TestReviewerAgent:
    """Tests for the reviewer_agent sub-agent."""

    def test_name(self):
        assert reviewer_agent.name == "reviewer_agent"

    def test_model(self):
        assert reviewer_agent.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert reviewer_agent.output_key == "final_response"

    def test_has_no_tools(self):
        assert not reviewer_agent.tools

    def test_instruction_checks_unsupported_claims(self):
        """Insight 1 — accuracy penalties dominate."""
        instruction = reviewer_agent.instruction.lower()
        assert "unsupported" in instruction or "factual" in instruction

    def test_instruction_checks_safety_caveats(self):
        """Insight 3 — emergency criteria are widespread."""
        instruction = reviewer_agent.instruction.lower()
        assert "safety" in instruction or "emergency" in instruction

    def test_instruction_checks_overconfident_language(self):
        """Insight 4 — penalty mass drives hard scores."""
        instruction = reviewer_agent.instruction.lower()
        assert "overconfident" in instruction or "qualified" in instruction

    def test_instruction_checks_tone(self):
        """Insight 6 — communication quality has high penalty frequency."""
        instruction = reviewer_agent.instruction.lower()
        assert "tone" in instruction or "language" in instruction

    def test_instruction_checks_instruction_compliance(self):
        """Insight 6 — instruction following has high penalty frequency."""
        instruction = reviewer_agent.instruction.lower()
        assert "instruction" in instruction or "format" in instruction


# ---------------------------------------------------------------------------
# Coordinator Agent tests
# ---------------------------------------------------------------------------


class TestCoordinatorAgent:
    """Tests for the coordinator_agent that routes to specialists."""

    def test_name(self):
        assert coordinator_agent.name == "coordinator_agent"

    def test_model(self):
        assert coordinator_agent.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert coordinator_agent.output_key == "specialist_response"

    def test_has_two_sub_agents(self):
        assert len(coordinator_agent.sub_agents) == 2

    def test_sub_agents_are_emergency_and_general(self):
        sub_agent_names = {a.name for a in coordinator_agent.sub_agents}
        assert sub_agent_names == {"emergency_agent", "general_health_agent"}

    def test_instruction_mentions_delegation(self):
        instruction = coordinator_agent.instruction.lower()
        assert "delegate" in instruction


# ---------------------------------------------------------------------------
# Root Agent (SequentialAgent) tests
# ---------------------------------------------------------------------------


class TestRootAgent:
    """Tests for the root_agent — the SequentialAgent orchestrator."""

    def test_root_agent_name(self):
        assert root_agent.name == "multi_agent"

    def test_root_agent_is_sequential(self):
        assert isinstance(root_agent, SequentialAgent)

    def test_root_agent_has_three_sub_agents(self):
        assert len(root_agent.sub_agents) == 3

    def test_pipeline_order_is_triage_coordinator_reviewer(self):
        names = [a.name for a in root_agent.sub_agents]
        assert names == ["triage_agent", "coordinator_agent", "reviewer_agent"]

    def test_root_agent_has_description(self):
        assert root_agent.description is not None
        assert len(root_agent.description) > 0


# ---------------------------------------------------------------------------
# Pipeline integration tests (structural, no LLM calls)
# ---------------------------------------------------------------------------


class TestPipelineStructure:
    """Tests that validate the end-to-end pipeline structure."""

    def test_triage_output_key_distinct_from_specialist(self):
        """Triage and specialist must use different output_keys."""
        assert triage_agent.output_key != emergency_agent.output_key
        assert triage_agent.output_key != general_health_agent.output_key

    def test_reviewer_output_key_is_final(self):
        assert reviewer_agent.output_key == "final_response"

    def test_emergency_and_general_share_output_key(self):
        """Both specialists write to the same key so reviewer reads consistently."""
        assert emergency_agent.output_key == general_health_agent.output_key

    def test_all_agents_use_same_model(self):
        """All agents use gemini-2.0-flash for fair architecture comparison."""
        agents = [triage_agent, emergency_agent, general_health_agent, reviewer_agent, coordinator_agent]
        models = {a.model for a in agents}
        assert models == {"gemini-2.0-flash"}

    def test_emergency_agent_has_emergency_flag_tool(self):
        from agents.tool_agent.tools import emergency_flag
        assert emergency_flag in emergency_agent.tools

    def test_general_agent_has_all_tools(self):
        from agents.tool_agent.tools import drug_reference, symptom_checker, emergency_flag
        assert drug_reference in general_health_agent.tools
        assert symptom_checker in general_health_agent.tools
        assert emergency_flag in general_health_agent.tools
