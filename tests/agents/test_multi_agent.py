"""Tests for agents/multi_agent/ — Architecture C.

Validates config-driven agent tree construction, routing orchestration,
sub-agent configuration, prompt loading, and the root_agent contract.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from google.adk.agents import LlmAgent, SequentialAgent

from agents.multi_pipeline import _pipeline, root_agent
from healthbench_agent.agent import AgentPipeline, RootAgentPipelineConfig, load_instruction
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline, build_agent_node
from healthbench_agent.agent.config import AgentNodeConfig

_PROMPT_PATH = Path(_pipeline.config.prompt_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_agent(name: str) -> LlmAgent:
    """Find an agent by name anywhere in the pipeline tree."""

    def _search(agent: LlmAgent | SequentialAgent) -> LlmAgent | None:
        if agent.name == name:
            return agent
        for sub in getattr(agent, "sub_agents", []):
            found = _search(sub)
            if found:
                return found
        return None

    found = _search(_pipeline.agent)
    assert found is not None, f"Agent '{name}' not found in pipeline tree"
    return found


# Convenience references resolved once from the config-driven tree.
_triage = _find_agent("triage_agent")
_emergency = _find_agent("emergency_agent")
_general_health = _find_agent("general_health_agent")
_coordinator = _find_agent("coordinator_agent")
_reviewer = _find_agent("reviewer_agent")


# ---------------------------------------------------------------------------
# AgentNodeConfig tests
# ---------------------------------------------------------------------------


class TestAgentNodeConfig:
    """Tests for AgentNodeConfig parsing, recursion, and defaults."""

    def test_minimal_config(self):
        config = AgentNodeConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.model == "gemini-2.0-flash"
        assert config.tools == []
        assert config.sub_agents == []
        assert config.output_key is None
        assert config.condition is None
        assert config.orchestration == "sequential"

    def test_nested_sub_agents(self):
        config = AgentNodeConfig(
            name="router",
            orchestration="routing",
            sub_agents=[
                AgentNodeConfig(name="child_a"),
                AgentNodeConfig(name="child_b", tools=["emergency_flag"]),
            ],
        )
        assert len(config.sub_agents) == 2
        assert config.sub_agents[0].name == "child_a"
        assert config.sub_agents[1].tools == ["emergency_flag"]

    def test_condition_field(self):
        config = AgentNodeConfig(
            name="emergency",
            condition="Query involves life-threatening symptoms.",
        )
        assert config.condition == "Query involves life-threatening symptoms."

    def test_root_config_is_agent_node_config(self):
        """RootAgentPipelineConfig IS-A AgentNodeConfig."""
        assert isinstance(_pipeline.config, AgentNodeConfig)

    def test_config_loaded_from_yaml(self):
        """Sub-agents in multi_agent.yaml are parsed as AgentNodeConfig."""
        config = _pipeline.config
        assert len(config.sub_agents) > 0
        assert all(isinstance(sa, AgentNodeConfig) for sa in config.sub_agents)

    def test_coordinator_has_nested_sub_agents_in_config(self):
        coordinator_configs = [sa for sa in _pipeline.config.sub_agents if sa.sub_agents]
        assert len(coordinator_configs) == 1
        assert coordinator_configs[0].name == "coordinator_agent"
        child_names = [c.name for c in coordinator_configs[0].sub_agents]
        assert "emergency_agent" in child_names
        assert "general_health_agent" in child_names

    def test_deeply_nested_config(self):
        """Three-level nesting works via recursive AgentNodeConfig."""
        config = AgentNodeConfig(
            name="root",
            sub_agents=[
                AgentNodeConfig(
                    name="mid",
                    sub_agents=[
                        AgentNodeConfig(name="leaf"),
                    ],
                ),
            ],
        )
        assert config.sub_agents[0].sub_agents[0].name == "leaf"


# ---------------------------------------------------------------------------
# Config-level routing + condition tests
# ---------------------------------------------------------------------------


class TestRoutingConfig:
    """Tests for routing orchestration and conditions in config."""

    def test_root_orchestration_is_sequential(self):
        assert _pipeline.config.orchestration == "sequential"

    def test_coordinator_orchestration_is_routing(self):
        coordinator_config = next(
            sa for sa in _pipeline.config.sub_agents if sa.name == "coordinator_agent"
        )
        assert coordinator_config.orchestration == "routing"

    def test_emergency_agent_has_condition(self):
        coordinator_config = next(
            sa for sa in _pipeline.config.sub_agents if sa.name == "coordinator_agent"
        )
        emergency_config = next(
            sa for sa in coordinator_config.sub_agents if sa.name == "emergency_agent"
        )
        assert emergency_config.condition is not None
        assert "life-threatening" in emergency_config.condition

    def test_general_health_agent_has_condition(self):
        coordinator_config = next(
            sa for sa in _pipeline.config.sub_agents if sa.name == "coordinator_agent"
        )
        general_config = next(
            sa for sa in coordinator_config.sub_agents if sa.name == "general_health_agent"
        )
        assert general_config.condition is not None
        assert "general health" in general_config.condition

    def test_triage_has_no_condition(self):
        triage_config = next(sa for sa in _pipeline.config.sub_agents if sa.name == "triage_agent")
        assert triage_config.condition is None

    def test_sub_agents_count(self):
        assert len(_pipeline.config.sub_agents) == 3

    def test_prompt_key_loaded(self):
        assert _pipeline.config.prompt_key == "coordinator_instruction"

    def test_sub_agent_names(self):
        names = [sa.name for sa in _pipeline.config.sub_agents]
        assert names == ["triage_agent", "coordinator_agent", "reviewer_agent"]

    def test_model_uses_default_for_sub_agents(self):
        """Sub-agents without explicit model get the default."""
        for sa in _pipeline.config.sub_agents:
            assert sa.model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Prompt YAML tests
# ---------------------------------------------------------------------------


class TestMultiAgentPromptYaml:
    """Tests for prompts/v1_structured.yaml structure and content."""

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

    def test_load_instruction_returns_string(self):
        instruction = load_instruction(_PROMPT_PATH, key="triage_instruction")
        assert isinstance(instruction, str)
        assert len(instruction) > 50

    def test_prompt_instructions_contain_conversation_placeholder(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        for key in [
            "triage_instruction",
            "emergency_instruction",
            "general_health_instruction",
            "reviewer_instruction",
        ]:
            assert "{{ conversation }}" in data[key], f"Missing conversation placeholder in {key}"


# ---------------------------------------------------------------------------
# Triage Agent tests
# ---------------------------------------------------------------------------


class TestTriageAgent:
    """Tests for the triage_agent built from config."""

    def test_name(self):
        assert _triage.name == "triage_agent"

    def test_model(self):
        assert _triage.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert _triage.output_key == "triage_classification"

    def test_has_no_tools(self):
        assert not _triage.tools

    def test_has_no_sub_agents(self):
        assert not _triage.sub_agents

    def test_instruction_contains_urgency_classification(self):
        instruction = _triage.instruction
        assert "URGENCY" in instruction
        assert "emergency" in instruction.lower()
        assert "urgent" in instruction.lower()
        assert "routine" in instruction.lower()

    def test_instruction_references_expertise_classification(self):
        instruction = _triage.instruction
        assert "EXPERTISE" in instruction

    def test_instruction_mentions_conversation_history(self):
        """Insight 5 — full conversation history must pass through triage."""
        instruction = _triage.instruction.lower()
        assert "previous turns" in instruction or "conversation" in instruction


# ---------------------------------------------------------------------------
# Emergency Agent tests
# ---------------------------------------------------------------------------


class TestEmergencyAgent:
    """Tests for the emergency_agent built from config."""

    def test_name(self):
        assert _emergency.name == "emergency_agent"

    def test_model(self):
        assert _emergency.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert _emergency.output_key == "specialist_response"

    def test_has_emergency_flag_tool(self):
        tool_names = [t.__name__ if callable(t) else str(t) for t in _emergency.tools]
        assert any("emergency_flag" in name for name in tool_names)

    def test_has_one_tool(self):
        assert len(_emergency.tools) == 1

    def test_instruction_mentions_emergency_care(self):
        instruction = _emergency.instruction.lower()
        assert "emergency" in instruction

    def test_description_includes_condition(self):
        """Condition is appended to description for routing visibility."""
        assert "Condition:" in _emergency.description


# ---------------------------------------------------------------------------
# General Health Agent tests
# ---------------------------------------------------------------------------


class TestGeneralHealthAgent:
    """Tests for the general_health_agent built from config."""

    def test_name(self):
        assert _general_health.name == "general_health_agent"

    def test_model(self):
        assert _general_health.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert _general_health.output_key == "specialist_response"

    def test_has_three_tools(self):
        assert len(_general_health.tools) == 3

    def test_instruction_mentions_drug_reference(self):
        instruction = _general_health.instruction
        assert "drug_reference" in instruction

    def test_instruction_mentions_symptom_checker(self):
        instruction = _general_health.instruction
        assert "symptom_checker" in instruction

    def test_instruction_mentions_hedging(self):
        """Insight 2 — agents must hedge on data interpretation."""
        instruction = _general_health.instruction.lower()
        assert "hedge" in instruction or "qualify" in instruction or "typical ranges" in instruction

    def test_description_includes_condition(self):
        """Condition is appended to description for routing visibility."""
        assert "Condition:" in _general_health.description


# ---------------------------------------------------------------------------
# Reviewer Agent tests
# ---------------------------------------------------------------------------


class TestReviewerAgent:
    """Tests for the reviewer_agent built from config."""

    def test_name(self):
        assert _reviewer.name == "reviewer_agent"

    def test_model(self):
        assert _reviewer.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert _reviewer.output_key == "final_response"

    def test_has_no_tools(self):
        assert not _reviewer.tools

    def test_instruction_checks_unsupported_claims(self):
        """Insight 1 — accuracy penalties dominate."""
        instruction = _reviewer.instruction.lower()
        assert "unsupported" in instruction or "factual" in instruction

    def test_instruction_checks_safety_caveats(self):
        """Insight 3 — emergency criteria are widespread."""
        instruction = _reviewer.instruction.lower()
        assert "safety" in instruction or "emergency" in instruction

    def test_instruction_checks_overconfident_language(self):
        """Insight 4 — penalty mass drives hard scores."""
        instruction = _reviewer.instruction.lower()
        assert "overconfident" in instruction or "qualified" in instruction

    def test_instruction_checks_tone(self):
        """Insight 6 — communication quality has high penalty frequency."""
        instruction = _reviewer.instruction.lower()
        assert "tone" in instruction or "language" in instruction

    def test_instruction_checks_instruction_compliance(self):
        """Insight 6 — instruction following has high penalty frequency."""
        instruction = _reviewer.instruction.lower()
        assert "instruction" in instruction or "format" in instruction


# ---------------------------------------------------------------------------
# Coordinator Agent tests (routing)
# ---------------------------------------------------------------------------


class TestCoordinatorAgent:
    """Tests for the coordinator_agent that routes to specialists."""

    def test_name(self):
        assert _coordinator.name == "coordinator_agent"

    def test_model(self):
        assert _coordinator.model == "gemini-2.0-flash"

    def test_output_key_set(self):
        assert _coordinator.output_key == "specialist_response"

    def test_is_regular_agent_not_sequential(self):
        """Routing agents are ADK Agent, not SequentialAgent."""
        assert isinstance(_coordinator, LlmAgent)
        assert not isinstance(_coordinator, SequentialAgent)

    def test_has_two_sub_agents(self):
        assert len(_coordinator.sub_agents) == 2

    def test_sub_agents_are_emergency_and_general(self):
        sub_agent_names = {a.name for a in _coordinator.sub_agents}
        assert sub_agent_names == {"emergency_agent", "general_health_agent"}

    def test_instruction_mentions_delegation(self):
        instruction = _coordinator.instruction.lower()
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
# Pipeline class tests
# ---------------------------------------------------------------------------


class TestMultiPipeline:
    """Tests for the multi-agent ADKAgentPipeline concrete class."""

    def test_is_agent_pipeline_subclass(self):
        assert isinstance(_pipeline, AgentPipeline)

    def test_config_is_root_agent_pipeline_config(self):
        assert isinstance(_pipeline.config, RootAgentPipelineConfig)

    def test_from_config_class_method_exists(self):
        assert callable(ADKAgentPipeline.from_config)

    def test_module_pipeline_is_multi_agent(self):
        assert isinstance(_pipeline, ADKAgentPipeline)
        assert _pipeline.agent.name == "multi_agent"

    def test_description_from_config(self):
        assert _pipeline.config.description == root_agent.description

    def test_unsupported_orchestration_raises(self):
        config = AgentNodeConfig(
            name="test",
            orchestration="unknown_type",
            sub_agents=[AgentNodeConfig(name="a", prompt_key="triage_instruction")],
            prompt_path=str(_PROMPT_PATH),
        )
        with pytest.raises(ValueError, match="Unsupported orchestration"):
            build_agent_node(config)


# ---------------------------------------------------------------------------
# Builder routing tests
# ---------------------------------------------------------------------------


class TestBuilderRouting:
    """Tests for build_agent_node orchestration handling."""

    def test_sequential_node_produces_sequential_agent(self):
        config = AgentNodeConfig(
            name="seq_test",
            orchestration="sequential",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(
            config,
            parent_prompt_path=str(_PROMPT_PATH),
        )
        assert isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    def test_routing_node_produces_agent_with_sub_agents(self):
        config = AgentNodeConfig(
            name="route_test",
            orchestration="routing",
            prompt_key="coordinator_instruction",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(
            config,
            parent_prompt_path=str(_PROMPT_PATH),
        )
        assert isinstance(agent, LlmAgent)
        assert not isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    def test_condition_appended_to_leaf_description(self):
        config = AgentNodeConfig(
            name="cond_test",
            description="Handles emergencies.",
            condition="Query is urgent",
            prompt_key="emergency_instruction",
        )
        agent = build_agent_node(
            config,
            parent_prompt_path=str(_PROMPT_PATH),
        )
        assert "Condition: Query is urgent" in agent.description
        assert "Handles emergencies." in agent.description

    def test_no_condition_leaves_description_unchanged(self):
        config = AgentNodeConfig(
            name="nocond_test",
            description="Handles triage.",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(
            config,
            parent_prompt_path=str(_PROMPT_PATH),
        )
        assert agent.description == "Handles triage."

    def test_prompt_path_inherited_from_parent(self):
        """Sub-agents with empty prompt_path inherit from parent."""
        config = AgentNodeConfig(
            name="inherit_test",
            prompt_key="triage_instruction",
        )
        # prompt_path is empty → inherits from parent_prompt_path
        agent = build_agent_node(
            config,
            parent_prompt_path=str(_PROMPT_PATH),
        )
        assert agent.instruction is not None
        assert len(agent.instruction) > 0


# ---------------------------------------------------------------------------
# Pipeline integration tests (structural, no LLM calls)
# ---------------------------------------------------------------------------


class TestPipelineStructure:
    """Tests that validate the end-to-end pipeline structure."""

    def test_triage_output_key_distinct_from_specialist(self):
        """Triage and specialist must use different output_keys."""
        assert _triage.output_key != _emergency.output_key
        assert _triage.output_key != _general_health.output_key

    def test_reviewer_output_key_is_final(self):
        assert _reviewer.output_key == "final_response"

    def test_emergency_and_general_share_output_key(self):
        """Both specialists write to the same key so reviewer reads consistently."""
        assert _emergency.output_key == _general_health.output_key

    def test_all_sub_agents_use_same_model(self):
        """All sub-agents use gemini-2.0-flash for fair architecture comparison."""
        agents = [_triage, _emergency, _general_health, _reviewer]
        models = {a.model for a in agents}
        assert models == {"gemini-2.0-flash"}

    def test_emergency_agent_has_emergency_flag_tool(self):
        from tools.emergency_flag import emergency_flag

        assert emergency_flag in _emergency.tools

    def test_general_agent_has_all_tools(self):
        from tools.drug_reference import drug_reference
        from tools.emergency_flag import emergency_flag
        from tools.symptom_checker import symptom_checker

        assert drug_reference in _general_health.tools
        assert symptom_checker in _general_health.tools
        assert emergency_flag in _general_health.tools
