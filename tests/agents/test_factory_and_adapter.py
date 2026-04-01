"""Tests for the pipeline factory, FrameworkAdapter, and ADKAgentPipeline.

Validates config-driven pipeline creation, framework dispatch, the ADK
adapter's agent construction, and the shared ADKAgentPipeline class.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

import pytest
from google.adk.agents import Agent, SequentialAgent

from healthbench_agent.agent import (
    AgentPipeline,
    FrameworkAdapter,
    RootAgentPipelineConfig,
    create_pipeline,
)
from healthbench_agent.agent.adapters.adk_adapter import (
    ADKAgentPipeline,
    ADKFrameworkAdapter,
    build_agent_node,
)
from healthbench_agent.agent.config import AgentNodeConfig

# ---------------------------------------------------------------------------
# FrameworkAdapter ABC tests
# ---------------------------------------------------------------------------


class TestFrameworkAdapterABC:
    """Tests for the FrameworkAdapter abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            FrameworkAdapter()

    def test_has_create_pipeline_method(self):
        assert hasattr(FrameworkAdapter, "create_pipeline")

    def test_adk_adapter_is_framework_adapter(self):
        adapter = ADKFrameworkAdapter()
        assert isinstance(adapter, FrameworkAdapter)


# ---------------------------------------------------------------------------
# ADKFrameworkAdapter tests
# ---------------------------------------------------------------------------


class TestADKFrameworkAdapter:
    """Tests for the ADK framework adapter."""

    def test_creates_baseline_pipeline(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        adapter = ADKFrameworkAdapter()
        pipeline = adapter.create_pipeline(config)
        assert isinstance(pipeline, ADKAgentPipeline)
        assert isinstance(pipeline, AgentPipeline)
        assert pipeline.agent.name == "baseline_agent"

    def test_creates_tool_pipeline(self):
        import agents.tool_agent  # noqa: F401 — trigger registration
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/tool_agent.yaml",
        )
        adapter = ADKFrameworkAdapter()
        pipeline = adapter.create_pipeline(config)
        assert isinstance(pipeline, ADKAgentPipeline)
        assert len(pipeline.agent.tools) == 3

    def test_creates_multi_agent_pipeline(self):
        import agents.tool_agent  # noqa: F401 — trigger registration
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/multi_agent.yaml",
        )
        adapter = ADKFrameworkAdapter()
        pipeline = adapter.create_pipeline(config)
        assert isinstance(pipeline, ADKAgentPipeline)
        assert isinstance(pipeline.agent, SequentialAgent)


# ---------------------------------------------------------------------------
# ADKAgentPipeline tests
# ---------------------------------------------------------------------------


class TestADKAgentPipeline:
    """Tests for the shared ADKAgentPipeline class."""

    def test_is_agent_pipeline_subclass(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        pipeline = ADKFrameworkAdapter().create_pipeline(config)
        assert isinstance(pipeline, AgentPipeline)

    def test_has_config_attribute(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        pipeline = ADKFrameworkAdapter().create_pipeline(config)
        assert pipeline.config is config

    def test_has_agent_attribute(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        pipeline = ADKFrameworkAdapter().create_pipeline(config)
        assert isinstance(pipeline.agent, (Agent, SequentialAgent))

    def test_has_generate_method(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        pipeline = ADKFrameworkAdapter().create_pipeline(config)
        assert hasattr(pipeline, "generate")
        assert callable(pipeline.generate)


# ---------------------------------------------------------------------------
# create_pipeline factory tests
# ---------------------------------------------------------------------------


class TestCreatePipeline:
    """Tests for the create_pipeline factory function."""

    def test_default_framework_is_adk(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        assert config.framework == "adk"

    def test_creates_adk_pipeline(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        pipeline = create_pipeline(config)
        assert isinstance(pipeline, ADKAgentPipeline)
        assert pipeline.agent.name == "baseline_agent"

    def test_unsupported_framework_raises(self):
        config = RootAgentPipelineConfig(
            name="test",
            framework="unknown_framework",
            prompt_path="prompts/baseline_agent/v1_baseline.yaml",
        )
        with pytest.raises(ValueError, match="Unsupported framework"):
            create_pipeline(config)

    def test_framework_field_in_config(self):
        config = AgentNodeConfig(name="test", framework="langgraph")
        assert config.framework == "langgraph"

    def test_framework_default_in_agent_node_config(self):
        config = AgentNodeConfig(name="test")
        assert config.framework == "adk"


# ---------------------------------------------------------------------------
# build_agent_node tests
# ---------------------------------------------------------------------------


class TestBuildAgentNode:
    """Tests for the build_agent_node function extracted from multi-agent."""

    _PROMPT_PATH = "prompts/multi_agent/v1_structured.yaml"

    def test_leaf_agent_is_plain_agent(self):
        config = AgentNodeConfig(
            name="leaf",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, Agent)
        assert not isinstance(agent, SequentialAgent)

    def test_sequential_produces_sequential_agent(self):
        config = AgentNodeConfig(
            name="seq",
            orchestration="sequential",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    def test_routing_produces_agent_with_sub_agents(self):
        config = AgentNodeConfig(
            name="router",
            orchestration="routing",
            prompt_key="coordinator_instruction",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, Agent)
        assert not isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    def test_unsupported_orchestration_raises(self):
        config = AgentNodeConfig(
            name="bad",
            orchestration="parallel",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
            ],
            prompt_path=self._PROMPT_PATH,
        )
        with pytest.raises(ValueError, match="Unsupported orchestration"):
            build_agent_node(config)

    def test_condition_appended_to_description(self):
        config = AgentNodeConfig(
            name="cond",
            description="Emergency handler.",
            condition="Life-threatening symptoms",
            prompt_key="emergency_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert "Condition: Life-threatening symptoms" in agent.description
        assert "Emergency handler." in agent.description

    def test_no_condition_preserves_description(self):
        config = AgentNodeConfig(
            name="nocond",
            description="Triage handler.",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.description == "Triage handler."

    def test_prompt_path_inheritance(self):
        config = AgentNodeConfig(
            name="inherit",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.instruction is not None
        assert len(agent.instruction) > 0

    def test_tools_resolved_from_registry(self):
        import agents.tool_agent  # noqa: F401 — trigger registration
        config = AgentNodeConfig(
            name="tools_test",
            tools=["emergency_flag"],
            prompt_key="emergency_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert len(agent.tools) == 1
