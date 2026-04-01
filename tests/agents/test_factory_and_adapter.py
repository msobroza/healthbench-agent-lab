"""Tests for the pipeline factory, FrameworkAdapter, and ADKAgentPipeline.

Validates config-driven pipeline creation, framework dispatch, the ADK
adapter's agent construction, and the shared ADKAgentPipeline class.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

import pytest
from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent

from healthbench_agent.agent import (
    AgentPipeline,
    FrameworkAdapter,
    PlannerConfig,
    RootAgentPipelineConfig,
    create_pipeline,
    register_callback,
)
from healthbench_agent.agent.adapters.adk_adapter import (
    ADKAgentPipeline,
    ADKFrameworkAdapter,
    build_agent_node,
)
from healthbench_agent.agent.callback_registry import _REGISTRY as _CB_REGISTRY
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
        import tools  # noqa: F401 — trigger registration

        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/tool_agent.yaml",
        )
        adapter = ADKFrameworkAdapter()
        pipeline = adapter.create_pipeline(config)
        assert isinstance(pipeline, ADKAgentPipeline)
        assert len(pipeline.agent.tools) == 3

    def test_creates_multi_agent_pipeline(self):
        import tools  # noqa: F401 — trigger registration

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
        assert isinstance(pipeline.agent, (LlmAgent, SequentialAgent))

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
    """Tests for the build_agent_node function."""

    _PROMPT_PATH = "prompts/multi_agent/v1_structured.yaml"

    def test_leaf_agent_is_plain_agent(self):
        config = AgentNodeConfig(
            name="leaf",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LlmAgent)
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
        assert isinstance(agent, LlmAgent)
        assert not isinstance(agent, SequentialAgent)
        assert len(agent.sub_agents) == 2

    def test_loop_produces_loop_agent(self):
        config = AgentNodeConfig(
            name="loop",
            orchestration="loop",
            max_iterations=5,
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LoopAgent)
        assert agent.max_iterations == 5
        assert len(agent.sub_agents) == 2

    def test_loop_without_max_iterations(self):
        config = AgentNodeConfig(
            name="loop_no_max",
            orchestration="loop",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LoopAgent)
        assert agent.max_iterations is None

    def test_parallel_produces_parallel_agent(self):
        config = AgentNodeConfig(
            name="par",
            orchestration="parallel",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
                AgentNodeConfig(name="b", prompt_key="reviewer_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, ParallelAgent)
        assert len(agent.sub_agents) == 2

    def test_unsupported_orchestration_raises(self):
        config = AgentNodeConfig(
            name="bad",
            orchestration="unknown_type",
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
        import tools  # noqa: F401 — trigger registration

        config = AgentNodeConfig(
            name="tools_test",
            tools=["emergency_flag"],
            prompt_key="emergency_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert len(agent.tools) == 1


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestBuildAgentNodePlanners:
    """Tests for planner configuration in build_agent_node."""

    _PROMPT_PATH = "prompts/multi_agent/v1_structured.yaml"

    def test_builtin_planner_attached(self):
        from google.adk.planners import BuiltInPlanner

        config = AgentNodeConfig(
            name="planned",
            prompt_key="triage_instruction",
            planner=PlannerConfig(
                type="builtin",
                thinking_budget=1024,
                include_thoughts=True,
            ),
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LlmAgent)
        assert isinstance(agent.planner, BuiltInPlanner)

    def test_plan_react_planner_attached(self):
        from google.adk.planners import PlanReActPlanner

        config = AgentNodeConfig(
            name="react_planned",
            prompt_key="triage_instruction",
            planner=PlannerConfig(type="plan_react"),
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LlmAgent)
        assert isinstance(agent.planner, PlanReActPlanner)

    def test_no_planner_by_default(self):
        config = AgentNodeConfig(
            name="no_planner",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.planner is None


# ---------------------------------------------------------------------------
# Multi-agent control field tests
# ---------------------------------------------------------------------------


class TestBuildAgentNodeControlFields:
    """Tests for multi-agent control fields in build_agent_node."""

    _PROMPT_PATH = "prompts/multi_agent/v1_structured.yaml"

    def test_include_contents_none(self):
        config = AgentNodeConfig(
            name="stateless",
            prompt_key="triage_instruction",
            include_contents="none",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.include_contents == "none"

    def test_include_contents_default(self):
        config = AgentNodeConfig(
            name="stateful",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.include_contents == "default"

    def test_disallow_transfer_to_parent(self):
        config = AgentNodeConfig(
            name="no_parent",
            prompt_key="triage_instruction",
            disallow_transfer_to_parent=True,
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.disallow_transfer_to_parent is True

    def test_disallow_transfer_to_peers(self):
        config = AgentNodeConfig(
            name="no_peers",
            prompt_key="triage_instruction",
            disallow_transfer_to_peers=True,
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.disallow_transfer_to_peers is True

    def test_global_instruction_passed_through(self):
        config = AgentNodeConfig(
            name="global_instr",
            prompt_key="triage_instruction",
            global_instruction="Always prioritize safety.",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.global_instruction == "Always prioritize safety."


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


class TestBuildAgentNodeCallbacks:
    """Tests for callback resolution in build_agent_node."""

    _PROMPT_PATH = "prompts/multi_agent/v1_structured.yaml"

    @pytest.fixture(autouse=True)
    def _clean_callback_registry(self):
        """Clear callback registry before and after each test."""
        saved = dict(_CB_REGISTRY)
        _CB_REGISTRY.clear()
        yield
        _CB_REGISTRY.clear()
        _CB_REGISTRY.update(saved)

    def test_before_agent_callback_resolved_on_leaf(self):
        @register_callback("test_before_agent")
        def _before_agent(ctx):
            return None

        config = AgentNodeConfig(
            name="cb_leaf",
            prompt_key="triage_instruction",
            before_agent_callback="test_before_agent",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.before_agent_callback is _before_agent

    def test_before_model_callback_resolved_on_leaf(self):
        @register_callback("test_before_model")
        def _before_model(ctx, req):
            return None

        config = AgentNodeConfig(
            name="cb_model",
            prompt_key="triage_instruction",
            before_model_callback="test_before_model",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.before_model_callback is _before_model

    def test_no_callbacks_by_default(self):
        config = AgentNodeConfig(
            name="no_cb",
            prompt_key="triage_instruction",
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert agent.before_agent_callback is None
        assert agent.after_agent_callback is None
        assert agent.before_model_callback is None
        assert agent.after_model_callback is None

    def test_before_agent_callback_on_sequential(self):
        @register_callback("test_seq_cb")
        def _seq_cb(ctx):
            return None

        config = AgentNodeConfig(
            name="seq_cb",
            orchestration="sequential",
            before_agent_callback="test_seq_cb",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, SequentialAgent)
        assert agent.before_agent_callback is _seq_cb

    def test_before_agent_callback_on_loop(self):
        @register_callback("test_loop_cb")
        def _loop_cb(ctx):
            return None

        config = AgentNodeConfig(
            name="loop_cb",
            orchestration="loop",
            max_iterations=3,
            before_agent_callback="test_loop_cb",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, LoopAgent)
        assert agent.before_agent_callback is _loop_cb

    def test_before_agent_callback_on_parallel(self):
        @register_callback("test_par_cb")
        def _par_cb(ctx):
            return None

        config = AgentNodeConfig(
            name="par_cb",
            orchestration="parallel",
            before_agent_callback="test_par_cb",
            sub_agents=[
                AgentNodeConfig(name="a", prompt_key="triage_instruction"),
            ],
        )
        agent = build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)
        assert isinstance(agent, ParallelAgent)
        assert agent.before_agent_callback is _par_cb

    def test_unregistered_callback_raises(self):
        config = AgentNodeConfig(
            name="bad_cb",
            prompt_key="triage_instruction",
            before_agent_callback="nonexistent_callback",
        )
        with pytest.raises(KeyError, match="not registered"):
            build_agent_node(config, parent_prompt_path=self._PROMPT_PATH)


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure existing configs still work after adding new fields."""

    def test_baseline_config_loads(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
        )
        agent = build_agent_node(config)
        assert isinstance(agent, LlmAgent)
        assert agent.planner is None
        assert agent.include_contents == "default"
        assert agent.disallow_transfer_to_parent is False

    def test_new_fields_have_safe_defaults(self):
        config = AgentNodeConfig(name="defaults_test")
        assert config.max_iterations is None
        assert config.planner is None
        assert config.include_contents == "default"
        assert config.disallow_transfer_to_parent is False
        assert config.disallow_transfer_to_peers is False
        assert config.global_instruction == ""
        assert config.before_agent_callback is None
        assert config.before_model_callback is None
        assert config.before_tool_callback is None
        assert config.after_agent_callback is None
        assert config.after_model_callback is None
        assert config.after_tool_callback is None
