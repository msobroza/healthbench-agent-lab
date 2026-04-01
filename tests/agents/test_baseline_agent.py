"""Tests for agents/baseline_pipeline.py — Architecture A.

Validates agent configuration, prompt loading, and the root_agent contract.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agents.baseline_pipeline import _pipeline, root_agent
from healthbench_agent.agent import AgentPipeline, RootAgentPipelineConfig, load_instruction
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline

_PROMPT_PATH = Path(_pipeline.config.prompt_path)


# ---------------------------------------------------------------------------
# Prompt YAML tests
# ---------------------------------------------------------------------------


class TestPromptYaml:
    """Tests for prompts/v1_baseline.yaml structure and content."""

    def test_prompt_file_exists(self):
        assert _PROMPT_PATH.exists(), f"Prompt file not found: {_PROMPT_PATH}"

    def test_prompt_yaml_has_required_keys(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        assert "version" in data
        assert "instruction" in data
        assert "rationale" in data
        assert "architecture" in data

    def test_prompt_version_is_semver(self):
        with open(_PROMPT_PATH) as f:
            data = yaml.safe_load(f)
        parts = data["version"].split(".")
        assert len(parts) == 3, f"Version '{data['version']}' is not semver"
        assert all(p.isdigit() for p in parts)

    def test_prompt_instruction_is_nonempty_string(self):
        instruction = load_instruction(_PROMPT_PATH)
        assert isinstance(instruction, str)
        assert len(instruction) > 100, "Instruction seems too short for a health prompt"

    def test_prompt_instruction_contains_conversation_placeholder(self):
        with open(_PROMPT_PATH) as f:
            raw = yaml.safe_load(f)["instruction"]
        assert "{{ conversation }}" in raw


# ---------------------------------------------------------------------------
# Agent configuration tests
# ---------------------------------------------------------------------------


class TestBaselineAgentConfig:
    """Tests for the root_agent object exported by the baseline agent module."""

    def test_root_agent_name(self):
        assert root_agent.name == "baseline_agent"

    def test_root_agent_model(self):
        assert root_agent.model == "gemini-2.0-flash"

    def test_root_agent_has_description(self):
        assert root_agent.description is not None
        assert len(root_agent.description) > 0

    def test_root_agent_has_no_tools(self):
        assert not root_agent.tools, "Baseline agent should have no tools"

    def test_root_agent_has_no_sub_agents(self):
        assert not root_agent.sub_agents, "Baseline agent should have no sub_agents"

    def test_root_agent_instruction_contains_safety_guidance(self):
        instruction = root_agent.instruction
        assert "emergency" in instruction.lower()
        assert "safety" in instruction.lower()

    def test_root_agent_instruction_contains_accuracy_guidance(self):
        instruction = root_agent.instruction
        assert "accuracy" in instruction.lower() or "accurate" in instruction.lower()

    def test_root_agent_instruction_contains_limitation_disclaimer(self):
        instruction = root_agent.instruction
        assert "healthcare provider" in instruction.lower() or "doctor" in instruction.lower()


# ---------------------------------------------------------------------------
# Pipeline class tests
# ---------------------------------------------------------------------------


class TestBaselinePipeline:
    """Tests for the baseline ADKAgentPipeline concrete class."""

    def test_is_agent_pipeline_subclass(self):
        assert isinstance(_pipeline, AgentPipeline)

    def test_config_is_agent_pipeline_config(self):
        assert isinstance(_pipeline.config, RootAgentPipelineConfig)

    def test_config_matches_yaml(self):
        assert _pipeline.config.name == "baseline_agent"
        assert _pipeline.config.model == "gemini-2.0-flash"
        assert _pipeline.config.prompt_version == "1.0.0"

    def test_from_config_creates_pipeline(self):
        pipeline = ADKAgentPipeline.from_config("config/agents/baseline_agent.yaml")
        assert isinstance(pipeline, ADKAgentPipeline)
        assert pipeline.agent.name == "baseline_agent"

    def test_description_from_config(self):
        assert _pipeline.config.description == root_agent.description
