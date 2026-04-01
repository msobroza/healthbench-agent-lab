"""Tests for agents/baseline_agent/agent.py — Architecture A.

Validates agent configuration, prompt loading, and the root_agent contract.
Does NOT test LLM inference (no network calls).
"""

from __future__ import annotations

import yaml
from pathlib import Path

from agents.baseline_agent.agent import root_agent, _load_instruction, _PROMPT_PATH


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
        instruction = _load_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 100, "Instruction seems too short for a health prompt"


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

    def test_root_agent_instruction_matches_yaml(self):
        expected = _load_instruction()
        assert root_agent.instruction == expected

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
