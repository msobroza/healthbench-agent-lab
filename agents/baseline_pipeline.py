"""Baseline single agent — Architecture A.

Simplest possible agent: one LLM call with a minimal health instruction,
no tools, no sub-agents. Establishes the performance floor against which
Architecture B (tool-augmented) and C (multi-agent) are compared.

Run:
    uv run adk web agents/baseline_agent
    uv run adk run agents/baseline_agent
"""

from __future__ import annotations

from healthbench_agent.agent import RootAgentPipelineConfig
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline
from healthbench_agent.agent.factory import create_pipeline

_DEFAULT_CONFIG_PATH = "config/agents/baseline_agent.yaml"


def from_config(
    config_path: str = _DEFAULT_CONFIG_PATH, **overrides,
) -> ADKAgentPipeline:
    """Create a baseline pipeline from a YAML config file.

    Args:
        config_path: Path to the agent YAML config.
        **overrides: Explicit field overrides for the config.

    Returns:
        A fully configured ADKAgentPipeline.
    """
    config = RootAgentPipelineConfig.from_yaml(config_path, **overrides)
    pipeline = create_pipeline(config)
    assert isinstance(pipeline, ADKAgentPipeline)
    return pipeline


_pipeline = from_config()
root_agent = _pipeline.agent
