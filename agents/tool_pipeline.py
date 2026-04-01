"""Tool-augmented agent — Architecture B.

Single agent with clinically-aware prompt and medical reference tools
resolved from the tool registry. The agent is instructed to use tools
proactively to ground its responses in verified data.

Run:
    uv run adk web agents/tool_agent
    uv run adk run agents/tool_agent
"""

from __future__ import annotations

from healthbench_agent.agent import RootAgentPipelineConfig
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline
from healthbench_agent.agent.factory import create_pipeline

_DEFAULT_CONFIG_PATH = "config/agents/tool_agent.yaml"


def from_config(
    config_path: str = _DEFAULT_CONFIG_PATH, **overrides,
) -> ADKAgentPipeline:
    """Create a tool-augmented pipeline from a YAML config file.

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
