"""Multi-agent pipeline — Architecture C.

Orchestrates a config-driven multi-agent pipeline. The agent tree is
built entirely from ``RootAgentPipelineConfig`` sub-agent definitions
via the ADK adapter's ``build_agent_node`` builder: leaf agents get
tools from the registry, routing agents delegate via LLM decisions,
and sequential agents run children in order.

Orchestration modes:
    - ``"sequential"``: children run in order (``SequentialAgent``).
    - ``"routing"``: parent LLM decides which child to delegate to.

Pipeline (default configuration):
    1. Triage Agent → classifies urgency/topic/expertise
    2. Coordinator (routing) → delegates to emergency or general_health
    3. Reviewer Agent → checks safety/quality

Run:
    uv run adk web agents/multi_agent
    uv run adk run agents/multi_agent
"""

from __future__ import annotations

import agents.tool_agent  # noqa: F401 — triggers @register_tool registration
from healthbench_agent.agent import RootAgentPipelineConfig
from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline
from healthbench_agent.agent.factory import create_pipeline

_DEFAULT_CONFIG_PATH = "config/agents/multi_agent.yaml"


def from_config(
    config_path: str = _DEFAULT_CONFIG_PATH, **overrides,
) -> ADKAgentPipeline:
    """Create a multi-agent pipeline from a YAML config file.

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
