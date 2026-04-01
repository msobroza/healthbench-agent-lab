"""Baseline single agent — Architecture A.

Simplest possible agent: one LLM call with a minimal health instruction,
no tools, no sub-agents. Establishes the performance floor against which
Architecture B (tool-augmented) and C (multi-agent) are compared.

Run:
    uv run adk web agents/baseline_agent
    uv run adk run agents/baseline_agent
"""

from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline

_pipeline = ADKAgentPipeline.from_config("config/agents/baseline_agent.yaml")
root_agent = _pipeline.agent
