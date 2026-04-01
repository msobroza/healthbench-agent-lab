"""Tool-augmented agent — Architecture B.

Single agent with clinically-aware prompt and medical reference tools
resolved from the tool registry. The agent is instructed to use tools
proactively to ground its responses in verified data.

Run:
    uv run adk web agents/tool_agent
    uv run adk run agents/tool_agent
"""

from healthbench_agent.agent.adapters.adk_adapter import ADKAgentPipeline

_pipeline = ADKAgentPipeline.from_config("config/agents/tool_agent.yaml")
root_agent = _pipeline.agent
