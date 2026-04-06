"""ADK entry point for the tool-augmented agent.

Re-exports ``root_agent`` from the tool pipeline so that
``adk web agents/`` discovers the agent automatically.
"""

from tool_pipeline import root_agent  # type: ignore[import-not-found]

__all__ = ["root_agent"]
