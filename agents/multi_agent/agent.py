"""ADK entry point for the multi-agent pipeline.

Re-exports ``root_agent`` from the multi pipeline so that
``adk web agents/`` discovers the agent automatically.
"""

from multi_pipeline import root_agent

__all__ = ["root_agent"]
