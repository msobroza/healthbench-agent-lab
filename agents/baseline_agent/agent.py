"""ADK entry point for the baseline agent.

Re-exports ``root_agent`` from the baseline pipeline so that
``adk web agents/baseline_agent`` discovers the agent automatically.
"""

from baseline_pipeline import root_agent  # type: ignore[import-not-found]

__all__ = ["root_agent"]
