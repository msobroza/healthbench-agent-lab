"""Framework-specific adapters for building agent pipelines.

Each adapter implements ``FrameworkAdapter`` and translates the
framework-agnostic ``RootAgentPipelineConfig`` into a runnable
``AgentPipeline`` backed by a specific orchestration framework.
"""

from .adk_adapter import ADKAgentPipeline, ADKFrameworkAdapter

__all__ = [
    "ADKAgentPipeline",
    "ADKFrameworkAdapter",
]
