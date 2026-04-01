"""Framework adapter abstraction for building agent pipelines.

Defines the ``FrameworkAdapter`` ABC that each orchestration framework
(ADK, LangGraph, etc.) must implement. An adapter translates the
framework-agnostic ``RootAgentPipelineConfig`` into a runnable
``AgentPipeline``, encapsulating three bridging concerns:

1. **Tool adaptation** — converting plain callables from the tool
   registry into the framework's tool format.
2. **Agent construction** — building the framework-specific agent
   graph or tree from the recursive config.
3. **Execution** — running the agent and extracting a string response.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from healthbench_agent.agent.agent_pipeline import AgentPipeline
from healthbench_agent.agent.config import RootAgentPipelineConfig


class FrameworkAdapter(ABC):
    """Translates framework-agnostic config into a runnable AgentPipeline.

    Subclasses encapsulate all framework-specific logic: tool wrapping,
    agent graph/tree construction, session management, and execution.
    The ``create_pipeline`` method is the single entry point.
    """

    @abstractmethod
    def create_pipeline(
        self,
        config: RootAgentPipelineConfig,
    ) -> AgentPipeline:
        """Build a framework-specific AgentPipeline from config.

        Args:
            config: Root agent pipeline configuration with optional
                sub-agent definitions, tool names, and orchestration
                mode.

        Returns:
            A fully configured AgentPipeline ready for ``generate()``
            calls.
        """
        ...
