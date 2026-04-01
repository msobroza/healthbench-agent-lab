"""Pipeline factory — config-driven AgentPipeline creation.

Dispatches on ``config.framework`` to select the appropriate
``FrameworkAdapter`` and build a ready-to-use ``AgentPipeline``.

Usage::

    from healthbench_agent.agent.factory import create_pipeline
    from healthbench_agent.agent.config import RootAgentPipelineConfig

    config = RootAgentPipelineConfig.from_yaml("config/agents/baseline_agent.yaml")
    pipeline = create_pipeline(config)
    response = await pipeline.generate(conversation)
"""

from __future__ import annotations

from healthbench_agent.agent.agent_pipeline import AgentPipeline
from healthbench_agent.agent.config import RootAgentPipelineConfig
from healthbench_agent.agent.framework_adapter import FrameworkAdapter


def _get_adapter(framework: str) -> FrameworkAdapter:
    """Resolve a FrameworkAdapter by name.

    Imports are deferred so that framework-specific dependencies
    (e.g. ``langgraph``) are only required when actually requested.

    Args:
        framework: Framework identifier (``"adk"``, ``"langgraph"``).

    Returns:
        An instance of the corresponding FrameworkAdapter.

    Raises:
        ValueError: If the framework is not supported.
    """
    if framework == "adk":
        from healthbench_agent.agent.adapters.adk_adapter import ADKFrameworkAdapter
        return ADKFrameworkAdapter()

    raise ValueError(
        f"Unsupported framework: {framework!r}. Supported: 'adk'"
    )


def create_pipeline(config: RootAgentPipelineConfig) -> AgentPipeline:
    """Create an AgentPipeline from config, dispatching on framework.

    The ``config.framework`` field determines which ``FrameworkAdapter``
    is used to build the pipeline. This is the primary entry point for
    config-driven pipeline creation.

    Args:
        config: Root agent pipeline configuration. The ``framework``
            field selects the adapter.

    Returns:
        A fully configured AgentPipeline ready for ``generate()`` calls.

    Raises:
        ValueError: If ``config.framework`` is not supported.
    """
    adapter = _get_adapter(config.framework)
    return adapter.create_pipeline(config)
