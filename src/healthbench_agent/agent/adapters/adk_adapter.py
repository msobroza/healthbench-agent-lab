"""Google ADK framework adapter.

Translates ``RootAgentPipelineConfig`` into ADK agent trees and provides
``ADKAgentPipeline`` — the shared ``AgentPipeline`` implementation for
all ADK-backed pipelines (baseline, tool-augmented, multi-agent).

The duplicated ``generate()`` method previously found in each concrete
agent module is unified here. The recursive ``_build_agent_node``
builder constructs the appropriate ADK agent type based on the config's
``orchestration`` and ``tools`` fields.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import ThinkingConfig

from healthbench_agent.agent.agent_pipeline import AgentPipeline
from healthbench_agent.agent.callback_registry import get_callback
from healthbench_agent.agent.config import (
    AgentNodeConfig,
    PlannerConfig,
    RootAgentPipelineConfig,
)
from healthbench_agent.agent.framework_adapter import FrameworkAdapter
from healthbench_agent.agent.prompt import format_conversation, load_instruction
from healthbench_agent.agent.tool_registry import get_tools
from healthbench_agent.domain.conversation import MessageList

_ADK_AGENT: TypeAlias = LlmAgent | SequentialAgent | LoopAgent | ParallelAgent


class ADKAgentPipeline(AgentPipeline):
    """Shared AgentPipeline for all ADK-backed agents.

    Encapsulates the ADK ``Runner`` + ``InMemorySessionService`` execution
    loop that was previously duplicated across baseline, tool, and
    multi-agent modules. Any ADK agent (plain ``Agent``, ``Agent`` with
    tools, or ``SequentialAgent`` tree) can be wrapped by this class.

    Attributes:
        config: The root pipeline configuration.
        agent: The top-level ADK agent.
    """

    def __init__(
        self,
        config: RootAgentPipelineConfig,
        agent: _ADK_AGENT,
    ) -> None:
        self.config = config
        self.agent = agent
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=agent,
            app_name=config.name,
            session_service=self._session_service,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str,
        **overrides: object,
    ) -> ADKAgentPipeline:
        """Create an ADKAgentPipeline from a YAML config file.

        Args:
            config_path: Path to the agent YAML config.
            **overrides: Explicit field overrides for the config.

        Returns:
            A fully configured ADKAgentPipeline.
        """
        config = RootAgentPipelineConfig.from_yaml(config_path, **overrides)
        agent = build_agent_node(config)
        return cls(config, agent)

    async def generate(self, conversation: MessageList) -> str:
        """Generate a response by running the ADK agent.

        Renders the Jinja2 prompt template with the formatted conversation,
        creates a new session, sends the rendered prompt as a user message,
        and extracts the final response text from the event stream.

        Args:
            conversation: Full conversation history (all turns).

        Returns:
            The agent's response text.
        """
        rendered = load_instruction(
            self.config.prompt_path,
            key=self.config.prompt_key,
            conversation=format_conversation(conversation),
        )
        session = await self._session_service.create_session(
            app_name=self.config.name,
            user_id="eval",
        )
        content = types.Content(
            role="user",
            parts=[types.Part(text=rendered)],
        )
        response_text = ""
        async for event in self._runner.run_async(
            user_id="eval",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text or ""
        return response_text


# ---------------------------------------------------------------------------
# Helper functions for build_agent_node
# ---------------------------------------------------------------------------


def _build_planner(
    planner_config: PlannerConfig,
) -> BuiltInPlanner | PlanReActPlanner:
    """Create an ADK planner from config.

    Args:
        planner_config: Planner configuration specifying type and
            optional thinking parameters.

    Returns:
        A ``BuiltInPlanner`` or ``PlanReActPlanner`` instance.
    """
    if planner_config.type == "plan_react":
        return PlanReActPlanner()
    return BuiltInPlanner(
        thinking_config=ThinkingConfig(
            thinking_budget=planner_config.thinking_budget,
            include_thoughts=planner_config.include_thoughts,
        ),
    )


def _resolve_callback(name: str | None) -> Callable[..., Any] | None:
    """Resolve a callback registry name to the actual function.

    Args:
        name: Registered callback name, or ``None`` to skip.

    Returns:
        The callback function, or ``None`` if name is ``None``.
    """
    return get_callback(name) if name else None


def _resolve_agent_callbacks(
    config: AgentNodeConfig,
) -> dict[str, Callable[..., Any] | None]:
    """Resolve before/after agent callbacks from config.

    These callbacks are available on all agent types (LlmAgent,
    SequentialAgent, LoopAgent, ParallelAgent).

    Args:
        config: Agent node configuration.

    Returns:
        Dict with ``before_agent_callback`` and ``after_agent_callback``
        keys, values are resolved callables or ``None``.
    """
    return {
        "before_agent_callback": _resolve_callback(config.before_agent_callback),
        "after_agent_callback": _resolve_callback(config.after_agent_callback),
    }


def _build_llm_agent(
    config: AgentNodeConfig,
    instruction: str,
    description: str,
    tools: list[Callable[..., Any]] | None = None,
    sub_agents: list[_ADK_AGENT] | None = None,
) -> LlmAgent:
    """Construct an LlmAgent with all config-driven fields.

    Centralises LlmAgent construction so that routing agents and leaf
    agents share the same planner, callback, and control-field wiring.

    Args:
        config: Agent node configuration.
        instruction: Rendered instruction text.
        description: Agent description (may include routing condition).
        tools: Resolved tool callables for leaf agents.
        sub_agents: Child ADK agents for routing agents.

    Returns:
        A fully configured ``LlmAgent``.
    """
    agent_callbacks = _resolve_agent_callbacks(config)
    return LlmAgent(
        name=config.name,
        model=config.model,
        description=description,
        instruction=instruction,
        tools=tools or [],
        sub_agents=sub_agents or [],
        output_key=config.output_key,
        planner=_build_planner(config.planner) if config.planner else None,
        include_contents=config.include_contents,
        disallow_transfer_to_parent=config.disallow_transfer_to_parent,
        disallow_transfer_to_peers=config.disallow_transfer_to_peers,
        global_instruction=config.global_instruction or "",
        before_model_callback=_resolve_callback(config.before_model_callback),
        after_model_callback=_resolve_callback(config.after_model_callback),
        before_tool_callback=_resolve_callback(config.before_tool_callback),
        after_tool_callback=_resolve_callback(config.after_tool_callback),
        **agent_callbacks,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_agent_node(
    config: AgentNodeConfig,
    parent_prompt_path: str = "",
) -> _ADK_AGENT:
    """Build an ADK agent from config, recursively.

    Resolves ``prompt_path`` inheritance (empty inherits from parent),
    builds children recursively, and selects the appropriate ADK agent
    type based on ``orchestration``:
        - ``"sequential"``: ``SequentialAgent`` wrapping children.
        - ``"routing"``: ``LlmAgent`` with ``sub_agents`` for LLM routing.
        - ``"loop"``: ``LoopAgent`` iterating over children.
        - ``"parallel"``: ``ParallelAgent`` running children concurrently.

    Leaf agents (no ``sub_agents``) are always ``LlmAgent`` nodes
    with tools resolved from the registry. When a leaf has a
    ``condition``, it is appended to the description so the parent
    routing agent can see it.

    Args:
        config: Agent node configuration.
        parent_prompt_path: Inherited prompt path from the parent
            agent. Used when ``config.prompt_path`` is empty.

    Returns:
        An ADK agent (LlmAgent, SequentialAgent, LoopAgent, or
        ParallelAgent).

    Raises:
        ValueError: If the orchestration type is not supported.
    """
    prompt_path = config.prompt_path or parent_prompt_path

    if config.sub_agents:
        children = [build_agent_node(child, prompt_path) for child in config.sub_agents]

        agent_callbacks = _resolve_agent_callbacks(config)

        if config.orchestration == "routing":
            instruction = load_instruction(
                prompt_path,
                key=config.prompt_key,
            )
            return _build_llm_agent(
                config,
                instruction=instruction,
                description=config.description,
                sub_agents=children,
            )

        if config.orchestration == "sequential":
            return SequentialAgent(
                name=config.name,
                description=config.description,
                sub_agents=children,
                **agent_callbacks,
            )

        if config.orchestration == "loop":
            return LoopAgent(
                name=config.name,
                description=config.description,
                sub_agents=children,
                max_iterations=config.max_iterations,
                **agent_callbacks,
            )

        if config.orchestration == "parallel":
            return ParallelAgent(
                name=config.name,
                description=config.description,
                sub_agents=children,
                **agent_callbacks,
            )

        raise ValueError(
            f"Unsupported orchestration: '{config.orchestration}'. "
            f"Supported: 'sequential', 'routing', 'loop', 'parallel'"
        )

    # Leaf agent — tools resolved from registry, condition in description.
    tools = get_tools(config.tools) if config.tools else []
    instruction = load_instruction(prompt_path, key=config.prompt_key)
    description = config.description
    if config.condition:
        description = f"{description} [Condition: {config.condition}]"

    return _build_llm_agent(
        config,
        instruction=instruction,
        description=description,
        tools=tools,
    )


class ADKFrameworkAdapter(FrameworkAdapter):
    """ADK framework adapter — builds ADK agent pipelines from config.

    Handles all three architecture types:
        - **Baseline**: no tools, no sub-agents -> single ``LlmAgent``.
        - **Tool-augmented**: tools list -> single ``LlmAgent`` with tools.
        - **Multi-agent**: sub-agents -> recursive agent tree.

    The adapter delegates agent construction to ``build_agent_node``
    and wraps the result in ``ADKAgentPipeline`` for unified execution.
    """

    def create_pipeline(
        self,
        config: RootAgentPipelineConfig,
    ) -> ADKAgentPipeline:
        """Build an ADK-backed AgentPipeline from config.

        Args:
            config: Root agent pipeline configuration.

        Returns:
            An ADKAgentPipeline wrapping the constructed ADK agent.
        """
        agent = build_agent_node(config)
        return ADKAgentPipeline(config, agent)
