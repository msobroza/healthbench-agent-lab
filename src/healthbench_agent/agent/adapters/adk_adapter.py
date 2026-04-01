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

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from healthbench_agent.agent.agent_pipeline import AgentPipeline
from healthbench_agent.agent.config import AgentNodeConfig, RootAgentPipelineConfig
from healthbench_agent.agent.framework_adapter import FrameworkAdapter
from healthbench_agent.agent.prompt import format_conversation, load_instruction
from healthbench_agent.agent.tool_registry import get_tools
from healthbench_agent.domain.conversation import MessageList


class ADKAgentPipeline(AgentPipeline):
    """Shared AgentPipeline for all ADK-backed agents.

    Encapsulates the ADK ``Runner`` + ``InMemorySessionService`` execution
    loop that was previously duplicated across baseline, tool, and
    multi-agent modules. Any ADK agent (plain ``Agent``, ``Agent`` with
    tools, or ``SequentialAgent`` tree) can be wrapped by this class.

    Attributes:
        config: The root pipeline configuration.
        agent: The top-level ADK agent (Agent or SequentialAgent).
    """

    def __init__(
        self,
        config: RootAgentPipelineConfig,
        agent: Agent | SequentialAgent,
    ) -> None:
        self.config = config
        self.agent = agent
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=agent,
            app_name=config.name,
            session_service=self._session_service,
        )

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
            app_name=self.config.name, user_id="eval",
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
            if (
                event.is_final_response()
                and event.content
                and event.content.parts
            ):
                response_text = event.content.parts[0].text or ""
        return response_text


def build_agent_node(
    config: AgentNodeConfig,
    parent_prompt_path: str = "",
) -> Agent | SequentialAgent:
    """Build an ADK agent from config, recursively.

    Resolves ``prompt_path`` inheritance (empty inherits from parent),
    builds children recursively, and selects the appropriate ADK agent
    type based on ``orchestration``:
        - ``"sequential"``: ``SequentialAgent`` wrapping children.
        - ``"routing"``: ``Agent`` with ``sub_agents`` for LLM routing.

    Leaf agents (no ``sub_agents``) are always plain ``Agent`` nodes
    with tools resolved from the registry. When a leaf has a
    ``condition``, it is appended to the description so the parent
    routing agent can see it.

    Args:
        config: Agent node configuration.
        parent_prompt_path: Inherited prompt path from the parent
            agent. Used when ``config.prompt_path`` is empty.

    Returns:
        An ADK Agent or SequentialAgent.

    Raises:
        ValueError: If the orchestration type is not supported.
    """
    prompt_path = config.prompt_path or parent_prompt_path

    if config.sub_agents:
        children = [
            build_agent_node(child, prompt_path)
            for child in config.sub_agents
        ]

        if config.orchestration == "routing":
            instruction = load_instruction(
                prompt_path, key=config.prompt_key,
            )
            return Agent(
                name=config.name,
                model=config.model,
                description=config.description,
                instruction=instruction,
                sub_agents=children,
                output_key=config.output_key,
            )

        if config.orchestration == "sequential":
            return SequentialAgent(
                name=config.name,
                description=config.description,
                sub_agents=children,
            )

        raise ValueError(
            f"Unsupported orchestration: '{config.orchestration}'. "
            f"Supported: 'sequential', 'routing'"
        )

    # Leaf agent — tools resolved from registry, condition in description.
    tools = get_tools(config.tools) if config.tools else []
    instruction = load_instruction(prompt_path, key=config.prompt_key)
    description = config.description
    if config.condition:
        description = f"{description} [Condition: {config.condition}]"

    return Agent(
        name=config.name,
        model=config.model,
        description=description,
        instruction=instruction,
        tools=tools,
        output_key=config.output_key,
    )


class ADKFrameworkAdapter(FrameworkAdapter):
    """ADK framework adapter — builds ADK agent pipelines from config.

    Handles all three architecture types:
        - **Baseline**: no tools, no sub-agents → single ``Agent``.
        - **Tool-augmented**: tools list → single ``Agent`` with tools.
        - **Multi-agent**: sub-agents → recursive agent tree.

    The adapter delegates agent construction to ``build_agent_node``
    and wraps the result in ``ADKAgentPipeline`` for unified execution.
    """

    def create_pipeline(
        self, config: RootAgentPipelineConfig,
    ) -> ADKAgentPipeline:
        """Build an ADK-backed AgentPipeline from config.

        Args:
            config: Root agent pipeline configuration.

        Returns:
            An ADKAgentPipeline wrapping the constructed ADK agent.
        """
        agent = build_agent_node(config)
        return ADKAgentPipeline(config, agent)
