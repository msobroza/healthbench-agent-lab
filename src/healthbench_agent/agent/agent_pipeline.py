"""Agent pipeline abstraction for generating responses to conversations.

Defines the contract that any agent (baseline, tool-augmented, multi-agent)
must implement to be evaluable by the llm_eval runner. Nothing in this
module imports from the rest of the project except conversation types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from healthbench_agent.domain.conversation import MessageList


class AgentPipeline(ABC):
    """Abstract base for agent pipelines that generate responses.

    Subclasses wrap a specific agent architecture (ADK Agent, multi-agent
    pipeline, or any custom inference) and expose a uniform async interface
    used by EvalRunner to generate responses before grading.
    """

    @abstractmethod
    async def generate(self, conversation: MessageList) -> str:
        """Generate a response for the given conversation.

        Args:
            conversation: Conversation history to respond to. The last
                message is typically a user turn.

        Returns:
            The agent's response text.
        """
        ...
