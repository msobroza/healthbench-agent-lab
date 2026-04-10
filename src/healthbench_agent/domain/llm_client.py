"""LLM client abstraction for model interaction.

Mirrors simple-evals SamplerBase / SamplerResponse under clearer names.
Nothing in this module imports from the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conversation import MessageList


@dataclass
class LLMResponse:
    """Response returned by an LLM client.

    Attributes:
        response_text: Final text output from the model.
        actual_queried_message_list: The exact message list sent to the model,
            after any prompt assembly or injection.
        response_metadata: Provider-specific metadata (token counts, finish
            reason, latency, etc.).
    """

    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]


class LLMClient:
    """Abstract base class for an LLM client.

    Subclasses wrap a specific model provider (e.g. Gemini, OpenAI) and
    expose a uniform call interface used by all Eval implementations.
    """

    def __call__(self, message_list: MessageList) -> LLMResponse:
        """Sample a response from the model.

        Args:
            message_list: Conversation history to condition the response on.

        Returns:
            An LLMResponse with the model output and request metadata.
        """
        raise NotImplementedError
