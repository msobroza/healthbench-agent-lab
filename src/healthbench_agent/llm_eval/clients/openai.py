"""OpenAI Chat Completions LLM client.

Implements :class:`LLMClient.__call__` against the OpenAI Chat Completions
API. Interchangeable with any other :class:`LLMClient` implementation
(e.g. :class:`~healthbench_agent.llm_eval.clients.gemini.GeminiChatClient`)
as the grader model without changing evaluation logic.

See SPEC §5.6.
"""

from __future__ import annotations

import os
from typing import Any

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.llm_client import LLMClient, LLMResponse


class OpenAIChatClient(LLMClient):
    """LLM client wrapping the OpenAI Chat Completions API.

    Accepts an explicit ``api_key`` or falls back to the
    ``OPENAI_API_KEY`` environment variable.

    Attributes:
        model: OpenAI model identifier (e.g. 'gpt-4.1-2025-04-14').
        temperature: Sampling temperature. 0.0 for deterministic grading.
        max_retries: Number of retries on transient failures.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 30,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            import openai

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OpenAI API key not configured — pass api_key= to "
                    "OpenAIChatClient, set OPENAI_API_KEY in the environment, "
                    "or provide JUDGE_OPENAI_API_KEY via JudgeConfig."
                )
            self._client = openai.OpenAI(
                api_key=key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def __call__(self, message_list: MessageList) -> LLMResponse:
        """Sample a response from OpenAI.

        Args:
            message_list: Conversation history as role/content dicts.

        Returns:
            LLMResponse with the model output and request metadata.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=message_list,
            temperature=self.temperature,
        )
        choice = response.choices[0]
        return LLMResponse(
            response_text=choice.message.content or "",
            actual_queried_message_list=message_list,
            response_metadata={
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                },
            },
        )
