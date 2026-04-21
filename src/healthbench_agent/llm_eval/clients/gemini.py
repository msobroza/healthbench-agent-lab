"""Google Gemini (GenAI) LLM client.

Implements :class:`LLMClient.__call__` against the Google GenAI client.
Interchangeable with any other :class:`LLMClient` implementation
(e.g. :class:`~healthbench_agent.llm_eval.clients.openai.OpenAIChatClient`)
as the grader model without changing evaluation logic.

See SPEC §5.6.
"""

from __future__ import annotations

import os
from typing import Any

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.llm_client import LLMClient, LLMResponse


class GeminiChatClient(LLMClient):
    """LLM client wrapping the Google Gemini (GenAI) API.

    Accepts an explicit ``api_key`` or falls back to the
    ``GOOGLE_API_KEY`` environment variable.

    Attributes:
        model: Gemini model identifier (e.g. 'gemini-2.5-flash').
        temperature: Sampling temperature. 0.0 for deterministic grading.
        max_retries: Number of retries on transient failures.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            import google.genai as genai

            key = self._api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError(
                    "Google API key not configured — pass api_key= to "
                    "GeminiChatClient, set GOOGLE_API_KEY in the environment, "
                    "or provide JUDGE_GOOGLE_API_KEY via JudgeConfig."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def __call__(self, message_list: MessageList) -> LLMResponse:
        """Sample a response from Gemini.

        Args:
            message_list: Conversation history as role/content dicts.

        Returns:
            LLMResponse with the model output and request metadata.
        """
        client = self._get_client()

        # Convert MessageList to Gemini format
        contents = []
        for msg in message_list:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature},
        )

        return LLMResponse(
            response_text=response.text or "",
            actual_queried_message_list=message_list,
            response_metadata={
                "model": self.model,
                "finish_reason": "stop",
            },
        )
