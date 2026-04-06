"""LLM sampler implementations for OpenAI and Gemini providers.

Both implement SamplerBase.__call__(message_list) -> SamplerResponse,
making them interchangeable as the grader model without changing
evaluation logic.

See SPEC §5.6.
"""

from __future__ import annotations

import os
from typing import Any

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.sampler import SamplerBase, SamplerResponse

from .config_grader import JudgeConfig


class OpenAIChatSampler(SamplerBase):
    """Sampler wrapping the OpenAI Chat Completions API.

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

            key = self._api_key or os.environ["OPENAI_API_KEY"]
            self._client = openai.OpenAI(
                api_key=key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """Sample a response from OpenAI.

        Args:
            message_list: Conversation history as role/content dicts.

        Returns:
            SamplerResponse with the model output and request metadata.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=message_list,
            temperature=self.temperature,
        )
        choice = response.choices[0]
        return SamplerResponse(
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


class GeminiChatSampler(SamplerBase):
    """Sampler wrapping the Google Gemini (GenAI) API.

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

            key = self._api_key or os.environ["GOOGLE_API_KEY"]
            self._client = genai.Client(api_key=key)
        return self._client

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """Sample a response from Gemini.

        Args:
            message_list: Conversation history as role/content dicts.

        Returns:
            SamplerResponse with the model output and request metadata.
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

        return SamplerResponse(
            response_text=response.text or "",
            actual_queried_message_list=message_list,
            response_metadata={
                "model": self.model,
                "finish_reason": "stop",
            },
        )


def create_sampler(config: JudgeConfig) -> SamplerBase:
    """Create a sampler from a JudgeConfig, injecting API keys from config.

    Falls back to environment variables if the config does not contain
    the relevant API key.

    Args:
        config: Judge configuration specifying provider, model, and
            optional API keys.

    Returns:
        A SamplerBase implementation matching the configured provider.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    if config.provider == "openai":
        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        return OpenAIChatSampler(
            model=config.model,
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
            api_key=api_key,
        )
    if config.provider == "gemini":
        api_key = config.google_api_key.get_secret_value() if config.google_api_key else None
        return GeminiChatSampler(
            model=config.model,
            temperature=config.temperature,
            max_retries=config.max_retries,
            api_key=api_key,
        )
    raise ValueError(f"Unknown provider: {config.provider!r}. Supported: 'openai', 'gemini'.")
