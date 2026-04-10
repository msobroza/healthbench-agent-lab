"""Per-provider :class:`LLMClient` implementations plus the factory.

Public surface:

* :class:`OpenAIChatClient` — OpenAI Chat Completions client.
* :class:`GeminiChatClient` — Google Gemini GenAI client.
* :func:`create_llm_client` — factory that picks the right client for a
  :class:`~healthbench_agent.llm_eval.grading.config.JudgeConfig`.
"""

from __future__ import annotations

from healthbench_agent.domain.llm_client import LLMClient

from ..grading.config import JudgeConfig
from .gemini import GeminiChatClient
from .openai import OpenAIChatClient


def create_llm_client(config: JudgeConfig) -> LLMClient:
    """Create an LLM client from a JudgeConfig, injecting API keys from config.

    Falls back to environment variables if the config does not contain
    the relevant API key.

    Args:
        config: Judge configuration specifying provider, model, and
            optional API keys.

    Returns:
        An LLMClient implementation matching the configured provider.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    if config.provider == "openai":
        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        return OpenAIChatClient(
            model=config.model,
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
            api_key=api_key,
        )
    if config.provider == "gemini":
        api_key = config.google_api_key.get_secret_value() if config.google_api_key else None
        return GeminiChatClient(
            model=config.model,
            temperature=config.temperature,
            max_retries=config.max_retries,
            api_key=api_key,
        )
    raise ValueError(f"Unknown provider: {config.provider!r}. Supported: 'openai', 'gemini'.")


__all__ = [
    "GeminiChatClient",
    "OpenAIChatClient",
    "create_llm_client",
]
