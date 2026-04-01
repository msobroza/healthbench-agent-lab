"""Tests for healthbench_agent.llm_eval.samplers — mocked provider clients.

Tests OpenAIChatSampler and GeminiChatSampler with mocked SDK clients.
No network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.llm_eval.samplers import GeminiChatSampler, OpenAIChatSampler
from healthbench_agent.llm_eval.config import JudgeConfig


# ---------------------------------------------------------------------------
# OpenAIChatSampler tests
# ---------------------------------------------------------------------------


class TestOpenAIChatSampler:
    """Tests for OpenAIChatSampler with mocked OpenAI client."""

    def test_default_model(self):
        sampler = OpenAIChatSampler()
        assert sampler.model == "gpt-4.1-2025-04-14"

    def test_custom_model(self):
        sampler = OpenAIChatSampler(model="gpt-4o")
        assert sampler.model == "gpt-4o"

    def test_default_temperature_is_zero(self):
        sampler = OpenAIChatSampler()
        assert sampler.temperature == 0.0

    def test_lazy_client_initialization(self):
        sampler = OpenAIChatSampler()
        assert sampler._client is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_returns_sampler_response(self):
        # Setup mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4.1-2025-04-14"
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        sampler = OpenAIChatSampler()
        sampler._client = mock_client  # inject mock directly

        result = sampler([{"role": "user", "content": "Hello"}])

        assert result.response_text == "Test response"
        assert result.response_metadata["model"] == "gpt-4.1-2025-04-14"
        assert result.response_metadata["finish_reason"] == "stop"
        assert result.response_metadata["usage"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# GeminiChatSampler tests
# ---------------------------------------------------------------------------


class TestGeminiChatSampler:
    """Tests for GeminiChatSampler with mocked Gemini client."""

    def test_default_model(self):
        sampler = GeminiChatSampler()
        assert sampler.model == "gemini-2.0-flash"

    def test_custom_model(self):
        sampler = GeminiChatSampler(model="gemini-2.5-pro")
        assert sampler.model == "gemini-2.5-pro"

    def test_default_temperature_is_zero(self):
        sampler = GeminiChatSampler()
        assert sampler.temperature == 0.0

    def test_lazy_client_initialization(self):
        sampler = GeminiChatSampler()
        assert sampler._client is None

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    def test_call_returns_sampler_response(self):
        mock_response = MagicMock()
        mock_response.text = "Gemini response"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        sampler = GeminiChatSampler()
        sampler._client = mock_client  # inject mock directly

        result = sampler([{"role": "user", "content": "Hello"}])

        assert result.response_text == "Gemini response"
        assert result.response_metadata["model"] == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# JudgeConfig tests
# ---------------------------------------------------------------------------


class TestJudgeConfig:
    """Tests for JudgeConfig pydantic-settings."""

    def test_default_values(self):
        config = JudgeConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4.1-2025-04-14"
        assert config.temperature == 0.0
        assert config.max_retries == 3
        assert config.max_workers == 120

    @patch.dict("os.environ", {"JUDGE_MODEL": "gemini-2.0-flash"})
    def test_env_var_override(self):
        config = JudgeConfig()
        assert config.model == "gemini-2.0-flash"

    @patch.dict("os.environ", {"JUDGE_PROVIDER": "gemini"})
    def test_provider_override(self):
        config = JudgeConfig()
        assert config.provider == "gemini"

    def test_temperature_validation_lower_bound(self):
        with pytest.raises(Exception):
            JudgeConfig(temperature=-0.1)

    def test_temperature_validation_upper_bound(self):
        with pytest.raises(Exception):
            JudgeConfig(temperature=1.1)

    def test_max_retries_validation(self):
        with pytest.raises(Exception):
            JudgeConfig(max_retries=0)

    def test_model_dump_for_mlflow(self):
        config = JudgeConfig()
        dump = config.model_dump()
        assert "provider" in dump
        assert "model" in dump
        assert "temperature" in dump
        # API keys should NOT be in the config
        assert "api_key" not in dump
        assert "OPENAI_API_KEY" not in dump
