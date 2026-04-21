"""Tests for healthbench_agent.llm_eval.clients — mocked LLMClient implementations.

Tests OpenAIChatClient, GeminiChatClient, and create_llm_client factory
with mocked SDK clients. No network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.llm_eval.clients import (
    GeminiChatClient,
    OpenAIChatClient,
    create_llm_client,
)
from healthbench_agent.llm_eval.grading.config import EvalMode, JudgeConfig

# ---------------------------------------------------------------------------
# OpenAIChatClient tests
# ---------------------------------------------------------------------------


class TestOpenAIChatClient:
    """Tests for OpenAIChatClient with mocked OpenAI client."""

    def test_default_model(self):
        llm_client = OpenAIChatClient()
        assert llm_client.model == "gpt-4.1-2025-04-14"

    def test_custom_model(self):
        llm_client = OpenAIChatClient(model="gpt-4o")
        assert llm_client.model == "gpt-4o"

    def test_default_temperature_is_zero(self):
        llm_client = OpenAIChatClient()
        assert llm_client.temperature == 0.0

    def test_lazy_client_initialization(self):
        llm_client = OpenAIChatClient()
        assert llm_client._client is None

    def test_explicit_api_key_stored(self):
        llm_client = OpenAIChatClient(api_key="test-key-123")
        assert llm_client._api_key == "test-key-123"

    def test_missing_api_key_raises_runtime_error_with_guidance(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm_client = OpenAIChatClient()
        with pytest.raises(RuntimeError, match="OpenAI API key not configured"):
            llm_client._get_client()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_returns_llm_response(self):
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

        llm_client = OpenAIChatClient()
        llm_client._client = mock_client  # inject mock directly

        result = llm_client([{"role": "user", "content": "Hello"}])

        assert result.response_text == "Test response"
        assert result.response_metadata["model"] == "gpt-4.1-2025-04-14"
        assert result.response_metadata["finish_reason"] == "stop"
        assert result.response_metadata["usage"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# GeminiChatClient tests
# ---------------------------------------------------------------------------


class TestGeminiChatClient:
    """Tests for GeminiChatClient with mocked Gemini client."""

    def test_default_model(self):
        llm_client = GeminiChatClient()
        assert llm_client.model == "gemini-2.5-flash"

    def test_custom_model(self):
        llm_client = GeminiChatClient(model="gemini-2.5-pro")
        assert llm_client.model == "gemini-2.5-pro"

    def test_default_temperature_is_zero(self):
        llm_client = GeminiChatClient()
        assert llm_client.temperature == 0.0

    def test_lazy_client_initialization(self):
        llm_client = GeminiChatClient()
        assert llm_client._client is None

    def test_explicit_api_key_stored(self):
        llm_client = GeminiChatClient(api_key="test-key-456")
        assert llm_client._api_key == "test-key-456"

    def test_missing_api_key_raises_runtime_error_with_guidance(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        llm_client = GeminiChatClient()
        with pytest.raises(RuntimeError, match="Google API key not configured"):
            llm_client._get_client()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    def test_call_returns_llm_response(self):
        mock_response = MagicMock()
        mock_response.text = "Gemini response"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        llm_client = GeminiChatClient()
        llm_client._client = mock_client  # inject mock directly

        result = llm_client([{"role": "user", "content": "Hello"}])

        assert result.response_text == "Gemini response"
        assert result.response_metadata["model"] == "gemini-2.5-flash"


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
        assert config.grader_max_workers == 8
        assert config.mode == EvalMode.ASYNC
        assert config.prompt_path == "prompts/llm_grader/v1_llm_grader.yaml"

    @patch.dict("os.environ", {"JUDGE_MODEL": "gemini-2.5-flash"})
    def test_env_var_override(self):
        config = JudgeConfig()
        assert config.model == "gemini-2.5-flash"

    @patch.dict("os.environ", {"JUDGE_PROVIDER": "gemini"})
    def test_provider_override(self):
        config = JudgeConfig()
        assert config.provider == "gemini"

    @patch.dict("os.environ", {"JUDGE_MODE": "batch"})
    def test_mode_override_from_env(self):
        config = JudgeConfig()
        assert config.mode == EvalMode.BATCH

    def test_mode_is_enum(self):
        config = JudgeConfig(mode=EvalMode.BATCH)
        assert config.mode == EvalMode.BATCH
        assert config.mode == "batch"

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
        assert "mode" in dump

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-123"})
    def test_openai_api_key_from_env(self):
        config = JudgeConfig()
        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-test-123"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "gcp-test-456"})
    def test_google_api_key_from_env(self):
        config = JudgeConfig()
        assert config.google_api_key is not None
        assert config.google_api_key.get_secret_value() == "gcp-test-456"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret"})
    def test_api_key_masked_in_model_dump(self):
        config = JudgeConfig()
        dump = config.model_dump()
        # SecretStr serializes as '**********', not the actual value
        assert dump.get("openai_api_key") != "sk-secret"

    def test_no_api_keys_by_default(self, monkeypatch):
        # Scrub env vars so the test is hermetic regardless of local .env
        for var in (
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "JUDGE_OPENAI_API_KEY",
            "JUDGE_GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        config = JudgeConfig(_env_file=None)
        assert config.openai_api_key is None
        assert config.google_api_key is None


# ---------------------------------------------------------------------------
# create_llm_client tests
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_creates_openai_client(self):
        config = JudgeConfig(provider="openai")
        llm_client = create_llm_client(config)
        assert isinstance(llm_client, OpenAIChatClient)
        assert llm_client.model == config.model
        assert llm_client.temperature == config.temperature

    def test_creates_gemini_client(self):
        config = JudgeConfig(provider="gemini", model="gemini-2.5-flash")
        llm_client = create_llm_client(config)
        assert isinstance(llm_client, GeminiChatClient)
        assert llm_client.model == "gemini-2.5-flash"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-config"})
    def test_passes_api_key_from_config(self):
        config = JudgeConfig(provider="openai")
        llm_client = create_llm_client(config)
        assert isinstance(llm_client, OpenAIChatClient)
        assert llm_client._api_key == "sk-from-config"

    def test_unknown_provider_raises(self):
        config = JudgeConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(config)
