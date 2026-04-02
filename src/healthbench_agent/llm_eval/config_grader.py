"""Judge configuration using pydantic-settings.

Type-safe configuration for the LLM-as-judge grader, with env var override,
optional YAML file loading, and MLflow-friendly serialization. API keys are
stored as SecretStr so they never appear in model_dump() or logs.

See SPEC §5.9.1.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class EvalMode(StrEnum):
    """Execution mode for the evaluation runner."""

    ASYNC = "async"
    BATCH = "batch"


class JudgeConfig(BaseSettings):
    """Configuration for the LLM-as-judge evaluation pipeline.

    Settings are resolved in priority order:
        1. Explicit constructor kwargs
        2. Environment variables (prefixed ``JUDGE_``)
        3. ``.env`` dotenv file
        4. YAML file (when loaded via ``from_yaml``)
        5. Field defaults

    Override at run time via env vars::

        JUDGE_MODEL=gemini-2.5-flash JUDGE_MAX_WORKERS=20 uv run ...

    Attributes:
        provider: Model provider — ``"openai"`` or ``"gemini"``.
        model: Exact model version string (always pin versions).
        temperature: Sampling temperature. Must be 0.0 for reproducibility.
        max_retries: Number of retries on transient API failures.
        timeout_seconds: Per-request timeout in seconds.
        max_workers: ThreadPool size for async evaluation mode.
        mode: Execution mode — async (ThreadPool) or batch (OpenAI Batch API).
        prompt_path: Path to the Jinja2 grader prompt YAML file.
        openai_api_key: OpenAI API key. Read from ``OPENAI_API_KEY`` or
            ``JUDGE_OPENAI_API_KEY``. Masked in ``model_dump()``.
        google_api_key: Google API key. Read from ``GOOGLE_API_KEY`` or
            ``JUDGE_GOOGLE_API_KEY``. Masked in ``model_dump()``.
    """

    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=".env")

    provider: str = "openai"
    model: str = "gpt-4.1-2025-04-14"
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_retries: int = Field(3, ge=1)
    timeout_seconds: int = 30
    max_workers: int = Field(120, ge=1)
    mode: EvalMode = EvalMode.ASYNC
    prompt_path: str = "prompts/llm_grader/v1_llm_grader.yaml"

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("JUDGE_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("JUDGE_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define settings source priority: init > env > dotenv > yaml > defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides: Any) -> JudgeConfig:
        """Create config loading non-secret settings from a YAML file.

        Priority: explicit overrides > env vars > dotenv > YAML > defaults.
        API keys should not be stored in YAML — they are read from env vars.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Explicit field overrides (highest priority).

        Returns:
            A JudgeConfig instance with settings merged from all sources.
        """
        yaml_config_class = type(
            f"{cls.__name__}WithYaml",
            (cls,),
            {
                "model_config": SettingsConfigDict(
                    env_prefix="JUDGE_",
                    env_file=".env",
                    yaml_file=yaml_path,
                ),
            },
        )
        return yaml_config_class(**overrides)
