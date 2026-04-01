"""Judge configuration using pydantic-settings.

Type-safe configuration for the LLM-as-judge grader, with env var override
and MLflow-friendly serialization. API keys (OPENAI_API_KEY, GOOGLE_API_KEY)
are NOT stored here — they remain as plain env vars for security.

See SPEC §5.9.1 and AGENT_DECISIONS.md §15.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeConfig(BaseSettings):
    """Configuration for the LLM-as-judge evaluation pipeline.

    Override at run time via env vars prefixed with JUDGE_:
        JUDGE_MODEL=gemini-2.0-flash JUDGE_MAX_WORKERS=20 uv run ...

    Attributes:
        provider: Model provider — "openai" or "gemini".
        model: Exact model version string (always pin versions).
        temperature: Sampling temperature. Must be 0.0 for reproducibility.
        max_retries: Number of retries on transient API failures.
        timeout_seconds: Per-request timeout in seconds.
        max_workers: ThreadPool size for async evaluation mode.
        prompt_path: Path to the Jinja2 grader prompt YAML file.
    """

    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=".env")

    provider: str = "openai"
    model: str = "gpt-4.1-2025-04-14"
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_retries: int = Field(3, ge=1)
    timeout_seconds: int = 30
    max_workers: int = Field(120, ge=1)
    prompt_path: str = "prompts/grader_v1.yaml"
