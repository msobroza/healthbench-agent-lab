"""Configuration classes for prompt optimization.

Each optimizer backend has its own config subclass with framework-specific
fields. All share a common base with env var override (prefix ``OPTIM_``).
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseOptimizationConfig(BaseSettings):
    """Shared settings for all prompt optimizers.

    Settings are resolved in priority order:
        1. Explicit constructor kwargs
        2. Environment variables (prefixed ``OPTIM_``)
        3. ``.env`` dotenv file
        4. Field defaults

    Attributes:
        optimizer: Backend identifier used by the registry to dispatch.
        max_trials: Maximum candidate prompts to evaluate.
        sample_size: Number of HealthBench samples per evaluation.
        seed: Random seed for reproducible sampling.
        meta_model: Model used by the optimizer to propose/critique prompts.
        meta_provider: Provider for the meta model.
        google_api_key: Google API key for Gemini-based optimization.
        openai_api_key: OpenAI API key for OpenAI-based optimization.
    """

    model_config = SettingsConfigDict(env_prefix="OPTIM_", env_file=".env")

    optimizer: str = "dspy"
    max_trials: int = Field(50, ge=1)
    sample_size: int = Field(20, ge=1)
    seed: int = 42
    meta_model: str = "gemini-2.5-flash"
    meta_provider: str = "gemini"

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPTIM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )


class DSPyConfig(BaseOptimizationConfig):
    """DSPy-specific optimization config.

    Attributes:
        dspy_optimizer: DSPy optimizer to use — ``"copro"`` or ``"miprov2"``.
        max_bootstrapped_demos: Number of bootstrapped demos. Set to 0 for
            instruction-only optimization (no few-shot examples).
    """

    optimizer: str = "dspy"
    dspy_optimizer: str = "copro"
    max_bootstrapped_demos: int = 0


class TextGradConfig(BaseOptimizationConfig):
    """TextGrad-specific optimization config.

    Attributes:
        steps: Number of text-gradient descent steps.
    """

    optimizer: str = "textgrad"
    steps: int = Field(10, ge=1)


class CritiqueRefineConfig(BaseOptimizationConfig):
    """Critique-refine (PromptWizard algorithm) optimization config.

    Attributes:
        mutation_rounds: Number of prompt mutation rounds per iteration.
        refine_iterations: Number of critique-and-refine cycles.
        style_variations: Number of thinking-style variations per mutation.
    """

    optimizer: str = "critique_refine"
    mutation_rounds: int = Field(3, ge=1)
    refine_iterations: int = Field(3, ge=1)
    style_variations: int = Field(5, ge=1)
