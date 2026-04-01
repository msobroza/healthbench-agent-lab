"""Agent pipeline configuration using pydantic-settings.

Type-safe configuration for agent pipelines, with env var override,
optional YAML file loading, and MLflow-friendly serialization. API keys
are stored as SecretStr so they never appear in model_dump() or logs.

Provides two config classes sharing the same agent fields:
    - ``AgentNodeConfig``: Recursive agent config (BaseModel). Used for
      sub-agents — lightweight, no env var loading.
    - ``RootAgentPipelineConfig``: Extends ``AgentNodeConfig`` with
      pydantic-settings machinery (env vars, dotenv, YAML loading).
      Used for the top-level pipeline config.

The same field set supports all pipeline types:
    - Baseline agent: no tools, no sub-agents.
    - Tool-augmented agent: ``tools`` lists registered tool names.
    - Multi-agent pipeline: ``sub_agents`` defines the agent tree,
      ``orchestration`` controls execution strategy.

Orchestration modes:
    - ``"sequential"``: children run in order (``SequentialAgent``).
    - ``"routing"``: parent LLM decides which child to delegate to.
    - ``"loop"``: children run iteratively (``LoopAgent``).
    - ``"parallel"``: children run concurrently (``ParallelAgent``).

"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class PlannerConfig(BaseModel):
    """Planner configuration for LlmAgent nodes.

    ADK supports two planner types:
        - ``"builtin"``: Leverages the model's built-in thinking
          capability (e.g. Gemini thinking). Controlled by
          ``thinking_budget`` and ``include_thoughts``.
        - ``"plan_react"``: Structured plan-action-reasoning format.
          Useful for models without built-in thinking.

    Attributes:
        type: Planner type discriminator.
        thinking_budget: Token budget for built-in thinking (builtin
            planner only).
        include_thoughts: Whether to include raw thinking in the
            response (builtin planner only).
    """

    type: Literal["builtin", "plan_react"] = "builtin"
    thinking_budget: int | None = None
    include_thoughts: bool | None = None


class AgentNodeConfig(BaseModel):
    """Recursive agent configuration node.

    Base configuration for any agent in the pipeline tree. Used directly
    for sub-agents (lightweight, no env var loading) and extended by
    ``RootAgentPipelineConfig`` for root-level configs with env var and
    YAML file support.

    Sub-agents with their own ``sub_agents`` act as parent agents. The
    ``orchestration`` field determines how children are composed:
        - ``"sequential"``: children run in order (``SequentialAgent``).
        - ``"routing"``: parent LLM decides which child to delegate to.
        - ``"loop"``: children run iteratively (``LoopAgent``).
        - ``"parallel"``: children run concurrently (``ParallelAgent``).

    Attributes:
        name: Unique identifier for this agent.
        model: Model version string.
        description: Short description of the agent's role.
        prompt_path: Path to the prompt YAML file. Inherited from parent
            when empty.
        prompt_key: Key to extract from the prompt YAML file.
        prompt_version: Prompt YAML version string.
        tools: Registered tool names to attach (leaf agents only).
        framework: Orchestration framework backend (``"adk"`` or
            ``"langgraph"``). Used by the pipeline factory to select
            the appropriate ``FrameworkAdapter``.
        orchestration: How children are composed.
        condition: Routing condition — describes when this agent should
            be activated by its parent routing agent.
        output_key: ADK session state key for this agent's output.
        sub_agents: Child agent configs (recursive).
        max_iterations: Maximum loop iterations (loop orchestration
            only). ``None`` means no limit.
        planner: Optional planner configuration (LlmAgent nodes only).
        include_contents: Whether to include conversation history.
            ``"default"`` sends history; ``"none"`` omits it.
        disallow_transfer_to_parent: Prevent this agent from
            transferring control back to its parent.
        disallow_transfer_to_peers: Prevent this agent from
            transferring control to sibling agents.
        global_instruction: System-wide instruction applied to this
            agent and all its descendants.
        before_agent_callback: Registered callback name to run before
            agent execution.
        after_agent_callback: Registered callback name to run after
            agent execution.
        before_model_callback: Registered callback name to run before
            LLM calls (LlmAgent only).
        after_model_callback: Registered callback name to run after
            LLM calls (LlmAgent only).
        before_tool_callback: Registered callback name to run before
            tool execution (LlmAgent only).
        after_tool_callback: Registered callback name to run after
            tool execution (LlmAgent only).
    """

    name: str = "agent"
    model: str = "gemini-2.0-flash"
    description: str = ""
    prompt_path: str = ""
    prompt_key: str = "instruction"
    prompt_version: str = ""
    tools: list[str] = Field(default_factory=list)
    framework: str = "adk"
    orchestration: str = "sequential"
    condition: str | None = None
    output_key: str | None = None
    sub_agents: list[AgentNodeConfig] = Field(default_factory=list)

    # Loop orchestration.
    max_iterations: int | None = None

    # Planner (LlmAgent nodes only).
    planner: PlannerConfig | None = None

    # Multi-agent control.
    include_contents: Literal["default", "none"] = "default"
    disallow_transfer_to_parent: bool = False
    disallow_transfer_to_peers: bool = False
    global_instruction: str = ""

    # Callbacks (registered names resolved at build time).
    before_agent_callback: str | None = None
    after_agent_callback: str | None = None
    before_model_callback: str | None = None
    after_model_callback: str | None = None
    before_tool_callback: str | None = None
    after_tool_callback: str | None = None


AgentNodeConfig.model_rebuild()


class RootAgentPipelineConfig(AgentNodeConfig, BaseSettings):
    """Root agent pipeline configuration with env var support.

    Extends ``AgentNodeConfig`` with pydantic-settings machinery for
    environment variable overrides, dotenv loading, and YAML config files.
    Sub-agents are ``AgentNodeConfig`` instances (no env var loading).

    Since ``RootAgentPipelineConfig`` IS-A ``AgentNodeConfig``, the
    config is structurally recursive: the root and every sub-agent share
    the same agent fields.

    Settings are resolved in priority order:
        1. Explicit constructor kwargs
        2. Environment variables (prefixed ``AGENT_``)
        3. ``.env`` dotenv file
        4. YAML file (when loaded via ``from_yaml``)
        5. Field defaults

    Override at run time via env vars::

        AGENT_MODEL=gemini-2.5-pro AGENT_NAME=tool_agent uv run ...

    Attributes:
        google_api_key: Google API key. Read from ``GOOGLE_API_KEY`` or
            ``AGENT_GOOGLE_API_KEY``. Masked in ``model_dump()``.
        openai_api_key: OpenAI API key. Read from ``OPENAI_API_KEY`` or
            ``AGENT_OPENAI_API_KEY``. Masked in ``model_dump()``.
    """

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # Override defaults for root-level configs.
    name: str = "baseline_agent"
    description: str = "Health assistant agent."
    prompt_version: str = "1.0.0"
    prompt_path: str = "prompts/baseline_agent/v1_baseline.yaml"

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
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
    def from_yaml(
        cls,
        yaml_path: str,
        **overrides: Any,
    ) -> RootAgentPipelineConfig:
        """Create config loading non-secret settings from a YAML file.

        Priority: explicit overrides > env vars > dotenv > YAML > defaults.
        API keys should not be stored in YAML — they are read from env vars.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Explicit field overrides (highest priority).

        Returns:
            A RootAgentPipelineConfig with settings merged from all sources.
        """
        yaml_config_class = type(
            f"{cls.__name__}WithYaml",
            (cls,),
            {
                "model_config": SettingsConfigDict(
                    env_prefix="AGENT_",
                    env_file=".env",
                    extra="ignore",
                    yaml_file=yaml_path,
                ),
            },
        )
        return yaml_config_class(**overrides)
