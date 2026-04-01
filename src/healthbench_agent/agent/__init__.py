"""Agent toolkit — pipeline abstraction, configuration, and prompt loading.

Provides the ``AgentPipeline`` abstract base class that all agent
architectures implement, ``RootAgentPipelineConfig`` for type-safe
configuration with YAML loading and environment variable overrides,
``load_instruction`` for Jinja2-templated prompt files, and
``create_pipeline`` for config-driven pipeline creation via framework
adapters.

Public API:
    - AgentPipeline: ABC for agent pipelines (async generate).
    - AgentNodeConfig: Recursive agent config node (BaseModel).
    - RootAgentPipelineConfig: Root config with env/YAML support.
    - FrameworkAdapter: ABC for framework-specific adapters.
    - create_pipeline: Config-driven factory (dispatches on framework).
    - load_instruction: Load and render a Jinja2 instruction from YAML.
"""

from .agent_pipeline import AgentPipeline
from .config import (
    AgentNodeConfig,
    RootAgentPipelineConfig,
)
from .factory import create_pipeline
from .framework_adapter import FrameworkAdapter
from .prompt import format_conversation, load_instruction
from .tool_registry import get_tool, get_tools, register_tool, registered_tools

__all__ = [
    "AgentPipeline",
    "AgentNodeConfig",
    "RootAgentPipelineConfig",
    "FrameworkAdapter",
    "create_pipeline",
    "format_conversation",
    "load_instruction",
    "register_tool",
    "get_tool",
    "get_tools",
    "registered_tools",
]
