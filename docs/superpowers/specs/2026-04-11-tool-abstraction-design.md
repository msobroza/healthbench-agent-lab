# Tool Abstraction — Design Spec

**Date:** 2026-04-11
**Status:** Approved — ready for implementation plan
**Scope:** Introduce a framework-agnostic `Tool` abstraction with a `@tool`
decorator and four framework adapters (ADK, PydanticAI, LangChain, CrewAI).
Refactor the three existing medical tools onto the new abstraction. Wire
the ADK pipeline adapter to resolve tools through the new registry.

## 1. Motivation

Today's tool-authoring surface lives in
[src/healthbench_agent/agent/tool_registry.py](../../../src/healthbench_agent/agent/tool_registry.py)
and stores bare `Callable[..., Any]` values under string names. Every tool is
authored with `@register_tool("name")` on a plain Python function; the ADK
pipeline adapter at
[src/healthbench_agent/agent/adapters/adk_adapter.py:322](../../../src/healthbench_agent/agent/adapters/adk_adapter.py#L322)
resolves the list of names to callables via `get_tools(...)` and hands them
straight to ADK's `LlmAgent(tools=[...])`.

This is tightly coupled to ADK. Swapping to (or additionally supporting) any
other agent framework means rewriting every tool module and the pipeline
resolution path. The goal of this PR is to introduce a thin `Tool` abstraction
that any framework adapter can consume, and to prove it portable across four
frameworks — ADK (which we use today), PydanticAI, LangChain/LangGraph, and
CrewAI — by shipping adapters for all four and exercising them in CI.

The abstraction is deliberately minimal: it holds only the fields every
framework needs (name, description, callable, optional Pydantic args schema)
plus three reproducibility fields (version, captured source text, source
hash). Framework-specific quirks (return_direct, timeout, confirmation
hooks, MCP annotations) are intentionally out of scope; they can be added
later as optional `Tool` fields without breaking existing consumers.

## 2. Goals and non-goals

### Goals

- Define a `Tool` Pydantic model and a single `@tool` decorator that is
  the only authoring entry point. Pydantic validators enforce
  identifier-style names, semver-style versions, and source/hash
  consistency.
- Define a `ToolAdapter` ABC and a decorator-based registry matching the
  existing `prompt_optimization` adapter pattern.
- Ship four adapters: ADK (required dep), PydanticAI, LangChain, CrewAI
  (optional deps under a new `tool-adapters` extra).
- Refactor the three existing medical tools (`drug_reference`,
  `symptom_checker`, `emergency_flag`) onto `@tool` with a two-line diff each.
- Wire the ADK pipeline adapter to resolve tools through
  `get_tool_adapter("adk").convert(t)` rather than by passing bare callables.
- Capture per-tool reproducibility metadata at decoration time: explicit
  `version` (semver-style, author-supplied, default `"1.0.0"`); `source`
  via `inspect.getsource`; `source_hash` via truncated SHA-256. Matches
  the existing `prompt_version` convention in `config/agents/*.yaml`.
- Install `--extra tool-adapters` in CI so every adapter is exercised on
  every run. Document this as a repeating convention in `CLAUDE.md`.

### Non-goals

- No context-injection feature (PydanticAI-style `RunContext`,
  ADK-style `tool_context`). Revisited when a real tool needs it.
- No `metadata: dict` escape hatch on `Tool`. Added later if a real tool
  needs framework-specific passthrough.
- No subclass-of-`Tool` authoring path, no polymorphic `@tool` dispatch on
  class or instance arguments. `@tool` is a plain function decorator.
- No new medical tools. The three existing ones are the reference examples.
- No MCP server exposure of the registered tools.
- No migration of [config/agents/tool_agent.yaml](../../../config/agents/tool_agent.yaml);
  tools are still referenced by string name in config.
- No public API for fetching tools by tag or category. Registry is flat
  name→`Tool`.

## 3. Research — cross-framework common ground

Research across six frameworks (LangChain, Google ADK, PydanticAI, CrewAI,
LlamaIndex, OpenAI Agents SDK) plus the MCP protocol found that every
framework models a tool with the **same four logical fields** under different
names:

| Logical field | LangChain | Google ADK | PydanticAI | CrewAI | LlamaIndex | OpenAI Agents | MCP |
|---|---|---|---|---|---|---|---|
| name | `name` | `func.__name__` | `name` | `name` | `name` | `name` | `name` |
| description | `description` / docstring | `func.__doc__` | `description` / docstring | `description` | `description` / docstring | `description` / docstring | `description` |
| callable | `func` + `coroutine` | `func` | `function` | `func` | `fn` + `async_fn` | `on_invoke_tool` | decorated fn |
| args schema | `args_schema` | inferred from type hints | inferred from type hints | `args_schema` (required Pydantic) | `fn_schema` (Pydantic) | `params_json_schema` (JSON) | `inputSchema` (JSON) |

**Common to all:** sync and async callables are supported; description falls
back to the docstring; tools are passed as a list of instances to an agent
constructor.

**Divergences that the abstraction has to choose on:**

- **Sync/async shape.** Two separate fields (LangChain, LlamaIndex,
  CrewAI via `_run`/`_arun`) vs one callable with runtime detection
  (ADK, PydanticAI, OpenAI, MCP). **Decision:** one callable field,
  detect async via `inspect.iscoroutinefunction` at adapter time.
- **Schema source.** Type-hint introspection vs explicit Pydantic model vs
  raw JSON Schema. **Decision:** optional Pydantic model, `None` means
  "each adapter uses its framework's native type-hint inference." CrewAI
  (which requires a schema) synthesises one internally when ours is `None`.
- **Description override.** ADK forbids it; everyone else allows it.
  **Decision:** our `Tool.description` is authoritative; the ADK adapter
  wraps the callable in a thin shim that overwrites `__name__`/`__doc__`
  before handing it to `FunctionTool(func=...)`.
- **Return contract.** ADK prefers dict with `status`; MCP requires a
  `content[]` array; most others accept anything. **Decision:** keep
  permissive `Any`. The ADK `dict` with `status` convention stays as a
  documented recommendation, not a type constraint.
- **Framework-specific flags** (`return_direct`, `timeout`, `max_retries`,
  `require_confirmation`, MCP `annotations`). **Decision:** cut entirely
  for this PR. None overlap between frameworks; speculative design. Add
  as optional fields to `Tool` when a real tool needs them.

## 4. File layout

**New subpackage** — `src/healthbench_agent/tools/`, mirroring the existing
`src/healthbench_agent/prompt_optimization/` layout:

```
src/healthbench_agent/tools/
├── __init__.py                   # re-exports public API
├── tool.py                       # Tool frozen Pydantic model + validators
├── registry.py                   # @tool decorator + accessors
├── tool_adapter.py               # ToolAdapter ABC + require_optional
├── tool_adapter_registry.py      # @register_tool_adapter decorator + get_tool_adapter
└── adapters/
    ├── __init__.py               # side-effect imports of all four adapters
    ├── adk_adapter.py            # ADKToolAdapter (required dep)
    ├── pydantic_ai_adapter.py    # PydanticAIToolAdapter (optional)
    ├── langchain_adapter.py      # LangChainToolAdapter (optional)
    └── crewai_adapter.py         # CrewAIToolAdapter (optional)
```

**`tools/__init__.py` public API** — five authoring symbols and four
adapter-registry symbols:

```python
from .registry import get_tool, get_tools, registered_tools, tool
from .tool import Tool
from .tool_adapter import ToolAdapter, require_optional
from .tool_adapter_registry import (
    get_tool_adapter,
    register_tool_adapter,
    registered_tool_adapters,
)

# Side-effect import so the built-in adapters self-register on package load.
from . import adapters  # noqa: F401

__all__ = [
    "Tool",
    "ToolAdapter",
    "get_tool",
    "get_tool_adapter",
    "get_tools",
    "register_tool_adapter",
    "registered_tool_adapters",
    "registered_tools",
    "require_optional",
    "tool",
]
```

**Name coexistence note.** `src/healthbench_agent/tools/` (the new
subpackage — the abstraction) and the repo-root [tools/](../../../tools/)
working directory (the concrete medical tool functions) are separate Python
packages with different import paths (`healthbench_agent.tools` vs `tools`).
They coexist cleanly; the former is the abstraction layer, the latter holds
concrete implementations — same relationship as
`src/healthbench_agent/agent/` (abstraction) vs
[agents/](../../../agents/) (concrete pipeline definitions).

**Filename note on `adk_adapter.py`.** Two files will be named
`adk_adapter.py` — one under
[src/healthbench_agent/agent/adapters/](../../../src/healthbench_agent/agent/adapters/)
(the pipeline adapter, builds agent trees) and one under
`src/healthbench_agent/tools/adapters/` (the tool adapter, converts a single
`Tool` to `google.adk.tools.FunctionTool`). The full import paths are
unambiguous and module docstrings clarify intent.

**Touched files (migrations):**

- [src/healthbench_agent/agent/tool_registry.py](../../../src/healthbench_agent/agent/tool_registry.py)
  — **deleted**. Content moves to `src/healthbench_agent/tools/registry.py`
  with updated semantics (stores `Tool` instances, not bare callables).
- [src/healthbench_agent/agent/__init__.py](../../../src/healthbench_agent/agent/__init__.py)
  — stops re-exporting `register_tool` / `get_tool` / `get_tools`.
  No compat shim.
- [src/healthbench_agent/agent/adapters/adk_adapter.py:38](../../../src/healthbench_agent/agent/adapters/adk_adapter.py#L38)
  — import update.
- [src/healthbench_agent/agent/adapters/adk_adapter.py:322](../../../src/healthbench_agent/agent/adapters/adk_adapter.py#L322)
  — two-line change to route through `get_tool_adapter("adk").convert(...)`.
- [tools/drug_reference.py](../../../tools/drug_reference.py),
  [tools/symptom_checker.py](../../../tools/symptom_checker.py),
  [tools/emergency_flag.py](../../../tools/emergency_flag.py) — swap the
  import path and swap `@register_tool("name")` for bare `@tool`.
- [tools/tools.py](../../../tools/tools.py) — import path update.
- [tests/agents/test_tool_agent.py](../../../tests/agents/test_tool_agent.py)
  — import path update; new pipeline integration test added.
- [pyproject.toml](../../../pyproject.toml) — add `tool-adapters` optional
  extra listing `pydantic-ai`, `langchain-core`, `crewai`.
- [.github/workflows/ci.yml](../../../.github/workflows/ci.yml) — add
  `--extra tool-adapters` to the `uv sync` invocation.
- [CLAUDE.md](../../../CLAUDE.md) — three updates (convention, project
  layout, architecture note). See Section 9.

**New test files** (under `tests/tools/`, mirroring the new subpackage):

```
tests/tools/
├── __init__.py
├── conftest.py                          # clean_tool_registry + clean_tool_adapter_registry autouse fixtures
├── test_tool.py                         # 9 tests
├── test_registry.py                     # 16 tests
├── test_tool_adapter.py                 # 4 tests
├── test_tool_adapter_registry.py        # 5 tests
└── adapters/
    ├── __init__.py
    ├── test_adk_adapter.py              # 6 tests (always runs)
    ├── test_pydantic_ai_adapter.py      # 5 tests (importorskip)
    ├── test_langchain_adapter.py        # 6 tests (importorskip)
    └── test_crewai_adapter.py           # 7 tests (importorskip)
```

**Total new test count: 58.** Plus one pipeline integration test added to
the existing [tests/agents/test_tool_agent.py](../../../tests/agents/test_tool_agent.py).

## 5. `Tool` Pydantic model

Defined in `src/healthbench_agent/tools/tool.py`:

```python
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Tool(BaseModel):
    """Framework-agnostic tool definition.

    Authored exclusively via the :func:`tool` decorator, which builds a
    Tool from a plain Python function by inferring ``name`` from
    ``func.__name__`` and ``description`` from ``inspect.getdoc(func)``.
    Framework adapters translate a Tool into their native tool type via
    the :class:`ToolAdapter` interface.

    The ``func`` field may be sync or async; adapters detect the
    difference at call time via ``inspect.iscoroutinefunction`` and
    route it into the framework's appropriate slot.

    When ``args_schema`` is ``None``, each adapter derives the argument
    schema from ``func``'s type hints and docstring using its
    framework's native introspection. When a Pydantic ``BaseModel`` is
    supplied, it becomes the single source of truth for the LLM-visible
    argument schema — every adapter consumes it. Adapters whose
    framework does not natively accept a Pydantic model (ADK) wrap the
    callable in a thin shim that surfaces the schema through the
    framework's own declaration hook.

    ``version``, ``source``, and ``source_hash`` support reproducibility:
    they are captured once at decoration time and logged by the
    experiment tracker alongside the prompt version and agent config.

    Attributes:
        name: Unique string identifier the LLM uses to call the tool.
            Must be a non-empty valid Python identifier — LLM-callable
            names cannot contain spaces, dashes, or punctuation.
        description: Human-readable explanation of what the tool does,
            shown to the LLM. Must be non-empty.
        func: The underlying Python callable, sync or async.
        args_schema: Optional Pydantic model describing the LLM-visible
            arguments. When ``None``, adapters infer from ``func``'s
            type hints + docstring natively.
        version: Dotted MAJOR.MINOR.PATCH version string (e.g.
            ``"1.0.0"``, ``"2.1.3-beta"``). Bumped manually on
            behavioral changes. Default ``"1.0.0"``.
        source: Full source text of ``func`` as captured by
            ``inspect.getsource`` at decoration time. Empty string
            when the source was not available (REPL, C extension,
            builtin).
        source_hash: Truncated SHA-256 (16 hex chars) of ``source``,
            or empty string when ``source`` is empty. Enables change
            detection when authors forget to bump ``version``.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,  # Callable is not a native Pydantic type.
    )

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    func: Callable[..., Any]
    args_schema: type[BaseModel] | None = None
    version: str = "1.0.0"
    source: str = ""
    source_hash: str = ""

    @field_validator("name")
    @classmethod
    def _name_must_be_identifier(cls, value: str) -> str:
        """Reject tool names that are not valid Python identifiers.

        LLMs call tools by name, and frameworks pass the name directly
        into function-calling JSON schemas where non-identifier
        characters break the schema. Enforcing ``str.isidentifier()``
        at construction time surfaces the error at decoration time
        rather than inside an adapter.
        """
        if not value.isidentifier():
            raise ValueError(
                f"Tool name {value!r} must be a valid Python identifier "
                "(LLM-callable tool names cannot contain spaces, dashes, "
                "or punctuation)."
            )
        return value

    @field_validator("version")
    @classmethod
    def _version_must_be_semver_like(cls, value: str) -> str:
        """Require a dotted numeric version string (e.g. ``1.2.3``).

        Matches the format already used by ``prompt_version`` fields
        in :file:`config/agents/*.yaml`. Does not enforce strict
        semver — pre-release tags and build metadata are allowed —
        but the leading ``MAJOR.MINOR.PATCH`` numeric triple must be
        present so the version sorts naturally in MLflow tag lists.
        """
        parts = value.split(".", maxsplit=3)
        if len(parts) < 3 or not all(p[:1].isdigit() for p in parts[:3]):
            raise ValueError(
                f"Tool version {value!r} must start with MAJOR.MINOR.PATCH "
                f'(e.g. "1.0.0", "2.1.3-beta"); got {value!r}.'
            )
        return value

    @field_validator("source_hash")
    @classmethod
    def _source_hash_matches_source(
        cls, value: str, info: ValidationInfo
    ) -> str:
        """Reject a source_hash that does not match the captured source.

        Tight invariant: either both ``source`` and ``source_hash`` are
        empty strings (source was unavailable at decoration time), or
        ``source_hash`` is the truncated SHA-256 of ``source``. The
        decorator is the only caller that sets these fields, so a
        mismatch here always indicates programmer error.
        """
        source = info.data.get("source", "")
        if not source and not value:
            return value
        expected = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        if value != expected:
            raise ValueError(
                f"source_hash {value!r} does not match SHA-256 of source "
                f"(expected {expected!r}). Did you construct a Tool manually?"
            )
        return value
```

**Field naming.** `func` (not `callable`, not `function`). `callable`
shadows the Python builtin `callable()`. `func` is the majority convention
across ADK, CrewAI, and LangChain; only PydanticAI uses `function`. `fn`
would violate the "no abbreviations" rule in
[CLAUDE.md](../../../CLAUDE.md).

**Why Pydantic instead of a frozen dataclass.** Every other configuration
surface in the project (`AgentNodeConfig`, `PlannerConfig`, `JudgeConfig`,
all of `prompt_optimization/config.py`) is a Pydantic model; using a
dataclass for `Tool` would be the odd one out. Pydantic also enforces the
invariants the dataclass version trusted to convention:

- ``name`` is a valid Python identifier — catches ``@tool(name="drug reference")``
  at decoration time before any adapter sees it.
- ``description`` is non-empty — caught by ``min_length=1`` even if the
  decorator is bypassed.
- ``version`` is semver-like — catches typos like ``"v1"`` or ``"1.0"``.
- ``source_hash`` matches ``source`` — tight invariant for anyone
  constructing ``Tool(...)`` manually.

**Pydantic args_schema is the escape hatch for richer validation.** Type
hints alone cannot express per-parameter descriptions, enum constraints,
validators, or regex patterns. A Pydantic model gives the author all of
them in one place, and every framework we target accepts a
`BaseModel`-derived schema (LangChain and CrewAI natively; PydanticAI
via flattening; ADK via a `BaseTool._get_declaration()` override).

**Reproducibility fields — version, source, source_hash.** Matches the
existing ``prompt_version`` convention in
[config/agents/tool_agent.yaml:5](../../../config/agents/tool_agent.yaml#L5).
``version`` is author-supplied (declarative semver, not auto-inferred —
semantic-versioning decisions need author judgment). ``source`` and
``source_hash`` are populated automatically by the decorator at
decoration time via ``inspect.getsource`` + ``hashlib.sha256``. Together
they enable MLflow-logged reproducibility and detect the case where the
author changes behavior but forgets to bump the version. Adapters ignore
these fields entirely — they are reproducibility metadata, not runtime
dispatch inputs.

## 6. `@tool` decorator

Defined in `src/healthbench_agent/tools/registry.py`:

```python
"""Tool registry — stores Tool instances and exposes the ``@tool`` decorator.

Tools are authored by decorating a plain Python function with ``@tool``.
The decorator infers ``name`` from ``func.__name__`` and ``description``
from ``inspect.getdoc(func)``, captures the function's source via
``inspect.getsource`` and computes a truncated SHA-256 for reproducibility,
builds a :class:`Tool`, and registers it in the module-level ``_REGISTRY``.
All inferred and defaulted fields can be overridden via kwargs:

* ``@tool(name=..., description=...)`` — rename / redescribe.
* ``@tool(args_schema=MyPydanticModel)`` — force a specific argument
  schema, bypassing type-hint inference.
* ``@tool(version="2.0.0")`` — bump the tool's semver-style version
  on behavioral change.

The decorator returns the original function unchanged, so tests and
other callers can invoke it directly — the Tool instance lives only in
the registry.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from .tool import Tool

_REGISTRY: dict[str, Tool] = {}

F = TypeVar("F", bound=Callable[..., Any])


def _insert(tool_instance: Tool) -> None:
    """Insert a Tool into the registry, erroring on name collision.

    Args:
        tool_instance: The Tool to register.

    Raises:
        ValueError: If a tool with the same name is already registered.
    """
    if tool_instance.name in _REGISTRY:
        existing = _REGISTRY[tool_instance.name].func
        raise ValueError(
            f"Tool '{tool_instance.name}' is already registered "
            f"(existing func: {existing.__module__}.{existing.__qualname__})"
        )
    _REGISTRY[tool_instance.name] = tool_instance


def _capture_source(fn: Callable[..., Any]) -> tuple[str, str]:
    """Capture a function's source text and truncated SHA-256 hash.

    Uses ``inspect.getsource``. Returns empty strings for callables
    whose source is not available at import time — builtins, C
    extensions, dynamically generated functions, and REPL code.

    Args:
        fn: The function to snapshot.

    Returns:
        Tuple ``(source, source_hash)``. ``source_hash`` is the first
        16 hex characters of the SHA-256 digest of ``source``, or the
        empty string if ``source`` could not be captured.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return ("", "")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return (source, source_hash)


@overload
def tool(func: F) -> F: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    version: str = "1.0.0",
) -> Callable[[F], F]: ...


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    version: str = "1.0.0",
) -> F | Callable[[F], F]:
    """Decorator that builds a :class:`Tool` from a function and registers it.

    Two forms:

    * ``@tool`` — bare. Name inferred from ``func.__name__``,
      description from ``inspect.getdoc(func)``, the argument schema
      inferred by each adapter from ``func``'s type hints, and
      ``version`` defaulted to ``"1.0.0"``.
    * ``@tool(name=..., description=..., args_schema=..., version=...)``
      — any kwarg overrides its inferred / defaulted value. Omitted
      kwargs keep their inference behavior.

    Use ``args_schema`` when type hints alone don't capture what the
    LLM needs to see: per-parameter descriptions, enum constraints,
    validators, or a restructured shape. The Pydantic model you pass
    becomes the authoritative LLM-visible schema across every adapter.

    Use ``version`` to track the tool's semver-style behavioral
    version. Bump it manually when the tool's logic changes in a way
    that should be reproducible in MLflow logs. The decorator also
    captures ``func``'s source text and SHA-256 hash automatically so
    silent behavioral drift (author changes code but forgets to bump
    the version) is detectable after the fact.

    The decorator returns the **original function** unchanged, so the
    decorated symbol remains directly callable by tests and other
    Python code.

    Args:
        func: The function to decorate (supplied by Python when used bare).
        name: Override the inferred tool name.
        description: Override the inferred description.
        args_schema: Optional Pydantic model that replaces type-hint
            inference as the authoritative argument schema.
        version: Dotted MAJOR.MINOR.PATCH string. Default ``"1.0.0"``.

    Returns:
        The original function, unchanged.

    Raises:
        ValueError: If the resolved description is empty (no docstring
            and no explicit override), or if the resolved name is
            already registered.
        pydantic.ValidationError: If ``Tool`` construction fails — for
            example when the resolved ``name`` is not a valid Python
            identifier or ``version`` is not semver-like.
    """

    def _apply(fn: F) -> F:
        resolved_name = name if name is not None else fn.__name__
        resolved_description = (
            description if description is not None else (inspect.getdoc(fn) or "")
        )
        if not resolved_description:
            raise ValueError(
                f"Tool '{resolved_name}' has no description. "
                f"Add a docstring to {fn.__qualname__} or pass "
                f"description=... to @tool."
            )
        source, source_hash = _capture_source(fn)
        _insert(
            Tool(
                name=resolved_name,
                description=resolved_description,
                func=fn,
                args_schema=args_schema,
                version=version,
                source=source,
                source_hash=source_hash,
            )
        )
        return fn

    if func is not None:
        return _apply(func)
    return _apply


def get_tool(name: str) -> Tool:
    """Look up a registered tool by name.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Tool '{name}' is not registered. Available tools: {available}"
        )
    return _REGISTRY[name]


def get_tools(names: list[str]) -> list[Tool]:
    """Resolve a list of tool names to Tool instances.

    Raises:
        KeyError: If any name is not registered.
    """
    return [get_tool(name) for name in names]


def registered_tools() -> dict[str, Tool]:
    """Return a copy of the current tool registry."""
    return dict(_REGISTRY)
```

**Authoring ergonomics — four forms, all on a plain function:**

```python
# Bare — inference for everything, version defaults to "1.0.0"
@tool
def drug_reference(drug_name: str) -> dict:
    """Look up drug information including dosage, interactions, and contraindications.

    Args:
        drug_name: The name of the drug to look up (generic or brand name).
    """
    ...

# Rename / redescribe only
@tool(name="drug_lookup_v2", description="Alternate drug lookup.")
def drug_reference(drug_name: str) -> dict:
    ...

# Explicit version bump after a behavioral change
@tool(version="2.0.0")
def drug_reference(drug_name: str) -> dict:
    """Look up drug information — now covers EU drug database."""
    ...

# Redefine the argument schema via a Pydantic model
from typing import Literal
from pydantic import BaseModel, Field


class DrugLookupArgs(BaseModel):
    drug_name: str = Field(description="Generic or brand name of the drug to look up.")
    country: Literal["US", "EU", "global"] = Field(
        default="global",
        description="Regulatory region to pull dosing guidelines from.",
    )


@tool(args_schema=DrugLookupArgs, version="2.1.0")
def drug_reference(drug_name: str, country: str = "global") -> dict:
    """Look up drug information by region."""
    ...
```

## 7. `ToolAdapter` abstraction and registry

### 7.1 `ToolAdapter` ABC

Defined in `src/healthbench_agent/tools/tool_adapter.py`:

```python
"""Tool adapter abstraction — translates a :class:`Tool` into a framework-native tool.

Defines the narrow contract every framework adapter implements. One
adapter per framework lives under :mod:`healthbench_agent.tools.adapters`
and self-registers via ``@register_tool_adapter``.

The contract is intentionally tiny: a single ``convert(tool) -> Any``
method. The return type is ``Any`` because each framework's native
tool type is different (``google.adk.tools.BaseTool``,
``pydantic_ai.Tool``, ``langchain_core.tools.BaseTool``,
``crewai.tools.BaseTool``) and the abstraction deliberately avoids
unifying them — pipeline-level adapters know which framework they
target and consume the concrete type directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tool import Tool


def require_optional(module: object | None, adapter_name: str) -> None:
    """Raise a helpful ImportError if an optional dependency is missing.

    Adapters lazy-import their backend (``pydantic_ai``, ``langchain_core``,
    ``crewai``) so the rest of the package keeps working without the
    ``tool-adapters`` extra. This helper centralises the error message
    so every adapter points users to the same install command.

    Args:
        module: The optionally-imported module, or ``None`` if the
            import failed.
        adapter_name: User-facing name of the adapter requesting the
            dependency.

    Raises:
        ImportError: If ``module`` is ``None``.
    """
    if module is None:
        raise ImportError(
            f"{adapter_name} requires the 'tool-adapters' extra. "
            "Install with: uv sync --extra tool-adapters"
        )


class ToolAdapter(ABC):
    """Abstract base for tool adapters.

    Subclasses translate a framework-agnostic :class:`Tool` into the
    concrete tool type their target framework expects. Each adapter is
    instantiated fresh by :func:`get_tool_adapter`; adapters are
    stateless by convention.
    """

    @abstractmethod
    def convert(self, tool: Tool) -> Any:
        """Translate a :class:`Tool` into a framework-native tool object.

        Args:
            tool: The framework-agnostic Tool to convert.

        Returns:
            A framework-native tool object. The concrete type depends
            on the adapter (e.g. ``google.adk.tools.BaseTool``,
            ``pydantic_ai.Tool``, ``langchain_core.tools.BaseTool``,
            ``crewai.tools.BaseTool``).

        Raises:
            ImportError: If the adapter's backend is not installed.
        """
```

### 7.2 Tool adapter registry

Defined in `src/healthbench_agent/tools/tool_adapter_registry.py` — an exact
shape mirror of
[src/healthbench_agent/prompt_optimization/optimizer_registry.py](../../../src/healthbench_agent/prompt_optimization/optimizer_registry.py):

```python
"""Registry for tool adapters.

Maps framework names to :class:`ToolAdapter` subclasses. Adding a new
framework requires only a new adapter file under ``tools/adapters/``
with ``@register_tool_adapter`` — no changes to this module (Open/Closed).
"""

from __future__ import annotations

from typing import Any

from .tool_adapter import ToolAdapter

_ADAPTER_REGISTRY: dict[str, type[ToolAdapter]] = {}


def register_tool_adapter(name: str) -> Any:
    """Class decorator that registers a :class:`ToolAdapter` subclass by name.

    Args:
        name: Unique identifier for the framework (``"adk"``,
            ``"pydantic_ai"``, ``"langchain"``, ``"crewai"``).

    Returns:
        The original class, unchanged.

    Raises:
        ValueError: If an adapter with the same name is already registered.
    """

    def decorator(cls: type[ToolAdapter]) -> type[ToolAdapter]:
        if name in _ADAPTER_REGISTRY:
            existing = _ADAPTER_REGISTRY[name]
            raise ValueError(
                f"Tool adapter '{name}' is already registered "
                f"(existing: {existing.__module__}.{existing.__name__})"
            )
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_tool_adapter(name: str) -> ToolAdapter:
    """Instantiate and return the adapter registered under ``name``.

    Args:
        name: Registered adapter name.

    Returns:
        A fresh :class:`ToolAdapter` instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY) or ["(none)"]
        raise ValueError(f"Unknown tool adapter: {name!r}. Available: {available}")
    return _ADAPTER_REGISTRY[name]()


def registered_tool_adapters() -> dict[str, type[ToolAdapter]]:
    """Return a copy of the current tool-adapter registry.

    Returns:
        Dict mapping adapter names to their :class:`ToolAdapter` subclasses.
    """
    return dict(_ADAPTER_REGISTRY)
```

### 7.3 `adapters/__init__.py`

```python
"""Tool adapters.

Importing this package triggers registration of all built-in adapters.
"""

from . import (
    adk_adapter,  # noqa: F401
    crewai_adapter,  # noqa: F401
    langchain_adapter,  # noqa: F401
    pydantic_ai_adapter,  # noqa: F401
)
```

Imported for side effects from `tools/__init__.py`.

### 7.4 Pipeline-adapter integration

The pipeline adapter at
[src/healthbench_agent/agent/adapters/adk_adapter.py](../../../src/healthbench_agent/agent/adapters/adk_adapter.py)
changes in two places:

1. **Line 38 imports** — swap
   `from healthbench_agent.agent.tool_registry import get_tools` for
   `from healthbench_agent.tools import get_tools` and add
   `from healthbench_agent.tools.tool_adapter_registry import get_tool_adapter`.
2. **Line 322 body** — from:

   ```python
   tools = get_tools(config.tools) if config.tools else []
   ```

   to:

   ```python
   if config.tools:
       adapter = get_tool_adapter("adk")
       tools = [adapter.convert(t) for t in get_tools(config.tools)]
   else:
       tools = []
   ```

The pipeline adapter never imports `ADKToolAdapter` directly — it looks it
up by name through the registry. This is the DIP boundary: `agent/` depends
on `tools/` through the abstract registry, not on a concrete class.

`get_tool_adapter` returns a fresh instance on every call, matching the
`prompt_optimization` `create_prompt_optimizer(config)` pattern where the
registry stores the class and instantiation is deferred. Adapters are
stateless by convention, so instantiation is cheap and fresh-per-call
avoids any accidental cross-pipeline state sharing.

## 8. Adapter implementations

Each adapter is one file under `src/healthbench_agent/tools/adapters/`,
self-registers via `@register_tool_adapter("name")`, and implements
`convert(tool: Tool) -> <framework_native_type>`.

### 8.1 `ADKToolAdapter` → `adapters/adk_adapter.py`

Registered as `"adk"`. ADK is a **required** project dependency; no
`require_optional` guard, no `try/except ImportError` at module load.

**Conversion logic:**

- **When `tool.args_schema is None`** — build a thin shim callable via
  `functools.wraps(tool.func)` and overwrite its `__name__` with
  `tool.name` and `__doc__` with `tool.description`. Pass the shim to
  `google.adk.tools.FunctionTool(func=shim)`. ADK's native introspection
  reads name and description from the callable's dunders; the shim is the
  minimum wrapping needed to honour author-supplied overrides.
- **When `tool.args_schema is not None`** — build a concrete `BaseTool`
  subclass on-the-fly that overrides `_get_declaration()` to return a
  `FunctionDeclaration` built from `tool.args_schema.model_json_schema()`,
  and `run_async()` to invoke `tool.func` with keyword arguments after
  validating the LLM's raw input through `tool.args_schema.model_validate(args)`.
  This gives full Pydantic validation before the callable runs — ADK alone
  cannot do this out of the box.

**Sync/async handling.** ADK detects `inspect.iscoroutinefunction(target)`
at call time inside `_invoke_callable`, so both sync and async functions
work with zero extra adapter code in the `args_schema is None` branch.
In the override branch, `run_async()` does the same detection and either
awaits or calls the callable directly.

### 8.2 `PydanticAIToolAdapter` → `adapters/pydantic_ai_adapter.py`

Registered as `"pydantic_ai"`. Optional dependency.

**Module preamble:**

```python
try:
    import pydantic_ai
except ImportError:
    pydantic_ai = None  # type: ignore[assignment]
```

`convert()` calls `require_optional(pydantic_ai, "PydanticAIToolAdapter")`
before any work.

**Conversion logic:**

- Construct `pydantic_ai.Tool(tool.func, name=tool.name, description=tool.description)`.
  PydanticAI's `Tool` dataclass accepts all three as kwargs and auto-detects
  `async def` via signature inspection — no branching needed.
- **When `tool.args_schema is not None`** — wrap `tool.func` in a thin
  adapter that accepts a single parameter typed as the Pydantic model, and
  let PydanticAI flatten it via its built-in single-`BaseModel`-arg
  handling. This skips the need for a custom `function_schema` and produces
  the same object-shaped schema the LLM expects.

### 8.3 `LangChainToolAdapter` → `adapters/langchain_adapter.py`

Registered as `"langchain"`. Optional dependency.

**Module preamble:**

```python
try:
    from langchain_core.tools import StructuredTool
except ImportError:
    StructuredTool = None  # type: ignore[assignment]
```

`convert()` calls `require_optional(StructuredTool, "LangChainToolAdapter")`
first.

**Conversion logic.** This adapter exercises the sync/async split most
directly — LangChain models `func` and `coroutine` as separate fields on
`StructuredTool`:

- **Sync callable** — `StructuredTool.from_function(func=tool.func, coroutine=None, name=tool.name, description=tool.description, args_schema=tool.args_schema, infer_schema=tool.args_schema is None)`.
- **Async callable** (detected via `inspect.iscoroutinefunction`) —
  `StructuredTool.from_function(func=None, coroutine=tool.func, name=tool.name, description=tool.description, args_schema=tool.args_schema, infer_schema=tool.args_schema is None)`.

LangChain's `args_schema` natively accepts a Pydantic `BaseModel` — direct
pass-through, no synthesis. When `args_schema is None`, `infer_schema=True`
tells LangChain to infer from `tool.func`'s type hints via its own machinery.

Do not set `parse_docstring=True` — our `Tool.description` is authoritative,
and asking LangChain to re-parse would risk conflicts.

### 8.4 `CrewAIToolAdapter` → `adapters/crewai_adapter.py`

Registered as `"crewai"`. Optional dependency. The most translation work of
the four adapters.

**Module preamble:**

```python
try:
    import crewai.tools as _crewai_tools
except ImportError:
    _crewai_tools = None  # type: ignore[assignment]
```

`convert()` calls `require_optional(_crewai_tools, "CrewAIToolAdapter")`
first.

**Conversion logic.** CrewAI's concrete `crewai.tools.Tool` subclass
requires `name`, `description`, `args_schema` (Pydantic model, **required**),
and `func`. No inference path — the adapter must always supply an
`args_schema`:

- **When `tool.args_schema is not None`** — pass through directly.
- **When `tool.args_schema is None`** — synthesise a Pydantic model from
  `tool.func`'s signature via a module-level helper
  `_synthesise_args_schema(func) -> type[BaseModel]`:
  - Iterate `inspect.signature(func).parameters`.
  - Skip `*args` / `**kwargs`.
  - For each remaining parameter, read its type hint from
    `typing.get_type_hints(func)` (so `from __future__ import annotations`
    resolves correctly) and its default from the signature.
  - Build `{param_name: (annotation, default_or_...)}` and pass to
    `pydantic.create_model(f"{tool.name}Args", **fields)`.
  - If a parameter has no annotation, default its type to `Any` and log
    a warning via the module logger — CrewAI cannot enforce anything on
    `Any`, but it is better than the whole adapter failing.

**Async handling.** `tool.func` may be async. CrewAI's base
`BaseTool.run()` handles async by calling `asyncio.run(result)` when `_run`
returns a coroutine, but that spawns a new event loop and is incorrect
when the caller is already inside one. The adapter builds a concrete
`BaseTool` subclass that overrides both `_run` (sync dispatch) and
`_arun` (async dispatch), choosing based on
`inspect.iscoroutinefunction(tool.func)`. This complexity stays inside
the adapter.

**Description-mutation quirk.** CrewAI's `model_post_init` rewrites
`self.description` to include the schema JSON prepended. Our code never
reads `.description` back from a CrewAI `Tool`, so the mutation is
harmless for our purposes.

### 8.5 Common properties across all four adapters

- Each file is ~60–100 lines including docstring, imports, the
  optional-dep guard, and the concrete class.
- Every adapter's `convert()` is the only method; adapters are stateless.
- None of the adapters read `tool.func.__name__` or `tool.func.__doc__`
  directly — they consume the explicit `tool.name` and `tool.description`
  from the Pydantic model and pass them through. The ADK
  `args_schema is None` path uses `functools.wraps` to transfer those
  explicit values onto the callable's dunders so ADK's native
  introspection picks them up.

## 9. CLAUDE.md updates

Three updates to [CLAUDE.md](../../../CLAUDE.md):

### 9.1 New "Optional extras in CI" convention

Added as a new bullet under **Key Conventions**, right after
`Prompt Versioning`:

> **Optional extras in CI.** Any pyproject `[project.optional-dependencies]`
> extra that wraps a framework or backend MUST be installed in CI.
> `prompt_optimization` installs `--extra optimization` (DSPy, TextGrad).
> `tools/adapters` installs `--extra tool-adapters` (PydanticAI, LangChain,
> CrewAI). New extras follow the same rule — do not hide behind
> `pytest.importorskip` alone, because adapters then silently regress. If
> the extra is too heavy for every CI run, split CI into a core job plus
> an extras job, but never skip coverage entirely.

### 9.2 Project-layout update

The **Project Layout** section gains a new `tools/` subpackage entry
alongside the existing `agent/`, `llm_eval/`, `prompt_optimization/`
entries:

```
  - `tools/` — framework-agnostic tool abstraction (→ domain)
    - `__init__.py` — re-exports `Tool`, `tool`, `get_tool`, `get_tools`,
      `registered_tools`, `ToolAdapter`, `register_tool_adapter`,
      `get_tool_adapter`, `registered_tool_adapters`
    - `tool.py` — `Tool` (frozen Pydantic model: name, description, func, args_schema, version, source, source_hash; validators enforce identifier names, semver versions, and source/hash consistency)
    - `registry.py` — `@tool` decorator (captures source + hash, accepts version kwarg), `_insert`, `get_tool`, `get_tools`, `registered_tools`
    - `tool_adapter.py` — `ToolAdapter` ABC with `convert(tool) -> Any`, plus `require_optional`
    - `tool_adapter_registry.py` — `@register_tool_adapter` decorator, `get_tool_adapter`, `registered_tool_adapters`
    - `adapters/` — per-framework adapter implementations (lazy-imported)
      - `adk_adapter.py` — `ADKToolAdapter` (→ `google.adk.tools.BaseTool`)
      - `pydantic_ai_adapter.py` — `PydanticAIToolAdapter` (→ `pydantic_ai.Tool`)
      - `langchain_adapter.py` — `LangChainToolAdapter` (→ `langchain_core.tools.BaseTool`)
      - `crewai_adapter.py` — `CrewAIToolAdapter` (→ `crewai.tools.BaseTool`)
```

The existing `agent/tool_registry.py` entry is **removed** (the file is
deleted). The `agent/` section's `__init__.py` re-export list drops
`register_tool`, `get_tool`, `get_tools`, `registered_tools` — those
symbols now live under `healthbench_agent.tools`.

### 9.3 Architecture note

Added under **Architecture** right after the three agent descriptions:

> **Tool abstraction.** All three agent architectures consume tools
> through the framework-agnostic `healthbench_agent.tools` subpackage.
> Authors define a tool with `@tool` on a plain function (inferring
> name/description from `__name__` and `inspect.getdoc`, and capturing
> the function's source + SHA-256 for reproducibility); the registry
> stores a frozen `Tool` Pydantic model with validators enforcing
> identifier names and semver versions; the pipeline adapter (currently
> only ADK) resolves the registered names to `Tool` instances and runs
> each through `get_tool_adapter("<framework>").convert(tool)` to
> obtain the framework-native tool type. Tool versions follow the same
> dotted MAJOR.MINOR.PATCH convention as `prompt_version` in
> `config/agents/*.yaml` and should be logged to MLflow alongside the
> prompt version for reproducible experiments. Four framework adapters
> ship in `tools/adapters/` — `adk`, `pydantic_ai`, `langchain`,
> `crewai` — and adding a fifth is one new file under `adapters/` with
> `@register_tool_adapter`, no other code touches.

## 10. Testing strategy

### 10.1 Registry isolation via autouse fixtures

Both registries live in module-level dicts that must be cleaned between
tests. `tests/tools/conftest.py` adds two autouse fixtures:

```python
@pytest.fixture(autouse=True)
def clean_tool_registry(monkeypatch):
    """Isolate each test with a fresh tool registry."""
    from healthbench_agent.tools import registry
    monkeypatch.setattr(registry, "_REGISTRY", {})


@pytest.fixture(autouse=True)
def clean_tool_adapter_registry(monkeypatch):
    """Isolate each test with a fresh tool-adapter registry.

    Adapter tests must re-import the adapter module after this fixture
    clears the dict, so the ``@register_tool_adapter`` decorators run
    again against the empty registry. Each adapter test file includes
    a small ``_register_adapter()`` helper that reimports its target
    adapter module.
    """
    from healthbench_agent.tools import tool_adapter_registry
    monkeypatch.setattr(tool_adapter_registry, "_ADAPTER_REGISTRY", {})
```

This pattern is already used in
[tests/prompt_optimization/test_optimizer_registry.py](../../../tests/prompt_optimization/test_optimizer_registry.py)
— no new testing infrastructure needed.

### 10.2 Test inventory

**`test_tool.py` (9 tests)** — Pydantic model contract:

- `test_tool_is_frozen` — mutating a field on an instance raises
  `pydantic.ValidationError` (not `FrozenInstanceError` — Pydantic
  raises its own exception type for frozen-model violations).
- `test_tool_requires_three_fields` — omitting `name`, `description`,
  or `func` raises `pydantic.ValidationError`.
- `test_tool_accepts_sync_and_async_func` (parametrised).
- `test_tool_args_schema_defaults_to_none`.
- `test_tool_name_must_be_identifier` — `Tool(name="has space", ...)`
  raises `pydantic.ValidationError` with the identifier message.
- `test_tool_name_empty_string_raises` — `min_length=1` catches
  empty string separately from the identifier check.
- `test_tool_version_defaults_to_1_0_0` — confirm default.
- `test_tool_version_must_be_semver_like` — `Tool(version="v1", ...)`,
  `Tool(version="1.0", ...)` both raise `pydantic.ValidationError`.
- `test_tool_source_hash_must_match_source` — constructing a Tool
  with mismatched `source`/`source_hash` fields manually raises
  `pydantic.ValidationError`.

**`test_registry.py` (16 tests)** — `@tool` decorator behavior:

- `test_bare_decorator_registers_with_inferred_name_and_description`
- `test_bare_decorator_returns_original_function`
- `test_parametrised_decorator_overrides_name`
- `test_parametrised_decorator_overrides_description`
- `test_parametrised_decorator_accepts_args_schema`
- `test_decorator_inherits_partial_overrides`
- `test_decorator_raises_on_empty_description`
- `test_decorator_raises_on_duplicate_name`
- `test_get_tool_raises_on_unknown_name`
- `test_get_tools_preserves_order`
- `test_get_tools_raises_on_any_unknown_name`
- `test_registered_tools_returns_copy`
- `test_decorator_captures_function_source` — decorate a function
  defined on disk, assert `get_tool("...").source` contains the
  function's source text.
- `test_decorator_computes_source_hash` — confirm `source_hash` is
  16 hex chars and equals the truncated SHA-256 of `source`.
- `test_decorator_accepts_version_kwarg_with_default` — bare
  `@tool` produces `version == "1.0.0"`; `@tool(version="2.1.0")`
  propagates to the registered Tool.
- `test_decorator_handles_unavailable_source` — wrap a `lambda` or
  a builtin (where `inspect.getsource` raises `OSError`/`TypeError`)
  and assert `source == ""` and `source_hash == ""` rather than a
  decoration-time crash.

**`test_tool_adapter.py` (4 tests)** — ABC + `require_optional`:

- `test_cannot_instantiate_abc`
- `test_concrete_subclass_must_implement_convert`
- `test_require_optional_passes_on_non_none_module`
- `test_require_optional_raises_on_none_with_install_hint`

**`test_tool_adapter_registry.py` (5 tests):**

- `test_register_and_get`
- `test_get_tool_adapter_returns_fresh_instance`
- `test_register_raises_on_duplicate_name`
- `test_get_raises_on_unknown_name`
- `test_registered_tool_adapters_returns_copy`

**`test_adk_adapter.py` (6 tests, no skip)** — exercises all three medical
tools:

- `test_convert_inference_path_wraps_three_medical_tools`
- `test_convert_infers_schema_from_type_hints_on_shim`
- `test_convert_override_path_uses_pydantic_model`
- `test_convert_override_path_validates_input_through_pydantic`
- `test_convert_async_function`
- `test_shim_propagates_name_and_description_to_dunders`

**`test_pydantic_ai_adapter.py` (5 tests, importorskip):**

- `test_convert_bare_function`
- `test_convert_with_override_propagates_name_and_description`
- `test_convert_with_pydantic_args_schema_flattens_single_arg_path`
- `test_convert_async_function`
- `test_require_optional_raises_clean_error`

**`test_langchain_adapter.py` (6 tests, importorskip):**

- `test_convert_sync_function_uses_func_slot`
- `test_convert_async_function_uses_coroutine_slot`
- `test_convert_with_override_propagates_name_and_description`
- `test_convert_with_args_schema_passes_through_directly`
- `test_convert_without_args_schema_infers_from_type_hints`
- `test_require_optional_raises_clean_error`

**`test_crewai_adapter.py` (7 tests, importorskip):**

- `test_convert_synthesises_args_schema_from_type_hints`
- `test_convert_passes_through_explicit_args_schema`
- `test_synthesis_handles_untyped_parameter_as_any_with_warning`
- `test_convert_async_function_routes_to_arun`
- `test_convert_sync_function_routes_to_run`
- `test_convert_medical_tool_round_trip`
- `test_require_optional_raises_clean_error`

**Total new test count: 58.** (9 up from 4 in `test_tool.py` for the
Pydantic validators, 16 up from 12 in `test_registry.py` for source
capture and version kwarg — everything else unchanged.)

### 10.3 Pipeline integration test

One new test added to
[tests/agents/test_tool_agent.py](../../../tests/agents/test_tool_agent.py):

- `test_tool_agent_pipeline_uses_tool_adapter_registry` — construct a
  `RootAgentPipelineConfig` loading
  [config/agents/tool_agent.yaml](../../../config/agents/tool_agent.yaml),
  call `create_pipeline(config)`, inspect the resulting ADK `LlmAgent`'s
  `.tools` attribute, assert all three tools are present as
  `google.adk.tools.BaseTool` instances with the correct names. Confirms
  that the pipeline adapter routes through `get_tool_adapter("adk")`
  rather than passing bare callables.

The existing behavioral tests for the three medical tool functions at
[tests/agents/test_tool_agent.py](../../../tests/agents/test_tool_agent.py)
lines 143–371 stay untouched — they test what the tools *do*, which is
orthogonal to the abstraction. They need only the two-line import update
at the top of the file.

### 10.4 Coverage targets

Per [CLAUDE.md](../../../CLAUDE.md) the project-wide floor is 80% per
module, with 100% expected on pure-Python functional modules:

- `tools/tool.py` — **100%**
- `tools/registry.py` — **100%**
- `tools/tool_adapter.py` — **100%**
- `tools/tool_adapter_registry.py` — **100%**
- `tools/adapters/adk_adapter.py` — **≥90%**
- `tools/adapters/pydantic_ai_adapter.py` — **≥85%**
- `tools/adapters/langchain_adapter.py` — **≥85%**
- `tools/adapters/crewai_adapter.py` — **≥85%**

### 10.5 CI matrix

[.github/workflows/ci.yml](../../../.github/workflows/ci.yml) changes one
line: `uv sync` → `uv sync --extra optimization --extra tool-adapters`.
Both extras are installed on every CI run. This enforces the
"Optional extras in CI" convention added to CLAUDE.md in Section 9.1.

## 11. Implementation order

Twelve steps, ordered so the tree stays green at every checkpoint. Release
points are marked.

1. **New subpackage skeleton.** Create empty
   `src/healthbench_agent/tools/__init__.py`,
   `tests/tools/__init__.py`, `tests/tools/adapters/__init__.py`. No
   re-exports yet — just package markers.
2. **`Tool` Pydantic model.** Implement `tools/tool.py` with the seven
   fields (`name`, `description`, `func`, `args_schema`, `version`,
   `source`, `source_hash`) and the three validators (identifier check
   on `name`, semver check on `version`, source/hash consistency).
   Re-export `Tool` from `tools/__init__.py`; add
   `tests/tools/test_tool.py` (9 tests).
3. **`@tool` decorator + accessors.** Implement `tools/registry.py`,
   including `_capture_source` and the `version` kwarg on `@tool`;
   re-export `tool`, `get_tool`, `get_tools`, `registered_tools`; add
   `tests/tools/conftest.py` (`clean_tool_registry` fixture); add
   `tests/tools/test_registry.py` (16 tests).
4. **`ToolAdapter` ABC + `require_optional`.** Implement
   `tools/tool_adapter.py`; re-export `ToolAdapter` and `require_optional`;
   add `tests/tools/test_tool_adapter.py` (4 tests).
5. **Tool adapter registry.** Implement `tools/tool_adapter_registry.py`;
   re-export all three registry symbols; extend `conftest.py` with
   `clean_tool_adapter_registry`; add
   `tests/tools/test_tool_adapter_registry.py` (5 tests).
6. **`ADKToolAdapter` + pipeline wiring** — **first release checkpoint.**
   Implement `tools/adapters/__init__.py` (with only `adk_adapter`
   imported) and `tools/adapters/adk_adapter.py`. Add
   `from . import adapters  # noqa: F401` at the bottom of
   `tools/__init__.py` so the side-effect registration runs on package
   load. Change
   [agent/adapters/adk_adapter.py:38](../../../src/healthbench_agent/agent/adapters/adk_adapter.py#L38)
   and
   [line 322](../../../src/healthbench_agent/agent/adapters/adk_adapter.py#L322)
   to route through the new registry. Add
   `tests/tools/adapters/test_adk_adapter.py` (6 tests). The old
   `agent/tool_registry.py` still exists at this checkpoint; deletion
   comes in step 11.
7. **`PydanticAIToolAdapter`.** Add `pydantic-ai>=0.0.14` to the
   `tool-adapters` extra in [pyproject.toml](../../../pyproject.toml); run
   `uv sync --extra tool-adapters`; implement
   `tools/adapters/pydantic_ai_adapter.py`; add to `adapters/__init__.py`;
   add `tests/tools/adapters/test_pydantic_ai_adapter.py` (5 tests).
8. **`LangChainToolAdapter`.** Add `langchain-core>=0.3.0`;
   `uv sync --extra tool-adapters`; implement
   `tools/adapters/langchain_adapter.py`; add to `adapters/__init__.py`;
   add `tests/tools/adapters/test_langchain_adapter.py` (6 tests).
9. **`CrewAIToolAdapter`.** Add `crewai>=0.80.0`;
   `uv sync --extra tool-adapters`; implement
   `tools/adapters/crewai_adapter.py` with `_synthesise_args_schema` and
   the async `BaseTool` subclass; add to `adapters/__init__.py`; add
   `tests/tools/adapters/test_crewai_adapter.py` (7 tests). **Second
   release checkpoint** — all four adapters live, abstraction proven
   portable.
10. **Medical tool refactor.** Two-line diff in each of
    [tools/drug_reference.py](../../../tools/drug_reference.py),
    [tools/symptom_checker.py](../../../tools/symptom_checker.py),
    [tools/emergency_flag.py](../../../tools/emergency_flag.py): swap
    `from healthbench_agent.agent import register_tool` → `from healthbench_agent.tools import tool`;
    swap `@register_tool("<name>")` → bare `@tool`. Update
    [tools/tools.py:6](../../../tools/tools.py#L6) and
    [tests/agents/test_tool_agent.py:16](../../../tests/agents/test_tool_agent.py#L16)
    imports. Add the pipeline integration test described in Section 10.3.
11. **Delete the old registry.** Delete
    [src/healthbench_agent/agent/tool_registry.py](../../../src/healthbench_agent/agent/tool_registry.py).
    Remove stale re-exports from
    [src/healthbench_agent/agent/__init__.py](../../../src/healthbench_agent/agent/__init__.py).
    Run `uv run ruff check .` and `uv run mypy .` to confirm no stale
    imports remain.
12. **CI workflow and documentation** — **final release checkpoint.**
    Change [.github/workflows/ci.yml](../../../.github/workflows/ci.yml)
    to add `--extra tool-adapters`. Apply the three
    [CLAUDE.md](../../../CLAUDE.md) updates from Section 9.

**File touch count:** 11 new files under `src/`, 9 new files under
`tests/`, 1 deleted file, 8 touched files (3 medical tool modules,
`tools/tools.py`, `agent/adapters/adk_adapter.py`, `agent/__init__.py`,
`tests/agents/test_tool_agent.py`, `pyproject.toml`,
`.github/workflows/ci.yml`, `CLAUDE.md`). Total PR diff: roughly 30 files.

## 12. SOLID check

- **SRP.** `Tool` holds data and its own validation invariants (via
  Pydantic validators) — one reason to change: the set of fields a
  Tool holds. `registry.py` owns name→Tool mapping.
  `tool_adapter.py` defines the one-method contract.
  `tool_adapter_registry.py` owns name→class mapping. Each adapter
  file knows exactly one framework.
- **OCP.** Adding a fifth framework is one new file under
  `tools/adapters/` with `@register_tool_adapter`, plus one line in
  `adapters/__init__.py`. Zero touches to the ABC, either registry, or
  existing adapters. Adding a new optional field to `Tool`
  (`metadata`, `return_direct`, `context_param`, ...) is a non-breaking
  extension — existing consumers ignore fields they do not know.
- **LSP.** No `Tool` hierarchy. The one ABC is `ToolAdapter` with a
  single `convert(tool) -> Any` contract that every subclass honours.
- **ISP.** Authors see the `@tool` decorator. Adapters see the `Tool`
  Pydantic model and the `ToolAdapter` ABC. Pipeline code sees the
  adapter registry. No consumer pays for a path it does not use.
- **DIP.** `healthbench_agent/tools/` has zero imports from
  `healthbench_agent/agent/`. The pipeline adapter in
  `agent/adapters/adk_adapter.py` depends on the abstract registry
  (`get_tool_adapter("adk")`), not on the concrete `ADKToolAdapter`
  class. Framework imports live only inside the adapter files that
  need them.
- **Pydantic validation as invariants.** Converting `Tool` from a
  `@dataclass(frozen=True)` to a `BaseModel` with `frozen=True` and
  field validators moves three previously-convention-only invariants
  — identifier-style names, non-empty descriptions, semver-style
  versions, and source/hash consistency — into the type itself. Any
  construction path (decorator, test fixture, future programmatic
  builder) that violates these raises `pydantic.ValidationError`
  before the object exists, not inside an adapter where the error
  would be confusing.
