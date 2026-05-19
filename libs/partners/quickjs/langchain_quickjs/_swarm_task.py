from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from deepagents._models import resolve_model
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel  # noqa: TC002
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

SwarmTaskMode = Literal["agent", "invoke"]

_VARIANT_TTL_S = 60.0


class SwarmSubAgent(BaseModel):
    """Subagent specification for swarm dispatch targets."""

    name: str
    """Identifier used to select this subagent in swarm dispatch calls."""

    description: str
    """Human-readable description of what this subagent does."""

    system_prompt: str
    """System prompt injected at the start of the subagent's conversation."""

    tools: list[BaseTool] = Field(default_factory=list)
    """Tools available to this subagent."""

    model: str | BaseChatModel | None = None
    """Model override for this subagent. Falls back to the tool's ``default_model``."""

    model_config = {"arbitrary_types_allowed": True}


def _is_schema_node(value: object) -> bool:
    """Check whether a value is a plain dict suitable for JSON Schema traversal."""
    return isinstance(value, dict)


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add ``additionalProperties: false`` to every object-typed node."""
    schema_type = schema.get("type")
    if schema_type not in ("object", "array"):
        return schema

    if schema_type == "array":
        result = {**schema}
        items = result.get("items")
        if isinstance(items, dict):
            result["items"] = normalize_schema(items)
        return result

    result = {**schema, "additionalProperties": False}
    props = schema.get("properties")
    if isinstance(props, dict):
        normalized: dict[str, Any] = {}
        for key, value in props.items():
            normalized[key] = (
                normalize_schema(value) if _is_schema_node(value) else value
            )
        result["properties"] = normalized
    return result


def _normalize_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a response schema and validate it has ``type: "object"``."""
    normalized = normalize_schema(schema)
    if normalized.get("type") != "object":
        got = json.dumps(normalized.get("type"))
        msg = f'response_schema must have type: "object", got: {got}'
        raise ValueError(msg)
    return normalized


class VariantCache:
    """TTL cache for compiled agent variants.

    Stores values keyed by string with a last-accessed timestamp. On every
    ``get_or_create`` call, entries that haven't been accessed within ``ttl_s``
    are swept first. Cache hits refresh the timestamp.
    """

    def __init__(self, ttl_s: float = _VARIANT_TTL_S) -> None:
        self._entries: dict[str, tuple[Any, float]] = {}
        self._ttl_s = ttl_s

    def get_or_create(self, key: str, factory: Any) -> Any:
        """Return a cached value or create one via ``factory`` on cache miss."""
        self._sweep()
        entry = self._entries.get(key)
        if entry is not None:
            value, _ = entry
            self._entries[key] = (value, time.monotonic())
            return value
        value = factory()
        self._entries[key] = (value, time.monotonic())
        return value

    @property
    def size(self) -> int:
        return len(self._entries)

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._entries.items() if now - ts > self._ttl_s]
        for k in expired:
            del self._entries[k]


def _content_to_string(content: Any) -> str:
    """Convert message content (string or content-block list) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(json.dumps(block))
        return "\n".join(parts)
    return json.dumps(content)


async def _invoke_model(
    model: str | BaseChatModel,
    description: str,
    response_schema: dict[str, Any] | None,
) -> str:
    """Direct model invocation — single LLM call with optional structured output."""
    resolved = resolve_model(model) if isinstance(model, str) else model
    messages = [HumanMessage(content=description)]

    if response_schema is not None:
        return await _invoke_with_structured_output(resolved, messages, response_schema)

    result = await resolved.ainvoke(messages)
    if isinstance(result, str):
        return result
    if isinstance(result, AIMessage):
        return _content_to_string(result.content)
    return json.dumps(result)


async def _invoke_with_structured_output(
    model: BaseChatModel,
    messages: list[HumanMessage],
    response_schema: dict[str, Any],
) -> str:
    """Bind a structured output tool to the model and extract the result."""
    normalized = _normalize_response_schema(response_schema)

    if not hasattr(model, "bind_tools"):
        msg = (
            "invoke mode with response_schema requires"
            " a model that supports bind_tools()."
        )
        raise ValueError(msg)

    tool_name = "structured_output"
    bound = model.bind_tools(
        [
            {
                "name": tool_name,
                "description": "Return the structured result.",
                "parameters": normalized,
            }
        ],
        tool_choice=tool_name,
    )
    response = await bound.ainvoke(messages)

    if not isinstance(response, AIMessage) or not response.tool_calls:
        msg = "invoke mode with response_schema: model did not return structured output"
        raise ValueError(msg)

    return json.dumps(response.tool_calls[0]["args"])


class _AgentSpec:
    """Minimal agent creation parameters preserved for recompilation."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str,
        tools: list[BaseTool],
        name: str,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.name = name


class _CompiledAgent:
    """Compiled agent alongside its creation spec."""

    def __init__(self, agent: Runnable, spec: _AgentSpec) -> None:
        self.agent = agent
        self.spec = spec


async def _invoke_agent(
    entry: _CompiledAgent,
    description: str,
    response_schema: dict[str, Any] | None,
    variant_cache: VariantCache,
) -> str:
    """Full agentic invocation with optional schema-constrained variant caching."""
    agent = entry.agent

    if response_schema is not None:
        normalized = _normalize_response_schema(response_schema)
        cache_key = f"{entry.spec.name}::{json.dumps(normalized, sort_keys=True)}"
        agent = variant_cache.get_or_create(
            cache_key,
            lambda: create_agent(
                model=entry.spec.model,
                system_prompt=entry.spec.system_prompt,
                tools=entry.spec.tools,
                name=entry.spec.name,
                response_format=normalized,
            ),
        )

    state = {"messages": [HumanMessage(content=description)]}
    result = await agent.ainvoke(state)

    if isinstance(result, dict):
        structured = result.get("structured_response")
        if structured is not None:
            return json.dumps(structured)

        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        if last is None:
            return "Task completed"
        content = last.content if hasattr(last, "content") else str(last)
        return _content_to_string(content)

    return str(result)


class _SwarmTaskInput(BaseModel):
    """Input schema for the swarm_task tool."""

    description: str = Field(
        description="The task to execute with the selected subagent."
    )
    subagent_type: str | None = Field(
        default=None,
        description="Name of the swarm subagent to use.",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema (type: 'object') for structured output.",
    )
    mode: SwarmTaskMode | None = Field(
        default=None,
        description=(
            'Dispatch mode. "agent" (default) runs a full'
            ' agentic loop. "invoke" makes a single model call.'
        ),
    )


def create_swarm_task_tool(
    *,
    subagents: list[SwarmSubAgent] | None = None,
    default_model: str | BaseChatModel,
) -> BaseTool:
    """Create a PTC-only tool for swarm subagent dispatch.

    The returned tool is designed to be passed into the CodeInterpreterMiddleware's
    ``ptc`` list. It is never exposed to the LLM — only callable from QuickJS skill
    code via ``tools.swarmTask()``.

    Supports two dispatch modes:

    - ``agent`` (default): Full agentic loop with tools. Schema-constrained
      agents are cached with a TTL.
    - ``invoke``: Direct model call with structured output. No tools, no iteration.

    Args:
        subagents: Subagent specifications for swarm dispatch targets.
        default_model: Default model used for subagents that don't specify their own,
            and for ``invoke`` mode direct model calls.

    Returns:
        A ``BaseTool`` suitable for the ``ptc`` config.
    """
    subs = subagents or []
    compiled: dict[str, _CompiledAgent] = {}

    for sub in subs:
        model = sub.model if sub.model is not None else default_model
        spec = _AgentSpec(
            model=model,
            system_prompt=sub.system_prompt,
            tools=sub.tools,
            name=sub.name,
        )
        agent = create_agent(
            model=model,
            system_prompt=sub.system_prompt,
            tools=sub.tools,
            name=sub.name,
        )
        compiled[sub.name] = _CompiledAgent(agent=agent, spec=spec)

    subagent_names = [s.name for s in subs]
    variant_cache = VariantCache()

    async def _run(
        description: str,
        subagent_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        mode: SwarmTaskMode | None = None,
    ) -> str:
        effective_mode = mode or "agent"

        if effective_mode == "invoke":
            return await _invoke_model(default_model, description, response_schema)

        if subagent_type is None:
            available = (
                ", ".join(subagent_names) if subagent_names else "(none configured)"
            )
            msg = f"agent mode requires subagent_type. Available: {available}"
            raise ValueError(msg)

        entry = compiled.get(subagent_type)
        if entry is None:
            available = ", ".join(subagent_names)
            msg = (
                f'Unknown swarm subagent type "{subagent_type}". Available: {available}'
            )
            raise ValueError(msg)

        return await _invoke_agent(entry, description, response_schema, variant_cache)

    return StructuredTool.from_function(
        name="swarm_task",
        description=(
            "Dispatch a task to a swarm subagent. Supports agent mode "
            "(full agentic loop) and invoke mode (direct model call)."
        ),
        coroutine=_run,
        args_schema=_SwarmTaskInput,
    )
