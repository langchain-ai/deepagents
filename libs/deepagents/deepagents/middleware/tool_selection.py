"""Middleware for filtering the tool list sent to the model per turn.

## Overview

The tool list an agent is constructed with is normally sent to the model on
every call, regardless of whether the current turn touches any of it. With a
few MCP servers and skills wired in, that can easily be 25-30+ schemas on
every request. `ToolSelectionMiddleware` scores tools against the latest
user message and only sends the top-K most relevant ones (plus an
always-include set) to the model, while leaving the full tool set registered
and callable.

## Escape hatch

A `discover_tools` tool is injected alongside the middleware so the model is
never permanently blind to a tool that didn't make the cut. Calling it
searches the full tool registry and pins the best match into the
always-include set for the rest of the conversation.

## Usage

```python
from deepagents.middleware.tool_selection import ToolSelectionMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[...],
    middleware=[ToolSelectionMiddleware(top_k=15)],
)
```

Scoring defaults to zero-dependency lexical token overlap between the latest
`HumanMessage` and each tool's name + description. Pass an `embeddings:
Embeddings` instance for cosine-similarity scoring instead.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

# NOTE: this module intentionally does NOT use `from __future__ import annotations`.
# `discover_tools` below relies on the `runtime: ToolRuntime` parameter annotation
# being a real, live type at function-definition time -- the tool node's injection
# detection (`StructuredTool._injected_args_keys`) inspects `inspect.signature(fn)`
# directly and only recognizes injected params when the annotation is an actual
# class, not a postponed (stringified) one. Same reasoning as `async_subagents.py`
# and `subagents.py`, which also omit the future import for this reason.

DEFAULT_ALWAYS_INCLUDE: frozenset[str] = frozenset({"read_file", "write_file", "edit_file", "ls", "task"})
DISCOVER_TOOLS_NAME = "discover_tools"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

DISCOVER_TOOLS_DESCRIPTION = """Search the full tool registry for a tool relevant to a query.

The tool list you see each turn is filtered down to the most relevant tools --
use this when you suspect a tool exists that isn't currently visible to you.
A match found here stays available for the rest of the conversation.
"""


class DiscoverToolsSchema(BaseModel):
    """Arguments for the `discover_tools` escape-hatch tool."""

    query: str = Field(description="What you're trying to do, in plain language.")


class ToolSelectionState(AgentState):
    """State schema for `ToolSelectionMiddleware`.

    Attributes:
        tool_selection_pinned: Tool names pinned by `discover_tools`, bypassing
            the top-k relevance cutoff for the rest of the conversation.
    """

    tool_selection_pinned: NotRequired[Annotated[list[str], PrivateStateAttr]]


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """Extract tool name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _tool_description(tool: BaseTool | dict[str, Any]) -> str:
    """Extract tool description from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        description = tool.get("description", "")
        return description if isinstance(description, str) else ""
    description = getattr(tool, "description", "")
    return description if isinstance(description, str) else ""


def _tokenize(text: str) -> set[str]:
    """Lowercase, alphanumeric-only tokenization."""
    return {match.group(0) for match in _TOKEN_RE.finditer(text.lower())}


def _lexical_score(tool: BaseTool | dict[str, Any], query_tokens: set[str]) -> float:
    """Jaccard overlap between a tool's name + description tokens and the query tokens."""
    if not query_tokens:
        return 0.0
    tool_tokens = _tokenize(f"{_tool_name(tool) or ''} {_tool_description(tool)}")
    if not tool_tokens:
        return 0.0
    union = tool_tokens | query_tokens
    if not union:
        return 0.0
    return len(tool_tokens & query_tokens) / len(union)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _latest_human_message_text(messages: list[AnyMessage]) -> str:
    """Text of the most recent `HumanMessage`, or an empty string if there isn't one."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


class ToolSelectionMiddleware(AgentMiddleware[ToolSelectionState, ContextT, ResponseT]):
    """Filters the tool list sent to the model down to the top-K most relevant tools.

    Should be placed late in the middleware stack (after all tool-injecting
    middleware) so it scores and filters the fully assembled tool list, same
    placement requirement as `_ToolExclusionMiddleware`.

    Args:
        top_k: Maximum number of tools sent to the model per turn, beyond
            `always_include`. If the request already has `top_k` tools or
            fewer, filtering is skipped entirely.
        always_include: Tool names that are never filtered out.
        embeddings: Optional `Embeddings` instance for cosine-similarity
            scoring. Falls back to zero-dependency lexical token overlap.
    """

    state_schema = ToolSelectionState

    def __init__(
        self,
        *,
        top_k: int = 15,
        always_include: frozenset[str] = DEFAULT_ALWAYS_INCLUDE,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Initialize the tool selection middleware."""
        self._top_k = top_k
        self._always_include = always_include | {DISCOVER_TOOLS_NAME}
        self._embeddings = embeddings
        self._embedding_cache: dict[str, list[float]] = {}
        self.tools = [self._build_discover_tools_tool()]

    def _scorer(self, query: str) -> Callable[[BaseTool | dict[str, Any]], float]:
        """Build a per-turn scoring function, doing any one-time query prep up front."""
        if self._embeddings is not None:
            embeddings = self._embeddings
            query_vector = embeddings.embed_query(query)

            def _embedding_score(tool: BaseTool | dict[str, Any]) -> float:
                name = _tool_name(tool) or repr(tool)
                if name not in self._embedding_cache:
                    text = f"{_tool_name(tool) or ''}: {_tool_description(tool)}"
                    self._embedding_cache[name] = embeddings.embed_documents([text])[0]
                return _cosine_similarity(self._embedding_cache[name], query_vector)

            return _embedding_score

        query_tokens = _tokenize(query)
        return lambda tool: _lexical_score(tool, query_tokens)

    def _build_discover_tools_tool(self) -> BaseTool:
        """Build the `discover_tools` escape-hatch tool bound to this middleware instance."""

        def discover_tools(query: str, runtime: ToolRuntime) -> Command:
            scorer = self._scorer(query)
            candidates = [t for t in runtime.tools if _tool_name(t) != DISCOVER_TOOLS_NAME]
            match = max(candidates, key=scorer, default=None)
            pinned: list[str] = list(runtime.state.get("tool_selection_pinned") or [])
            if match is not None and scorer(match) > 0:
                name = _tool_name(match)
                if name is not None and name not in pinned:
                    pinned.append(name)
                content = f"Found tool `{name}`: {_tool_description(match)}\n\nIt is now available for the rest of this conversation."
            else:
                content = f"No tool found matching: {query!r}"
            return Command(
                update={
                    "tool_selection_pinned": pinned,
                    "messages": [ToolMessage(content, tool_call_id=runtime.tool_call_id)],
                }
            )

        return StructuredTool.from_function(
            name=DISCOVER_TOOLS_NAME,
            func=discover_tools,
            description=DISCOVER_TOOLS_DESCRIPTION,
            infer_schema=False,
            args_schema=DiscoverToolsSchema,
        )

    def _filter_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Filter `request.tools` down to the top-K most relevant, plus always-include."""
        if len(request.tools) <= self._top_k:
            return request

        pinned = set(request.state.get("tool_selection_pinned") or [])
        keep_names = self._always_include | pinned
        candidates = [t for t in request.tools if _tool_name(t) not in keep_names]

        query = _latest_human_message_text(request.messages)
        scorer = self._scorer(query)
        ranked = sorted(candidates, key=scorer, reverse=True)
        selected_names = {_tool_name(t) for t in ranked[: self._top_k]}

        filtered = [t for t in request.tools if _tool_name(t) in keep_names or _tool_name(t) in selected_names]
        if len(filtered) == len(request.tools):
            return request
        return request.override(tools=filtered)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Filter the tool list down to the top-K most relevant tools for this turn."""
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        return await handler(self._filter_request(request))
