"""Middleware for selecting a subset of tools per model call."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse


DEFAULT_ALWAYS_TOOL_NAMES = (
    "write_todos",
    "read_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "task",
)


@dataclass(frozen=True)
class ToolSelectorConfig:
    """Configuration for tool selection."""

    k: int = 7
    fallback_k: int = 20
    last_n_messages: int = 6
    always_tool_names: Sequence[str] = DEFAULT_ALWAYS_TOOL_NAMES
    include_tool_calls: bool = True
    include_system_prompt: bool = False
    include_schema_fields: bool = True
    allow_unindexed_tools: bool = True
    allow_empty_selection: bool = False


@dataclass(frozen=True)
class ToolDoc:
    name: str
    text: str


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _normalize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _tool_name(tool: Any) -> str | None:
    if hasattr(tool, "name"):
        return tool.name
    if isinstance(tool, dict):
        return tool.get("name")
    return None


def _tool_description(tool: Any) -> str:
    if hasattr(tool, "description"):
        return tool.description or ""
    if isinstance(tool, dict):
        return tool.get("description") or ""
    return ""


def _tool_schema_fields(tool: Any) -> list[str]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return []
    try:
        if hasattr(args_schema, "model_fields"):
            return list(args_schema.model_fields.keys())
        if hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
            return list(schema.get("properties", {}).keys())
        if hasattr(args_schema, "schema"):
            schema = args_schema.schema()
            return list(schema.get("properties", {}).keys())
    except Exception:
        return []
    return []


def _build_tool_text(tool: Any, include_schema_fields: bool) -> str:
    name = _tool_name(tool) or ""
    parts = [name, _tool_description(tool)]
    if include_schema_fields:
        parts.append(" ".join(_tool_schema_fields(tool)))
    return "\n".join(part for part in parts if part)


def _dedupe_by_name(tools: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for tool in tools:
        name = _tool_name(tool)
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(tool)
    return result


class TinyIndex:
    """Tiny in-memory index for tool name/description matching."""

    def __init__(self, docs: Sequence[ToolDoc]) -> None:
        self._tokens_by_name: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for doc in docs:
            tokens = set(_normalize(doc.text))
            self._tokens_by_name[doc.name] = tokens
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(len(docs), 1)
        self._idf = {tok: math.log((n_docs + 1) / (count + 1)) + 1.0 for tok, count in df.items()}

    def search(self, query: str, k: int, candidates: set[str] | None = None) -> list[str]:
        q_tokens = set(_normalize(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for name, tokens in self._tokens_by_name.items():
            if candidates is not None and name not in candidates:
                continue
            inter = q_tokens & tokens
            if not inter:
                continue
            score = sum(self._idf.get(tok, 0.0) for tok in inter) + 0.05 * len(inter)
            scored.append((score, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in scored[:k]]


class ToolSelectorMiddleware(AgentMiddleware):
    """Select a subset of tools per model call to reduce token usage."""

    def __init__(
        self,
        tools: Sequence[Any],
        config: ToolSelectorConfig | None = None,
        log_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or ToolSelectorConfig()
        self._log_fn = log_fn

        self._tool_by_name: dict[str, Any] = {}
        for tool in tools:
            name = _tool_name(tool)
            if name:
                self._tool_by_name[name] = tool

        self._always_names = set(self.config.always_tool_names)
        docs: list[ToolDoc] = []
        for name, tool in self._tool_by_name.items():
            if name in self._always_names:
                continue
            text = _build_tool_text(tool, include_schema_fields=self.config.include_schema_fields)
            docs.append(ToolDoc(name=name, text=text))
        self._index = TinyIndex(docs)

    def _build_query(self, request: ModelRequest) -> str:
        parts: list[str] = []
        if self.config.include_system_prompt and request.system_prompt:
            parts.append(request.system_prompt)

        messages = request.messages[-self.config.last_n_messages :] if self.config.last_n_messages > 0 else request.messages
        for message in messages:
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)

            if self.config.include_tool_calls:
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        if isinstance(call, dict):
                            name = call.get("name")
                        else:
                            name = getattr(call, "name", None)
                        if name:
                            parts.append(name)

        return "\n".join(parts)

    def _select_tools(self, request: ModelRequest) -> tuple[list[Any], dict[str, Any]]:
        available_by_name: dict[str, Any] = {}
        for tool in request.tools:
            name = _tool_name(tool)
            if name:
                available_by_name[name] = tool

        always_tools = [available_by_name[name] for name in self._always_names if name in available_by_name]
        unindexed_names = sorted(name for name in available_by_name if name not in self._tool_by_name)
        unindexed_tools = [available_by_name[name] for name in unindexed_names]

        candidate_names = [
            name for name in self._tool_by_name.keys() if name in available_by_name and name not in self._always_names
        ]

        query = self._build_query(request)
        selected_names = self._index.search(query, self.config.k, candidates=set(candidate_names))
        fallback_used = False
        if not selected_names:
            selected_names = candidate_names[: self.config.fallback_k]
            fallback_used = True

        picked_tools = [available_by_name[name] for name in selected_names if name in available_by_name]
        selected = always_tools + picked_tools
        if self.config.allow_unindexed_tools and not self.config.allow_empty_selection:
            selected.extend(unindexed_tools)
        selected_tools = _dedupe_by_name(selected)

        log_payload = {
            "query": query,
            "selected_names": [name for name in selected_names if name in available_by_name],
            "selected_tools": [name for name in (_tool_name(t) for t in selected_tools) if name],
            "fallback_used": fallback_used,
            "unindexed_tools": unindexed_names if self.config.allow_unindexed_tools else [],
        }
        return selected_tools, log_payload

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not request.tools:
            return handler(request)

        selected_tools, log_payload = self._select_tools(request)
        if self._log_fn is not None:
            self._log_fn(log_payload)

        if (selected_tools or self.config.allow_empty_selection) and len(selected_tools) < len(request.tools):
            request = request.override(tools=selected_tools)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not request.tools:
            return await handler(request)

        selected_tools, log_payload = self._select_tools(request)
        if self._log_fn is not None:
            self._log_fn(log_payload)

        if (selected_tools or self.config.allow_empty_selection) and len(selected_tools) < len(request.tools):
            request = request.override(tools=selected_tools)
        return await handler(request)
