"""Shared helpers for baseline code-interpreter extensions."""

from __future__ import annotations

import asyncio
import json
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_quickjs._format import coerce_tool_output_for_ptc
from langchain_quickjs._repl import (
    _inject_tool_args_for_ptc,
    _synth_tool_call_id,
    _tool_uses_injected_tool_call_id,
)
from langchain_quickjs._swarm._task import _validate_response_schema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import BackendProtocol
    from langchain.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

BASELINE_RESERVED_GLOBALS: frozenset[str] = frozenset(
    {"subagent", "llm", "fs", "glob", "editFile"}
)

READ_LIMIT = 1_000_000
_UNKNOWN_SUBAGENT_MARKER = (
    "because it does not exist, the only allowed types are"
)


def error(code: str, msg: str) -> None:
    """Raise a normalized runtime error surfaced to the REPL."""
    error_msg = f"{code}: {msg}"
    raise RuntimeError(error_msg)


def require_dict(payload: Any, call: str) -> dict[str, Any]:
    """Require a dict payload."""
    if not isinstance(payload, dict):
        error("ERR_INVALID_ARG_TYPE", f"{call} expects an options object")
    return payload


def require_str(payload: dict[str, Any], key: str, call: str) -> str:
    """Require a string value in a payload."""
    value = payload.get(key)
    if not isinstance(value, str):
        error("ERR_INVALID_ARG_TYPE", f"{call}: `{key}` must be a string")
    return value


def reject_unknown_keys(
    payload: dict[str, Any], *, allowed: set[str], call: str
) -> None:
    """Reject unknown keys in a payload."""
    unknown = sorted(set(payload) - allowed)
    if unknown:
        error(
            "ERR_INVALID_ARG",
            f"{call} got unsupported keys: {', '.join(unknown)}",
        )


def normalize_abs_path(path: str, *, cwd: str = "/") -> str:
    """Resolve a path against cwd and normalize to an absolute POSIX path."""
    base = path if path.startswith("/") else posixpath.join(cwd, path)
    normalized = posixpath.normpath(base)
    return normalized if normalized.startswith("/") else "/" + normalized


def map_backend_error(message: str) -> str:
    """Map backend error strings to normalized codes."""
    lower = message.lower()
    if (
        "file_not_found" in lower
        or "not found" in lower
        or "enoent" in lower
        or "path_not_found" in lower
    ):
        return "ENOENT"
    if "already exists" in lower or "eexist" in lower:
        return "EEXIST"
    if "is_directory" in lower or "is a directory" in lower:
        return "EISDIR"
    if "permission_denied" in lower or "permission denied" in lower:
        return "EACCES"
    if "not supported" in lower or "not implemented" in lower:
        return "ENOTSUP"
    return "ERR_BACKEND"


def raise_backend_error(message: str, *, fallback_code: str = "ERR_BACKEND") -> None:
    """Raise a backend error with normalized code."""
    code = map_backend_error(message)
    error(code if code != "ERR_BACKEND" else fallback_code, message)


def is_missing_error(message: str) -> bool:
    """Return whether a backend error means the path was missing."""
    lower = message.lower()
    return (
        "file_not_found" in lower
        or "not found" in lower
        or "enoent" in lower
        or "path_not_found" in lower
    )


def parse_schema(value: Any, *, call: str) -> dict[str, Any] | None:
    """Validate optional JSON schema payload."""
    if value is None:
        return None
    if not isinstance(value, dict):
        error("ERR_SCHEMA_INVALID", f"{call}.responseSchema must be an object")
    _validate_response_schema(value)
    return value


def parse_structured_output(value: Any, *, surface: str) -> Any:
    """Parse structured tool output and throw on invalid payloads."""
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        error(
            "ERR_STRUCTURED_OUTPUT_INVALID",
            f"{surface} produced a non-JSON structured payload",
        )
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        error(
            "ERR_STRUCTURED_OUTPUT_INVALID",
            f"{surface} returned invalid JSON for structured output: {exc}",
        )
    unreachable_msg = "unreachable"
    raise AssertionError(unreachable_msg)


def is_unknown_subagent_type_message(value: str) -> bool:
    """Return whether a task response indicates an unknown subagent type."""
    return _UNKNOWN_SUBAGENT_MARKER in value.lower()


def coerce_file_text(value: Any) -> str:
    """Normalize backend file content into string form."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [part for part in value if isinstance(part, str)]
        if len(parts) == len(value):
            return "\n".join(parts)
    return str(value)


def find_tool(runtime: ToolRuntime, name: str) -> BaseTool | None:
    """Find a runtime tool by name."""
    tools = getattr(runtime, "tools", []) or []
    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    return None


async def invoke_runtime_tool(
    runtime: ToolRuntime,
    *,
    tool_name: str,
    payload: dict[str, Any],
) -> Any:
    """Invoke a runtime tool and coerce output for REPL marshalling."""
    tool = find_tool(runtime, tool_name)
    if tool is None:
        error(
            "ERR_NOT_SUPPORTED",
            f"required runtime tool `{tool_name}` is not available",
        )
    tool_call_id = _synth_tool_call_id(tool.name)
    args = _inject_tool_args_for_ptc(tool, payload, runtime, tool_call_id)
    result = await tool.arun(
        args,
        tool_call_id=tool_call_id if _tool_uses_injected_tool_call_id(tool) else None,
    )
    return coerce_tool_output_for_ptc(result)


async def with_timeout(
    awaitable: Awaitable[Any],
    *,
    timeout_s: float | None,
    surface: str,
) -> Any:
    """Apply an optional timeout to an awaitable."""
    if timeout_s is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError:
        error("ETIMEDOUT", f"{surface} call exceeded timeout ({timeout_s}s)")
    unreachable_msg = "unreachable"
    raise AssertionError(unreachable_msg)


@dataclass
class CapacityGuard:
    """Simple in-flight guard used per eval and per capability."""

    surface: str
    max_in_flight: int
    in_flight: int = 0

    async def run(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        if self.in_flight >= self.max_in_flight:
            error(
                "ERR_CAPACITY_EXCEEDED",
                f"{self.surface} max_in_flight={self.max_in_flight} exceeded",
            )
        self.in_flight += 1
        try:
            return await fn()
        finally:
            self.in_flight -= 1


def infer_model_from_runtime(runtime: ToolRuntime | None) -> str | None:  # noqa: C901
    """Infer effective model from runtime context/config mappings."""
    if runtime is None:
        return None

    model_keys = ("effective_model", "model", "default_model", "model_name")

    def _from_mapping(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        for key in model_keys:
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        configurable = value.get("configurable")
        if isinstance(configurable, dict):
            for key in model_keys:
                candidate = configurable.get(key)
                if isinstance(candidate, str):
                    return candidate
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            for key in model_keys:
                candidate = metadata.get(key)
                if isinstance(candidate, str):
                    return candidate
        return None

    def _from_object(value: Any) -> str | None:
        if value is None or isinstance(value, dict):
            return None
        for key in model_keys:
            candidate = getattr(value, key, None)
            if isinstance(candidate, str):
                return candidate
        return None

    context_value = getattr(runtime, "context", None)
    config_value = getattr(runtime, "config", None)
    return (
        _from_mapping(context_value)
        or _from_object(context_value)
        or _from_mapping(config_value)
        or _from_object(config_value)
    )


def require_backend(
    backend: BackendProtocol | None, *, surface: str
) -> BackendProtocol:
    """Require backend availability for a capability call."""
    if backend is None:
        error(
            "ERR_BACKEND_REQUIRED",
            f"{surface} requires a configured backend",
        )
    return backend

