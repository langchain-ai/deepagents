"""Opt-in OpenAI Responses native editing and local shell middleware."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse, TracePolicy, omit_payload
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.middleware._native_patch import execute_patch
from deepagents.middleware.filesystem import FilesystemMiddleware

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.language_models import BaseChatModel
    from langgraph.types import Command

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.filesystem import FilesystemPermission
    from deepagents.profiles.harness.harness_profiles import HarnessProfile


def _responses_compatible(model: BaseChatModel) -> bool:
    return (
        getattr(model, "supports_native_agent_tools", False) is True
        and any(cls.__module__.startswith("langchain_openai.") for cls in type(model).__mro__)
        and getattr(model, "use_responses_api", None) is not False
    )


def _name(tool: BaseTool | dict[str, Any]) -> str:
    return tool.name if isinstance(tool, BaseTool) else str(tool.get("name", tool.get("type", "")))


def _shell_action(action: object) -> tuple[list[str], float | None, int | None]:
    if not isinstance(action, Mapping) or set(action) - {"commands", "timeout_ms", "max_output_length"}:
        msg = "Expected a shell action with commands, timeout_ms, and max_output_length"
        raise ValueError(msg)
    commands = action.get("commands")
    if not isinstance(commands, list) or any(not isinstance(command, str) or not command.strip() for command in commands):
        msg = "commands must be a list of nonempty strings"
        raise ValueError(msg)
    timeout = _positive_integer(action.get("timeout_ms"), "timeout_ms")
    maximum = _positive_integer(action.get("max_output_length"), "max_output_length")
    return cast("list[str]", commands), timeout / 1000 if timeout is not None else None, maximum


def _positive_integer(value: object, name: str) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _shell_batch(backend: LocalShellBackend, action: object) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    commands, timeout, maximum = _shell_action(action)
    budget = timeout if timeout is not None else backend._default_timeout
    if not math.isfinite(budget) or budget <= 0:
        msg = "Shell timeout must be finite and positive"
        raise ValueError(msg)
    deadline = time.monotonic() + budget
    full: list[dict[str, Any]] = []
    for command in commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        response = backend.execute(command, timeout=remaining)
        if response.stdout is None or response.stderr is None:
            msg = "Native shell requires structured stdout and stderr from LocalShellBackend"
            raise RuntimeError(msg)
        outcome = {"type": "timeout"} if response.timed_out else {"type": "exit", "exit_code": response.exit_code}
        full.append({"stdout": response.stdout, "stderr": response.stderr, "outcome": outcome})
        if response.timed_out:
            break
    visible: list[dict[str, Any]] = []
    left = maximum
    for result in full:
        stdout, stderr = result["stdout"], result["stderr"]
        if left is not None:
            stdout = stdout[:left]
            left -= len(stdout)
            stderr = stderr[:left]
            left -= len(stderr)
        visible.append({**result, "stdout": stdout, "stderr": stderr})
    return visible, full


def _invalid_arguments(call_id: str, name: str, key: str) -> ToolMessage:
    message = f"Expected only {key} in native tool arguments"
    content = (
        {"type": "apply_patch_call_output", "status": "failed", "output": message}
        if name == "apply_patch"
        else {"type": "shell_call_output", "output": []}
    )
    return ToolMessage(content=[content], tool_call_id=call_id, name=name, status="error", artifact={"error": message, "native_tool_output": content})


class NativeToolsMiddleware(AgentMiddleware):
    """Execute opt-in native tools without bypassing backend mutation methods.

    !!! warning "Experimental"
        Native tool support requires an OpenAI Responses-compatible model.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        enable_apply_patch: bool = False,
        enable_shell: bool = False,
        permissions: list[FilesystemPermission] | None = None,
        custom_tool_names: frozenset[str] = frozenset(),
        excluded_tools: frozenset[str] = frozenset(),
    ) -> None:
        """Configure native tools while preserving caller-owned tool surfaces."""
        self.backend = backend
        self.permissions = permissions or []
        self.patch = enable_apply_patch and not ({"apply_patch", "write_file", "edit_file", "delete"} & (custom_tool_names | excluded_tools))
        candidate = backend.default if type(backend) is CompositeBackend else backend
        self.shell_backend = candidate if type(candidate) is LocalShellBackend else None
        self.shell = enable_shell and not ({"shell", "execute"} & (custom_tool_names | excluded_tools))
        if self.shell and self.shell_backend is None:
            msg = "Native shell requires a stock LocalShellBackend or a stock CompositeBackend with that default"
            raise ValueError(msg)
        self.excluded = excluded_tools

    def _tools(self, request: ModelRequest) -> list[BaseTool | dict[str, Any]]:
        tools = list(request.tools)
        if (self.patch or self.shell) and not _responses_compatible(request.model):
            msg = "Native tools require an OpenAI Responses-compatible model"
            raise ValueError(msg)
        names = {_name(tool) for tool in tools}
        if self.patch and not {"write_file", "edit_file", "delete"} <= names:
            msg = "Native apply_patch requires the built-in write_file, edit_file, and delete tools"
            raise ValueError(msg)
        if self.shell and "execute" not in names:
            msg = "Native shell requires the built-in execute tool"
            raise ValueError(msg)
        replacements: dict[str, set[str]] = {}
        if (
            self.patch
            and {"write_file", "edit_file", "delete"} <= names
            and not ({"apply_patch", "write_file", "edit_file", "delete"} & self.excluded)
        ):
            replacements["apply_patch"] = {"write_file", "edit_file", "delete"}
        if self.shell and "execute" in names and not ({"execute", "shell"} & self.excluded):
            replacements["shell"] = {"execute"}
        hidden = set().union(*replacements.values()) if replacements else set()
        return [tool for tool in tools if _name(tool) not in hidden] + [
            {"type": name, **({"environment": {"type": "local"}} if name == "shell" else {})} for name in replacements
        ]

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Advertise native built-ins only on a compatible request."""
        messages = [self._restore_output(message) if isinstance(message, ToolMessage) else message for message in request.messages]
        return handler(request.override(tools=self._tools(request), messages=messages))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        """Advertise native built-ins only on a compatible async request."""
        messages = [self._restore_output(message) if isinstance(message, ToolMessage) else message for message in request.messages]
        return await handler(request.override(tools=self._tools(request), messages=messages))

    def _execute(self, request: ToolCallRequest) -> ToolMessage:
        call = request.tool_call
        name = call["name"]
        args = call["args"]
        key = "operation" if name == "apply_patch" else "action"
        if not call["id"]:
            msg = "Native tool calls require a nonempty call id"
            raise ValueError(msg)
        try:
            if set(args) != {key}:
                return _invalid_arguments(call["id"], name, key)
            if name == "apply_patch":
                output = execute_patch(self.backend, args[key], self.permissions)
                block = {"type": "apply_patch_call_output", "status": "completed", "output": output}
                return ToolMessage(content=[block], tool_call_id=call["id"], name=name, artifact={"native_tool_output": block})
            if self.shell_backend is None:
                msg = "Native shell requires a stock LocalShellBackend"
                raise RuntimeError(msg)
            visible, full = _shell_batch(self.shell_backend, args[key])
            return ToolMessage(
                content=[{"type": "shell_call_output", "output": visible}],
                artifact={"output": full, "native_tool_output": {"type": "shell_call_output", "output": visible}},
                tool_call_id=call["id"],
                name=name,
            )
        except (ValueError, OSError, UnicodeError) as error:
            block = (
                {"type": "apply_patch_call_output", "status": "failed", "output": str(error)}
                if name == "apply_patch"
                else {"type": "shell_call_output", "output": []}
            )
            return ToolMessage(
                content=[block], tool_call_id=call["id"], name=name, status="error", artifact={"error": str(error), "native_tool_output": block}
            )

    def _handles(self, request: ToolCallRequest) -> bool:
        name = request.tool_call["name"]
        return request.tool is None and name not in self.excluded and ((name == "apply_patch" and self.patch) or (name == "shell" and self.shell))

    @staticmethod
    def _restore_output(result: ToolMessage | Command) -> ToolMessage | Command:
        if not isinstance(result, ToolMessage) or not isinstance(result.artifact, dict):
            return result
        block = result.artifact.get("native_tool_output")
        if not isinstance(block, dict):
            return result
        artifact = dict(result.artifact)
        if result.content != [block] and result.content != json.dumps([block], ensure_ascii=False):
            artifact["native_tool_diagnostic"] = result.content
        return result.model_copy(update={"content": [block], "artifact": artifact})

    def _dynamic_tool(self, request: ToolCallRequest) -> BaseTool:
        name = request.tool_call["name"]
        key = "operation" if name == "apply_patch" else "action"

        def execute(**args: Any) -> ToolMessage:
            return self._execute(request.override(tool_call={**request.tool_call, "args": args}))

        async def aexecute(**args: Any) -> ToolMessage:
            return await asyncio.to_thread(execute, **args)

        return StructuredTool(
            name=name,
            description=f"Execute a native {name} call.",
            args_schema={"type": "object", "properties": {key: {"type": "object"}}, "required": [key], "additionalProperties": False},
            func=execute,
            coroutine=aexecute,
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        """Resolve native tools without skipping downstream approval middleware."""
        if not self._handles(request):
            return handler(request)
        return self._restore_output(handler(request.override(tool=self._dynamic_tool(request))))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        """Execute native calls without blocking the event loop."""
        if not self._handles(request):
            return await handler(request)
        return self._restore_output(await handler(request.override(tool=self._dynamic_tool(request))))


def native_tools_middleware(
    profile: HarnessProfile,
    backend: BackendProtocol,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None,
    middleware: Sequence[AgentMiddleware],
) -> list[AgentMiddleware]:
    """Build native middleware using the final filesystem permission policy."""
    if any(isinstance(entry, NativeToolsMiddleware) for entry in middleware):
        return []
    if not profile.enable_native_apply_patch and not profile.enable_native_shell:
        return []

    filesystem = next((entry for entry in middleware if isinstance(entry, FilesystemMiddleware)), None)
    if filesystem is None:
        return []
    names = frozenset(
        [_name(tool) if isinstance(tool, (BaseTool, dict)) else tool.__name__ for tool in tools or []]
        + [_name(tool) for entry in middleware if not isinstance(entry, FilesystemMiddleware) for tool in getattr(entry, "tools", [])]
    )
    return [
        NativeToolsMiddleware(
            backend=backend,
            enable_apply_patch=bool(profile.enable_native_apply_patch)
            and {"write_file", "edit_file", "delete"} <= {tool.name for tool in filesystem.tools},
            enable_shell=bool(profile.enable_native_shell) and "execute" in {tool.name for tool in filesystem.tools},
            permissions=filesystem._permissions,
            custom_tool_names=names,
            excluded_tools=profile.excluded_tools,
        )
    ]
