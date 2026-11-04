"""Session-local opt-ins for provider-native editing and shell tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolMessage
    from langchain_core.messages.tool import ToolCall
    from langchain_core.tools import BaseTool
    from langgraph.types import Command


def _enabled(key: str) -> bool:
    from deepagents_code.config_manifest import (
        _emit_ranked_diagnostics,
        _resolve_option,
        get_option,
    )

    option = get_option(key)
    if option is None:
        return False
    resolved = _resolve_option(option)
    _emit_ranked_diagnostics(option, resolved)
    return resolved.value is True


def _native_safety_middleware(
    middleware: Sequence[AgentMiddleware[Any, Any]],
    names: frozenset[str],
) -> AgentMiddleware[Any, Any]:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        TracePolicy,
        omit_payload,
    )
    from langchain_core.messages import ToolMessage

    from deepagents_code._cli_context import CLIContextSchema
    from deepagents_code.approval_mode import ApprovalMode
    from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware

    memory_guards = [
        item for item in middleware if isinstance(item, ManagedMemoryGuardMiddleware)
    ]

    class NativeFeatureSafetyMiddleware(AgentMiddleware):
        """Fail closed where native schemas lack dcode policy adapters."""

        trace_policy = TracePolicy(process_inputs=omit_payload)

        @staticmethod
        def _rejection(
            request: ToolCallRequest, mode: ApprovalMode
        ) -> ToolMessage | None:
            name = request.tool_call["name"]
            if name not in names:
                return None
            context = CLIContextSchema.from_payload(request.runtime.context)
            reason = None
            if mode is ApprovalMode.AUTO:
                reason = (
                    "Native tools do not support Auto approval yet; use Manual "
                    "or YOLO mode, or disable the native feature."
                )
            elif context and set(context.hooks_server_events) & {
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
            }:
                reason = (
                    "Native tools do not support configured tool-use hooks yet; "
                    "disable the native feature to preserve hook enforcement."
                )
            elif name == "apply_patch":
                operation = request.tool_call.get("args", {}).get("operation")
                if isinstance(operation, Mapping):
                    call: ToolCall = {
                        **request.tool_call,
                        "name": "delete"
                        if operation.get("type") == "delete_file"
                        else "write_file",
                        "args": {"file_path": operation.get("path")},
                    }
                    probe = request.override(tool_call=call)
                    if any(
                        guard._guarded_path(probe) is not None
                        for guard in memory_guards
                    ):
                        reason = (
                            "Use edit_file for managed memory files so the "
                            "existing memory-block guard can protect them."
                        )
            if reason is None:
                return None
            return ToolMessage(
                content=reason,
                name=name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            from deepagents_code.agent import _resolve_approval_mode

            if request.tool_call["name"] not in names:
                return handler(request)
            mode = _resolve_approval_mode(
                request.runtime.context, request.runtime.store
            )
            return self._rejection(request, mode) or handler(request)

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            from deepagents_code.agent import _aresolve_approval_mode

            if request.tool_call["name"] not in names:
                return await handler(request)
            mode = await _aresolve_approval_mode(
                request.runtime.context, request.runtime.store
            )
            if (rejected := self._rejection(request, mode)) is not None:
                return rejected
            return await handler(request)

    return NativeFeatureSafetyMiddleware()


def _extend_native_stack(
    *,
    backend: BackendProtocol,
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]],
    middleware: list[AgentMiddleware[Any, Any]],
    apply_patch: bool,
    shell: bool,
) -> None:
    from deepagents.middleware.filesystem import FilesystemMiddleware

    from deepagents_code.agent import _get_cli_harness_profile
    from deepagents_code.extensions.hosting import _tool_name

    try:
        from deepagents.middleware.native_tools import native_tools_middleware
    except ImportError as error:
        msg = (
            "Native features require a Deep Agents SDK with native-tools support "
            "and a compatible model adapter."
        )
        raise ValueError(msg) from error

    profile = _get_cli_harness_profile(model)
    if not any(isinstance(item, FilesystemMiddleware) for item in middleware):
        middleware.append(
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=dict(profile.tool_description_overrides),
            )
        )
    owned = {_tool_name(tool) for tool in tools}
    owned.update(
        _tool_name(tool) for item in middleware for tool in getattr(item, "tools", [])
    )
    names = frozenset(
        name
        for name, enabled in (("apply_patch", apply_patch), ("shell", shell))
        if enabled and name not in owned
    )
    profile = replace(
        profile, enable_native_apply_patch=apply_patch, enable_native_shell=shell
    )
    native = native_tools_middleware(profile, backend, tools, middleware)
    if native:
        safety = _native_safety_middleware(middleware, names)
        middleware.extend([safety, *native])


def configure_native_features(
    *,
    backend: BackendProtocol,
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]],
    middleware: list[AgentMiddleware[Any, Any]],
    subagents: list[SubAgent | CompiledSubAgent],
    fs_tools: Sequence[str] | None,
    shell_allow_list: Sequence[str] | None,
) -> None:
    """Add opted-in native tools without modifying the model-profile registry.

    Args:
        backend: Backend shared by the CLI graphs.
        model: Model used to preserve harness filesystem descriptions.
        tools: Explicit main-agent tools to preserve.
        middleware: Final main-agent middleware, extended in place.
        subagents: Declarative specs to extend; compiled agents remain unchanged.
        fs_tools: Explicit filesystem tool restriction, if any.
        shell_allow_list: Restrictive execution policy, if any.

    Raises:
        ValueError: If native tools would bypass an existing execution policy.
    """
    apply_patch = _enabled("features.apply_patch")
    shell = _enabled("features.native_shell")
    if not (apply_patch or shell):
        return
    if shell and shell_allow_list:
        msg = (
            "features.native_shell cannot enforce shell.allow_list; disable "
            "native_shell to preserve the execution policy."
        )
        raise ValueError(msg)
    if fs_tools is not None and (
        (shell and "execute" not in fs_tools)
        or (
            apply_patch and not {"write_file", "edit_file", "delete"}.issubset(fs_tools)
        )
    ):
        msg = (
            "Native features conflict with --allow-fs-tools; disable the native "
            "feature rather than expanding filesystem permissions."
        )
        raise ValueError(msg)
    _extend_native_stack(
        backend=backend,
        model=model,
        tools=tools,
        middleware=middleware,
        apply_patch=apply_patch,
        shell=shell,
    )
    for subagent in subagents:
        if "runnable" in subagent:
            continue
        stack = list(subagent.get("middleware", []))
        _extend_native_stack(
            backend=backend,
            model=subagent.get("model", model),
            tools=subagent.get("tools", tools),
            middleware=stack,
            apply_patch=apply_patch,
            shell=shell,
        )
        subagent["middleware"] = stack
