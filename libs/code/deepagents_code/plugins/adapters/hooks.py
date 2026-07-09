"""Adapter and execution helpers for plugin hooks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess  # noqa: S404  # Trusted plugin hooks execute fixed argv without a shell.
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from deepagents_code.plugins.substitution import (
    plugin_environment,
    substitute_string,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

    from deepagents_code.plugins.models import PluginInstance

logger = logging.getLogger(__name__)
HookDecision = Literal["pass", "allow", "deny", "ask"]
_DEFAULT_TIMEOUT = 60.0
_EVENT_TIMEOUTS = {
    "PermissionRequest": 600.0,
    "PreToolUse": 600.0,
    "UserPromptSubmit": 30.0,
}
_EVENT_MAP: dict[str, str] = {
    "PreToolUse": "tool.use",
    "PostToolUse": "tool.result",
    "PostToolUseFailure": "tool.result",
    "UserPromptSubmit": "user.prompt",
    "SessionStart": "session.start",
    "SessionEnd": "session.end",
    "Stop": "task.complete",
    "SubagentStart": "tool.use",
    "SubagentStop": "tool.result",
    "Notification": "permission.request",
    "PreCompact": "context.compact",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginHook:
    """Normalized command hook from a plugin."""

    event: str
    source_event: str
    command: tuple[str, ...]
    plugin_id: str
    matcher: str | None = None
    timeout: float = _DEFAULT_TIMEOUT
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    blocking: bool = False

    def matches(self, *, tool_name: str | None = None) -> bool:
        """Return whether this hook matches the supplied tool."""
        if not self.matcher:
            return True
        candidate = tool_name or ""
        if "|" in self.matcher:
            return candidate in self.matcher.split("|")
        try:
            return re.search(self.matcher, candidate) is not None
        except re.error:
            return candidate == self.matcher


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginHookResult:
    """Decision returned by blocking plugin hooks."""

    decision: HookDecision
    reason: str = ""
    additional_context: str = ""


_DECISION_LOCK = threading.Lock()
_PRE_TOOL_DECISIONS: dict[str, PluginHookResult] = {}
_TOOL_NAME_MAP = {
    "execute": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "delete": "Write",
    "task": "Task",
    "web_search": "WebSearch",
    "fetch_url": "WebFetch",
}


class PluginHookMiddleware(AgentMiddleware):
    """Apply blocking plugin hooks before tool execution."""

    def __init__(self, hooks: tuple[PluginHook, ...], *, session_cwd: Path) -> None:
        """Initialize middleware with an immutable hook set."""
        self.hooks = hooks
        self.session_cwd = session_cwd

    def _decision(self, request: ToolCallRequest) -> PluginHookResult:
        call_id = request.tool_call.get("id")
        with _DECISION_LOCK:
            cached = (
                _PRE_TOOL_DECISIONS.pop(call_id, None)
                if isinstance(call_id, str)
                else None
            )
        if cached is not None:
            if cached.decision == "ask":
                return PluginHookResult(
                    decision="allow",
                    additional_context=cached.additional_context,
                )
            return cached
        return evaluate_pre_tool_request(
            self.hooks,
            request,
            session_cwd=self.session_cwd,
        )

    def _post_context(
        self,
        request: ToolCallRequest,
        response: object,
        *,
        failed: bool,
    ) -> str:
        raw_args = request.tool_call.get("args")
        tool_input = raw_args if isinstance(raw_args, dict) else {}
        return run_post_tool_hooks(
            self.hooks,
            tool_name=request.tool_call["name"],
            tool_input=tool_input,
            tool_output=response,
            failed=failed,
            session_id=_request_session_id(request),
            cwd=self.session_cwd,
        )

    @staticmethod
    def _rejection(request: ToolCallRequest, result: PluginHookResult) -> ToolMessage:
        reason = result.reason or (
            "Plugin hook requires interactive approval"
            if result.decision == "ask"
            else "Plugin hook denied the tool call"
        )
        return ToolMessage(
            content=reason,
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run blocking hooks before a synchronous tool call.

        Returns:
            Tool result, command, or hook rejection.
        """
        result = self._decision(request)
        if result.decision in {"deny", "ask"}:
            return self._rejection(request, result)
        try:
            response = handler(request)
        except Exception as exc:
            self._post_context(request, exc, failed=True)
            raise
        failed = isinstance(response, ToolMessage) and response.status == "error"
        post_context = self._post_context(request, response, failed=failed)
        context = "\n".join(
            value for value in (result.additional_context, post_context) if value
        )
        return _append_context(response, context)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run blocking hooks before an asynchronous tool call.

        Returns:
            Tool result, command, or hook rejection.
        """
        result = await asyncio.to_thread(self._decision, request)
        if result.decision in {"deny", "ask"}:
            return self._rejection(request, result)
        try:
            response = await handler(request)
        except Exception as exc:
            await asyncio.to_thread(self._post_context, request, exc, failed=True)
            raise
        failed = isinstance(response, ToolMessage) and response.status == "error"
        post_context = await asyncio.to_thread(
            self._post_context,
            request,
            response,
            failed=failed,
        )
        context = "\n".join(
            value for value in (result.additional_context, post_context) if value
        )
        return _append_context(response, context)


def _append_context(
    response: ToolMessage | Command[Any], context: str
) -> ToolMessage | Command[Any]:
    if not context or not isinstance(response, ToolMessage):
        return response
    content = response.content
    if isinstance(content, str):
        updated = f"{content}\n\n{context}"
    else:
        updated = [*content, {"type": "text", "text": context}]
    return response.model_copy(update={"content": updated})


def map_hook_tool_call(
    tool_name: str, tool_input: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Map a dcode tool call to the compatibility hook vocabulary.

    Args:
        tool_name: Native dcode tool name.
        tool_input: Native dcode tool input.

    Returns:
        Compatibility tool name and input.
    """
    mapped_name = _TOOL_NAME_MAP.get(tool_name, tool_name)
    mapped_input = dict(tool_input)
    if tool_name == "delete":
        mapped_input.setdefault("operation", "delete")
    return mapped_name, mapped_input


def _request_session_id(request: ToolCallRequest) -> str:
    context = request.runtime.context
    if isinstance(context, dict):
        value = context.get("thread_id")
    else:
        value = getattr(context, "thread_id", None)
    return value if isinstance(value, str) else ""


def evaluate_pre_tool_request(
    hooks: tuple[PluginHook, ...],
    request: ToolCallRequest,
    *,
    session_cwd: Path,
) -> PluginHookResult:
    """Evaluate and cache the plugin decision for one tool request.

    Args:
        hooks: Active plugin hooks.
        request: Pending tool execution.
        session_cwd: Session working directory.

    Returns:
        Strictest plugin hook decision.
    """
    raw_args = request.tool_call.get("args")
    tool_input = raw_args if isinstance(raw_args, dict) else {}
    result = run_pre_tool_hooks(
        hooks,
        tool_name=request.tool_call["name"],
        tool_input=tool_input,
        session_id=_request_session_id(request),
        cwd=session_cwd,
    )
    call_id = request.tool_call.get("id")
    if isinstance(call_id, str):
        with _DECISION_LOCK:
            _PRE_TOOL_DECISIONS[call_id] = result
    return result


def _load_hook_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin hooks file %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    hooks = raw.get("hooks", raw)
    return hooks if isinstance(hooks, dict) else {}


def _hook_objects(plugin: PluginInstance) -> tuple[dict[str, Any], ...]:
    objects = tuple(_load_hook_object(path) for path in plugin.inventory.hooks_files)
    inline = plugin.manifest.inline_hooks if plugin.manifest else ()
    normalized_inline = tuple(
        value.get("hooks", value) if isinstance(value.get("hooks"), dict) else value
        for value in inline
    )
    return (*objects, *normalized_inline)


def plugin_hooks(
    plugins: tuple[PluginInstance, ...],
) -> tuple[PluginHook, ...]:
    """Load normalized hooks from trusted active plugins.

    Args:
        plugins: Active plugin instances.

    Returns:
        Normalized command hooks.
    """
    result: list[PluginHook] = []
    for plugin in plugins:
        if not plugin.trusted:
            continue
        env = {
            **os.environ,
            **plugin_environment(
                plugin_root=plugin.root,
                plugin_data=plugin.data_dir,
            ),
        }
        for config in _hook_objects(plugin):
            for source_event, matchers in config.items():
                event = _EVENT_MAP.get(source_event)
                if event is None or not isinstance(matchers, list):
                    continue
                for matcher_config in matchers:
                    if not isinstance(matcher_config, dict):
                        continue
                    matcher = matcher_config.get("matcher")
                    hooks = matcher_config.get("hooks")
                    if not isinstance(hooks, list):
                        continue
                    for hook in hooks:
                        if not isinstance(hook, dict) or hook.get("type") != "command":
                            continue
                        raw_command = hook.get("command")
                        if not isinstance(raw_command, str):
                            continue
                        command = substitute_string(
                            raw_command,
                            plugin_root=plugin.root,
                            plugin_data=plugin.data_dir,
                            warning_key=plugin.plugin_id,
                        )
                        try:
                            argv = tuple(shlex.split(command))
                        except ValueError:
                            logger.warning(
                                "Skipping malformed hook command in %s",
                                plugin.plugin_id,
                            )
                            continue
                        if not argv:
                            continue
                        timeout = hook.get("timeout")
                        result.append(
                            PluginHook(
                                event=event,
                                source_event=source_event,
                                command=argv,
                                plugin_id=plugin.plugin_id,
                                matcher=matcher if isinstance(matcher, str) else None,
                                timeout=float(timeout)
                                if isinstance(timeout, (int, float)) and timeout > 0
                                else _EVENT_TIMEOUTS.get(
                                    source_event, _DEFAULT_TIMEOUT
                                ),
                                env=env,
                                cwd=str(plugin.root),
                                blocking=source_event == "PreToolUse",
                            )
                        )
    return tuple(result)


def run_pre_tool_hooks(
    hooks: tuple[PluginHook, ...],
    *,
    tool_name: str,
    tool_input: Mapping[str, Any],
    session_id: str = "",
    cwd: Path | None = None,
) -> PluginHookResult:
    """Run blocking pre-tool hooks and return the strictest decision.

    Args:
        hooks: Active plugin hooks.
        tool_name: Tool requested by the model.
        tool_input: Parsed tool input.
        session_id: Active session identifier.
        cwd: Session working directory.

    Returns:
        Strictest decision returned by matching hooks.
    """
    mapped_name, mapped_input = map_hook_tool_call(tool_name, tool_input)
    payload = json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "",
            "cwd": str(cwd.resolve()) if cwd is not None else str(Path.cwd()),
            "hook_event_name": "PreToolUse",
            "tool_name": mapped_name,
            "tool_input": mapped_input,
        },
        default=str,
    ).encode()
    decision: HookDecision = "pass"
    reasons: list[str] = []
    contexts: list[str] = []
    for hook in hooks:
        if not hook.blocking or not hook.matches(tool_name=mapped_name):
            continue
        try:
            completed = subprocess.run(  # noqa: S603
                hook.command,
                input=payload,
                capture_output=True,
                cwd=hook.cwd,
                env=dict(hook.env) if hook.env is not None else None,
                timeout=hook.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            decision = "ask"
            reasons.append(str(exc))
            continue
        stderr = completed.stderr.decode(errors="replace").strip()
        stdout = completed.stdout.decode(errors="replace").strip()
        if completed.returncode == 2:  # noqa: PLR2004
            decision = "deny"
            reasons.append(stderr or "Plugin hook denied the tool call")
            continue
        if not stdout:
            continue
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(response, dict):
            continue
        specific = response.get("hookSpecificOutput")
        output = specific if isinstance(specific, dict) else response
        context = output.get("additionalContext")
        if isinstance(context, str) and context:
            contexts.append(context)
        value = output.get("permissionDecision")
        if value == "deny":
            decision = "deny"
            reason = output.get("permissionDecisionReason")
            reasons.append(
                reason
                if isinstance(reason, str)
                else "Plugin hook denied the tool call"
            )
        elif value == "ask" and decision != "deny":
            decision = "ask"
            reason = output.get("permissionDecisionReason")
            if isinstance(reason, str):
                reasons.append(reason)
        elif value == "allow" and decision == "pass":
            decision = "allow"
    return PluginHookResult(
        decision=decision,
        reason="; ".join(reasons),
        additional_context="\n".join(contexts),
    )


def run_post_tool_hooks(
    hooks: tuple[PluginHook, ...],
    *,
    tool_name: str,
    tool_input: Mapping[str, Any],
    tool_output: object,
    failed: bool,
    session_id: str = "",
    cwd: Path | None = None,
) -> str:
    """Run post-tool hooks and return model context.

    Args:
        hooks: Active plugin hooks.
        tool_name: Executed tool name.
        tool_input: Parsed tool input.
        tool_output: Tool result or failure.
        failed: Whether execution failed.
        session_id: Active session identifier.
        cwd: Session working directory.

    Returns:
        Additional model context emitted by hooks.
    """
    mapped_name, mapped_input = map_hook_tool_call(tool_name, tool_input)
    source_event = "PostToolUseFailure" if failed else "PostToolUse"
    payload = json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "",
            "cwd": str(cwd.resolve()) if cwd is not None else str(Path.cwd()),
            "hook_event_name": source_event,
            "tool_name": mapped_name,
            "tool_input": mapped_input,
            "tool_response": tool_output,
        },
        default=str,
    ).encode()
    contexts: list[str] = []
    for hook in hooks:
        if hook.source_event != source_event or not hook.matches(tool_name=mapped_name):
            continue
        try:
            completed = subprocess.run(  # noqa: S603
                hook.command,
                input=payload,
                capture_output=True,
                cwd=hook.cwd,
                env=dict(hook.env) if hook.env is not None else None,
                timeout=hook.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Post-tool plugin hook failed: %s", exc)
            continue
        stderr = completed.stderr.decode(errors="replace").strip()
        stdout = completed.stdout.decode(errors="replace").strip()
        if completed.returncode == 2 and stderr:  # noqa: PLR2004
            contexts.append(stderr)
        if not stdout:
            continue
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(response, dict):
            continue
        specific = response.get("hookSpecificOutput")
        output = specific if isinstance(specific, dict) else response
        context = output.get("additionalContext")
        if isinstance(context, str) and context:
            contexts.append(context)
        if output.get("decision") == "block":
            reason = output.get("reason")
            if isinstance(reason, str) and reason:
                contexts.append(reason)
    return "\n".join(contexts)


def run_user_prompt_hooks(
    hooks: tuple[PluginHook, ...],
    *,
    prompt: str,
    session_id: str = "",
    cwd: Path | None = None,
) -> PluginHookResult:
    """Run user-prompt hooks before submission.

    Args:
        hooks: Active plugin hooks.
        prompt: Submitted user input.
        session_id: Active session identifier.
        cwd: Session working directory.

    Returns:
        Prompt decision and additional model context.
    """
    payload = json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "",
            "cwd": str(cwd.resolve()) if cwd is not None else str(Path.cwd()),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    ).encode()
    decision: HookDecision = "pass"
    reasons: list[str] = []
    contexts: list[str] = []
    for hook in hooks:
        if hook.source_event != "UserPromptSubmit":
            continue
        try:
            completed = subprocess.run(  # noqa: S603
                hook.command,
                input=payload,
                capture_output=True,
                cwd=hook.cwd,
                env=dict(hook.env) if hook.env is not None else None,
                timeout=hook.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("User-prompt plugin hook failed: %s", exc)
            continue
        stderr = completed.stderr.decode(errors="replace").strip()
        stdout = completed.stdout.decode(errors="replace").strip()
        if completed.returncode == 2:  # noqa: PLR2004
            decision = "deny"
            reasons.append(stderr or "Plugin hook blocked the prompt")
            continue
        if not stdout:
            continue
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError:
            contexts.append(stdout)
            continue
        if not isinstance(response, dict):
            continue
        specific = response.get("hookSpecificOutput")
        output = specific if isinstance(specific, dict) else {}
        context = output.get("additionalContext")
        if isinstance(context, str) and context:
            contexts.append(context)
        if response.get("decision") == "block" or response.get("continue") is False:
            decision = "deny"
            reason = response.get("reason", response.get("stopReason"))
            reasons.append(
                reason if isinstance(reason, str) else "Plugin hook blocked the prompt"
            )
    return PluginHookResult(
        decision=decision,
        reason="; ".join(reasons),
        additional_context="\n".join(contexts),
    )
