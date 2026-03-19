"""Pluggable output sanitizer middleware for deep agents.

Intercepts tool output via wrap_tool_call and runs it through registered
SanitizerProvider implementations to redact sensitive content before it
reaches the LLM.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypedDict, cast, runtime_checkable

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages.content import ContentBlock
    from langchain_core.tools import BaseTool
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_GUIDANCE_TEMPLATE = (
    "\n---\n"
    "Note: {count} secret(s) were redacted from this output by the sanitizer.\n"
    "The redacted values (shown as <REDACTED:...>) are still present in the\n"
    "environment — you can use them in scripts and commands, but you cannot\n"
    "read or display the raw values directly."
)


class SanitizeFinding(TypedDict):
    """A single redaction finding. No secret values — only metadata."""

    rule_id: str
    redacted_as: str


class SanitizeResult(TypedDict):
    """Result of a sanitization pass."""

    content: str
    findings: list[SanitizeFinding]


@runtime_checkable
class SanitizerProvider(Protocol):
    """Interface for output sanitization providers."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def sanitize(self, content: str) -> SanitizeResult:
        """Sanitize *content* synchronously and return a SanitizeResult."""
        ...

    async def asanitize(self, content: str) -> SanitizeResult:
        """Sanitize *content* asynchronously and return a SanitizeResult."""
        ...


class SanitizerMiddleware(AgentMiddleware):
    """Middleware that redacts secrets from tool output before it reaches the LLM.

    Chains registered SanitizerProvider instances over every ToolMessage returned
    by a tool call.  Only tools whose names appear in ``target_tools`` are
    processed; when ``target_tools`` is empty *all* tools are processed.
    """

    tools: ClassVar[list[BaseTool]] = []

    def __init__(
        self,
        providers: list[SanitizerProvider],
        target_tools: list[str] | None = None,
    ) -> None:
        """Initialise with a list of providers and an optional tool filter."""
        self._providers = providers
        self._target_tools: list[str] = target_tools or []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_text(self, text: str) -> tuple[str, list[SanitizeFinding]]:
        """Run *text* through all providers synchronously, chaining results."""
        all_findings: list[SanitizeFinding] = []
        current = text
        for provider in self._providers:
            result = provider.sanitize(current)
            current = result["content"]
            all_findings.extend(result["findings"])
        return current, all_findings

    async def _asanitize_text(self, text: str) -> tuple[str, list[SanitizeFinding]]:
        """Run *text* through all providers asynchronously, chaining results."""
        all_findings: list[SanitizeFinding] = []
        current = text
        for provider in self._providers:
            result = await provider.asanitize(current)
            current = result["content"]
            all_findings.extend(result["findings"])
        return current, all_findings

    def _log_findings(self, tool_name: str, findings: list[SanitizeFinding]) -> None:
        logger.warning(
            "sanitizer: redacted %d secret(s) from tool '%s' output: %s",
            len(findings),
            tool_name,
            [f["rule_id"] for f in findings],
        )

    def _emit_audit_event(self, tool_name: str, findings: list[SanitizeFinding]) -> None:
        """Dispatch a custom audit event best-effort (never raises)."""
        with contextlib.suppress(Exception):
            dispatch_custom_event(
                "sanitizer_redaction",
                {
                    "tool": tool_name,
                    "findings": findings,
                },
            )

    def _apply_to_toolmessage(self, msg: ToolMessage, tool_name: str) -> ToolMessage:
        """Sanitize a ToolMessage (sync) and return it."""
        content = msg.content

        if isinstance(content, str):
            sanitized, findings = self._sanitize_text(content)
            if findings:
                sanitized += _GUIDANCE_TEMPLATE.format(count=len(findings))
                msg.content = sanitized
                self._log_findings(tool_name, findings)
                self._emit_audit_event(tool_name, findings)
            return msg

        if isinstance(content, list):
            all_findings: list[SanitizeFinding] = []
            new_blocks: list[ContentBlock] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    sanitized_text, findings = self._sanitize_text(block.get("text", ""))
                    all_findings.extend(findings)
                    new_blocks.append(cast("ContentBlock", {**block, "text": sanitized_text}))
                else:
                    new_blocks.append(cast("ContentBlock", block))
            if all_findings:
                guidance = _GUIDANCE_TEMPLATE.format(count=len(all_findings))
                new_blocks.append(cast("ContentBlock", {"type": "text", "text": guidance}))
                msg.content = new_blocks  # type: ignore[assignment]
                self._log_findings(tool_name, all_findings)
                self._emit_audit_event(tool_name, all_findings)
            elif new_blocks != list(content):
                msg.content = new_blocks  # type: ignore[assignment]
            return msg

        return msg

    async def _aapply_to_toolmessage(self, msg: ToolMessage, tool_name: str) -> ToolMessage:
        """Sanitize a ToolMessage (async variant)."""
        content = msg.content

        if isinstance(content, str):
            sanitized, findings = await self._asanitize_text(content)
            if findings:
                sanitized += _GUIDANCE_TEMPLATE.format(count=len(findings))
                msg.content = sanitized
                self._log_findings(tool_name, findings)
                self._emit_audit_event(tool_name, findings)
            return msg

        if isinstance(content, list):
            all_findings: list[SanitizeFinding] = []
            new_blocks: list[ContentBlock] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    sanitized_text, findings = await self._asanitize_text(block.get("text", ""))
                    all_findings.extend(findings)
                    new_blocks.append(cast("ContentBlock", {**block, "text": sanitized_text}))
                else:
                    new_blocks.append(cast("ContentBlock", block))
            if all_findings:
                guidance = _GUIDANCE_TEMPLATE.format(count=len(all_findings))
                new_blocks.append(cast("ContentBlock", {"type": "text", "text": guidance}))
                msg.content = new_blocks  # type: ignore[assignment]
                self._log_findings(tool_name, all_findings)
                self._emit_audit_event(tool_name, all_findings)
            elif new_blocks != list(content):
                msg.content = new_blocks  # type: ignore[assignment]
            return msg

        return msg

    def _should_process(self, tool_name: str) -> bool:
        """Return True if this tool's output should be sanitized."""
        if not self._target_tools:
            return True
        return tool_name in self._target_tools

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept sync tool output and redact secrets via registered providers."""
        tool_name: str = request.tool_call["name"]
        tool_result = handler(request)

        if not self._should_process(tool_name):
            return tool_result

        if isinstance(tool_result, ToolMessage):
            return self._apply_to_toolmessage(tool_result, tool_name)

        return tool_result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept async tool output and redact secrets via registered providers."""
        tool_name: str = request.tool_call["name"]
        tool_result = await handler(request)

        if not self._should_process(tool_name):
            return tool_result

        if isinstance(tool_result, ToolMessage):
            return await self._aapply_to_toolmessage(tool_result, tool_name)

        return tool_result
