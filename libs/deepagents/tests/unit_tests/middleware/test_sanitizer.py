"""Unit tests for SanitizerProvider protocol and SanitizerMiddleware."""

from unittest.mock import MagicMock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents.middleware.sanitizer import (
    SanitizeFinding,
    SanitizeResult,
    SanitizerMiddleware,
    SanitizerProvider,
)


class DummyProvider:
    """Minimal provider that satisfies the protocol."""

    @property
    def name(self) -> str:
        return "dummy"

    def sanitize(self, content: str) -> SanitizeResult:
        return SanitizeResult(content=content, findings=[])

    async def asanitize(self, content: str) -> SanitizeResult:
        return SanitizeResult(content=content, findings=[])


class RedactingProvider:
    """Provider that replaces 'SECRET123' with a redaction marker."""

    @property
    def name(self) -> str:
        return "test-redactor"

    def sanitize(self, content: str) -> SanitizeResult:
        if "SECRET123" in content:
            redacted = content.replace("SECRET123", "<REDACTED:test-secret>")
            return SanitizeResult(
                content=redacted,
                findings=[SanitizeFinding(rule_id="test-secret", redacted_as="<REDACTED:test-secret>")],
            )
        return SanitizeResult(content=content, findings=[])

    async def asanitize(self, content: str) -> SanitizeResult:
        return self.sanitize(content)


def _make_request(tool_name: str = "execute") -> ToolCallRequest:
    request = MagicMock(spec=ToolCallRequest)
    request.tool_call = {"name": tool_name, "args": {}, "id": "call_1"}
    request.runtime = MagicMock()
    return request


# ---------------------------------------------------------------------------
# Task 1: SanitizerProvider Protocol conformance
# ---------------------------------------------------------------------------


def test_dummy_provider_satisfies_protocol():
    provider: SanitizerProvider = DummyProvider()
    result = provider.sanitize("hello")
    assert result["content"] == "hello"
    assert result["findings"] == []


@pytest.mark.asyncio
async def test_dummy_provider_async():
    provider: SanitizerProvider = DummyProvider()
    result = await provider.asanitize("hello")
    assert result["content"] == "hello"
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# Task 2: SanitizerMiddleware — ToolMessage Redaction
# ---------------------------------------------------------------------------


def test_middleware_redacts_toolmessage_str_content():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("execute")
    msg = ToolMessage(content="output contains SECRET123 here", tool_call_id="call_1")

    result = mw.wrap_tool_call(request, lambda _r: msg)

    assert isinstance(result, ToolMessage)
    assert "<REDACTED:test-secret>" in result.content
    assert "SECRET123" not in result.content
    assert "secret(s) were redacted" in result.content


def test_middleware_skips_clean_output():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("execute")
    msg = ToolMessage(content="nothing sensitive here", tool_call_id="call_1")

    result = mw.wrap_tool_call(request, lambda _r: msg)

    assert result.content == "nothing sensitive here"


def test_middleware_respects_target_tools_filter():
    mw = SanitizerMiddleware(providers=[RedactingProvider()], target_tools=["execute"])
    request = _make_request("write_todos")
    msg = ToolMessage(content="SECRET123", tool_call_id="call_1")

    result = mw.wrap_tool_call(request, lambda _r: msg)

    # Should NOT redact because write_todos is not in target_tools
    assert result.content == "SECRET123"
