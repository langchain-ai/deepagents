"""Unit tests for SanitizerProvider protocol and SanitizerMiddleware."""

from unittest.mock import MagicMock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

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


# ---------------------------------------------------------------------------
# Task 3: Command Result Handling
# ---------------------------------------------------------------------------


def test_middleware_redacts_command_nested_messages():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("edit_file")
    inner_msg = ToolMessage(content="diff contains SECRET123", tool_call_id="call_1")
    command = Command(update={"messages": [inner_msg]})
    result = mw.wrap_tool_call(request, lambda r: command)
    assert isinstance(result, Command)
    msgs = result.update["messages"]
    assert len(msgs) == 1
    assert "<REDACTED:test-secret>" in msgs[0].content
    assert "SECRET123" not in msgs[0].content


def test_middleware_passes_command_without_messages():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("edit_file")
    command = Command(update={"files": {"a.txt": "data"}})
    result = mw.wrap_tool_call(request, lambda r: command)
    assert isinstance(result, Command)
    assert result.update.get("files") == {"a.txt": "data"}


# ---------------------------------------------------------------------------
# Task 4: Content Block Test
# ---------------------------------------------------------------------------


def test_middleware_sanitizes_text_blocks_only():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("read_file")
    content = [
        {"type": "text", "text": "config has SECRET123"},
        {"type": "image", "source": {"type": "base64", "data": "abc123"}},
        {"type": "text", "text": "no secrets here"},
    ]
    msg = ToolMessage(content=content, tool_call_id="call_1")
    result = mw.wrap_tool_call(request, lambda r: msg)
    assert isinstance(result.content, list)
    assert "<REDACTED:test-secret>" in result.content[0]["text"]
    assert result.content[1]["type"] == "image"
    assert result.content[2]["text"] == "no secrets here"
    assert "secret(s) were redacted" in result.content[-1]["text"]


# ---------------------------------------------------------------------------
# Task 4b: Async, Provider Chaining, Audit Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_async_redacts_toolmessage():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("execute")
    msg = ToolMessage(content="output contains SECRET123 here", tool_call_id="call_1")

    async def async_handler(r):
        return msg

    result = await mw.awrap_tool_call(request, async_handler)
    assert isinstance(result, ToolMessage)
    assert "<REDACTED:test-secret>" in result.content
    assert "SECRET123" not in result.content


class AnotherSecretProvider:
    @property
    def name(self) -> str:
        return "another"

    def sanitize(self, content: str) -> SanitizeResult:
        if "ANOTHER_SECRET" in content:
            return SanitizeResult(
                content=content.replace("ANOTHER_SECRET", "<REDACTED:another>"),
                findings=[SanitizeFinding(rule_id="another", redacted_as="<REDACTED:another>")],
            )
        return SanitizeResult(content=content, findings=[])

    async def asanitize(self, content: str) -> SanitizeResult:
        return self.sanitize(content)


def test_middleware_chains_multiple_providers():
    mw = SanitizerMiddleware(providers=[RedactingProvider(), AnotherSecretProvider()])
    request = _make_request("execute")
    msg = ToolMessage(content="has SECRET123 and ANOTHER_SECRET", tool_call_id="call_1")
    result = mw.wrap_tool_call(request, lambda r: msg)
    assert "SECRET123" not in result.content
    assert "ANOTHER_SECRET" not in result.content
    assert "<REDACTED:test-secret>" in result.content
    assert "<REDACTED:another>" in result.content


def test_middleware_emits_audit_event():
    mw = SanitizerMiddleware(providers=[RedactingProvider()])
    request = _make_request("execute")
    msg = ToolMessage(content="SECRET123", tool_call_id="call_1")
    from unittest.mock import patch

    with patch("deepagents.middleware.sanitizer.dispatch_custom_event") as mock_event:
        mw.wrap_tool_call(request, lambda r: msg)
        mock_event.assert_called_once()
        event_data = mock_event.call_args[0][1]
        assert event_data["tool"] == "execute"
