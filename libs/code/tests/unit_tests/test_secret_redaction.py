"""Tests for inbound tool-result secret redaction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from deepagents_code.offload_middleware import CLICompactionMiddleware
from deepagents_code.secret_redaction import redact_secrets
from deepagents_code.tool_display import format_tool_message_content


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("langsmith", "lsv2_pt_1234567890abcdef1234_123456"),
        ("openai", "sk-" + "A" * 24),
        ("anthropic", "sk-ant-" + "A" * 24),
        ("github", "ghp_" + "A" * 28),
        ("github_pat", "github_pat_" + "A" * 40),
        ("slack", "xoxb-" + "A" * 20),
        ("aws", "AKIA" + "A" * 16),
        ("google", "AIza" + "A" * 35),
        ("tavily", "tvly-" + "A" * 16),
        ("jwt", "eyJ" + "A" * 16 + "." + "B" * 16 + "." + "C"),
    ],
)
def test_credential_patterns_are_redacted(kind: str, value: str) -> None:
    redacted, hits = redact_secrets(f"value={value}")

    assert value not in redacted
    assert hits[0].kind == kind
    assert f"[REDACTED_SECRET {kind} " in redacted


def test_pylon_ticket_preserves_surrounding_prose() -> None:
    token = "lsv2_pt_1234567890abcdef1234_123456"
    body = (
        f'Customer reports a failure. LANGCHAIN_API_KEY="{token}" Please investigate.'
    )

    redacted, hits = redact_secrets(body)

    assert redacted == (
        'Customer reports a failure. LANGCHAIN_API_KEY="'
        f"[REDACTED_SECRET langsmith {hits[0].fingerprint}]"
        '" Please investigate.'
    )
    assert hits


def test_fingerprint_is_stable() -> None:
    token = "sk-" + "A" * 24

    first, first_hits = redact_secrets(token)
    second, second_hits = redact_secrets(token)

    assert first == second
    assert first_hits == second_hits


@pytest.mark.parametrize(
    "value",
    [
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "550e8400-e29b-41d4-a716-446655440000",
        "iVBORw0KGgo" + "A" * 200,
    ],
)
def test_high_entropy_non_secrets_are_not_redacted(value: str) -> None:
    assert redact_secrets(value) == (value, ())


async def test_async_tool_result_is_redacted_before_model_receives_it() -> None:
    token = "lsv2_pt_1234567890abcdef1234_123456"
    middleware = CLICompactionMiddleware(cast("Any", SimpleNamespace()))
    request = ToolCallRequest(
        tool_call={"name": "pylon_get_issue", "args": {}, "id": "call-1"},
        tool=cast("Any", SimpleNamespace(metadata={"_deepagents_code_mcp": True})),
        state={},
        runtime=cast("Any", Runtime()),
    )

    handler = AsyncMock(
        return_value=ToolMessage(content=f"ticket body: {token}", tool_call_id="call-1")
    )

    result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert token not in result.content
    assert "[REDACTED_SECRET langsmith " in result.content
    assert "1 secret redacted from result" in format_tool_message_content(
        result.content
    )
