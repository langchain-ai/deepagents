"""OAuth callback contracts at the Talon/MCP boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_talon.authorization import (
    CallbackURLRequested,
    reset_authorization_handler,
    set_authorization_handler,
)
from deepagents_talon.mcp import _run_authorized
from deepagents_talon.mcp_auth import _channel_handlers, _interactive_handlers

if TYPE_CHECKING:
    from deepagents_talon.authorization import AuthorizationEvent


@pytest.mark.parametrize("channel", [False, True])
@pytest.mark.parametrize("issuer", [None, "https://auth.example/tenant"])
async def test_callbacks_preserve_issuer(
    monkeypatch: pytest.MonkeyPatch, issuer: str | None, *, channel: bool
) -> None:
    redirect_uri = "http://localhost:3000/callback"
    url = f"{redirect_uri}?code=example-code&state=example-state"
    if issuer is not None:
        url += f"&iss={issuer}"
    monkeypatch.setattr("builtins.input", lambda _prompt: url)

    async def handler(event: AuthorizationEvent) -> str | None:
        if isinstance(event, CallbackURLRequested):
            assert event.binding.invocation_id == "call-42"
            return url
        return None

    redirect, callback = (
        _channel_handlers("remote", redirect_uri)
        if channel
        else _interactive_handlers(redirect_uri)
    )

    async def authorize():
        await redirect("https://auth.example/authorize")
        return await callback()

    token = set_authorization_handler(handler)
    try:
        result = await _run_authorized("call-42", authorize)
    finally:
        reset_authorization_handler(token)

    assert (result.code, result.state, result.iss) == ("example-code", "example-state", issuer)
