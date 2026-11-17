"""Tests for the built-in extension provenance display."""

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import get_slash_commands

if TYPE_CHECKING:
    import pytest


def test_extension_command_autocomplete_requires_experimental_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled experiment is absent from user-facing autocomplete."""
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    assert "/extensions" not in {item.name for item in get_slash_commands()}

    monkeypatch.setenv(EXPERIMENTAL, "1")
    assert "/extensions" in {item.name for item in get_slash_commands()}


async def test_extension_command_rejects_when_experimental_mode_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing the hidden command directly cannot reach the server endpoint."""
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    app = object.__new__(DeepAgentsApp)
    mount_message = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", mount_message)

    await app._handle_extensions_command("/extensions")

    assert "DEEPAGENTS_CODE_EXPERIMENTAL=1" in str(
        mount_message.await_args_list[-1].args[0]._content
    )


async def test_extension_command_bypasses_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extension provenance goes directly to the local server."""
    import httpx

    monkeypatch.setenv(EXPERIMENTAL, "1")
    monkeypatch.setenv("HTTP_PROXY", "http://company-gateway:8080")
    monkeypatch.setenv("http_proxy", "http://company-gateway:8080")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    direct = MagicMock(return_value=httpx.Response(200, json={"registrations": []}))
    proxy = MagicMock(return_value=httpx.Response(502))
    monkeypatch.setattr(
        httpx.AsyncClient,
        "_init_transport",
        MagicMock(return_value=httpx.MockTransport(direct)),
    )
    monkeypatch.setattr(
        httpx.AsyncClient,
        "_init_proxy_transport",
        MagicMock(return_value=httpx.MockTransport(proxy)),
    )
    app = object.__new__(DeepAgentsApp)
    monkeypatch.setattr(
        app,
        "_server_proc",
        SimpleNamespace(url="http://127.0.0.1:43210"),
        raising=False,
    )
    monkeypatch.setattr(app, "_mount_message", AsyncMock())

    await app._handle_extensions_command("/extensions")

    assert direct.call_args.args[0].url.path == "/extensions"
    proxy.assert_not_called()


def test_extension_display_escapes_endpoint_metadata() -> None:
    """Paths and errors cannot inject markdown structure into the transcript."""
    rendered = DeepAgentsApp._render_extensions(
        {
            "registrations": [
                {
                    "kind": "tool",
                    "name": "name|forged",
                    "source": {"scope": "project", "path": "<b>/tmp/[extension]</b>"},
                }
            ],
            "errors": ["bad\n# injected"],
            "restart_required": True,
        }
    )

    assert "name\\|forged" in rendered
    assert "\\<b\\>" in rendered
    assert "bad # injected" in rendered
    assert "Run `/restart`" in rendered
