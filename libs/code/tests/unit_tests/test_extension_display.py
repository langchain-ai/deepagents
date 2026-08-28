"""Tests for the built-in extension provenance display."""

from unittest.mock import AsyncMock

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import get_slash_commands


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
