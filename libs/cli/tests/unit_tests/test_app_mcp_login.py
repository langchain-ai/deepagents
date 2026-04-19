"""Unit tests for the /mcp login <server> TUI path."""

from __future__ import annotations

from typing import get_args

import pytest

from deepagents_cli.app import DeferredActionKind, _parse_mcp_login_argv


def test_deferred_action_kind_includes_mcp_login() -> None:
    """`mcp_login` must be a valid deferred-action kind for deduplication."""
    assert "mcp_login" in get_args(DeferredActionKind)


def test_parse_mcp_login_argv_valid() -> None:
    assert _parse_mcp_login_argv("/mcp login notion") == "notion"


def test_parse_mcp_login_argv_extra_whitespace() -> None:
    assert _parse_mcp_login_argv("/mcp login   github  ") == "github"


def test_parse_mcp_login_argv_missing_server() -> None:
    with pytest.raises(ValueError, match="Usage: /mcp login <server>"):
        _parse_mcp_login_argv("/mcp login")


def test_parse_mcp_login_argv_multiple_args() -> None:
    with pytest.raises(ValueError, match="Usage: /mcp login <server>"):
        _parse_mcp_login_argv("/mcp login notion github")


def test_parse_mcp_login_argv_bad_name() -> None:
    with pytest.raises(ValueError, match="Invalid server name"):
        _parse_mcp_login_argv("/mcp login bad.name")


def test_parse_mcp_login_argv_bad_chars() -> None:
    with pytest.raises(ValueError, match="Invalid server name"):
        _parse_mcp_login_argv("/mcp login foo/bar")


from unittest.mock import AsyncMock, patch  # noqa: E402


class _StubApp:
    """Minimal stand-in for `DeepAgentsApp` exercising one method at a time.

    We don't construct the real Textual app because it binds to a terminal
    and mounts widgets; we only need attribute access to test the helpers.
    """

    def __init__(
        self,
        mcp_preload_kwargs: dict | None,
        mcp_server_info: list | None = None,
    ) -> None:
        self._mcp_preload_kwargs = mcp_preload_kwargs
        self._mcp_server_info = mcp_server_info
        self._mcp_tool_count = 0


@pytest.mark.asyncio
async def test_refresh_mcp_server_info_happy_path() -> None:
    """Successful refresh replaces the cached list and updates the count."""
    from deepagents_cli.app import DeepAgentsApp
    from deepagents_cli.mcp_tools import MCPServerInfo

    info = [MCPServerInfo(name="notion", transport="http")]
    info[0].tools = [object(), object()]  # two "tools"
    preload_kwargs = {
        "mcp_config_path": None,
        "no_mcp": False,
        "trust_project_mcp": None,
    }
    app = _StubApp(mcp_preload_kwargs=preload_kwargs)

    with patch(
        "deepagents_cli.main._preload_session_mcp_server_info",
        new=AsyncMock(return_value=info),
    ):
        await DeepAgentsApp._refresh_mcp_server_info(app)  # type: ignore[arg-type]

    assert app._mcp_server_info is info
    assert app._mcp_tool_count == 2


@pytest.mark.asyncio
async def test_refresh_mcp_server_info_no_preload_kwargs() -> None:
    """When preload kwargs are `None` (MCP disabled), refresh is a no-op."""
    from deepagents_cli.app import DeepAgentsApp

    app = _StubApp(mcp_preload_kwargs=None, mcp_server_info=None)
    await DeepAgentsApp._refresh_mcp_server_info(app)  # type: ignore[arg-type]
    assert app._mcp_server_info is None


@pytest.mark.asyncio
async def test_refresh_mcp_server_info_preload_failure_propagates() -> None:
    """Exceptions in the preloader propagate so the caller can warn."""
    from deepagents_cli.app import DeepAgentsApp

    preload_kwargs = {
        "mcp_config_path": None,
        "no_mcp": False,
        "trust_project_mcp": None,
    }
    app = _StubApp(mcp_preload_kwargs=preload_kwargs)

    with (
        patch(
            "deepagents_cli.main._preload_session_mcp_server_info",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await DeepAgentsApp._refresh_mcp_server_info(app)  # type: ignore[arg-type]


from contextlib import contextmanager  # noqa: E402


class _RecordingApp(_StubApp):
    """Stub that records `_mount_message` calls and stubs `suspend`."""

    def __init__(
        self,
        mcp_preload_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            mcp_preload_kwargs=mcp_preload_kwargs
            or {
                "mcp_config_path": "/tmp/mcp.json",
                "no_mcp": False,
                "trust_project_mcp": None,
            }
        )
        self.mounted: list = []
        self.suspend_entered = 0

        @contextmanager
        def _suspend() -> object:
            self.suspend_entered += 1
            yield

        self.suspend = _suspend  # type: ignore[method-assign]

    async def _mount_message(self, msg: object) -> None:
        self.mounted.append(msg)


@pytest.mark.asyncio
async def test_run_mcp_login_success_refreshes_and_notifies() -> None:
    """Exit 0 → refresh cache, mount success AppMessage, suspend was used."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    refresh = AsyncMock()
    app._refresh_mcp_server_info = refresh  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(return_value=0),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert app.suspend_entered == 1
    refresh.assert_awaited_once()
    assert any(
        isinstance(m, AppMessage) and "Logged in to MCP server 'notion'" in m._content
        for m in app.mounted
    ), app.mounted


@pytest.mark.asyncio
async def test_run_mcp_login_exit_1_generic_failure() -> None:
    """Exit 1 → mount generic failure, no refresh."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    refresh = AsyncMock()
    app._refresh_mcp_server_info = refresh  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(return_value=1),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    refresh.assert_not_called()
    assert any(
        isinstance(m, AppMessage) and "Login failed" in m._content for m in app.mounted
    )


@pytest.mark.asyncio
async def test_run_mcp_login_exit_2_no_config() -> None:
    """Exit 2 → mount "No MCP config found" message."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    app._refresh_mcp_server_info = AsyncMock()  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(return_value=2),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert any(
        isinstance(m, AppMessage) and "No MCP config found" in m._content
        for m in app.mounted
    )


@pytest.mark.asyncio
async def test_run_mcp_login_keyboard_interrupt() -> None:
    """Ctrl+C during paste-back → mount cancellation message."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    app._refresh_mcp_server_info = AsyncMock()  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(side_effect=KeyboardInterrupt()),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert any(
        isinstance(m, AppMessage) and "MCP login cancelled" in m._content
        for m in app.mounted
    )


@pytest.mark.asyncio
async def test_run_mcp_login_unexpected_exception() -> None:
    """Unexpected exceptions → mount failure message, do not raise."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    app._refresh_mcp_server_info = AsyncMock()  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(side_effect=RuntimeError("oops")),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert any(
        isinstance(m, AppMessage)
        and "MCP login failed" in m._content
        and "oops" in m._content
        for m in app.mounted
    )


@pytest.mark.asyncio
async def test_run_mcp_login_refresh_failure_still_succeeds() -> None:
    """Refresh failure → success message plus warning suffix."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp()
    app._refresh_mcp_server_info = AsyncMock(side_effect=RuntimeError("probe failed"))  # type: ignore[method-assign]

    with patch(
        "deepagents_cli.mcp_commands.run_mcp_login",
        new=AsyncMock(return_value=0),
    ):
        await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert any(
        isinstance(m, AppMessage)
        and "Logged in" in m._content
        and "couldn't refresh /mcp viewer" in m._content
        for m in app.mounted
    )


@pytest.mark.asyncio
async def test_run_mcp_login_no_preload_kwargs_errors() -> None:
    """When MCP is disabled (no preload kwargs) the command errors cleanly."""
    from deepagents_cli.app import AppMessage, DeepAgentsApp

    app = _RecordingApp(mcp_preload_kwargs=None)  # type: ignore[arg-type]
    # Force the stub to actually carry None
    app._mcp_preload_kwargs = None
    app._refresh_mcp_server_info = AsyncMock()  # type: ignore[method-assign]

    await DeepAgentsApp._run_mcp_login_interactive(app, "notion")  # type: ignore[arg-type]

    assert app.suspend_entered == 0  # should not suspend
    assert any(
        isinstance(m, AppMessage) and "MCP is disabled" in m._content
        for m in app.mounted
    )
