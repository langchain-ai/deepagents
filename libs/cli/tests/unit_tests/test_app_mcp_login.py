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
