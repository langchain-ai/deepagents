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
