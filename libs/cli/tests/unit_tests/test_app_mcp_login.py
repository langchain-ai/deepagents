"""Unit tests for the /mcp login <server> TUI path."""

from __future__ import annotations

from typing import get_args

from deepagents_cli.app import DeferredActionKind


def test_deferred_action_kind_includes_mcp_login() -> None:
    """`mcp_login` must be a valid deferred-action kind for deduplication."""
    assert "mcp_login" in get_args(DeferredActionKind)
