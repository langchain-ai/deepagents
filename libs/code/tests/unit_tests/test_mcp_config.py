"""Tests for MCP configuration environment-variable expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.mcp_config import resolve_mcp_server_env

if TYPE_CHECKING:
    import pytest


class TestResolveMcpServerEnv:
    """Tests for supported `.mcp.json` interpolation fields."""

    def test_resolves_remote_url_and_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`url` and header values resolve multiple references and defaults."""
        monkeypatch.setenv("MCP_HOST", "mcp.example.com")
        monkeypatch.setenv("MCP_TOKEN", "token")
        monkeypatch.delenv("MCP_SCHEME", raising=False)

        resolved = resolve_mcp_server_env(
            "remote",
            {
                "url": "${MCP_SCHEME:-https}://${MCP_HOST}/mcp",
                "headers": {
                    "Authorization": "Bearer ${MCP_TOKEN}",
                    "X-Origin": "${MCP_SCHEME:-https}-${MCP_HOST}",
                },
            },
        )

        assert resolved["url"] == "https://mcp.example.com/mcp"
        assert resolved["headers"] == {
            "Authorization": "Bearer token",
            "X-Origin": "https-mcp.example.com",
        }

    def test_resolved_value_containing_brace_is_not_rescanned(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A resolved value that itself contains `${` is emitted verbatim.

        The malformed-reference guard runs against the raw config string, so a
        substituted value that happens to contain `${...}` neither re-expands
        nor trips the malformed check.
        """
        monkeypatch.setenv("MCP_LITERAL", "keep-${NOT_A_REF}-literal")

        resolved = resolve_mcp_server_env("srv", {"command": "${MCP_LITERAL}"})

        assert resolved["command"] == "keep-${NOT_A_REF}-literal"
