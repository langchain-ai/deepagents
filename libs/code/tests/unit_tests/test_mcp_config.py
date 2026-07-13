"""Tests for MCP configuration environment-variable expansion."""

from __future__ import annotations

from typing import Any

import pytest

from deepagents_code.mcp_config import resolve_mcp_server_env


class TestResolveMcpServerEnv:
    """Tests for supported `.mcp.json` interpolation fields."""

    def test_resolves_stdio_fields_without_mutating_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`command`, `args`, and `env` resolve in a copied config."""
        monkeypatch.setenv("MCP_HOME", "/opt/mcp")
        monkeypatch.setenv("MCP_TOKEN", "secret")
        monkeypatch.delenv("MCP_CACHE", raising=False)
        config: dict[str, Any] = {
            "command": "${MCP_HOME}/bin/server",
            "args": ["--root", "${MCP_HOME}", "${MCP_CACHE:-/tmp/cache}"],
            "env": {
                "TOKEN": "prefix-${MCP_TOKEN}",
                "CACHE": "${MCP_CACHE:-/tmp/cache}",
            },
        }

        resolved = resolve_mcp_server_env("discourse", config)

        assert resolved == {
            "command": "/opt/mcp/bin/server",
            "args": ["--root", "/opt/mcp", "/tmp/cache"],
            "env": {"TOKEN": "prefix-secret", "CACHE": "/tmp/cache"},
        }
        assert config["command"] == "${MCP_HOME}/bin/server"
        assert config["env"]["TOKEN"] == "prefix-${MCP_TOKEN}"

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

    def test_empty_variable_uses_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The `:-` form uses its default for an empty variable."""
        monkeypatch.setenv("MCP_SCHEME", "")

        resolved = resolve_mcp_server_env(
            "remote",
            {"url": "${MCP_SCHEME:-https}://example.com"},
        )

        assert resolved["url"] == "https://example.com"

    def test_plain_dollar_and_unsupported_fields_are_unchanged(self) -> None:
        """Only braced references in the supported field allowlist expand."""
        config = {
            "command": "$HOME/bin/server",
            "allowedTools": ["${SHOULD_NOT_EXPAND}"],
        }

        resolved = resolve_mcp_server_env("srv", config)

        assert resolved == config

    def test_unset_variable_reports_exact_field_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing required variables identify the server and field."""
        monkeypatch.delenv("MISSING_MCP_PATH", raising=False)

        with pytest.raises(
            RuntimeError,
            match=r"mcpServers\.srv\.args\[1\].*MISSING_MCP_PATH",
        ):
            resolve_mcp_server_env(
                "srv",
                {"command": "node", "args": ["--root", "${MISSING_MCP_PATH}"]},
            )

    def test_non_string_supported_value_reports_exact_field_path(self) -> None:
        """Malformed supported values fail with a field-specific error."""
        with pytest.raises(TypeError, match=r"mcpServers\.srv\.env\.PORT"):
            resolve_mcp_server_env("srv", {"command": "node", "env": {"PORT": 1}})

    def test_non_string_args_element_reports_indexed_field_path(self) -> None:
        """A non-string element inside `args` names its index."""
        with pytest.raises(TypeError, match=r"mcpServers\.srv\.args\[0\]"):
            resolve_mcp_server_env("srv", {"command": "node", "args": [1]})

    def test_non_string_header_value_reports_field_path(self) -> None:
        """A non-string header value names the offending header."""
        with pytest.raises(TypeError, match=r"mcpServers\.srv\.headers\.X-Bad"):
            resolve_mcp_server_env(
                "srv",
                {"url": "https://x", "headers": {"X-Bad": 1}},
            )

    def test_args_not_a_list_reports_field_path(self) -> None:
        """`args` must be a list, not a bare string."""
        with pytest.raises(TypeError, match=r"mcpServers\.srv\.args must be a list"):
            resolve_mcp_server_env("srv", {"command": "node", "args": "solo"})

    def test_mapping_field_not_a_dict_reports_field_path(self) -> None:
        """`env`/`headers` must be dictionaries."""
        with pytest.raises(
            TypeError,
            match=r"mcpServers\.srv\.headers must be a dictionary",
        ):
            resolve_mcp_server_env("srv", {"url": "https://x", "headers": ["nope"]})

    @pytest.mark.parametrize(
        "value",
        ["${VAR-default}", "${VAR:default}", "prefix-${VAR"],
    )
    def test_malformed_reference_is_rejected(self, value: str) -> None:
        """An unparseable `${...}` fails instead of being emitted verbatim."""
        with pytest.raises(RuntimeError, match=r"malformed"):
            resolve_mcp_server_env("srv", {"command": value})
