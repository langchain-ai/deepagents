"""Tests for MCP configuration environment-variable expansion."""

from __future__ import annotations

import pytest

from deepagents_code.mcp_config import resolve_mcp_server_env


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

    def test_resolves_slack_app_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom credentials expand without mutating the source config."""
        monkeypatch.setenv("SLACK_CLIENT_ID", "custom-app")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "private-value")
        config = {
            "url": "https://mcp.slack.com/mcp",
            "auth": "oauth",
            "oauth": {
                "client_id": "${SLACK_CLIENT_ID}",
                "client_secret": "${SLACK_CLIENT_SECRET}",
            },
        }
        resolved = resolve_mcp_server_env("slack", config)
        assert resolved["oauth"] == {
            "client_id": "custom-app",
            "client_secret": "private-value",
        }
        assert config["oauth"]["client_secret"] == "${SLACK_CLIENT_SECRET}"

    @pytest.mark.parametrize(
        "oauth",
        [
            None,
            [],
            {},
            {"client_id": ""},
            {"client_id": 123},
            {"client_id": "app", "client_secret": ""},
            {"client_id": "app", "client_secret": "private\nvalue"},
            {"client_id": "app", "unknown": "private-value"},
        ],
    )
    def test_rejects_invalid_slack_credentials(self, oauth: object) -> None:
        """Invalid settings fail without reflecting credential values."""
        with pytest.raises((TypeError, ValueError)) as error:
            resolve_mcp_server_env(
                "slack",
                {
                    "url": "https://mcp.slack.com/mcp",
                    "auth": "oauth",
                    "oauth": oauth,
                },
            )
        assert "private" not in str(error.value)

    @pytest.mark.parametrize(
        "url", ["https://example.com/mcp", "https://slack.com.evil.test/mcp"]
    )
    def test_rejects_app_credentials_for_other_hosts(self, url: str) -> None:
        """Slack-specific app settings cannot affect another provider."""
        with pytest.raises(ValueError, match="only for Slack"):
            resolve_mcp_server_env(
                "server",
                {
                    "url": url,
                    "auth": "oauth",
                    "oauth": {"client_id": "app"},
                },
            )
