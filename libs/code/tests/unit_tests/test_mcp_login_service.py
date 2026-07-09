"""Tests for the UI-agnostic MCP login service layer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from deepagents_code.mcp_login_service import (
    ConfigErrorKind,
    ConfigResolution,
    ConfigResolutionError,
    ServerSelection,
    format_untrusted_project_notice,
    resolve_mcp_config,
    select_server,
)

if TYPE_CHECKING:
    import pytest


class TestResolveMcpConfigExplicit:
    """Explicit `--mcp-config <path>` resolution path."""

    def test_loads_valid_config_file(self, tmp_path: Path) -> None:
        """A valid explicit config returns a `ConfigResolution`."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        result = resolve_mcp_config(str(cfg))
        assert isinstance(result, ConfigResolution)
        assert result.used_paths == (Path(str(cfg)),)
        assert "notion" in result.config["mcpServers"]

    def test_invalid_explicit_config_returns_error(self, tmp_path: Path) -> None:
        """Invalid explicit configs surface a structured error, never a print."""
        cfg = tmp_path / "broken.json"
        cfg.write_text("not json")
        result = resolve_mcp_config(str(cfg))
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.EXPLICIT_LOAD_FAILED
        assert "Failed to load MCP config" in result.message

    def test_missing_explicit_config_returns_error(self, tmp_path: Path) -> None:
        """A missing explicit config still surfaces a structured error."""
        result = resolve_mcp_config(str(tmp_path / "nope.json"))
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.EXPLICIT_LOAD_FAILED

    def test_permission_error_on_explicit_config_returns_error(
        self, tmp_path: Path
    ) -> None:
        """An unreadable explicit config surfaces a structured error."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text('{"mcpServers":{}}')
        cfg.chmod(0o000)
        try:
            result = resolve_mcp_config(str(cfg))
        finally:
            cfg.chmod(0o644)
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.EXPLICIT_LOAD_FAILED


class TestResolveMcpConfigAutodiscover:
    """Auto-discovery resolution path."""

    def test_no_discovered_configs_returns_no_config_found(self) -> None:
        """Empty discovery yields the `NO_CONFIG_FOUND` reason."""
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[],
        ):
            result = resolve_mcp_config(None)
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_CONFIG_FOUND

    def test_untrusted_only_returns_no_usable_config_with_paths(
        self,
        tmp_path: Path,
    ) -> None:
        """An untrusted-only discovery returns the project paths it skipped."""
        project_cfg = tmp_path / "project.json"
        project_cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert result.untrusted_project_paths == (project_cfg,)

    def test_user_level_config_is_loaded_without_trust_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """User-level configs bypass the trust gate."""
        fake_home = tmp_path / "home"
        user_dir = fake_home / ".deepagents"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / ".mcp.json"
        user_cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[user_cfg],
        ):
            result = resolve_mcp_config(None)
        assert isinstance(result, ConfigResolution)
        assert result.used_paths == (user_cfg,)
        assert result.untrusted_project_paths == ()

    def test_user_config_with_untrusted_project_config_succeeds_with_notice(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """User config loads OK while an untrusted project config is noted."""
        fake_home = tmp_path / "home"
        user_dir = fake_home / ".deepagents"
        user_dir.mkdir(parents=True)
        user_cfg = user_dir / ".mcp.json"
        user_cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        project_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"}}}'
        )
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[user_cfg, project_cfg],
        ):
            result = resolve_mcp_config(None)
        # Resolution succeeds because the user config is usable.
        assert isinstance(result, ConfigResolution)
        assert user_cfg in result.used_paths
        # The untrusted project config is recorded so callers can surface the hint.
        assert result.untrusted_project_paths == (project_cfg,)
        # Only the user server is in the merged config; the project server is excluded.
        assert "notion" in result.config["mcpServers"]
        assert "slack" not in result.config["mcpServers"]


class TestSelectServer:
    """`select_server` server lookup and validation."""

    def test_unknown_server_returns_error(self, tmp_path: Path) -> None:
        """A name not in `mcpServers` returns a structured error."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        resolution = resolve_mcp_config(str(cfg))
        assert isinstance(resolution, ConfigResolution)
        result = select_server(resolution, "missing")
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.UNKNOWN_SERVER
        assert "missing" in result.message

    def test_invalid_server_config_returns_error(self) -> None:
        """Path-traversal server names are rejected at the selection step.

        Auto-discovery uses lenient loading, so a `../evil` entry can
        reach `select_server` even though strict loaders reject it
        upfront. The selection layer is the last line of defense.
        """
        resolution = ConfigResolution(
            config={
                "mcpServers": {
                    "../evil": {
                        "transport": "http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": "oauth",
                    }
                }
            },
            used_paths=(Path("/tmp/fake.json"),),
        )
        result = select_server(resolution, "../evil")
        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.INVALID_SERVER_CONFIG

    def test_happy_path_returns_selection(self, tmp_path: Path) -> None:
        """A valid lookup returns the server entry and a search label."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"mcpServers":{"notion":{"transport":"http",'
            '"url":"https://mcp.notion.com/mcp","auth":"oauth"}}}'
        )
        resolution = resolve_mcp_config(str(cfg))
        assert isinstance(resolution, ConfigResolution)
        selection = select_server(resolution, "notion")
        assert isinstance(selection, ServerSelection)
        assert selection.server_name == "notion"
        assert selection.server_config["url"] == "https://mcp.notion.com/mcp"
        assert str(cfg) in selection.search_label


class TestFormatUntrustedProjectNotice:
    """`format_untrusted_project_notice` rendering."""

    def test_empty_returns_empty_string(self) -> None:
        """No untrusted paths means no notice."""
        assert format_untrusted_project_notice(()) == ""

    def test_includes_each_path_and_trust_hint(self, tmp_path: Path) -> None:
        """The rendered notice names each skipped path and the trust hint."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        notice = format_untrusted_project_notice((a, b))
        assert str(a) in notice
        assert str(b) in notice
        assert "pass --mcp-config <path> to use it explicitly" in notice
