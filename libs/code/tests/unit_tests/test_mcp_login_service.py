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
    format_legacy_ignored_notice,
    format_untrusted_project_notice,
    resolve_mcp_config,
    select_server,
)

if TYPE_CHECKING:
    import pytest


def _project_approval_config(
    project_root: Path,
    name: str,
    server: object,
    *,
    disabled: list[str] | None = None,
) -> str:
    """Build user config text with one scoped project MCP approval."""
    from deepagents_code.model_config import fingerprint_mcp_server_config

    text = (
        "[mcp]\n"
        "enabled_project_server_approvals = ["
        f'{{ project_root = "{project_root}", name = "{name}", '
        f'fingerprint = "{fingerprint_mcp_server_config(server)}" }}]\n'
    )
    if disabled:
        quoted = ", ".join(f'"{item}"' for item in disabled)
        text += f"disabled_project_servers = [{quoted}]\n"
    return text


def _isolate_project_mcp_trust_lists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_text: str = "[mcp]\n",
) -> Path:
    """Point MCP trust lists at a test-only user config."""
    from deepagents_code import _env_vars

    user_config = tmp_path / "config.toml"
    user_config.write_text(config_text)
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", user_config)
    monkeypatch.delenv(_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS, raising=False)
    monkeypatch.delenv(_env_vars.DISABLED_PROJECT_MCP_SERVERS, raising=False)
    return user_config


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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An untrusted-only discovery returns the project paths it skipped."""
        _isolate_project_mcp_trust_lists(monkeypatch, tmp_path)
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

    def test_legacy_enabled_project_servers_is_surfaced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A legacy flat allowlist is reported so login can explain the change.

        `mcp login` is non-interactive (no approval prompt), so a user who
        relied on the removed `[mcp].enabled_project_servers` key must be told
        why their server stopped loading rather than have it vanish silently.
        """
        _isolate_project_mcp_trust_lists(
            monkeypatch, tmp_path, '[mcp]\nenabled_project_servers = ["notion"]\n'
        )
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
        assert result.legacy_ignored == ("notion",)

    def test_unreadable_policy_fails_closed_and_surfaces_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken trust policy skips even a previously-approved project server.

        Regression guard: a valid approval plus a wrong-typed
        `disabled_project_servers` sets `read_error` and empties the deny set, so
        `is_enabled` would still match the live approval. The login resolver must
        instead fail closed (a revoked-but-mistyped deny must not be bypassed)
        and surface the policy read error rather than a bare "no usable config".
        """
        from deepagents_code.model_config import fingerprint_mcp_server_config

        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        slack = {"type": "http", "url": "https://slack.com/mcp", "auth": "oauth"}
        project_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"}}}'
        )
        # Valid approval for slack, but a wrong-typed deny list -> read_error.
        config_text = (
            "[mcp]\n"
            "enabled_project_server_approvals = ["
            f'{{ project_root = "{project_cfg.parent}", name = "slack", '
            f'fingerprint = "{fingerprint_mcp_server_config(slack)}" }}]\n'
            "disabled_project_servers = 123\n"
        )
        _isolate_project_mcp_trust_lists(monkeypatch, tmp_path, config_text)
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert result.untrusted_project_paths == (project_cfg,)
        assert "config.toml" in result.message

    def test_broken_project_config_surfaces_parse_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A malformed `.mcp.json` surfaces its parse error, not a bare not-found."""
        _isolate_project_mcp_trust_lists(monkeypatch, tmp_path)
        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        project_cfg.write_text("{not valid json")
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert "load errors" in result.message
        assert str(project_cfg) in result.message

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
        _isolate_project_mcp_trust_lists(monkeypatch, tmp_path)
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

    def test_allowlisted_project_server_is_loaded_for_login(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An always-allowed project server is available to `mcp login`."""
        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        slack = {"type": "http", "url": "https://slack.com/mcp", "auth": "oauth"}
        project_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"},'
            '"other":{"type":"http","url":"https://example.com/mcp",'
            '"auth":"oauth"}}}'
        )
        _isolate_project_mcp_trust_lists(
            monkeypatch,
            tmp_path,
            _project_approval_config(project_cfg.parent, "slack", slack),
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolution)
        assert result.used_paths == (project_cfg,)
        assert set(result.config["mcpServers"]) == {"slack"}
        assert result.untrusted_project_paths == (project_cfg,)

    def test_changed_override_hides_approved_lower_precedence_server(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Login cannot select a stale approval shadowed by a changed server."""
        project = tmp_path / "project"
        nested_cfg = project / ".deepagents" / ".mcp.json"
        root_cfg = project / ".mcp.json"
        nested_cfg.parent.mkdir(parents=True)
        approved = {"type": "http", "url": "https://safe.test/mcp"}
        nested_cfg.write_text(
            '{"mcpServers":{"docs":{"type":"http","url":"https://safe.test/mcp"}}}'
        )
        root_cfg.write_text(
            '{"mcpServers":{"docs":{"type":"http","url":"https://changed.test/mcp"}}}'
        )
        _isolate_project_mcp_trust_lists(
            monkeypatch,
            tmp_path,
            _project_approval_config(project, "docs", approved),
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[nested_cfg, root_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert result.untrusted_project_paths == (root_cfg,)

    def test_env_approval_survives_unreadable_trust_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Login retains explicit env approvals when trust TOML is unreadable."""
        from deepagents_code import _env_vars

        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        project_cfg.write_text(
            '{"mcpServers":{"docs":{"type":"http",'
            '"url":"https://docs.test/mcp"},'
            '"other":{"type":"http","url":"https://other.test/mcp"}}}'
        )
        _isolate_project_mcp_trust_lists(monkeypatch, tmp_path, "[[not valid toml")
        monkeypatch.setenv(_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS, "docs")
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolution)
        assert set(result.config["mcpServers"]) == {"docs"}
        assert result.used_paths == (project_cfg,)
        assert result.untrusted_project_paths == (project_cfg,)

    def test_invalid_unapproved_sibling_does_not_block_login(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An invalid unapproved entry cannot block an approved login target."""
        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        slack = {"type": "http", "url": "https://slack.com/mcp", "auth": "oauth"}
        project_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"},'
            '"broken":{"args":[]}}}'
        )
        _isolate_project_mcp_trust_lists(
            monkeypatch,
            tmp_path,
            _project_approval_config(project_cfg.parent, "slack", slack),
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolution)
        assert result.used_paths == (project_cfg,)
        assert set(result.config["mcpServers"]) == {"slack"}
        assert result.untrusted_project_paths == (project_cfg,)

    def test_symlinked_project_config_uses_containing_project_scope(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Approving a symlink target project does not approve the symlink repo."""
        approved_project = tmp_path / "approved"
        attack_project = tmp_path / "attack"
        approved_project.mkdir()
        attack_project.mkdir()
        approved_cfg = approved_project / ".mcp.json"
        attack_cfg = attack_project / ".mcp.json"
        slack = {"type": "http", "url": "https://slack.com/mcp", "auth": "oauth"}
        approved_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"}}}'
        )
        attack_cfg.symlink_to(approved_cfg)
        _isolate_project_mcp_trust_lists(
            monkeypatch,
            tmp_path,
            _project_approval_config(approved_project, "slack", slack),
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[attack_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert result.untrusted_project_paths == (attack_cfg,)

    def test_disabled_project_server_overrides_allowlist_for_login(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A disabled project server stays unavailable even if also enabled."""
        project_cfg = tmp_path / "project" / ".mcp.json"
        project_cfg.parent.mkdir()
        slack = {"type": "http", "url": "https://slack.com/mcp", "auth": "oauth"}
        project_cfg.write_text(
            '{"mcpServers":{"slack":{"type":"http",'
            '"url":"https://slack.com/mcp","auth":"oauth"}}}'
        )
        _isolate_project_mcp_trust_lists(
            monkeypatch,
            tmp_path,
            _project_approval_config(
                project_cfg.parent, "slack", slack, disabled=["slack"]
            ),
        )
        with patch(
            "deepagents_code.mcp_tools.discover_mcp_configs",
            return_value=[project_cfg],
        ):
            result = resolve_mcp_config(None)

        assert isinstance(result, ConfigResolutionError)
        assert result.kind is ConfigErrorKind.NO_USABLE_CONFIG
        assert result.untrusted_project_paths == (project_cfg,)


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
        assert "pass --mcp-config <path> to use the file explicitly" in notice


class TestFormatLegacyIgnoredNotice:
    """`format_legacy_ignored_notice` rendering."""

    def test_empty_returns_empty_string(self) -> None:
        """No ignored names means no notice."""
        assert format_legacy_ignored_notice(()) == ""

    def test_names_and_migration_hint(self) -> None:
        """The notice names each ignored server and how to re-approve."""
        notice = format_legacy_ignored_notice(("docs", "slack"))
        assert "docs" in notice
        assert "slack" in notice
        assert "enabled_project_servers is no longer used" in notice
        assert "dcode" in notice
