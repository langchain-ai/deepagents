"""Unit tests for main entry point."""

import asyncio
import inspect
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import AppResult, DeepAgentsApp, run_textual_app
from deepagents_cli.config import build_langsmith_thread_url, reset_langsmith_url_cache
from deepagents_cli.main import (
    _ripgrep_install_hint,
    build_missing_tool_notification,
    check_optional_tools,
    format_tool_warning_cli,
    run_textual_cli_async,
)


class TestResumeHintLogic:
    """Test that resume hint logic is correct.

    The actual condition in `cli_main` is::

        thread_id and return_code == 0 and asyncio.run(thread_exists(thread_id))

    These tests mirror the three-part condition. `thread_exists` is
    represented as a boolean to keep the tests as pure unit tests.
    """

    def test_resume_hint_condition_error_case(self) -> None:
        """Resume hint should NOT be shown when return_code is non-zero."""
        thread_id = "test123"
        return_code = 1
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "Resume hint should not be shown on error"

    def test_resume_hint_condition_success_case(self) -> None:
        """Resume hint SHOULD be shown on success with checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown on success"

    def test_resume_hint_shown_for_resumed_threads(self) -> None:
        """Resume hint SHOULD be shown for resumed threads too."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown for resumed threads"

    def test_resume_hint_not_shown_without_checkpoints(self) -> None:
        """Resume hint should NOT appear when thread has no checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = False

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "No hint when thread_exists returns False"


class TestLangSmithTeardownUrl:
    """Test LangSmith thread URL display logic on teardown."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_thread_url_requires_all_components(self) -> None:
        """LangSmith link requires thread_id, project_name, and project_url."""
        thread_url = build_langsmith_thread_url("abc123")
        # Without LangSmith configured, should return None
        assert thread_url is None

    def test_thread_url_not_shown_for_none_thread_id(self) -> None:
        """Guard condition: thread_url and thread_exists both needed."""
        thread_url = None
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_not_shown_when_no_checkpoints(self) -> None:
        """Guard condition: thread must have checkpointed content."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = False
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_shown_when_all_conditions_met(self) -> None:
        """Guard condition: both thread_url and thread_exists must be truthy."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert show_link


class TestAppResult:
    """Tests for the AppResult dataclass."""

    def test_fields_accessible(self) -> None:
        """AppResult should expose return_code and thread_id."""
        result = AppResult(return_code=0, thread_id="tid-abc")
        assert result.return_code == 0
        assert result.thread_id == "tid-abc"

    def test_thread_id_none(self) -> None:
        """AppResult should accept None for thread_id."""
        result = AppResult(return_code=1, thread_id=None)
        assert result.thread_id is None

    def test_frozen(self) -> None:
        """AppResult should be immutable."""
        from dataclasses import FrozenInstanceError

        result = AppResult(return_code=0, thread_id="tid")
        with pytest.raises(FrozenInstanceError):
            result.return_code = 1  # type: ignore[misc]


class TestRunTextualAppReturnType:
    """Test that run_textual_app returns AppResult."""

    async def test_run_textual_app_returns_app_result(self) -> None:
        """run_textual_app should return an AppResult."""
        sig = inspect.signature(run_textual_app)
        annotation = sig.return_annotation
        assert annotation in (AppResult, "AppResult"), (
            f"run_textual_app should return AppResult, got {annotation}"
        )


class TestRunTextualCliAsyncReturnType:
    """Test that run_textual_cli_async returns AppResult."""

    def test_run_textual_cli_async_returns_app_result(self) -> None:
        """run_textual_cli_async should return an AppResult."""
        sig = inspect.signature(run_textual_cli_async)
        assert sig.return_annotation in (AppResult, "AppResult"), (
            "run_textual_cli_async should return AppResult, "
            f"got {sig.return_annotation}"
        )


class TestThreadMessage:
    """Test thread info display format.

    Thread info is now displayed in the WelcomeBanner widget rather than via
    pre-TUI console output, so we verify the banner receives the thread ID.
    """

    def test_thread_id_forwarded_to_app(self) -> None:
        """run_textual_cli_async passes thread_id to run_textual_app."""
        source = inspect.getsource(run_textual_cli_async)
        assert "thread_id=thread_id" in source, (
            "thread_id should be forwarded to run_textual_app"
        )


class TestRunTextualCliAsyncMcp:
    """Tests for MCP/server kwargs forwarding in interactive server mode.

    Server startup and MCP preload now happen inside the TUI via deferred
    kwargs rather than being invoked directly in `run_textual_cli_async`.
    """

    async def test_passes_server_and_mcp_kwargs_to_textual_app(self) -> None:
        """TUI should receive server, mcp_preload, and model kwargs."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with patch("deepagents_cli.app.run_textual_app", new=_run_textual_app_stub):
            result = await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-4o",
            )

        assert result == app_result

        # Server kwargs forwarded for deferred startup inside the TUI
        assert captured_kwargs["server_kwargs"] is not None
        assert captured_kwargs["server_kwargs"]["assistant_id"] == "agent"
        assert captured_kwargs["server_kwargs"]["interactive"] is True
        # auto_approve must NOT be in server_kwargs — the interactive server
        # must always compile with full HITL interrupts so Shift+Tab works.
        assert "auto_approve" not in captured_kwargs["server_kwargs"]

        # MCP preload kwargs forwarded (no_mcp=False by default)
        assert captured_kwargs["mcp_preload_kwargs"] is not None
        assert captured_kwargs["mcp_preload_kwargs"]["no_mcp"] is False

        # Model kwargs forwarded for deferred create_model() inside the TUI
        assert captured_kwargs["model_kwargs"] is not None
        assert captured_kwargs["model_kwargs"]["model_spec"] == "openai:gpt-4o"
        assert captured_kwargs["model_kwargs"]["extra_kwargs"] is None

    async def test_no_mcp_kwargs_when_disabled(self) -> None:
        """mcp_preload_kwargs should be None when no_mcp=True."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with patch("deepagents_cli.app.run_textual_app", new=_run_textual_app_stub):
            await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-4o",
                no_mcp=True,
            )

        assert captured_kwargs["mcp_preload_kwargs"] is None

    async def test_onboarding_trigger_reaches_textual_app(self) -> None:
        """First-run onboarding state should control the app launch flag."""
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with (
            patch("deepagents_cli.app.run_textual_app", new=_run_textual_app_stub),
            patch("deepagents_cli.onboarding.should_run_onboarding", return_value=True),
        ):
            await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                model_name="openai:gpt-4o",
            )

        assert captured_kwargs["launch_init"] is True


class TestServerCleanupLifecycle:
    """Verify server_proc.stop() is guaranteed after the TUI exits."""

    async def test_server_proc_stopped_after_app_exits(self) -> None:
        """run_textual_app must call server_proc.stop() in the finally block."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new_callable=AsyncMock,
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # type: ignore[invalid-argument-type]

        server_proc.stop.assert_called_once_with()

    async def test_server_proc_stopped_even_on_crash(self) -> None:
        """server_proc.stop() must fire even when run_async raises."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with (
            patch.object(
                DeepAgentsApp,
                "run_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # type: ignore[invalid-argument-type]

        server_proc.stop.assert_called_once_with()

    async def test_deferred_server_proc_stopped_after_app_exits(self) -> None:
        """server_proc set by the background worker must still be cleaned up."""
        server_proc = SimpleNamespace(stop=MagicMock())

        async def _fake_run_async(self: DeepAgentsApp) -> None:  # noqa: RUF029
            # Simulate the background worker having set _server_proc
            self._server_proc = server_proc

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new=_fake_run_async,
        ):
            await run_textual_app(
                server_kwargs={"assistant_id": "a"},
                thread_id="t-1",
            )

        server_proc.stop.assert_called_once_with()


class TestCheckOptionalTools:
    """Tests for check_optional_tools() function."""

    @pytest.fixture(autouse=True)
    def _tavily_available(self) -> Iterator[None]:
        """Patch settings.has_tavily to True so ripgrep-only tests stay isolated."""
        with patch(
            "deepagents_cli.config.settings",
            SimpleNamespace(has_tavily=True),
        ):
            yield

    def test_returns_tool_name_when_rg_not_found(self) -> None:
        """Returns `['ripgrep']` when `rg` is not on PATH."""
        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools()

        assert missing == ["ripgrep"]

    def test_returns_empty_when_rg_found(self) -> None:
        """Returns empty list when `rg` is found on PATH."""
        with patch("deepagents_cli.main.shutil.which", return_value="/usr/bin/rg"):
            missing = check_optional_tools()

        assert missing == []

    def test_warning_suppressed_via_config(self, tmp_path: Path) -> None:
        """Returns empty list when ripgrep warning is suppressed in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == []

    def test_malformed_config_does_not_suppress(self, tmp_path: Path) -> None:
        """Malformed TOML config degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml [[[")

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_non_list_suppress_does_not_crash(self, tmp_path: Path) -> None:
        """Non-list `suppress` value degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[warnings]\nsuppress = true\n")

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_unrelated_suppress_key_does_not_suppress(self, tmp_path: Path) -> None:
        """Suppressing a different key does not suppress the ripgrep warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["something_else"]\n')

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_returns_tavily_when_key_missing(self) -> None:
        """Returns `'tavily'` when TAVILY_API_KEY is not set."""
        with (
            patch("deepagents_cli.main.shutil.which", return_value="/usr/bin/rg"),
            patch(
                "deepagents_cli.config.settings",
                SimpleNamespace(has_tavily=False),
            ),
        ):
            missing = check_optional_tools()

        assert missing == ["tavily"]

    def test_omits_tavily_when_key_present(self) -> None:
        """Does not include `'tavily'` when TAVILY_API_KEY is set."""
        with patch("deepagents_cli.main.shutil.which", return_value="/usr/bin/rg"):
            missing = check_optional_tools()

        assert "tavily" not in missing

    def test_tavily_warning_suppressed_via_config(self, tmp_path: Path) -> None:
        """Returns empty list when tavily warning is suppressed in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["tavily"]\n')

        with (
            patch("deepagents_cli.main.shutil.which", return_value="/usr/bin/rg"),
            patch(
                "deepagents_cli.config.settings",
                SimpleNamespace(has_tavily=False),
            ),
        ):
            missing = check_optional_tools(config_path=config_path)

        assert missing == []


class TestRipgrepInstallHint:
    """Tests for platform-specific ripgrep install hints."""

    def test_macos_brew(self) -> None:
        """Returns brew command on macOS when brew is available."""

        def _which(cmd: str) -> str | None:
            return "/opt/homebrew/bin/brew" if cmd == "brew" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "brew install ripgrep"

    def test_macos_port(self) -> None:
        """Falls back to MacPorts when brew is absent."""

        def _which(cmd: str) -> str | None:
            return "/opt/local/bin/port" if cmd == "port" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "sudo port install ripgrep"

    def test_linux_apt(self) -> None:
        """Returns apt-get command on Debian/Ubuntu."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/apt-get" if cmd == "apt-get" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo apt-get install ripgrep"

    def test_linux_dnf(self) -> None:
        """Returns dnf command on Fedora/RHEL."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/dnf" if cmd == "dnf" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo dnf install ripgrep"

    def test_linux_pacman(self) -> None:
        """Returns pacman command on Arch."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/pacman" if cmd == "pacman" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo pacman -S ripgrep"

    def test_linux_zypper(self) -> None:
        """Returns zypper command on openSUSE."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/zypper" if cmd == "zypper" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo zypper install ripgrep"

    def test_linux_apk(self) -> None:
        """Returns apk command on Alpine."""

        def _which(cmd: str) -> str | None:
            return "/sbin/apk" if cmd == "apk" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "sudo apk add ripgrep"

    def test_linux_nix(self) -> None:
        """Returns nix-env command on NixOS."""

        def _which(cmd: str) -> str | None:
            if cmd == "nix-env":
                return "/nix/var/nix/profiles/default/bin/nix-env"
            return None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "linux"
            assert _ripgrep_install_hint() == "nix-env -iA nixpkgs.ripgrep"

    def test_win32_choco(self) -> None:
        """Returns choco command on Windows when available."""

        def _which(cmd: str) -> str | None:
            if cmd == "choco":
                return "C:\\ProgramData\\chocolatey\\bin\\choco.exe"
            return None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "choco install ripgrep"

    def test_win32_scoop(self) -> None:
        """Returns scoop command on Windows when available."""

        def _which(cmd: str) -> str | None:
            if cmd == "scoop":
                return "C:\\Users\\user\\scoop\\shims\\scoop.exe"
            return None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "scoop install ripgrep"

    def test_win32_winget(self) -> None:
        """Returns winget command on Windows when available."""

        def _which(cmd: str) -> str | None:
            return "C:\\winget.exe" if cmd == "winget" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "win32"
            assert _ripgrep_install_hint() == "winget install BurntSushi.ripgrep"

    def test_darwin_no_manager_falls_through(self) -> None:
        """Falls through to cross-platform on macOS without brew/port."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/cargo" if cmd == "cargo" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "darwin"
            assert _ripgrep_install_hint() == "cargo install ripgrep"

    def test_linux_no_manager_falls_through(self) -> None:
        """Falls through to URL on Linux without any package manager."""
        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", return_value=None),
        ):
            mock_sys.platform = "linux"
            assert "github.com/BurntSushi/ripgrep" in _ripgrep_install_hint()

    def test_cargo_fallback(self) -> None:
        """Falls back to cargo when no system package manager found."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/cargo" if cmd == "cargo" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "freebsd"
            assert _ripgrep_install_hint() == "cargo install ripgrep"

    def test_conda_fallback(self) -> None:
        """Falls back to conda when no other manager found."""

        def _which(cmd: str) -> str | None:
            return "/usr/bin/conda" if cmd == "conda" else None

        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", side_effect=_which),
        ):
            mock_sys.platform = "freebsd"
            assert _ripgrep_install_hint() == "conda install -c conda-forge ripgrep"

    def test_url_fallback(self) -> None:
        """Returns GitHub URL when nothing is detected."""
        with (
            patch("deepagents_cli.main.sys") as mock_sys,
            patch("deepagents_cli.main.shutil.which", return_value=None),
        ):
            mock_sys.platform = "freebsd"
            hint = _ripgrep_install_hint()
            assert hint.startswith("https://")
            assert "github.com/BurntSushi/ripgrep" in hint


class TestFormatToolWarnings:
    """Tests for the CLI warning formatter and the notification builder."""

    def test_cli_format_contains_install_hint(self) -> None:
        """CLI format includes a platform-specific install hint."""
        hint_patch = patch(
            "deepagents_cli.main._ripgrep_install_hint",
            return_value="brew install ripgrep",
        )
        with hint_patch:
            msg = format_tool_warning_cli("ripgrep")
        assert "brew install ripgrep" in msg

    def test_cli_format_wraps_url_in_rich_link(self) -> None:
        """CLI format wraps URL fallback in Rich `[link]` markup."""
        url = "https://github.com/BurntSushi/ripgrep#installation"
        hint_patch = patch(
            "deepagents_cli.main._ripgrep_install_hint",
            return_value=url,
        )
        with hint_patch:
            msg = format_tool_warning_cli("ripgrep")
        assert f"[link={url}]" in msg
        assert "[/link]" in msg

    def test_cli_format_contains_config_hint(self) -> None:
        """CLI format references config.toml for suppression."""
        msg = format_tool_warning_cli("ripgrep")
        assert "config.toml" in msg
        assert 'suppress = \\["ripgrep"]' in msg

    def test_cli_format_unknown_tool_fallback(self) -> None:
        """Unknown tools get a generic CLI message."""
        assert format_tool_warning_cli("foo") == "foo is not installed."

    def test_cli_format_tavily_contains_env_hint(self) -> None:
        """CLI format for tavily mentions the env var with Rich link."""
        msg = format_tool_warning_cli("tavily")
        assert "TAVILY_API_KEY" in msg
        assert "[link=https://tavily.com]" in msg

    def test_cli_format_tavily_contains_config_hint(self) -> None:
        """CLI tavily format references config.toml for suppression."""
        msg = format_tool_warning_cli("tavily")
        assert "config.toml" in msg
        assert 'suppress = \\["tavily"]' in msg


class TestBuildMissingToolNotification:
    """Tests for `build_missing_tool_notification` registry factory."""

    def test_ripgrep_with_package_manager_hint(self) -> None:
        """Ripgrep with install command offers copy + open-website + suppress."""
        from deepagents_cli.main import _RIPGREP_URL
        from deepagents_cli.notifications import ActionId, MissingDepPayload

        with patch(
            "deepagents_cli.main._ripgrep_install_hint",
            return_value="brew install ripgrep",
        ):
            entry = build_missing_tool_notification("ripgrep")
        assert entry.key == "dep:ripgrep"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "ripgrep"
        assert entry.payload.install_command == "brew install ripgrep"
        assert entry.payload.url == _RIPGREP_URL
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [
            ActionId.COPY_INSTALL,
            ActionId.OPEN_WEBSITE,
            ActionId.SUPPRESS,
        ]
        assert entry.actions[0].primary is True

    def test_ripgrep_url_fallback_opens_website(self) -> None:
        """Ripgrep with URL fallback offers open-website + suppress."""
        from deepagents_cli.notifications import ActionId, MissingDepPayload

        url = "https://github.com/BurntSushi/ripgrep#installation"
        with patch(
            "deepagents_cli.main._ripgrep_install_hint",
            return_value=url,
        ):
            entry = build_missing_tool_notification("ripgrep")
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.url == url
        assert entry.payload.install_command is None
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [ActionId.OPEN_WEBSITE, ActionId.SUPPRESS]

    def test_tavily_offers_website_and_suppress(self) -> None:
        """Tavily entry links to tavily.com and offers suppression."""
        from deepagents_cli.notifications import ActionId, MissingDepPayload

        entry = build_missing_tool_notification("tavily")
        assert entry.key == "dep:tavily"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "tavily"
        assert entry.payload.url == "https://tavily.com"
        assert entry.payload.install_command is None
        action_ids = [a.action_id for a in entry.actions]
        assert action_ids == [ActionId.OPEN_WEBSITE, ActionId.SUPPRESS]
        assert "TAVILY_API_KEY" in entry.body

    def test_unknown_tool_only_suppresses_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown tools fall back to a bare suppress action and log a warning."""
        import logging

        from deepagents_cli.notifications import ActionId, MissingDepPayload

        with caplog.at_level(logging.WARNING, logger="deepagents_cli.main"):
            entry = build_missing_tool_notification("foo")
        assert entry.key == "dep:foo"
        assert isinstance(entry.payload, MissingDepPayload)
        assert entry.payload.tool == "foo"
        assert [a.action_id for a in entry.actions] == [ActionId.SUPPRESS]
        assert any("No install hint" in record.message for record in caplog.records)


class TestRunTextualCliAsyncModelConfigError:
    """Verify ModelConfigError is caught cleanly before launching the TUI."""

    async def test_returns_error_code_on_no_credentials(self) -> None:
        """ModelConfigError from _get_default_model_spec gives return code 1."""
        from deepagents_cli.model_config import ModelConfigError

        with (
            patch(
                "deepagents_cli.config._get_default_model_spec",
                side_effect=ModelConfigError("No credentials configured"),
            ),
            patch("deepagents_cli.config._get_console") as mock_console_fn,
        ):
            mock_console = MagicMock()
            mock_console_fn.return_value = mock_console

            result = await run_textual_cli_async("agent")

        assert result.return_code == 1
        assert result.thread_id is None

    async def test_no_error_when_model_name_provided(self) -> None:
        """Explicit model_name bypasses _get_default_model_spec."""
        app_result = AppResult(return_code=0, thread_id="t-1")

        async def _stub(**_kwargs: Any) -> AppResult:  # noqa: RUF029  # must be async for run_textual_app signature
            return app_result

        with patch("deepagents_cli.app.run_textual_app", new=_stub):
            result = await run_textual_cli_async("agent", model_name="openai:gpt-4o")

        assert result.return_code == 0
