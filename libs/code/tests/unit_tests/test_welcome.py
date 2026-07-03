"""Unit tests for the welcome banner widget."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.style import Style
from textual.content import Content

from deepagents_code._env_vars import (
    DEBUG,
    HIDE_CWD,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_VERSION,
)
from deepagents_code._version import __version__
from deepagents_code.widgets.welcome import (
    WelcomeBanner,
    _home_prefixed,
)

_EDITABLE = "deepagents_code.widgets.welcome._is_editable_install"
_PROJECT_NAME = "deepagents_code.widgets.welcome.get_langsmith_project_name"


def _make_banner(
    *,
    model_provider: str = "anthropic",
    model_name: str = "claude-opus-4-8",
    cwd: str | None = "/work/project",
    thread_id: str | None = None,
    mcp_tool_count: int = 0,
    project_name: str | None = None,
    env: dict[str, str] | None = None,
) -> WelcomeBanner:
    """Create a `WelcomeBanner` with a controlled environment.

    Args:
        model_provider: Model provider to display.
        model_name: Model name to display.
        cwd: Working directory to display.
        thread_id: Thread ID to display (only shown in debug mode).
        mcp_tool_count: MCP tool count to display.
        project_name: LangSmith project name to inject (or `None`).
        env: Environment variables to set while constructing.

    Returns:
        A `WelcomeBanner` instance ready for testing.
    """
    with (
        patch(_PROJECT_NAME, return_value=project_name),
        patch.dict("os.environ", env or {}, clear=True),
    ):
        return WelcomeBanner(
            model_provider=model_provider,
            model_name=model_name,
            cwd=cwd,
            thread_id=thread_id,
            mcp_tool_count=mcp_tool_count,
        )


class TestHomePrefixed:
    """Tests for the `_home_prefixed` helper."""

    def test_collapses_home_to_tilde(self) -> None:
        """Paths under the home directory render with a `~` prefix."""
        path = str(Path.home() / "Documents" / "Dev")
        assert _home_prefixed(path) == "~/Documents/Dev"

    def test_leaves_non_home_path_unchanged(self) -> None:
        """Paths outside the home directory are returned as-is."""
        assert _home_prefixed("/tmp/work") == "/tmp/work"


class TestTitle:
    """Tests for the banner title line."""

    def test_shows_product_name(self) -> None:
        """The banner shows the `dcode` title."""
        assert "dcode" in _make_banner()._build_banner().plain

    def test_shows_version_by_default(self) -> None:
        """The version is shown when not hidden and not editable."""
        with patch(_EDITABLE, return_value=False):
            plain = _make_banner()._build_banner().plain
        assert f"v{__version__}" in plain
        assert "(local)" not in plain

    def test_hides_version_when_env_set(self) -> None:
        """`HIDE_SPLASH_VERSION` removes the version from the title."""
        plain = _make_banner(env={HIDE_SPLASH_VERSION: "1"})._build_banner().plain
        assert f"v{__version__}" not in plain
        assert "(local)" not in plain

    def test_marks_editable_install_as_local(self) -> None:
        """Editable installs append a `(local)` tag to the version."""
        with patch(_EDITABLE, return_value=True):
            plain = _make_banner()._build_banner().plain
        assert f"v{__version__}" in plain
        assert "(local)" in plain


class TestModelLine:
    """Tests for the model row."""

    def test_shows_provider_and_model(self) -> None:
        """The model row renders `provider:model`."""
        plain = (
            _make_banner(model_provider="anthropic", model_name="claude-opus-4-8")
            ._build_banner()
            .plain
        )
        assert "model:" in plain
        assert "anthropic:claude-opus-4-8" in plain

    def test_omits_provider_prefix_when_empty(self) -> None:
        """With no provider, only the bare model name is shown."""
        plain = (
            _make_banner(model_provider="", model_name="claude-opus-4-8")
            ._build_banner()
            .plain
        )
        assert "claude-opus-4-8" in plain
        assert ":claude-opus-4-8" not in plain

    def test_no_model_line_when_unset(self) -> None:
        """No model row is rendered when the model name is empty."""
        plain = _make_banner(model_provider="", model_name="")._build_banner().plain
        assert "model:" not in plain

    def test_update_model_refreshes_line(self) -> None:
        """`update_model` updates the rendered model row."""
        widget = _make_banner(model_provider="anthropic", model_name="claude-opus-4-8")
        with patch.object(widget, "update"):
            widget.update_model(provider="openai", model="gpt-5")
        plain = widget._build_banner().plain
        assert "openai:gpt-5" in plain
        assert "claude-opus-4-8" not in plain


class TestDirectoryLine:
    """Tests for the directory row."""

    def test_shows_directory(self) -> None:
        """The directory row renders the working directory."""
        plain = _make_banner(cwd="/work/project")._build_banner().plain
        assert "directory:" in plain
        assert "/work/project" in plain

    def test_home_prefixed(self) -> None:
        """The directory is home-prefixed with `~`."""
        cwd = str(Path.home() / "code" / "app")
        plain = _make_banner(cwd=cwd)._build_banner().plain
        assert "~/code/app" in plain

    def test_hidden_when_hide_cwd_env_set(self) -> None:
        """`HIDE_CWD` removes the directory row."""
        plain = (
            _make_banner(cwd="/work/project", env={HIDE_CWD: "1"})._build_banner().plain
        )
        assert "directory:" not in plain
        assert "/work/project" not in plain


class TestTracingLine:
    """Tests for the LangSmith tracing project row."""

    def test_shows_project_name(self) -> None:
        """The tracing row renders the LangSmith project name."""
        plain = _make_banner(project_name="dcode-johannes")._build_banner().plain
        assert "tracing:" in plain
        assert "'dcode-johannes'" in plain

    def test_omitted_when_no_project(self) -> None:
        """No tracing row when the project name is `None`."""
        plain = _make_banner(project_name=None)._build_banner().plain
        assert "tracing:" not in plain

    def test_hidden_when_hide_langsmith_env_set(self) -> None:
        """`HIDE_LANGSMITH_TRACING` removes the tracing row."""
        plain = (
            _make_banner(
                project_name="dcode-johannes",
                env={HIDE_LANGSMITH_TRACING: "1"},
            )
            ._build_banner()
            .plain
        )
        assert "tracing:" not in plain


class TestThreadLine:
    """Tests for the thread ID row (shown only in debug mode)."""

    def test_shows_thread_id_when_debug_enabled(self) -> None:
        """The thread row renders the thread ID when debug mode is on."""
        plain = (
            _make_banner(thread_id="abc-123", env={DEBUG: "1"})._build_banner().plain
        )
        assert "thread:" in plain
        assert "abc-123" in plain

    def test_omitted_when_debug_disabled(self) -> None:
        """No thread row when debug mode is off."""
        plain = _make_banner(thread_id="abc-123")._build_banner().plain
        assert "thread:" not in plain
        assert "abc-123" not in plain

    def test_omitted_when_no_thread_id(self) -> None:
        """No thread row in debug mode when the thread ID is unset."""
        plain = _make_banner(thread_id=None, env={DEBUG: "1"})._build_banner().plain
        assert "thread:" not in plain


class TestMcpToolLine:
    """Tests for the MCP tool count row."""

    def test_shows_tool_count(self) -> None:
        """The mcp row renders the tool count."""
        plain = _make_banner(mcp_tool_count=5)._build_banner().plain
        assert "mcp:" in plain
        assert "5 tools" in plain

    def test_singular_tool_label(self) -> None:
        """A count of 1 uses the singular `tool` label."""
        plain = _make_banner(mcp_tool_count=1)._build_banner().plain
        assert "1 tool" in plain
        assert "1 tools" not in plain

    def test_omitted_when_zero(self) -> None:
        """No mcp row when the tool count is zero."""
        plain = _make_banner(mcp_tool_count=0)._build_banner().plain
        assert "mcp:" not in plain


class TestRemovedSections:
    """The banner does not show the old splash tips/footer content."""

    def test_no_legacy_sections(self) -> None:
        """None of the old splash footer sections appear."""
        plain = _make_banner()._build_banner().plain
        for absent in ("Ready to code", "Tip:"):
            assert absent not in plain


class TestReturnType:
    """Tests for `_build_banner` return value."""

    def test_returns_content(self) -> None:
        """`_build_banner` returns a `Content` object."""
        assert isinstance(_make_banner()._build_banner(), Content)


class TestCompatibilityMethods:
    """Retained no-op/state methods keep the app's call sites working."""

    def test_set_connected_tracks_counts(self) -> None:
        """`set_connected` stores counts without raising."""
        widget = _make_banner()
        widget.set_connected(
            5, mcp_unauthenticated=1, mcp_errored=2, mcp_awaiting_reconnect=3
        )
        assert widget._mcp_tool_count == 5
        assert widget._mcp_unauthenticated == 1
        assert widget._mcp_errored == 2
        assert widget._mcp_awaiting_reconnect == 3

    def test_set_connecting_and_idle_are_noops(self) -> None:
        """`set_connecting`/`set_idle` do not raise and render nothing extra."""
        widget = _make_banner()
        widget.set_connecting()
        widget.set_idle()
        assert "Ready to code" not in widget._build_banner().plain

    def test_update_thread_id_tracks_without_rendering(self) -> None:
        """`update_thread_id` stores the id but does not display it without debug."""
        widget = _make_banner()
        widget.update_thread_id("abc123")
        assert widget._cli_thread_id == "abc123"
        assert "abc123" not in widget._build_banner().plain

    def test_update_thread_id_renders_in_debug(self) -> None:
        """`update_thread_id` re-renders to show the id when debug mode is on."""
        widget = _make_banner(env={DEBUG: "1"})
        with patch.object(widget, "update"):
            widget.update_thread_id("abc123")
        plain = widget._build_banner().plain
        assert "abc123" in plain
        assert "thread:" in plain


class TestAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`WelcomeBanner` should disable Textual's `auto_links`."""
        assert WelcomeBanner.auto_links is False


_WEBBROWSER_OPEN = "deepagents_code.widgets._links.webbrowser.open"


class TestOnClickOpensLink:
    """Tests for `WelcomeBanner.on_click` opening Rich-style hyperlinks."""

    def test_click_on_link_opens_browser(self) -> None:
        """Clicking a Rich link should call `webbrowser.open`."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_called_once_with("https://example.com")
        event.stop.assert_called_once()

    def test_click_without_link_is_noop(self) -> None:
        """Clicking on non-link text should not open the browser."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style()

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()


class TestPointerShapeOnHover:
    """Tests for the hand pointer shown when hovering link spans."""

    def test_mouse_move_over_link_sets_pointer(self) -> None:
        """Hovering a link span should show the hand pointer."""
        widget = _make_banner()
        event = MagicMock()
        event.style = Style(link="https://example.com")

        widget.on_mouse_move(event)

        assert widget.styles.pointer == "pointer"

    def test_mouse_move_off_link_resets_pointer(self) -> None:
        """Hovering non-link text should reset to the default pointer."""
        widget = _make_banner()
        widget.styles.pointer = "pointer"
        event = MagicMock()
        event.style = Style()

        widget.on_mouse_move(event)

        assert widget.styles.pointer == "default"

    def test_leave_resets_pointer(self) -> None:
        """Leaving the banner should reset to the default pointer."""
        widget = _make_banner()
        widget.styles.pointer = "pointer"

        widget.on_leave()

        assert widget.styles.pointer == "default"
