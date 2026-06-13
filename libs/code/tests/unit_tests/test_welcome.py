"""Unit tests for the welcome banner widget."""

from unittest.mock import MagicMock, patch

import pytest
from rich.style import Style
from textual.content import Content
from textual.style import Style as TStyle

from deepagents_code._env_vars import (
    DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER,
    HIDE_CWD,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_TIPS,
    HIDE_SPLASH_VERSION,
)
from deepagents_code._version import __version__
from deepagents_code.widgets.welcome import (
    _TIPS,
    WelcomeBanner,
    build_welcome_footer,
)


@pytest.fixture(autouse=True)
def _clear_startup_splash_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local startup splash overrides from affecting tests."""
    monkeypatch.delenv(DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER, raising=False)
    monkeypatch.delenv(HIDE_SPLASH_TIPS, raising=False)


def _extract_links(banner: Content, text_start: int, text_end: int) -> list[str]:
    """Extract link URLs from spans covering the given text range.

    Args:
        banner: The Content object to inspect.
        text_start: Start index in the plain text.
        text_end: End index in the plain text.

    Returns:
        List of link URL strings found on spans covering the range.
    """
    links: list[str] = []
    for span in banner._spans:
        style = span.style
        if (
            isinstance(style, TStyle)
            and style.link
            and span.start <= text_start
            and span.end >= text_end
        ):
            links.append(style.link)
    return links


def _make_banner(
    thread_id: str | None = None,
    project_name: str | None = None,
    *,
    hide_langsmith_tracing: bool = False,
) -> WelcomeBanner:
    """Create a `WelcomeBanner` with all env vars cleared.

    Args:
        thread_id: Optional thread ID to display.
        project_name: If set, simulates LangSmith being configured.
        hide_langsmith_tracing: Whether to hide tracing info from the splash.

    Returns:
        A `WelcomeBanner` instance ready for testing.
    """
    import deepagents_code.config as _cfg

    env = {}
    if project_name:
        env["LANGSMITH_API_KEY"] = "fake-key"
        env["LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_PROJECT"] = project_name
        env["DEEPAGENTS_CODE_LANGSMITH_PROJECT"] = project_name
    if hide_langsmith_tracing:
        env[HIDE_LANGSMITH_TRACING] = "1"

    # Temporarily clear the cached settings singleton so _get_settings()
    # re-creates it from the patched env vars inside the context manager.
    saved = _cfg.__dict__.pop("settings", None)
    saved_bootstrap = _cfg._bootstrap_done
    _cfg._bootstrap_done = False
    try:
        with patch.dict("os.environ", env, clear=True):
            return WelcomeBanner(thread_id=thread_id)
    finally:
        _cfg._bootstrap_done = saved_bootstrap
        if saved is not None:
            _cfg.__dict__["settings"] = saved
        else:
            _cfg.__dict__.pop("settings", None)


class TestBuildBannerThreadLink:
    """Tests for thread ID display in `_build_banner`."""

    def test_thread_id_plain_when_no_project_url(self) -> None:
        """Thread ID should be plain dim text when `project_url` is `None`."""
        widget = _make_banner(thread_id="12345")
        banner = widget._build_banner(project_url=None)

        assert "Thread: 12345" in banner.plain
        assert "\n  Thread: 12345\n" in banner.plain

        # Verify no link style on the thread portion
        thread_start = banner.plain.index("Thread: 12345")
        thread_end = thread_start + len("Thread: 12345")
        links = _extract_links(banner, thread_start, thread_end)
        assert not links, "Thread ID should not have a link when project_url is None"

    def test_thread_id_linked_when_project_url_provided(self) -> None:
        """Thread ID should be a hyperlink when `project_url` is provided."""
        project_url = "https://smith.langchain.com/o/org/projects/p/abc123"
        widget = _make_banner(thread_id="99999")
        banner = widget._build_banner(project_url=project_url)

        assert "Thread: 99999" in banner.plain
        assert "\n  Thread: 99999\n" in banner.plain

        # Find a span with a link on the thread ID text
        thread_id_start = banner.plain.index("99999")
        thread_id_end = thread_id_start + len("99999")
        links = _extract_links(banner, thread_id_start, thread_id_end)
        assert links, "Expected a link style on the thread ID text"
        assert links[0] == f"{project_url}/t/99999?utm_source=deepagents-code"

    def test_no_thread_line_when_thread_id_is_none(self) -> None:
        """Banner should not contain a thread line when `thread_id` is `None`."""
        widget = _make_banner(thread_id=None)
        banner = widget._build_banner(project_url=None)
        assert "Thread:" not in banner.plain

    def test_no_thread_line_when_project_url_but_no_thread_id(self) -> None:
        """Banner should not contain a thread line even with `project_url`."""
        widget = _make_banner(thread_id=None)
        banner = widget._build_banner(
            project_url="https://smith.langchain.com/o/org/projects/p/abc123"
        )
        assert "Thread:" not in banner.plain

    def test_trailing_slash_on_project_url_normalized(self) -> None:
        """Trailing slash on `project_url` should not cause double-slash in URL."""
        project_url = "https://smith.langchain.com/o/org/projects/p/abc123/"
        widget = _make_banner(thread_id="55555")
        banner = widget._build_banner(project_url=project_url)

        thread_id_start = banner.plain.index("55555")
        thread_id_end = thread_id_start + len("55555")
        links = _extract_links(banner, thread_id_start, thread_id_end)
        assert links
        # Path portion (after ://) should not contain double slashes
        path = links[0].split("://", 1)[1]
        assert "//" not in path

    def test_thread_link_coexists_with_langsmith_project(self) -> None:
        """Thread link should work when LangSmith project info is also shown."""
        project_url = "https://smith.langchain.com/o/org/projects/p/abc123"
        widget = _make_banner(thread_id="77777", project_name="my-project")
        banner = widget._build_banner(project_url=project_url)

        assert "my-project" in banner.plain
        assert "Thread: 77777" in banner.plain
        assert "\n  Thread: 77777\n" in banner.plain

        thread_id_start = banner.plain.index("77777")
        thread_id_end = thread_id_start + len("77777")
        links = _extract_links(banner, thread_id_start, thread_id_end)
        assert links
        assert links[0] == f"{project_url}/t/77777?utm_source=deepagents-code"

    def test_hide_langsmith_tracing_env_var_hides_project_and_thread(self) -> None:
        """Tracing splash frontmatter should hide when the env var is enabled."""
        widget = _make_banner(
            thread_id="77777",
            project_name="my-project",
            hide_langsmith_tracing=True,
        )
        banner = widget._build_banner(
            project_url="https://smith.langchain.com/o/org/projects/p/abc123"
        )

        assert "LangSmith tracing:" not in banner.plain
        assert "my-project" not in banner.plain
        assert "Thread:" not in banner.plain
        assert "77777" not in banner.plain


class TestUpdateThreadId:
    """Tests for `update_thread_id`."""

    def test_update_thread_id_changes_internal_state(self) -> None:
        """After `update_thread_id`, `_build_banner` should reflect the new ID."""
        widget = _make_banner(thread_id="old_id")
        assert "Thread: old_id" in widget._build_banner().plain

        # Patch Static.update to avoid needing an active Textual app context
        with patch.object(widget, "update"):
            widget.update_thread_id("new_id")

        banner = widget._build_banner()
        assert "Thread: new_id" in banner.plain
        assert "old_id" not in banner.plain

    def test_update_thread_id_preserves_project_url(self) -> None:
        """Thread link should use the cached project URL after update."""
        project_url = "https://smith.langchain.com/o/org/projects/p/abc123"
        widget = _make_banner(thread_id="old_id")
        widget._project_url = project_url

        with patch.object(widget, "update") as mock_update:
            widget.update_thread_id("new_id")

        # Verify update_thread_id passed the correct banner to Static.update
        mock_update.assert_called_once()
        banner = mock_update.call_args[0][0]
        assert "Thread: new_id" in banner.plain
        thread_start = banner.plain.index("new_id")
        thread_end = thread_start + len("new_id")
        links = _extract_links(banner, thread_start, thread_end)
        assert links
        assert links[0] == f"{project_url}/t/new_id?utm_source=deepagents-code"


class TestBuildBannerEditableInstall:
    """Tests for the editable-install path in `_build_banner`."""

    def test_build_banner_with_editable_install(self) -> None:
        """Banner should include install path when running from editable install."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "deepagents_code.widgets.welcome._is_editable_install",
                return_value=True,
            ),
            patch(
                "deepagents_code.widgets.welcome._get_editable_install_path",
                return_value="~/dev/deepagents",
            ),
        ):
            widget = WelcomeBanner()
            banner = widget._build_banner()
        assert "Installed from: ~/dev/deepagents" in banner.plain

    def test_build_banner_without_editable_install(self) -> None:
        """Banner should not include install path for non-editable installs."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "deepagents_code.widgets.welcome._is_editable_install",
                return_value=False,
            ),
            patch(
                "deepagents_code.widgets.welcome._get_editable_install_path",
                return_value=None,
            ),
        ):
            widget = WelcomeBanner()
            banner = widget._build_banner()
        assert "Installed from:" not in banner.plain

    def test_hide_splash_version_env_var_hides_local_install_details(self) -> None:
        """Splash version override should hide version and local install details."""
        with (
            patch.dict("os.environ", {HIDE_SPLASH_VERSION: "1"}, clear=True),
            patch(
                "deepagents_code.widgets.welcome._is_editable_install",
                return_value=True,
            ) as editable,
            patch(
                "deepagents_code.widgets.welcome._get_editable_install_path",
                return_value="~/dev/deepagents",
            ) as editable_path,
        ):
            widget = WelcomeBanner()
            banner = widget._build_banner()
        editable.assert_not_called()
        editable_path.assert_not_called()
        assert f"v{__version__}" not in banner.plain
        assert "(local)" not in banner.plain
        assert "Installed from:" not in banner.plain

    def test_hide_cwd_env_var_hides_editable_install_path(self) -> None:
        """Cwd privacy override should hide the local editable install path."""
        with (
            patch.dict("os.environ", {HIDE_CWD: "1"}, clear=True),
            patch(
                "deepagents_code.widgets.welcome._is_editable_install",
                return_value=True,
            ),
            patch(
                "deepagents_code.widgets.welcome._get_editable_install_path",
                return_value="~/oss/deepagents/libs/code",
            ) as editable_path,
        ):
            widget = WelcomeBanner()
            banner = widget._build_banner()
        editable_path.assert_not_called()
        assert f"v{__version__}" in banner.plain
        assert "Installed from:" not in banner.plain
        assert "~/oss/deepagents/libs/code" not in banner.plain


class TestBuildBannerReturnType:
    """Tests for `_build_banner` return value."""

    def test_returns_content(self) -> None:
        """`_build_banner` should return a `Content` object."""
        widget = _make_banner(thread_id="abc")
        result = widget._build_banner()
        assert isinstance(result, Content)


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
        widget = _make_banner(thread_id="abc")
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_called_once_with("https://example.com")
        event.stop.assert_called_once()

    def test_click_without_link_is_noop(self) -> None:
        """Clicking on non-link text should not open the browser."""
        widget = _make_banner(thread_id="abc")
        event = MagicMock()
        event.style = Style()

        with patch(_WEBBROWSER_OPEN) as mock_open:
            widget.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()

    def test_click_with_browser_error_is_graceful(self) -> None:
        """Browser failure should not crash the widget."""
        widget = _make_banner(thread_id="abc")
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN, side_effect=OSError("no display")):
            widget.on_click(event)  # should not raise

        event.stop.assert_not_called()


class TestPointerShapeOnHover:
    """Tests for the hand pointer shown when hovering link spans."""

    def test_mouse_move_over_link_sets_pointer(self) -> None:
        """Hovering a link span should show the hand pointer."""
        widget = _make_banner(thread_id="abc")
        event = MagicMock()
        event.style = Style(link="https://example.com")

        widget.on_mouse_move(event)

        assert widget.styles.pointer == "pointer"

    def test_mouse_move_off_link_resets_pointer(self) -> None:
        """Hovering non-link text should reset to the default pointer."""
        widget = _make_banner(thread_id="abc")
        widget.styles.pointer = "pointer"
        event = MagicMock()
        event.style = Style()

        widget.on_mouse_move(event)

        assert widget.styles.pointer == "default"

    def test_leave_resets_pointer(self) -> None:
        """Leaving the banner should reset to the default pointer."""
        widget = _make_banner(thread_id="abc")
        widget.styles.pointer = "pointer"

        widget.on_leave()

        assert widget.styles.pointer == "default"


class TestBuildWelcomeFooter:
    """Tests for the `build_welcome_footer` standalone function."""

    def test_returns_content(self) -> None:
        """Footer should return a `Content` object."""
        assert isinstance(build_welcome_footer(), Content)

    def test_contains_ready_prompt(self) -> None:
        """Footer should include the ready-to-code prompt."""
        assert (
            "Ready to code! What would you like to build?"
            in build_welcome_footer().plain
        )

    def test_startup_subheader_env_var_overrides_ready_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup subheader override should replace the default ready prompt."""
        monkeypatch.setenv(DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER, "Ship it.")

        plain = build_welcome_footer(tip="Use /help").plain

        assert "Ship it." in plain
        assert "Ready to code! What would you like to build?" not in plain
        assert "Tip: Use /help" in plain

    def test_contains_tip(self) -> None:
        """Footer should include a tip from the rotating tips list."""
        plain = build_welcome_footer().plain
        assert "Tip: " in plain
        assert any(tip in plain for tip in _TIPS)

    def test_hide_splash_tips_env_var_hides_tip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Splash tips should hide when the env var is enabled."""
        monkeypatch.setenv(HIDE_SPLASH_TIPS, "1")

        plain = build_welcome_footer(tip="Use /help").plain

        assert "Ready to code! What would you like to build?" in plain
        assert "Tip: " not in plain
        assert "Use /help" not in plain
        assert plain.split("\n") == [
            "",
            "Ready to code! What would you like to build?",
        ]

    def test_hide_splash_tips_env_var_skips_random_tip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabling splash tips should avoid selecting a random tip."""
        monkeypatch.setenv(HIDE_SPLASH_TIPS, "1")

        with patch("deepagents_code.widgets.welcome._pick_tip") as pick_tip:
            build_welcome_footer()

        pick_tip.assert_not_called()

    def test_startup_cmd_tip_registered(self) -> None:
        """New `--startup-cmd` flag must have a discoverability tip."""
        assert any("--startup-cmd" in tip for tip in _TIPS)

    def test_incognito_shell_tip_registered(self) -> None:
        """New `!!` shell mode must have a discoverability tip."""
        assert any("!!" in tip and "incognito" in tip.lower() for tip in _TIPS)

    def test_copy_command_tip_registered(self) -> None:
        """The `/copy` command must have a discoverability tip."""
        assert "Use /copy to copy the latest assistant message" in _TIPS

    def test_tip_varies_across_calls(self) -> None:
        """Tips should rotate (not always the same)."""
        seen = {build_welcome_footer().plain for _ in range(50)}
        assert len(seen) > 1, "Expected different tips across multiple calls"

    def test_ready_line_is_first_content_line(self) -> None:
        """The ready prompt must be the first non-blank line."""
        lines = build_welcome_footer().plain.strip().splitlines()
        assert lines[0].strip() == "Ready to code! What would you like to build?"

    def test_tip_line_is_last(self) -> None:
        """The tip line must be the last line after the ready prompt."""
        lines = build_welcome_footer().plain.strip().splitlines()
        assert lines[-1].strip().startswith("Tip: ")

    def test_blank_line_precedes_ready_prompt(self) -> None:
        """A blank line must precede the ready prompt (leading newline)."""
        raw = build_welcome_footer().plain
        assert raw.startswith("\n")

    def test_exactly_three_lines_with_leading_blank(self) -> None:
        """Footer: blank line, ready prompt, tip."""
        lines = build_welcome_footer().plain.split("\n")
        # Leading \n produces ['', 'Ready to code...', 'Tip: ...']
        assert lines[0] == ""
        assert lines[1].startswith("Ready to code")
        assert lines[2].startswith("Tip: ")
        assert len(lines) == 3


class TestBannerFooterPosition:
    """Tests that the footer is always the last content in the full banner."""

    def test_footer_is_last_in_minimal_banner(self) -> None:
        """With no thread/project/MCP, footer lines are still last."""
        widget = _make_banner()
        lines = widget._build_banner().plain.strip().splitlines()
        assert "Ready to code" in lines[-2]
        assert lines[-1].strip().startswith("Tip: ")

    def test_hide_splash_tips_env_var_hides_tip_in_banner(self) -> None:
        """Full startup banner should omit tips when the env var is enabled."""
        with patch.dict("os.environ", {HIDE_SPLASH_TIPS: "1"}, clear=True):
            widget = WelcomeBanner()
        plain = widget._build_banner().plain
        lines = plain.strip().splitlines()
        assert "Ready to code" in lines[-1]
        assert "Tip: " not in plain

    def test_footer_is_last_with_thread_id(self) -> None:
        """Footer remains last when a thread ID is displayed."""
        widget = _make_banner(thread_id="tid-123")
        lines = widget._build_banner().plain.strip().splitlines()
        assert "Ready to code" in lines[-2]
        assert lines[-1].strip().startswith("Tip: ")

    def test_footer_is_last_with_langsmith_project(self) -> None:
        """Footer remains last when LangSmith project info is shown."""
        widget = _make_banner(project_name="my-proj")
        lines = widget._build_banner().plain.strip().splitlines()
        assert "Ready to code" in lines[-2]
        assert lines[-1].strip().startswith("Tip: ")

    def test_footer_is_last_with_mcp_tools(self) -> None:
        """Footer remains last when MCP tools are loaded."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner(mcp_tool_count=5)
        lines = widget._build_banner().plain.strip().splitlines()
        assert "Ready to code" in lines[-2]
        assert lines[-1].strip().startswith("Tip: ")

    def test_footer_is_last_with_all_info(self) -> None:
        """Footer remains last when all info lines are present."""
        env = {
            "LANGSMITH_API_KEY": "fake-key",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "proj",
        }
        with patch.dict("os.environ", env, clear=True):
            widget = WelcomeBanner(thread_id="t-1", mcp_tool_count=3)
        lines = widget._build_banner().plain.strip().splitlines()
        assert "Ready to code" in lines[-2]
        assert lines[-1].strip().startswith("Tip: ")

    def test_blank_line_separates_info_from_footer(self) -> None:
        """A blank line should appear between info lines and footer."""
        widget = _make_banner(thread_id="tid")
        plain = widget._build_banner().plain
        # The ready prompt should be preceded by a double newline
        idx = plain.index("Ready to code")
        assert plain[idx - 1] == "\n"
        assert plain[idx - 2] == "\n"


class TestBannerConnectionState:
    """WelcomeBanner keeps identity content while the status bar shows progress."""

    def test_connecting_keeps_ready_footer(self) -> None:
        """The banner should not render app connection progress."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner()
        plain = widget._build_banner().plain
        assert "Connecting to server" not in plain
        assert "Resuming" not in plain
        assert "Ready to code" in plain

    def test_set_connecting_updates_state_without_progress_footer(self) -> None:
        """Mid-session restarts update lifecycle state without a second spinner."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner()
        with patch.object(widget, "update"):
            widget.set_connecting()
        plain = widget._build_banner().plain
        assert "Connecting to server" not in plain
        assert "Ready to code" in plain


class TestMcpServerCounters:
    """Tests for the MCP server status counter lines on the splash banner."""

    def test_unauthenticated_line_singular(self) -> None:
        """A single unauthenticated server reads `'server'`, not `'servers'`."""
        # Suppress the random splash tip; one tip mentions `/mcp login`, which
        # would otherwise spuriously match the negative assertion below.
        with patch.dict(
            "os.environ", {"DEEPAGENTS_CODE_HIDE_SPLASH_TIPS": "1"}, clear=True
        ):
            widget = WelcomeBanner(mcp_unauthenticated=1)
        plain = widget._build_banner().plain
        assert "1 MCP server needs login — open /mcp" in plain
        assert "/mcp login" not in plain

    def test_errored_line_plural(self) -> None:
        """Two errored servers read `'servers'` and route to `/mcp` for details."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner(mcp_errored=2)
        plain = widget._build_banner().plain
        assert "2 MCP servers failed to load" in plain
        assert "open /mcp for details" in plain

    def test_awaiting_reconnect_line_singular(self) -> None:
        """A single awaiting-reconnect server prompts `/mcp reconnect`."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner(mcp_awaiting_reconnect=1)
        plain = widget._build_banner().plain
        assert "1 MCP server ready to load" in plain
        assert "/mcp reconnect" in plain

    def test_awaiting_reconnect_line_plural(self) -> None:
        """Multiple awaiting-reconnect servers use plural noun."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner(mcp_awaiting_reconnect=3)
        plain = widget._build_banner().plain
        assert "3 MCP servers ready to load" in plain

    def test_no_counter_lines_when_all_zero(self) -> None:
        """Banner has no MCP status warning when all counters are zero."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner()
        plain = widget._build_banner().plain
        assert "need login" not in plain
        assert "ready to load" not in plain
        assert "failed to load" not in plain

    def test_all_three_counters_render_independently(self) -> None:
        """Unauth, errored, and awaiting-reconnect lines can coexist."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner(
                mcp_unauthenticated=1,
                mcp_errored=1,
                mcp_awaiting_reconnect=1,
            )
        plain = widget._build_banner().plain
        assert "needs login" in plain
        assert "failed to load" in plain
        assert "ready to load" in plain

    def test_set_connected_updates_awaiting_reconnect(self) -> None:
        """`set_connected` plumbs the new counter onto the banner."""
        with patch.dict("os.environ", {}, clear=True):
            widget = WelcomeBanner()
        with patch.object(widget, "update"):
            widget.set_connected(0, mcp_awaiting_reconnect=2)
        assert widget._mcp_awaiting_reconnect == 2
        assert "2 MCP servers ready to load" in widget._build_banner().plain
