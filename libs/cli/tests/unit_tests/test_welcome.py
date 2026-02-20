"""Unit tests for the welcome banner widget."""

from unittest.mock import MagicMock, patch

from rich.style import Style
from rich.text import Text

from deepagents_cli.widgets.welcome import WelcomeBanner


def _extract_links(banner: Text, text_start: int, text_end: int) -> list[str]:
    """Extract link URLs from spans covering the given text range.

    Note: This relies on `rich.text.Text._spans` internals and may need
    updating if the Rich library changes its internal representation.

    Args:
        banner: The Rich Text object to inspect.
        text_start: Start index in the plain text.
        text_end: End index in the plain text.

    Returns:
        List of link URL strings found on spans covering the range.
    """
    links: list[str] = []
    for start, end, style in banner._spans:
        if not isinstance(style, Style):
            continue
        if start <= text_start and end >= text_end and style.link:
            links.append(style.link)
    return links


def _make_banner(
    thread_id: str | None = None,
    project_name: str | None = None,
) -> WelcomeBanner:
    """Create a `WelcomeBanner` with all env vars cleared.

    Args:
        thread_id: Optional thread ID to display.
        project_name: If set, simulates LangSmith being configured.

    Returns:
        A `WelcomeBanner` instance ready for testing.
    """
    env = {}
    if project_name:
        env["LANGSMITH_API_KEY"] = "fake-key"
        env["LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_PROJECT"] = project_name

    with patch.dict("os.environ", env, clear=True):
        return WelcomeBanner(thread_id=thread_id)


class TestBuildBannerThreadLink:
    """Tests for thread ID display in `_build_banner`."""

    def test_thread_id_plain_when_no_project_url(self) -> None:
        """Thread ID should be plain dim text when `project_url` is `None`."""
        widget = _make_banner(thread_id="12345")
        banner = widget._build_banner(project_url=None)

        assert "Thread: 12345" in banner.plain

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

        # Find a span with a link on the thread ID text
        thread_id_start = banner.plain.index("99999")
        thread_id_end = thread_id_start + len("99999")
        links = _extract_links(banner, thread_id_start, thread_id_end)
        assert links, "Expected a link style on the thread ID text"
        assert links[0] == f"{project_url}/t/99999?utm_source=deepagents-cli"

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

        thread_id_start = banner.plain.index("77777")
        thread_id_end = thread_id_start + len("77777")
        links = _extract_links(banner, thread_id_start, thread_id_end)
        assert links
        assert links[0] == f"{project_url}/t/77777?utm_source=deepagents-cli"


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
        assert links[0] == f"{project_url}/t/new_id?utm_source=deepagents-cli"


class TestBuildBannerReturnType:
    """Tests for `_build_banner` return value."""

    def test_returns_rich_text(self) -> None:
        """`_build_banner` should return a `rich.text.Text` object."""
        widget = _make_banner(thread_id="abc")
        result = widget._build_banner()
        assert isinstance(result, Text)


class TestAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`WelcomeBanner` should disable Textual's `auto_links`."""
        assert WelcomeBanner.auto_links is False


_WEBBROWSER_OPEN = "deepagents_cli.widgets._links.webbrowser.open"


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
