"""Unit tests for the welcome banner widget."""

from unittest.mock import patch

from rich.style import Style
from rich.text import Text

from deepagents_cli.widgets.welcome import WelcomeBanner


def _extract_links(banner: Text, text_start: int, text_end: int) -> list[str]:
    """Extract link URLs from spans covering the given text range.

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
    """Create a `WelcomeBanner` with LangSmith env vars cleared.

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
        assert links[0] == f"{project_url}/t/99999"

    def test_no_thread_line_when_thread_id_is_none(self) -> None:
        """Banner should not contain a thread line when `thread_id` is `None`."""
        widget = _make_banner(thread_id=None)
        banner = widget._build_banner(project_url=None)
        assert "Thread:" not in banner.plain

    def test_thread_line_still_present_with_project_url_and_no_thread_id(self) -> None:
        """Banner should not contain a thread line even with `project_url`."""
        widget = _make_banner(thread_id=None)
        banner = widget._build_banner(
            project_url="https://smith.langchain.com/o/org/projects/p/abc123"
        )
        assert "Thread:" not in banner.plain


class TestBuildBannerReturnType:
    """Tests for `_build_banner` return value."""

    def test_returns_rich_text(self) -> None:
        """``_build_banner`` should return a `rich.text.Text` object."""
        widget = _make_banner(thread_id="abc")
        result = widget._build_banner()
        assert isinstance(result, Text)
