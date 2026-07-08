"""Unit tests for the welcome banner widget."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.content import Content

from deepagents_code._env_vars import (
    DEBUG,
    HIDE_CWD,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_VERSION,
    SHOW_LANGSMITH_REPLICA_TRACING,
    SPLASH_SHOW_CWD,
    SPLASH_SHOW_MODEL,
)
from deepagents_code._version import __version__
from deepagents_code.tui.widgets.welcome import (
    WelcomeBanner,
    _home_prefixed,
    _langsmith_project_link,
    _langsmith_project_link_style,
    _local_tag_style,
)

_EDITABLE = "deepagents_code.tui.widgets.welcome._is_editable_install"
_EDITABLE_PATH = "deepagents_code.tui.widgets.welcome._get_editable_install_path"
_PROJECT_NAME = "deepagents_code.tui.widgets.welcome.get_langsmith_project_name"
_REPLICA_PROJECT = "deepagents_code.tui.widgets.welcome.get_langsmith_replica_project"
_FETCH_URL = "deepagents_code.tui.widgets.welcome.fetch_langsmith_project_url"


def _make_banner(
    *,
    model_provider: str = "anthropic",
    model_name: str = "claude-opus-4-8",
    cwd: str | None = "/work/project",
    thread_id: str | None = None,
    mcp_tool_count: int = 0,
    mcp_unauthenticated: int = 0,
    mcp_errored: int = 0,
    mcp_awaiting_reconnect: int = 0,
    project_name: str | None = None,
    replica_project: str | None = None,
    project_urls: dict[str, str] | None = None,
    show_model: bool = True,
    show_cwd: bool = False,
    env: dict[str, str] | None = None,
) -> WelcomeBanner:
    """Create a `WelcomeBanner` with a controlled environment.

    Args:
        model_provider: Model provider to display.
        model_name: Model name to display.
        cwd: Working directory to display (only shown when `show_cwd`).
        thread_id: Thread ID to display (only shown in debug mode).
        mcp_tool_count: MCP tool count to display.
        mcp_unauthenticated: Number of MCP servers awaiting login.
        mcp_errored: Number of MCP servers that failed to load.
        mcp_awaiting_reconnect: Number of MCP servers awaiting reconnect.
        project_name: LangSmith project name to inject (or `None`).
        replica_project: Replica LangSmith project name to inject (or `None`).
        project_urls: LangSmith project URLs keyed by project name.
        show_model: Set `SPLASH_SHOW_MODEL` so the model row renders. Defaults to
            `True` so model tests exercise the row; the real default is off.
        show_cwd: Set `SPLASH_SHOW_CWD` so the directory row renders.
        env: Additional environment variables to set while constructing.

    Returns:
        A `WelcomeBanner` instance ready for testing.
    """
    resolved_env: dict[str, str] = {}
    if show_model:
        resolved_env[SPLASH_SHOW_MODEL] = "1"
    if show_cwd:
        resolved_env[SPLASH_SHOW_CWD] = "1"
    if env:
        resolved_env.update(env)
    with (
        patch(_PROJECT_NAME, return_value=project_name),
        patch(_REPLICA_PROJECT, return_value=replica_project),
        patch.dict("os.environ", resolved_env, clear=True),
    ):
        widget = WelcomeBanner(
            model_provider=model_provider,
            model_name=model_name,
            cwd=cwd,
            thread_id=thread_id,
            mcp_tool_count=mcp_tool_count,
            mcp_unauthenticated=mcp_unauthenticated,
            mcp_errored=mcp_errored,
            mcp_awaiting_reconnect=mcp_awaiting_reconnect,
        )
        if project_urls:
            widget._project_urls = project_urls
        return widget


class TestHomePrefixed:
    """Tests for the `_home_prefixed` helper."""

    def test_collapses_home_to_tilde(self) -> None:
        """Paths under the home directory render with a `~` prefix."""
        path = str(Path.home() / "Documents" / "Dev")
        assert _home_prefixed(path) == "~/Documents/Dev"

    def test_leaves_non_home_path_unchanged(self) -> None:
        """Paths outside the home directory are returned as-is."""
        assert _home_prefixed("/tmp/work") == "/tmp/work"

    def test_exact_home_collapses_to_bare_tilde(self) -> None:
        """The home directory itself renders as `~`, not `~/.`."""
        assert _home_prefixed(str(Path.home())) == "~"

    def test_falls_back_to_absolute_path_when_home_unresolved(self) -> None:
        """When `Path.home()` raises `RuntimeError`, the absolute path is returned."""
        with patch(
            "deepagents_code.tui.widgets.welcome.Path.home",
            side_effect=RuntimeError("no home"),
        ):
            assert _home_prefixed("/srv/app") == "/srv/app"

    def test_falls_back_to_absolute_path_on_value_error(self) -> None:
        """A `ValueError` from path comparison (e.g. embedded NUL) is absorbed."""
        with patch(
            "deepagents_code.tui.widgets.welcome.Path.home",
            side_effect=ValueError("embedded null byte"),
        ):
            assert _home_prefixed("/srv/app") == "/srv/app"


class TestLangsmithLinkHelpers:
    """Tests for the LangSmith link helper functions."""

    def test_link_appends_utm_source(self) -> None:
        """`_langsmith_project_link` appends the UTM source tag."""
        result = _langsmith_project_link("https://smith.langchain.com/o/org/p/proj")
        assert "utm_source=deepagents-code" in result

    def test_link_style_non_ansi_has_link(self) -> None:
        """Non-ANSI link style carries the project URL as a link."""
        from deepagents_code.theme import DARK_COLORS

        style = _langsmith_project_link_style(
            "https://smith.langchain.com/o/org/p/proj",
            ansi=False,
            colors=DARK_COLORS,
        )
        assert style.link is not None
        assert "utm_source=deepagents-code" in style.link

    def test_link_style_ansi_is_bold(self) -> None:
        """ANSI link style is bold with a link."""
        from deepagents_code.theme import DARK_COLORS

        style = _langsmith_project_link_style(
            "https://smith.langchain.com/o/org/p/proj",
            ansi=True,
            colors=DARK_COLORS,
        )
        assert style.bold is True
        assert style.link is not None


class TestLocalTagStyle:
    """Tests for the editable-install `(local)` tag style."""

    def test_ansi_uses_bold_markup(self) -> None:
        """Under ANSI themes the tag stays visible via bold terminal text."""
        from deepagents_code.theme import DARK_COLORS

        assert _local_tag_style(ansi=True, colors=DARK_COLORS) == "bold"

    def test_non_ansi_uses_themed_color(self) -> None:
        """Non-ANSI themes color the tag with the theme's tool color."""
        from textual.color import Color as TColor
        from textual.style import Style as TStyle

        from deepagents_code.theme import DARK_COLORS

        style = _local_tag_style(ansi=False, colors=DARK_COLORS)
        assert isinstance(style, TStyle)
        assert style.bold is True
        assert style.foreground == TColor.parse(DARK_COLORS.tool)


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
        """`update_model` re-renders (calls `update`) and shows the new model."""
        widget = _make_banner(model_provider="anthropic", model_name="claude-opus-4-8")
        with patch.object(widget, "update") as mock_update:
            widget.update_model(provider="openai", model="gpt-5")
            mock_update.assert_called_once()
        plain = widget._build_banner().plain
        assert "openai:gpt-5" in plain
        assert "claude-opus-4-8" not in plain

    def test_update_model_does_not_render_when_hidden(self) -> None:
        """`update_model` tracks the model but skips re-render when the row is off."""
        widget = _make_banner(
            model_provider="anthropic", model_name="claude-opus-4-8", show_model=False
        )
        with patch.object(widget, "update") as mock_update:
            widget.update_model(provider="openai", model="gpt-5")
            mock_update.assert_not_called()
        assert widget._model_name == "gpt-5"

    def test_hidden_without_show_model_flag(self) -> None:
        """No model row when `SPLASH_SHOW_MODEL` is not set (opt-in)."""
        plain = _make_banner(show_model=False)._build_banner().plain
        assert "model:" not in plain


class TestDirectoryLine:
    """Tests for the opt-in directory row (`SPLASH_SHOW_CWD`)."""

    def test_shows_directory_when_flag_set(self) -> None:
        """The directory row renders the working directory when enabled."""
        plain = _make_banner(cwd="/work/project", show_cwd=True)._build_banner().plain
        assert "directory:" in plain
        assert "/work/project" in plain

    def test_home_prefixed(self) -> None:
        """The directory is home-prefixed with `~`."""
        cwd = str(Path.home() / "code" / "app")
        plain = _make_banner(cwd=cwd, show_cwd=True)._build_banner().plain
        assert "~/code/app" in plain

    def test_hidden_without_flag(self) -> None:
        """No directory row when `SPLASH_SHOW_CWD` is not set (opt-in)."""
        plain = _make_banner(cwd="/work/project")._build_banner().plain
        assert "directory:" not in plain

    def test_update_cwd_refreshes_when_shown(self) -> None:
        """`update_cwd` re-renders (calls `update`) the directory row when enabled."""
        widget = _make_banner(cwd="/work/project", show_cwd=True)
        with patch.object(widget, "update") as mock_update:
            widget.update_cwd("/work/other")
            mock_update.assert_called_once()
        plain = widget._build_banner().plain
        assert "/work/other" in plain
        assert "/work/project" not in plain

    def test_update_cwd_does_not_render_when_hidden(self) -> None:
        """`update_cwd` tracks the path but skips re-render when the row is off."""
        widget = _make_banner(cwd="/work/project", show_cwd=False)
        with patch.object(widget, "update") as mock_update:
            widget.update_cwd("/work/other")
            mock_update.assert_not_called()
        assert widget._cwd == "/work/other"


class TestTracingLine:
    """Tests for the LangSmith tracing project row."""

    def test_shows_project_name_without_url(self) -> None:
        """The tracing row renders the project name even before the URL resolves."""
        plain = _make_banner(project_name="dcode-johannes")._build_banner().plain
        assert "tracing:" in plain
        assert "'dcode-johannes'" in plain

    def test_project_name_is_clickable_when_url_resolved(self) -> None:
        """The project name is a hyperlink when the URL has been fetched."""
        from textual.style import Style as TStyle

        widget = _make_banner(
            project_name="dcode-johannes",
            project_urls={
                "dcode-johannes": "https://smith.langchain.com/o/org/p/dcode-johannes"
            },
        )
        content = widget._build_banner()
        linked_spans = [
            s for s in content.spans if isinstance(s.style, TStyle) and s.style.link
        ]
        assert any(
            "dcode-johannes" in content._text[s.start : s.end] for s in linked_spans
        )

    def test_project_name_not_clickable_without_url(self) -> None:
        """The project name has no link when the URL has not been fetched."""
        from textual.style import Style as TStyle

        widget = _make_banner(project_name="dcode-johannes")
        content = widget._build_banner()
        linked = [
            s for s in content.spans if isinstance(s.style, TStyle) and s.style.link
        ]
        assert not linked

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

    async def test_fetch_and_update_sets_url(self) -> None:
        """`_fetch_and_update` fetches the URL and re-renders the banner."""
        widget = _make_banner(project_name="dcode-johannes")
        with (
            patch(
                _FETCH_URL,
                return_value="https://smith.langchain.com/o/org/p/dcode-johannes",
            ),
            patch.object(widget, "update"),
        ):
            await widget._fetch_and_update()
        assert widget._project_urls["dcode-johannes"] is not None
        assert "dcode-johannes" in widget._project_urls["dcode-johannes"]

    async def test_fetch_and_update_handles_timeout(self) -> None:
        """`_fetch_and_update` does not crash on timeout."""
        widget = _make_banner(project_name="dcode-johannes")

        def _raise_timeout(*_args: object, **_kwargs: object) -> str:
            raise TimeoutError

        with (
            patch(_FETCH_URL, side_effect=_raise_timeout),
            patch.object(widget, "update"),
        ):
            await widget._fetch_and_update()
        assert widget._project_urls == {}


class TestReplicaTracingLine:
    """Tests for the LangSmith replica tracing project row."""

    def test_shows_replica_project_by_default(self) -> None:
        """The replica row renders when a primary project and replica are set."""
        plain = (
            _make_banner(
                project_name="dcode-primary",
                replica_project="dcode-replica",
            )
            ._build_banner()
            .plain
        )
        assert "tracing:" in plain
        assert "'dcode-primary'" in plain
        assert "replica:" in plain
        assert "'dcode-replica'" in plain

    def test_hidden_when_show_replica_flag_disabled(self) -> None:
        """The replica row respects `SHOW_LANGSMITH_REPLICA_TRACING`."""
        plain = (
            _make_banner(
                project_name="dcode-primary",
                replica_project="dcode-replica",
                env={SHOW_LANGSMITH_REPLICA_TRACING: "0"},
            )
            ._build_banner()
            .plain
        )
        assert "tracing:" in plain
        assert "replica:" not in plain
        assert "dcode-replica" not in plain

    def test_hidden_when_primary_tracing_hidden(self) -> None:
        """Replica tracing is hidden with the primary tracing row."""
        plain = (
            _make_banner(
                project_name="dcode-primary",
                replica_project="dcode-replica",
                env={HIDE_LANGSMITH_TRACING: "1"},
            )
            ._build_banner()
            .plain
        )
        assert "tracing:" not in plain
        assert "replica:" not in plain
        assert "dcode-replica" not in plain

    def test_replica_project_is_clickable_when_url_resolved(self) -> None:
        """The replica project is a hyperlink when the URL has been fetched."""
        from textual.style import Style as TStyle

        widget = _make_banner(
            project_name="dcode-primary",
            replica_project="dcode-replica",
            project_urls={
                "dcode-replica": "https://smith.langchain.com/o/org/p/dcode-replica"
            },
        )
        content = widget._build_banner()
        linked_spans = [
            s for s in content.spans if isinstance(s.style, TStyle) and s.style.link
        ]
        assert any(
            "dcode-replica" in content._text[s.start : s.end] for s in linked_spans
        )

    async def test_fetch_and_update_sets_primary_and_replica_urls(self) -> None:
        """`_fetch_and_update` fetches URLs for primary and replica projects."""
        widget = _make_banner(
            project_name="dcode-primary",
            replica_project="dcode-replica",
        )
        urls = {
            "dcode-primary": "https://smith.langchain.com/o/org/p/dcode-primary",
            "dcode-replica": "https://smith.langchain.com/o/org/p/dcode-replica",
        }

        def _fetch_url(project: str) -> str:
            return urls[project]

        with (
            patch(_FETCH_URL, side_effect=_fetch_url),
            patch.object(widget, "update"),
        ):
            await widget._fetch_and_update()
        assert widget._project_urls == urls


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


class TestMcpWarnings:
    """Tests for MCP server warning lines."""

    def test_shows_unauthenticated_warning(self) -> None:
        """An unauthenticated-server warning line is rendered."""
        plain = _make_banner(mcp_unauthenticated=2)._build_banner().plain
        assert "2 MCP servers need login" in plain
        assert "open /mcp" in plain

    def test_singular_unauthenticated(self) -> None:
        """A single unauthenticated server uses singular wording."""
        plain = _make_banner(mcp_unauthenticated=1)._build_banner().plain
        assert "1 MCP server needs login" in plain

    def test_shows_errored_warning(self) -> None:
        """An errored-server warning line is rendered."""
        plain = _make_banner(mcp_errored=1)._build_banner().plain
        assert "1 MCP server failed to load" in plain
        assert "open /mcp for details" in plain

    def test_shows_awaiting_reconnect_warning(self) -> None:
        """An awaiting-reconnect warning line is rendered."""
        plain = _make_banner(mcp_awaiting_reconnect=3)._build_banner().plain
        assert "3 MCP servers ready to load" in plain
        assert "/mcp reconnect" in plain

    def test_no_warnings_when_all_zero(self) -> None:
        """No warning lines when all warning counts are zero."""
        plain = _make_banner()._build_banner().plain
        assert "login" not in plain
        assert "failed to load" not in plain
        assert "reconnect" not in plain

    def test_set_connected_updates_warnings(self) -> None:
        """`set_connected` updates warning counts and re-renders."""
        widget = _make_banner()
        with patch.object(widget, "update"):
            widget.set_connected(
                5, mcp_unauthenticated=1, mcp_errored=2, mcp_awaiting_reconnect=3
            )
        assert widget._mcp_tool_count == 5
        assert widget._mcp_unauthenticated == 1
        assert widget._mcp_errored == 2
        assert widget._mcp_awaiting_reconnect == 3
        plain = widget._build_banner().plain
        assert "1 MCP server needs login" in plain
        assert "2 MCP servers failed to load" in plain
        assert "3 MCP servers ready to load" in plain


class TestEditableInstallPath:
    """Tests for the editable-install path row."""

    def test_shows_install_path_for_editable(self) -> None:
        """The install path is shown for editable installs."""
        with (
            patch(_EDITABLE, return_value=True),
            patch(_EDITABLE_PATH, return_value="~/oss/deepagents/libs/code"),
        ):
            plain = _make_banner()._build_banner().plain
        assert "installed:" in plain
        assert "~/oss/deepagents/libs/code" in plain

    def test_no_install_path_for_non_editable(self) -> None:
        """No install path row for non-editable installs."""
        with (
            patch(_EDITABLE, return_value=False),
            patch(_EDITABLE_PATH, return_value=None),
        ):
            plain = _make_banner()._build_banner().plain
        assert "installed:" not in plain

    def test_no_install_path_when_version_hidden(self) -> None:
        """No install path row when the version is hidden."""
        with (
            patch(_EDITABLE, return_value=True),
            patch(_EDITABLE_PATH, return_value="~/code"),
        ):
            plain = _make_banner(env={HIDE_SPLASH_VERSION: "1"})._build_banner().plain
        assert "installed:" not in plain

    def test_no_install_path_when_cwd_hidden(self) -> None:
        """No install path row when local path displays are hidden."""
        with (
            patch(_EDITABLE, return_value=True),
            patch(_EDITABLE_PATH, return_value="~/code"),
        ):
            plain = _make_banner(env={HIDE_CWD: "1"})._build_banner().plain
        assert "installed:" not in plain
        assert "~/code" not in plain


class TestRemovedSections:
    """The banner does not show the old splash tips/footer content."""

    def test_no_legacy_sections(self) -> None:
        """None of the old splash footer sections appear."""
        plain = _make_banner()._build_banner().plain
        for absent in ("Ready to code", "Tip:", "tip:"):
            assert absent not in plain


class TestReturnType:
    """Tests for `_build_banner` return value."""

    def test_returns_content(self) -> None:
        """`_build_banner` returns a `Content` object."""
        assert isinstance(_make_banner()._build_banner(), Content)


class TestThreadIdUpdates:
    """`update_thread_id` tracks the id and only re-renders in debug mode."""

    def test_update_thread_id_tracks_without_rendering(self) -> None:
        """`update_thread_id` stores the id but does not re-render without debug."""
        widget = _make_banner()
        with patch.object(widget, "update") as mock_update:
            widget.update_thread_id("abc123")
            mock_update.assert_not_called()
        assert widget._cli_thread_id == "abc123"
        assert "abc123" not in widget._build_banner().plain

    def test_update_thread_id_renders_in_debug(self) -> None:
        """`update_thread_id` re-renders (calls `update`) to show the id in debug."""
        widget = _make_banner(env={DEBUG: "1"})
        with patch.object(widget, "update") as mock_update:
            widget.update_thread_id("abc123")
            mock_update.assert_called_once()
        plain = widget._build_banner().plain
        assert "abc123" in plain
        assert "thread:" in plain


class TestAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`WelcomeBanner` should disable Textual's `auto_links`."""
        assert WelcomeBanner.auto_links is False
