"""Welcome banner widget."""

from __future__ import annotations

import asyncio
import os
import random
from typing import TYPE_CHECKING, Any

from textual.color import Color as TColor
from textual.content import Content
from textual.style import Style as TStyle
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.events import Click
    from textual.timer import Timer

from deepagents_code import theme
from deepagents_code._env_vars import (
    DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER,
    HIDE_CWD,
    HIDE_LANGSMITH_TRACING,
    HIDE_SPLASH_TIPS,
    HIDE_SPLASH_VERSION,
    is_env_truthy,
)
from deepagents_code._version import __version__
from deepagents_code.config import (
    _get_editable_install_path,
    _is_editable_install,
    fetch_langsmith_project_url,
    get_banner,
    get_glyphs,
    get_langsmith_project_name,
)
from deepagents_code.widgets._links import open_style_link

_TIPS: dict[str, int] = {
    "Use @ to reference files and / for commands": 3,
    "Try /threads to resume a previous conversation": 2,
    "Use /offload when your conversation gets long": 2,
    "Use /copy to copy the latest assistant message": 3,
    "Use /mcp to search your MCP servers and inspect tool parameters": 1,
    "Use /mcp login <server> to authenticate MCP OAuth servers without leaving the TUI": 1,  # noqa: E501
    "Use /remember to save learnings from this conversation": 1,
    "Use /model to switch models mid-conversation": 2,
    "Press ctrl+x to compose prompts in your external editor": 1,
    "Press ctrl+u to delete to the start of the line in the chat input": 1,
    "Use /skill:<name> to invoke a skill directly": 1,
    "Type /update to check for and install updates": 1,
    "Use /install <extra> to add optional dependencies (e.g. /install quickjs)": 1,
    "Use /theme to customize the TUI's colors": 1,
    "In /theme, press N to toggle labels/keys, T to set for the current terminal": 1,
    "Use /skill-creator to build reusable agent skills": 1,
    "Use /auto-update to toggle automatic updates": 1,
    "Use /agents to browse and switch between your available agents": 2,
    "In /agents, press Ctrl+S to set the highlighted agent as your default": 1,
    "Press Shift+Tab to toggle auto-approve mode": 2,
    "Use --startup-cmd to run a shell command before the first prompt": 1,
    "Use !! for incognito shell commands that stay out of model context": 1,
    "Deep Agents can explain its own features and look up its docs. Ask it how to use.": 3,  # noqa: E501
}
"""Rotating tips shown in the welcome footer, with relative selection weights.

One is picked per session. Higher weights are picked more often.
"""

_CONNECTING_FOOTER_DELAY_SECONDS = 5.0
"""Upper bound on how long the banner waits before revealing "Connecting...".

Startup is usually fast enough that flashing the spinner makes the app feel
slower than it is; the welcome footer renders immediately and the connecting
footer only appears if startup is genuinely taking a while or the user
submits a message before the agent is reachable. The timer is cancelled
early when `set_connected`, `set_idle`, or `set_connecting` runs first, so
this delay is the maximum — not a fixed wait.
"""

_DOT_FRAMES: tuple[str, ...] = (".", "..", "...")
"""Ellipsis animation frames cycled by the connecting-footer dot timer."""

_DOT_INTERVAL = 0.4
"""Seconds between connecting-footer ellipsis frame advances."""


def _pick_tip() -> str:
    """Pick a tip from `_TIPS` weighted by its associated weight.

    Returns:
        A single tip string, selected with probability proportional to its
        weight in `_TIPS`.
    """
    tips = list(_TIPS.keys())
    weights = list(_TIPS.values())
    return random.choices(tips, weights=weights, k=1)[0]  # noqa: S311


class WelcomeBanner(Static):
    """Welcome banner displayed at startup."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        thread_id: str | None = None,
        mcp_tool_count: int = 0,
        *,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
        connecting: bool = False,
        resuming: bool = False,
        local_server: bool = False,
        defer_connecting_display: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the welcome banner.

        Args:
            thread_id: Optional thread ID to display in the banner.
            mcp_tool_count: Number of MCP tools loaded at startup.
            mcp_unauthenticated: Number of MCP servers awaiting login.
            mcp_errored: Number of MCP servers that failed to load.
            mcp_awaiting_reconnect: Number of MCP servers that completed OAuth
                login but are waiting for `/mcp reconnect` before their tools
                can load.
            connecting: When `True`, show a "Connecting..." footer instead of
                the normal ready prompt. Call `set_connected` to transition.
            resuming: When `True`, the connecting footer says "Resuming..."
                instead of any `'Connecting...'` variant.
            local_server: When `True`, the connecting footer qualifies the
                server as "local" (i.e. a server process).

                Ignored when `resuming` is `True`.
            defer_connecting_display: When `True` and `connecting` is `True`,
                suppress the connecting footer initially so a fast startup
                feels instantaneous; the welcome footer remains visible until
                startup resolves. The connecting footer is revealed by
                `reveal_connecting_footer` (called when the user submits a
                message during startup) or automatically after
                `_CONNECTING_FOOTER_DELAY_SECONDS`.
            **kwargs: Additional arguments passed to parent.
        """
        # Avoid collision with Widget._thread_id (Textual internal int)
        self._cli_thread_id: str | None = thread_id
        self._mcp_tool_count = mcp_tool_count
        self._mcp_unauthenticated = mcp_unauthenticated
        self._mcp_errored = mcp_errored
        self._mcp_awaiting_reconnect = mcp_awaiting_reconnect
        self._connecting = connecting
        self._resuming = resuming
        self._local_server = local_server
        self._reconnecting = False
        self._idle = False
        self._defer_connecting_display = defer_connecting_display and connecting
        self._defer_timer: Timer | None = None
        self._dot_frame: int = len(_DOT_FRAMES) - 1
        self._dot_timer: Timer | None = None
        self._hide_langsmith_tracing = is_env_truthy(HIDE_LANGSMITH_TRACING)
        self._hide_splash_tips = is_env_truthy(HIDE_SPLASH_TIPS)
        self._project_name: str | None = (
            None if self._hide_langsmith_tracing else get_langsmith_project_name()
        )
        self._project_url: str | None = None
        self._tip: str | None = None if self._hide_splash_tips else _pick_tip()

        super().__init__(self._build_banner(), **kwargs)

    def on_mount(self) -> None:
        """Kick off background fetch for LangSmith project URL."""
        self.watch(self.app, "theme", self._on_theme_change, init=False)
        if self._project_name:
            self.run_worker(self._fetch_and_update, exclusive=True)
        if self._defer_connecting_display:
            self._defer_timer = self.set_timer(
                _CONNECTING_FOOTER_DELAY_SECONDS, self._on_defer_timer_fired
            )
        elif self._connecting:
            self._start_dot_animation()

    def _cancel_defer_timer(self) -> None:
        """Stop and drop the deferred-display timer if it is still pending."""
        if self._defer_timer is not None:
            self._defer_timer.stop()
            self._defer_timer = None

    def _start_dot_animation(self) -> None:
        """Start the ellipsis animation for the connecting footer.

        No-op when the widget is not yet running (e.g., called before mount
        or from sync test code): `set_interval` requires a live event loop.
        """
        if self._dot_timer is not None or not self._running:
            return
        self._dot_frame = len(_DOT_FRAMES) - 1
        self._dot_timer = self.set_interval(_DOT_INTERVAL, self._tick_dots)

    def _stop_dot_animation(self) -> None:
        """Stop the ellipsis animation and reset to full dots."""
        if self._dot_timer is not None:
            self._dot_timer.stop()
            self._dot_timer = None
        self._dot_frame = len(_DOT_FRAMES) - 1

    def _tick_dots(self) -> None:
        """Advance the ellipsis frame and re-render the banner."""
        self._dot_frame = (self._dot_frame + 1) % len(_DOT_FRAMES)
        self.update(self._build_banner(self._project_url))

    def _on_defer_timer_fired(self) -> None:
        """Reveal the connecting footer once the deferral window expires."""
        self._defer_timer = None
        self.reveal_connecting_footer()

    def reveal_connecting_footer(self) -> None:
        """Stop deferring the "Connecting..." footer and render it now.

        No-op once the deferred state has been cleared (by reveal, connect,
        idle, or because deferral was never active). Two callers reach this:
        the deferral timer (`_on_defer_timer_fired`) when the wait window
        elapses, and the app when the user submits a message during startup
        so the queued state has explicit feedback.
        """
        if not self._defer_connecting_display:
            return
        self._cancel_defer_timer()
        self._defer_connecting_display = False
        if self._connecting:
            self._start_dot_animation()
            self.update(self._build_banner(self._project_url))

    def _on_theme_change(self) -> None:
        """Re-render the banner when the app theme changes."""
        self.update(self._build_banner(self._project_url))

    async def _fetch_and_update(self) -> None:
        """Fetch the LangSmith URL in a thread and update the banner."""
        if not self._project_name:
            return
        try:
            project_url = await asyncio.wait_for(
                asyncio.to_thread(fetch_langsmith_project_url, self._project_name),
                timeout=2.0,
            )
        except (TimeoutError, OSError):
            project_url = None
        if project_url:
            self._project_url = project_url
            self.update(self._build_banner(project_url))

    def update_thread_id(self, thread_id: str) -> None:
        """Update the displayed thread ID and re-render the banner.

        Args:
            thread_id: The new thread ID to display.
        """
        self._cli_thread_id = thread_id
        self.update(self._build_banner(self._project_url))

    def set_connected(
        self,
        mcp_tool_count: int = 0,
        *,
        mcp_unauthenticated: int = 0,
        mcp_errored: int = 0,
        mcp_awaiting_reconnect: int = 0,
    ) -> None:
        """Transition from "connecting" to "ready" state.

        Args:
            mcp_tool_count: Number of MCP tools loaded during connection.
            mcp_unauthenticated: Number of MCP servers awaiting login.
            mcp_errored: Number of MCP servers that failed to load.
            mcp_awaiting_reconnect: Number of MCP servers that completed OAuth
                login but are waiting for `/mcp reconnect` before their tools
                can load.
        """
        self._connecting = False
        self._reconnecting = False
        self._idle = False
        self._defer_connecting_display = False
        self._cancel_defer_timer()
        self._stop_dot_animation()
        self._mcp_tool_count = mcp_tool_count
        self._mcp_unauthenticated = mcp_unauthenticated
        self._mcp_errored = mcp_errored
        self._mcp_awaiting_reconnect = mcp_awaiting_reconnect
        self.update(self._build_banner(self._project_url))

    def set_connecting(self) -> None:
        """Transition back to the "connecting" state.

        Used when the server is being restarted mid-session (e.g., switching
        agents via `/agents`), so the banner reflects that no agent is
        currently reachable. Mid-session swaps show the connecting footer
        immediately — only the initial app launch defers it.
        """
        self._stop_dot_animation()
        self._connecting = True
        self._reconnecting = True
        self._idle = False
        self._resuming = False
        self._defer_connecting_display = False
        self._cancel_defer_timer()
        self._start_dot_animation()
        self.update(self._build_banner(self._project_url))

    def set_idle(self) -> None:
        """Transition to a neutral state with no connecting spinner or footer.

        Used after a fatal startup failure so the banner stops claiming
        progress (the failure is communicated via the chat surface). The
        banner keeps its identity rows (title, version, install path,
        LangSmith project, thread ID) but appends no footer line, leaving
        the chat error as the sole source of failure context.
        """
        self._connecting = False
        self._reconnecting = False
        self._idle = True
        self._defer_connecting_display = False
        self._cancel_defer_timer()
        self._stop_dot_animation()
        self.update(self._build_banner(self._project_url))

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)

    def _build_banner(self, project_url: str | None = None) -> Content:
        """Build the banner content.

        When a `project_url` is provided and a thread ID is set, the thread ID
        is rendered as a clickable hyperlink to the LangSmith thread view.

        Args:
            project_url: LangSmith project URL used for linking the project
                name and thread ID. When `None`, text is rendered without links.

        Returns:
            Content object containing the formatted banner.
        """
        parts: list[str | tuple[str, str | TStyle] | Content] = []
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in {"ansi-dark", "ansi-light"}

        banner = get_banner()
        primary_style: str | TStyle = (
            "bold"
            if ansi
            else TStyle(foreground=TColor.parse(colors.primary), bold=True)
        )

        hide_version = is_env_truthy(HIDE_SPLASH_VERSION)
        if not hide_version and not ansi and _is_editable_install():
            # Highlight local-install version tag with tool accent; art stays primary.
            dev_style = TStyle(foreground=TColor.parse(colors.tool), bold=True)
            version_tag = f"v{__version__} (local)"
            idx = banner.rfind(version_tag)
            if idx >= 0:
                parts.extend(
                    [
                        (banner[:idx], primary_style),
                        (version_tag, dev_style),
                        (banner[idx + len(version_tag) :] + "\n", primary_style),
                    ]
                )
            else:
                parts.append((banner + "\n", primary_style))
        else:
            parts.append((banner + "\n", primary_style))

        # For ANSI theme, use "bold" (terminal foreground) instead of hex
        accent: str | TStyle = "bold" if ansi else colors.primary
        success_color: str = "bold green" if ansi else colors.success

        hide_editable_path = hide_version or is_env_truthy(HIDE_CWD)
        editable_path = None if hide_editable_path else _get_editable_install_path()
        if editable_path:
            parts.extend([("Installed from: ", "dim"), (editable_path, "dim"), "\n"])

        if self._project_name:
            parts.extend(
                [
                    (f"{get_glyphs().checkmark} ", success_color),
                    "LangSmith tracing: ",
                ]
            )
            if project_url:
                link_style: str | TStyle
                if ansi:
                    url = f"{project_url}?utm_source=deepagents-code"
                    link_style = TStyle(bold=True, link=url)
                else:
                    link_style = TStyle(
                        foreground=TColor.parse(colors.primary),
                        link=f"{project_url}?utm_source=deepagents-code",
                    )
                parts.append((f"'{self._project_name}'", link_style))
            else:
                parts.append((f"'{self._project_name}'", accent))
            parts.append("\n")

        if self._cli_thread_id and not self._hide_langsmith_tracing:
            if project_url:
                thread_url = (
                    f"{project_url.rstrip('/')}/t/{self._cli_thread_id}"
                    "?utm_source=deepagents-code"
                )
                parts.extend(
                    [
                        ("  Thread: ", "dim"),
                        (self._cli_thread_id, TStyle(dim=True, link=thread_url)),
                        ("\n", "dim"),
                    ]
                )
            else:
                parts.append((f"  Thread: {self._cli_thread_id}\n", "dim"))

        if self._mcp_tool_count > 0:
            parts.append((f"{get_glyphs().checkmark} ", success_color))
            label = "MCP tool" if self._mcp_tool_count == 1 else "MCP tools"
            parts.append(f"Loaded {self._mcp_tool_count} {label}\n")

        warn_color: str = "bold yellow" if ansi else colors.warning
        if self._mcp_unauthenticated > 0:
            server_label = "server" if self._mcp_unauthenticated == 1 else "servers"
            unauth_text = (
                f"{self._mcp_unauthenticated} MCP {server_label} need login "
                "— open /mcp or run `/mcp login <server>`\n"
            )
            parts.extend(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (unauth_text, "dim"),
                ]
            )
        if self._mcp_errored > 0:
            server_label = "server" if self._mcp_errored == 1 else "servers"
            errored_text = (
                f"{self._mcp_errored} MCP {server_label} failed to load "
                "— open /mcp for details\n"
            )
            parts.extend(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (errored_text, "dim"),
                ]
            )
        if self._mcp_awaiting_reconnect > 0:
            server_label = "server" if self._mcp_awaiting_reconnect == 1 else "servers"
            awaiting_text = (
                f"{self._mcp_awaiting_reconnect} MCP {server_label} ready to load "
                "— run `/mcp reconnect`\n"
            )
            parts.extend(
                [
                    (f"{get_glyphs().warning} ", warn_color),
                    (awaiting_text, "dim"),
                ]
            )

        show_connecting = self._connecting and not self._defer_connecting_display
        if show_connecting:
            parts.append(
                build_connecting_footer(
                    resuming=self._resuming,
                    local_server=self._local_server,
                    reconnecting=self._reconnecting,
                    dots=_DOT_FRAMES[self._dot_frame],
                )
            )
        elif not self._idle:
            ready_color = "bold" if ansi else colors.primary
            parts.append(
                build_welcome_footer(
                    primary_color=ready_color,
                    tip=self._tip,
                    show_tip=not self._hide_splash_tips,
                )
            )
        # `_idle` ⇒ no footer; chat-surface owns the failure message.
        return Content.assemble(*parts)


def build_connecting_footer(
    *,
    resuming: bool = False,
    local_server: bool = False,
    reconnecting: bool = False,
    dots: str = "...",
) -> Content:
    """Build a footer shown while waiting for the server to connect.

    `resuming` wins over the other branches; otherwise `local_server`
    selects between the local and generic variants, and `reconnecting`
    swaps the verb only when `local_server` is `True`.

    Args:
        resuming: Show `'Resuming...'` instead of any `'Connecting...'` variant.
        local_server: Qualify the server as "local" in the connecting message.
            Honored only when `resuming` is `False`.
        reconnecting: Use `'Reconnecting'` instead of `'Connecting'` for
            mid-session restarts. Honored only when `local_server` is `True`
            and `resuming` is `False`.
        dots: Ellipsis string appended to the status text. Pass an animated
            frame (e.g. `"."`, `".."`, `"..."`) to show a cycling indicator.

    Returns:
        Content with a connecting status message.
    """
    if resuming:
        text = f"\nResuming{dots}\n"
    elif local_server:
        verb = "Reconnecting" if reconnecting else "Connecting"
        text = f"\n{verb} to local server{dots}\n"
    else:
        text = f"\nConnecting to server{dots}\n"
    return Content.styled(text, "dim")


def build_welcome_footer(
    *,
    primary_color: str = theme.PRIMARY,
    tip: str | None = None,
    show_tip: bool | None = None,
) -> Content:
    """Build the footer shown at the bottom of the welcome banner.

    Includes a tip to help users discover features unless tips are disabled.

    Args:
        primary_color: Color string for the ready prompt.

            Defaults to the module-level ANSI `PRIMARY` constant; widget callers
            should pass the active theme's hex value.
        tip: Tip text to display. When `None`, a random tip is selected.

            Pass an explicit value to keep the tip stable across re-renders.
        show_tip: Whether to show the tip. When `None`, the startup splash tips
            env var controls visibility.

    Returns:
        Content with the ready prompt and, when enabled, a tip.
    """
    if show_tip is None:
        show_tip = not is_env_truthy(HIDE_SPLASH_TIPS)
    if show_tip and tip is None:
        tip = _pick_tip()
    subheader = (
        os.environ.get(DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER)
        or "Ready to code! What would you like to build?"
    )
    parts: list[tuple[str, str]] = [(f"\n{subheader}", primary_color)]
    if show_tip and tip is not None:
        parts.append((f"\nTip: {tip}", "dim italic"))
    return Content.assemble(*parts)
