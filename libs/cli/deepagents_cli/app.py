"""Textual UI application for deepagents-cli."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import sys
import time
import uuid
import webbrowser
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from textual.app import App, ScreenStackError
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.notifications import Notification as _Notification, Notify as _Notify
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.theme import Theme
from textual.widgets import Static
from textual.widgets._toast import (
    Toast as _Toast,  # noqa: PLC2701  # for Toast click routing
)

from deepagents_cli import theme
from deepagents_cli._cli_context import CLIContext
from deepagents_cli._git import (
    read_git_branch_from_filesystem,
    read_git_branch_via_subprocess,
)
from deepagents_cli._session_stats import (
    SessionStats,
    SpinnerStatus,
    format_token_count,
)

# Only is_ascii_mode is needed before first paint (on_mount scrollbar config).
# All other config imports — settings, create_model, detect_provider, etc. — are
# deferred to local imports at their call sites since they are only accessed
# after user interaction begins.
from deepagents_cli._version import CHANGELOG_URL, DOCS_URL
from deepagents_cli.config import is_ascii_mode
from deepagents_cli.notifications import (
    ActionId,
    MissingDepPayload,
    NotificationAction,
    NotificationRegistry,
    PendingNotification,
    UpdateAvailablePayload,
)
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.loading import LoadingWidget
from deepagents_cli.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    ErrorMessage,
    QueuedUserMessage,
    SkillMessage,
    ToolCallMessage,
    UserMessage,
)
from deepagents_cli.widgets.status import StatusBar
from deepagents_cli.widgets.welcome import WelcomeBanner

logger = logging.getLogger(__name__)
_monotonic = time.monotonic

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends import CompositeBackend
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel
    from textual.app import ComposeResult
    from textual.events import Click, MouseUp, Paste
    from textual.scrollbar import ScrollUp
    from textual.widget import Widget
    from textual.worker import Worker

    from deepagents_cli._ask_user_types import AskUserWidgetResult, Question
    from deepagents_cli.mcp_tools import MCPServerInfo
    from deepagents_cli.remote_client import RemoteAgent
    from deepagents_cli.server import ServerProcess
    from deepagents_cli.skills.load import ExtendedSkillMetadata
    from deepagents_cli.textual_adapter import TextualUIAdapter
    from deepagents_cli.widgets.approval import ApprovalMenu
    from deepagents_cli.widgets.ask_user import AskUserMenu

# iTerm2 Cursor Guide Workaround
# ===============================
# iTerm2's cursor guide (highlight cursor line) causes visual artifacts when
# Textual takes over the terminal in alternate screen mode. We disable it at
# module load and restore on exit. Both atexit and exit() override are used
# for defense-in-depth: atexit catches abnormal termination (SIGTERM, unhandled
# exceptions), while exit() ensures restoration before Textual's cleanup.

# Detection: check env vars AND that stderr is a TTY (avoids false positives
# when env vars are inherited but running in non-TTY context like CI)
_IS_ITERM = (
    (
        os.environ.get("LC_TERMINAL", "") == "iTerm2"
        or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
    )
    and hasattr(os, "isatty")
    and os.isatty(2)
)

# iTerm2 cursor guide escape sequences (OSC 1337)
# Format: OSC 1337 ; HighlightCursorLine=<yes|no> ST
# Where OSC = ESC ] (0x1b 0x5d) and ST = ESC \ (0x1b 0x5c)
_ITERM_CURSOR_GUIDE_OFF = "\x1b]1337;HighlightCursorLine=no\x1b\\"
_ITERM_CURSOR_GUIDE_ON = "\x1b]1337;HighlightCursorLine=yes\x1b\\"


def _write_iterm_escape(sequence: str) -> None:
    """Write an iTerm2 escape sequence to stderr.

    Silently fails if the terminal is unavailable (redirected, closed, broken
    pipe). This is a cosmetic feature, so failures should never crash the app.
    """
    if not _IS_ITERM:
        return
    try:
        import sys

        if sys.__stderr__ is not None:
            sys.__stderr__.write(sequence)
            sys.__stderr__.flush()
    except OSError:
        # Terminal may be unavailable (redirected, closed, broken pipe)
        pass


# Disable cursor guide at module load (before Textual takes over)
_write_iterm_escape(_ITERM_CURSOR_GUIDE_OFF)

if _IS_ITERM:
    import atexit

    def _restore_cursor_guide() -> None:
        """Restore iTerm2 cursor guide on exit.

        Registered with atexit to ensure the cursor guide is re-enabled
        when the CLI exits, regardless of how the exit occurs.
        """
        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    atexit.register(_restore_cursor_guide)


def _load_theme_preference() -> str:
    """Load the saved theme name from config, or return the default.

    Returns:
        A Textual theme name (e.g., `'langchain'`, `'langchain-light'`).
    """
    import tomllib

    try:
        from deepagents_cli.model_config import DEFAULT_CONFIG_PATH

        if not DEFAULT_CONFIG_PATH.exists():
            return theme.DEFAULT_THEME

        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as exc:
        logger.warning("Could not read config for theme preference: %s", exc)
        return theme.DEFAULT_THEME

    name = data.get("ui", {}).get("theme")
    if isinstance(name, str) and name in theme.ThemeEntry.REGISTRY:
        return name
    if isinstance(name, str):
        logger.warning(
            "Unknown theme '%s' in config; falling back to default",
            name,
        )
    return theme.DEFAULT_THEME


def save_theme_preference(name: str) -> bool:
    """Persist theme preference to `~/.deepagents/config.toml`.

    Args:
        name: Textual theme name to save.

    Returns:
        `True` if the preference was saved, `False` if any error occurred.
    """
    if name not in theme.ThemeEntry.REGISTRY:
        logger.warning("Refusing to save unknown theme '%s'", name)
        return False

    import contextlib
    import tempfile

    try:
        import tomllib

        import tomli_w

        from deepagents_cli.model_config import DEFAULT_CONFIG_PATH

        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DEFAULT_CONFIG_PATH.exists():
            with DEFAULT_CONFIG_PATH.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "ui" not in data:
            data["ui"] = {}
        data["ui"]["theme"] = name

        fd, tmp_path = tempfile.mkstemp(dir=DEFAULT_CONFIG_PATH.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(DEFAULT_CONFIG_PATH)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except Exception:
        logger.exception("Could not save theme preference")
        return False
    return True


def _extract_model_params_flag(raw_arg: str) -> tuple[str, dict[str, Any] | None]:
    """Extract `--model-params` and its JSON value from a `/model` arg string.

    Handles quoted (`'...'` / `"..."`) and bare `{...}` values with balanced
    braces so that JSON containing spaces works without quoting.

    Note:
        The bare-brace mode counts `{` / `}` characters without awareness of
        JSON string contents. Values that contain literal braces inside strings
        (e.g., `{"stop": "end}here"}`) will mis-parse. Users should quote the
        value in that case.

    Args:
        raw_arg: The argument string after `/model `.

    Returns:
        Tuple of `(remaining_args, parsed_dict | None)`. Returns `None` for the
            dict when the flag is absent.

    Raises:
        ValueError: If the value is missing, has unclosed quotes,
            unbalanced braces, or is not valid JSON.
        TypeError: If the parsed JSON is not a dict.
    """
    flag = "--model-params"
    idx = raw_arg.find(flag)
    if idx == -1:
        return raw_arg, None

    before = raw_arg[:idx].rstrip()
    after = raw_arg[idx + len(flag) :].lstrip()

    if not after:
        msg = "--model-params requires a JSON object value"
        raise ValueError(msg)

    # Determine the JSON string boundaries.
    if after[0] in {"'", '"'}:
        quote = after[0]
        end = -1
        backslash_count = 0
        for i, ch in enumerate(after[1:], start=1):
            if ch == "\\":
                backslash_count += 1
                continue
            if ch == quote and backslash_count % 2 == 0:
                end = i
                break
            backslash_count = 0
        if end == -1:
            msg = f"Unclosed {quote} in --model-params value"
            raise ValueError(msg)
        # Parse the quoted token with shlex so escaped quotes are unescaped.
        json_str = shlex.split(after[: end + 1], posix=True)[0]
        rest = after[end + 1 :].lstrip()
    elif after[0] == "{":
        # Walk forward to find the matching closing brace.
        depth = 0
        end = -1
        for i, ch in enumerate(after):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            msg = "Unbalanced braces in --model-params value"
            raise ValueError(msg)
        json_str = after[: end + 1]
        rest = after[end + 1 :].lstrip()
    else:
        # Non-brace, non-quoted — take the next whitespace-delimited token.
        parts = after.split(None, 1)
        json_str = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    remaining = f"{before} {rest}".strip()
    try:
        params = json.loads(json_str)
    except json.JSONDecodeError:
        msg = (
            f"Invalid JSON in --model-params: {json_str!r}. "
            'Expected format: --model-params \'{"key": "value"}\''
        )
        raise ValueError(msg) from None
    if not isinstance(params, dict):
        msg = "--model-params must be a JSON object, got " + type(params).__name__
        raise TypeError(msg)
    return remaining, params


InputMode = Literal["normal", "shell", "command"]

_TYPING_IDLE_THRESHOLD_SECONDS: float = 2.0
"""Seconds since the last keystroke after which the user is considered idle and
a pending approval widget can be shown.

Two seconds balances responsiveness with avoiding accidental approval
key presses.
"""

_DEFERRED_APPROVAL_TIMEOUT_SECONDS: float = 30.0
"""Maximum seconds the deferred-approval worker will wait for the user to stop
typing before showing the approval widget regardless."""


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """Represents a queued user message awaiting processing."""

    text: str
    """The message text content."""

    mode: InputMode
    """The input mode that determines message routing."""


DeferredActionKind = Literal[
    "model_switch",
    "thread_switch",
    "chat_output",
    "agent_switch",
]
"""Valid `DeferredAction.kind` values for type-checked deduplication."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredAction:
    """An action deferred until the current busy state resolves."""

    kind: DeferredActionKind
    """Identity key for deduplication — one of `DeferredActionKind`."""

    execute: Callable[[], Awaitable[None]]
    """Async callable that performs the actual work."""


@dataclass(frozen=True, slots=True)
class _ThreadHistoryPayload:
    """Data returned by `_fetch_thread_history_data`."""

    messages: list[MessageData]
    """Converted message data ready for bulk loading."""

    context_tokens: int
    """Persisted `_context_tokens` from the checkpoint (0 if absent)."""


def _new_thread_id() -> str:
    """Deferred-import wrapper around `sessions.generate_thread_id`.

    Returns:
        UUID7 string.
    """
    from deepagents_cli.sessions import generate_thread_id

    return generate_thread_id()


def _action_label(entry: PendingNotification, action_id: ActionId) -> str:
    """Return the user-facing label for *action_id* on *entry*, or the id itself."""
    for action in entry.actions:
        if action.action_id == action_id:
            return action.label
    return str(action_id)


def _truncate(text: str, *, limit: int) -> str:
    """Return *text* truncated to *limit* characters with an ellipsis suffix."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class TextualSessionState:
    """Session state for the Textual app."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        thread_id: str | None = None,
    ) -> None:
        """Initialize session state.

        Args:
            auto_approve: Whether to auto-approve tool calls
            thread_id: Optional thread ID (generates UUID7 if not provided)
        """
        self.auto_approve = auto_approve
        self.thread_id = thread_id or _new_thread_id()

    def reset_thread(self) -> str:
        """Reset to a new thread.

        Returns:
            The new thread_id.
        """
        self.thread_id = _new_thread_id()
        return self.thread_id


_COMMAND_URLS: dict[str, str] = {
    "/changelog": CHANGELOG_URL,
    "/docs": DOCS_URL,
    "/feedback": "https://github.com/langchain-ai/deepagents/issues/new/choose",
}
"""Slash-command to URL mapping for commands that just open a browser."""


_toast_internals_warned: list[bool] = [False]
"""Single-slot flag; once `_Toast._notification` is missing, log warning once.

Reset by `DeepAgentsApp.reset_toast_internals_warned` before tests so the
one-shot semantics can be exercised deterministically.
"""


def _toast_identity(
    widget: _Toast,
    *,
    app: App | None = None,
) -> str | None:
    """Return the identity of the notification backing *widget*, or `None`.

    `_Toast._notification` is a Textual internal. If a future upgrade
    renames it, toast-click routing silently becomes inert. Logs a
    single warning, and — when *app* is supplied — also posts a
    one-shot user-visible toast pointing users at the `ctrl+n`
    fallback so the regression isn't invisible outside the debug log.
    """
    notif = getattr(widget, "_notification", None)
    if notif is None:
        if not _toast_internals_warned[0]:
            logger.warning(
                "Textual Toast no longer exposes `_notification`; "
                "toast-click routing is disabled.",
            )
            if app is not None:
                app.notify(
                    "Toast click routing disabled after a Textual upgrade. "
                    "Press ctrl+n to view notifications.",
                    severity="warning",
                    timeout=10,
                    markup=False,
                )
            _toast_internals_warned[0] = True
        return None
    return getattr(notif, "identity", None)


class DeepAgentsApp(App):
    """Main Textual application for deepagents-cli."""

    TITLE = "Deep Agents"
    """Textual application title."""

    CSS_PATH = "app.tcss"
    """Path to the Textual CSS stylesheet for the app layout."""

    ENABLE_COMMAND_PALETTE = False
    """Disable Textual's built-in command palette in favor of the custom slash
    command system."""

    SCROLL_SENSITIVITY_Y = 1.0
    """Vertical scroll speed (reduced from Textual default for finer control)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding(
            "ctrl+c",
            "quit_or_interrupt",
            "Quit/Interrupt",
            show=False,
            priority=True,
        ),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        Binding(
            "shift+tab",
            "toggle_auto_approve",
            "Toggle Auto-Approve",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+o",
            "toggle_tool_output",
            "Toggle Tool Output",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+x",
            "open_editor",
            "Open Editor",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+n",
            "open_notifications",
            "Notifications",
            show=False,
            priority=True,
        ),
        # Approval menu keys (handled at App level for reliability)
        Binding("up", "approval_up", "Up", show=False),
        Binding("k", "approval_up", "Up", show=False),
        Binding("down", "approval_down", "Down", show=False),
        Binding("j", "approval_down", "Down", show=False),
        Binding("enter", "approval_select", "Select", show=False),
        Binding("y", "approval_yes", "Yes", show=False),
        Binding("1", "approval_yes", "Yes", show=False),
        Binding("2", "approval_auto", "Auto", show=False),
        Binding("a", "approval_auto", "Auto", show=False),
        Binding("3", "approval_no", "No", show=False),
        Binding("n", "approval_no", "No", show=False),
    ]
    """App-level keybindings for interrupt, quit, toggles, and approval menu
    navigation."""

    class ServerReady(Message):
        """Posted by the background server-startup worker on success."""

        def __init__(  # noqa: D107
            self,
            agent: Any,  # noqa: ANN401
            server_proc: Any,  # noqa: ANN401
            mcp_server_info: list[Any] | None,
        ) -> None:
            super().__init__()
            self.agent = agent
            self.server_proc = server_proc
            self.mcp_server_info = mcp_server_info

    class ServerStartFailed(Message):
        """Posted by the background server-startup worker on failure."""

        def __init__(self, error: Exception) -> None:  # noqa: D107
            super().__init__()
            self.error = error

    def __init__(
        self,
        *,
        agent: Pregel | None = None,
        assistant_id: str | None = None,
        backend: CompositeBackend | None = None,
        auto_approve: bool = False,
        cwd: str | Path | None = None,
        thread_id: str | None = None,
        resume_thread: str | None = None,
        initial_prompt: str | None = None,
        initial_skill: str | None = None,
        startup_cmd: str | None = None,
        mcp_server_info: list[MCPServerInfo] | None = None,
        profile_override: dict[str, Any] | None = None,
        server_proc: ServerProcess | None = None,
        server_kwargs: dict[str, Any] | None = None,
        mcp_preload_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Deep Agents application.

        Args:
            agent: Pre-configured LangGraph agent, or `None` when server
                startup is deferred via `server_kwargs`.
            assistant_id: Agent identifier for memory storage
            backend: Backend for file operations
            auto_approve: Whether to start with auto-approve enabled
            cwd: Current working directory to display
            thread_id: Thread ID for the session.

                `None` when `resume_thread` is provided (resolved asynchronously).
            resume_thread: Raw resume intent from `-r` flag.

                `'__MOST_RECENT__'` for bare `-r`, a thread ID string for
                `-r <id>`, or `None` for new sessions.

                Resolved via `_resolve_resume_thread`
                during `_start_server_background`.

                Requires `server_kwargs` to be set; ignored otherwise.
            initial_prompt: Optional prompt to auto-submit when session starts
            initial_skill: Optional skill name to invoke when session starts.
            startup_cmd: Optional shell command to run at startup before the
                first prompt is accepted.

                Output is rendered in the transcript and non-zero exits warn but
                do not abort the session.
            mcp_server_info: MCP server metadata for the `/mcp` viewer.
            profile_override: Extra profile fields from `--profile-override`,
                retained so later profile-aware behavior stays consistent with
                the CLI override, including model selection details,
                offload budget display, and on-demand `create_model()`
                calls such as `/offload`.
            server_proc: LangGraph server process for the interactive session.
            server_kwargs: When provided, server startup is deferred.

                The app shows a "Connecting..." state and starts the server in
                the background using these kwargs
                for `start_server_and_get_agent`.
            mcp_preload_kwargs: Kwargs for `_preload_session_mcp_server_info`,
                run concurrently with server startup when `server_kwargs` is set.
            model_kwargs: Kwargs for deferred `create_model()`.

                When provided, model creation runs in a background worker after
                first paint instead of blocking startup.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)

        self._register_custom_themes()

        self.theme = _load_theme_preference()
        """Active Textual theme name.

        Loaded from the user's saved preference (or the default) so the app
        boots with consistent colors before `/theme` runs.
        """

        # Injected session config
        self._agent = agent
        """Pre-configured agent (local `Pregel` or `RemoteAgent`).

        `None` when server startup is deferred via `server_kwargs`; filled in
        by the server-ready handler (and by the agent-swap worker when
        `/agents` restarts the subprocess).
        """

        self._assistant_id = assistant_id
        """Current agent identity.

        Scopes per-agent memory (`~/.deepagents/<id>/`) and skill discovery,
        keys `FileOpTracker` file-op history, and is attached to LangSmith
        traces as `assistant_id` / `agent_name`. Mutated by `/agents` swaps
        and by `-r` resume when the resumed thread belongs to a different
        agent.
        """

        self._backend = backend
        """Filesystem/storage backend for agent file operations."""

        self._auto_approve = auto_approve
        """Current auto-approve state for tool calls.

        Initialized from `--auto-approve` and toggled at runtime via
        Ctrl+T / Shift+Tab or the approval menu's 'Auto' option; kept in
        sync with `_session_state.auto_approve`.
        """

        self._cwd = str(cwd) if cwd else str(Path.cwd())
        """Session cwd.

        Shown in the status bar; used as the root for `@` file-mention
        completion in the chat input.
        """

        self._lc_thread_id = thread_id
        """LangChain thread identifier.

        Named `_lc_thread_id` to avoid collision with Textual's `App._thread_id`.
        """

        self._resume_thread_intent = resume_thread
        """Raw `-r` intent (`None`, `'__MOST_RECENT__'`, or a thread id).

        Resolved into a concrete `_lc_thread_id` by `_resolve_resume_thread`
        during background startup.
        """

        self._initial_prompt = initial_prompt
        """Prompt to auto-submit after first paint (from `-m`)."""

        self._initial_skill = (
            initial_skill.strip().lower()
            if initial_skill and initial_skill.strip()
            else None
        )
        """Skill name to auto-invoke after first paint (from `--skill`).

        Normalized to lowercase; `None` when not provided.
        """

        self._startup_cmd = (
            startup_cmd.strip() if startup_cmd and startup_cmd.strip() else None
        )
        """Shell command to run once before the first prompt, from
        `--startup-cmd`.

        Cleared to `None` after it runs so later server swaps cannot re-run it.
        """

        self._mcp_server_info = mcp_server_info
        """MCP server metadata surfaced in the `/mcp` viewer."""

        self._mcp_tool_count = sum(len(s.tools) for s in (mcp_server_info or []))
        """Total tool count across MCP servers, displayed in the status bar."""

        self._profile_override = profile_override
        """Extra profile fields from `--profile-override`, retained so later
        profile-aware behavior (model selection, offload budget display,
        on-demand `create_model()`) stays consistent with the CLI override."""

        self._server_proc = server_proc
        """Handle to the langgraph dev subprocess, when the CLI owns one.

        `None` in remote-server mode (the CLI connects to an external server
        and cannot restart it).
        """

        self._server_kwargs = server_kwargs
        """Cached kwargs for `start_server_and_get_agent`.

        When non-`None`, startup is deferred and the UI begins in
        "Connecting..." state.

        Re-used so downstream features that restart the server (e.g. `/agents`)
        start from the same config.
        """

        self._mcp_preload_kwargs = mcp_preload_kwargs
        """Kwargs for `_preload_session_mcp_server_info`, run concurrently
        with server startup when `server_kwargs` is set."""

        self._model_kwargs = model_kwargs
        """Kwargs for deferred `create_model()`.

        When non-`None`, model creation runs in a background worker after
        first paint; consumed by the startup worker and reset to `None`.
        """

        raw = (server_kwargs or {}).get("sandbox_type")
        """Raw argparse sandbox value from `server_kwargs` before normalization.

        `ServerConfig.__post_init__` maps `"none"` to `None`, but
        `server_kwargs` still carries the argparse string, so `_sandbox_type`
        below guards against both representations.
        """

        self._sandbox_type: str | None = raw if raw and raw != "none" else None
        """Normalized sandbox type (or `None`), attached to trace metadata."""

        # Per-turn model overrides
        self._model_override: str | None = None
        """Per-turn model override set via `/model`; `None` uses session default."""

        self._model_params_override: dict[str, Any] | None = None
        """Per-turn model params override set via `/model --model-params`."""

        # Widget refs (populated in compose/on_mount)
        self._status_bar: StatusBar | None = None
        """Status bar widget; populated in `on_mount`."""

        self._chat_input: ChatInput | None = None
        """Chat input widget; populated in `on_mount`."""

        self._loading_widget: LoadingWidget | None = None
        """Active spinner widget; populated by `_set_spinner(status)` and
        cleared when status resolves to `None`."""

        self._ui_adapter: TextualUIAdapter | None = None
        """Bridge that renders agent events into widgets; set in `on_mount`."""

        self._approval_placeholder: Static | None = None
        """'Waiting for typing to finish...' placeholder mounted in place of
        the approval menu while the user is mid-type, so stray keys (`y`,
        `n`, `1`-`3`) can't trigger approval decisions. Swapped for the real
        `ApprovalMenu` by `_deferred_show_approval` once typing settles."""

        self._pending_approval_widget: ApprovalMenu | None = None
        """Currently-mounted HITL approval widget awaiting a decision."""

        self._pending_ask_user_widget: AskUserMenu | None = None
        """Currently-mounted `ask_user` prompt awaiting an answer."""

        # Agent & shell run state
        self._agent_worker: Worker[None] | None = None
        """Active `_run_agent_task` worker, tracked so it can be cancelled
        on interrupt (`Ctrl+C`) or exit."""

        self._agent_running = False
        """True while the agent worker is streaming a response."""

        self._shell_process: asyncio.subprocess.Process | None = None
        """Shell command process tracking for interruption (! commands)."""

        self._shell_worker: Worker[None] | None = None
        """Active `!` shell-command worker, tracked for interruption."""

        self._shell_running = False
        """True while a `!` shell command is executing."""

        # Lifecycle flags & re-entry guards
        self._connecting = server_kwargs is not None
        """True while the backing server is being started or restarted.

        Gates message handling so user input is queued until the agent is
        actually reachable.
        """

        self._server_startup_error: str | None = None
        """Set when the background server fails to start; persists for the
        session lifetime (server failure is terminal).

        Shown in place of the generic 'Agent not configured' message.
        """

        self._quit_pending = False
        """True after a first `Ctrl+C` so a second press within the window quits."""

        self._thread_switching = False
        """Re-entry guard for `/threads` switches; blocks message handling
        until the new thread's history finishes loading."""

        self._model_switching = False
        """Re-entry guard for `/model` switches while the new model is being
        resolved."""

        self._agent_switching = False
        """Re-entry guard for `/agents` switches while the backing server is
        being restarted with a new `assistant_id`."""

        self._processing_pending = False
        """Re-entry guard for `_process_next_from_queue` so only one drain
        loop runs at a time."""

        self._startup_sequence_running = False
        """True while post-connect startup work is still being sequenced.

        Covers resumed-history hydration, `--startup-cmd`, and the handoff to
        the first queued or initial submission so user input stays serialized
        until the session reaches its first stable busy/idle state.
        """

        # Message queue & store
        self._pending_messages: deque[QueuedMessage] = deque()
        """User message queue for sequential processing."""

        self._queued_widgets: deque[QueuedUserMessage] = deque()
        """Placeholder widgets mounted for messages still sitting in
        `_pending_messages`, removed as the queue drains."""

        self._message_store = MessageStore()
        """Message virtualization store."""

        self._deferred_actions: list[DeferredAction] = []
        """Deferred actions executed after the current busy state resolves."""

        # Session stats & tokens
        self._session_stats: SessionStats = SessionStats()
        """Cumulative usage stats across all turns in this session."""

        self._inflight_turn_stats: SessionStats | None = None
        """Stats for the currently executing turn.

        Held here so `exit()` can merge them synchronously before the event loop
        tears down (e.g. `Ctrl+D` during a pending tool call).
        """

        self._inflight_turn_start: float = 0.0
        """Monotonic timestamp when the current turn started."""

        self._context_tokens: int = 0
        """Local cache of the last total-context token count.

        Source of truth is `_context_tokens` in graph state; this is a sync
        copy for the status bar.
        """

        self._tokens_approximate: bool = False
        """Whether the cached token count is stale (interrupted generation)."""

        # Session lazy state & startup
        self._session_state: TextualSessionState | None = None
        """Auto-approve + thread state shared with `execute_task_textual`.

        Lazily constructed by the session-init worker so we don't block
        startup on it.
        """

        self._startup_task: asyncio.Task[None] | None = None
        """Startup task reference (set in on_mount)."""

        self._git_branch_refresh_task: asyncio.Task[None] | None = None
        """Latest background git-branch refresh task, if one is running."""

        self._last_typed_at: float | None = None
        """Typing-aware approval deferral state."""

        self._update_available: tuple[bool, str | None] = (False, None)
        """Update availability state.

        Set by `_check_for_updates` when PyPI reports a newer version;
        read at shutdown (for the exit banner), by `_handle_version_command`
        (for the `/version` update hint), and by downstream callers. Does
        *not* drive missing-dep toast suppression — that's gated on
        `_update_modal_pending`.
        """

        self._update_check_done = asyncio.Event()
        """Set by `_check_for_updates` when it returns (success, failure, or
        no-op). Lets `_check_optional_tools_background` defer posting
        missing-dep toasts until we know whether the update modal is about
        to clear them."""

        self._update_modal_pending = asyncio.Event()
        """Set only immediately before the update modal is scheduled.

        Used by `_check_optional_tools_background` to decide whether to
        suppress missing-dep toasts: we only suppress when a modal is
        actually about to open, not merely when an update was detected.
        A detected-but-throttled update (already notified within
        `CACHE_TTL`) leaves this clear so missing-dep toasts still fire.
        """

        # Skills cache
        self._discovered_skills: list[ExtendedSkillMetadata] = []
        """Cached skill metadata (populated by startup discovery worker,
        refreshed on `/reload`).

        Used by `_invoke_skill` to skip re-walking all skill directories on
        every invocation.
        """

        self._skill_allowed_roots: list[Path] = []
        """Pre-resolved skill root directories for containment checks in
        `load_skill_content`.

        Built alongside `_discovered_skills`.
        """

        # Media
        # Lazily imported here to avoid pulling image dependencies into
        # argument parsing paths.
        from deepagents_cli.input import MediaTracker

        self._image_tracker = MediaTracker()
        """Tracks image/media pastes in the chat input so they can be
        attached to outgoing messages and cleared after submission."""

        self._notice_registry = NotificationRegistry()
        """Pending actionable notifications.

        Startup workers register notices (missing deps, update available)
        here; the user opens them via toast click or `ctrl+n`.
        """

    def _remote_agent(self) -> RemoteAgent | None:
        """Return the agent narrowed to `RemoteAgent`, or `None`.

        Returns `None` when:

        - No agent is configured (`self._agent is None`).
        - The agent is a local `Pregel` graph (e.g. ACP mode, test harnesses).

        Used to gate features that require a server-backed agent (e.g. model
        switching via `ConfigurableModelMiddleware`, checkpointer fallback).
        Checks the agent type rather than server ownership so this works for
        both CLI-spawned servers and externally managed ones.

        Returns:
            The `RemoteAgent` instance, or `None` for local agents.
        """
        from deepagents_cli.remote_client import RemoteAgent

        return self._agent if isinstance(self._agent, RemoteAgent) else None

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Return custom CSS variable defaults for the current theme.

        Most styling uses Textual's built-in variables (`$primary`,
        `$text-muted`, `$error-muted`, etc.).  This override injects the
        app-specific variables (`$mode-bash`, `$mode-command`, `$skill`,
        `$skill-hover`, `$tool`, `$tool-hover`) that have no Textual equivalent.

        Returns:
            Dict of CSS variable names to hex color values.
        """
        colors = theme.get_theme_colors(self)
        return theme.get_css_variable_defaults(colors=colors)

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        Yields:
            UI components for the main chat area and status bar.
        """
        # Main chat area with scrollable messages
        # VerticalScroll tracks user scroll intent for better auto-scroll behavior
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(
                thread_id=self._lc_thread_id,
                mcp_tool_count=self._mcp_tool_count,
                connecting=self._connecting,
                resuming=self._resume_thread_intent is not None,
                local_server=self._server_kwargs is not None,
                id="welcome-banner",
            )
            yield Container(id="messages")
        with Container(id="bottom-app-container"):
            yield ChatInput(
                cwd=self._cwd,
                image_tracker=self._image_tracker,
                id="input-area",
            )

        # Status bar at bottom
        yield StatusBar(cwd=self._cwd, id="status-bar")

    async def on_mount(self) -> None:
        """Initialize components after mount.

        Only widget queries and lightweight config go here — anything that
        would delay the first rendered frame (subprocess calls, heavy
        imports) is deferred to `_post_paint_init` via `call_after_refresh`.
        """
        # Move all objects allocated during import/compose into the permanent
        # generation so the cyclic GC skips them during first-paint rendering.
        import gc

        gc.freeze()

        chat = self.query_one("#chat", VerticalScroll)
        chat.anchor()
        if is_ascii_mode():
            chat.styles.scrollbar_size_vertical = 0

        self._status_bar = self.query_one("#status-bar", StatusBar)
        self._chat_input = self.query_one("#input-area", ChatInput)

        # Apply any skill commands discovered before the widget was mounted
        if self._discovered_skills:
            from deepagents_cli.command_registry import (
                SLASH_COMMANDS,
                build_skill_commands,
            )

            cmds = build_skill_commands(self._discovered_skills)
            merged = list(SLASH_COMMANDS) + cmds
            self._chat_input.update_slash_commands(merged)

        # Set initial auto-approve state
        if self._auto_approve:
            self._status_bar.set_auto_approve(enabled=True)

        # Focus the input immediately so the cursor is visible on first paint
        self._chat_input.focus_input()

        # Prewarm heavy imports in a thread while the first frame renders.
        # The user can't type yet, so GIL contention is harmless.  By the
        # time _post_paint_init fires its inline imports are dict lookups.
        self.run_worker(
            asyncio.to_thread(self._prewarm_deferred_imports),
            exclusive=True,
            group="startup-import-prewarm",
        )

        # Start branch resolution immediately — the thread launches now
        # (during on_mount) so by the time the first frame finishes painting
        # the filesystem probe is already done. _post_paint_init fires the
        # heavier workers (server, model creation) afterward.
        self._startup_task = asyncio.create_task(
            self._resolve_git_branch_and_continue()
        )

    async def _refresh_git_branch(self) -> None:
        """Resolve the current git branch and update the status bar.

        Reads repository metadata from `self._cwd` inline so the common path is
        just local file I/O. Falls back to a thread-offloaded `git rev-parse`
        only for unusual repository layouts. Swallows all errors — the status
        bar simply stays empty (or keeps its prior value on unexpected failure)
        if git is unavailable.
        """
        try:
            cwd = self._cwd
            branch = read_git_branch_from_filesystem(cwd)
            if branch is None:
                branch = await asyncio.to_thread(read_git_branch_via_subprocess, cwd)
            if self._status_bar:
                self._status_bar.branch = branch
        except Exception:
            logger.warning("Git branch resolution failed", exc_info=True)

    async def _refresh_git_branch_subprocess_fallback(self, cwd: str) -> None:
        """Run the `git rev-parse` fallback off-thread for unusual repo layouts."""
        try:
            branch = await asyncio.to_thread(read_git_branch_via_subprocess, cwd)
        except Exception:
            logger.warning("Git branch subprocess fallback failed", exc_info=True)
            return
        if self._status_bar:
            self._status_bar.branch = branch

    def _cancel_git_branch_refresh_task(self) -> None:
        """Cancel and clear any in-flight background branch refresh task."""
        prior_task = self._git_branch_refresh_task
        if prior_task is not None and not prior_task.done():
            prior_task.cancel()
        self._git_branch_refresh_task = None

    def _schedule_git_branch_refresh(self) -> None:
        """Refresh the git branch, inline when possible.

        The filesystem probe is sub-millisecond for the common repo layout, so
        we run it synchronously and only spawn a background task for the
        `git rev-parse` fallback. Keeping the hot path inline avoids an
        event-loop tick plus a reactive watcher hop between a tool exiting and
        the footer updating.
        """
        if self._exit:
            return

        cwd = self._cwd
        try:
            branch = read_git_branch_from_filesystem(cwd)
        except Exception:
            logger.warning("Git branch filesystem probe failed", exc_info=True)
            return

        if branch is not None:
            if self._status_bar:
                self._status_bar.branch = branch
            self._cancel_git_branch_refresh_task()
            return

        # Unusual repo layout — hop to a thread for `git rev-parse`.
        self._cancel_git_branch_refresh_task()
        refresh_task = asyncio.create_task(
            self._refresh_git_branch_subprocess_fallback(cwd)
        )
        self._git_branch_refresh_task = refresh_task

        def _finalize_git_branch_refresh(task: asyncio.Task[None]) -> None:
            if self._git_branch_refresh_task is task:
                self._git_branch_refresh_task = None
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "Background git branch refresh failed unexpectedly",
                    exc_info=True,
                )

        refresh_task.add_done_callback(_finalize_git_branch_refresh)

    async def _resolve_git_branch_and_continue(self) -> None:
        """Resolve git branch, then schedule remaining init workers.

        Launched via `asyncio.create_task()` during `on_mount` so branch
        detection runs concurrently with first-paint rendering.
        `_post_paint_init` is scheduled via `call_after_refresh` regardless
        of whether branch resolution succeeds.
        """
        try:
            await self._refresh_git_branch()
        finally:
            # Always schedule post-paint init — even if branch resolution
            # fails, the app must still start the server, session, etc.
            self.call_after_refresh(self._post_paint_init)

    async def _post_paint_init(self) -> None:
        """Fire background workers for remaining startup work.

        Everything here is non-blocking: workers and thread-offloaded calls
        so the UI stays responsive.
        """
        # Create UI adapter unconditionally — it only holds UI callbacks and
        # doesn't depend on the agent. The agent is injected later at
        # execute_task_textual() call time.
        from deepagents_cli.textual_adapter import TextualUIAdapter

        self._ui_adapter = TextualUIAdapter(
            mount_message=self._mount_message,
            update_status=self._update_status,
            request_approval=self._request_approval,
            on_auto_approve_enabled=self._on_auto_approve_enabled,
            set_spinner=self._set_spinner,
            set_active_message=self._set_active_message,
            sync_message_content=self._sync_message_content,
            request_ask_user=self._request_ask_user,
            on_tool_complete=self._schedule_git_branch_refresh,
        )
        # Wire token display callbacks
        self._ui_adapter._on_tokens_update = self._on_tokens_update
        self._ui_adapter._on_tokens_hide = self._hide_tokens
        self._ui_adapter._on_tokens_show = self._show_tokens

        # Fire-and-forget workers — none of these block the event loop.

        # Discover skills first so /skill: autocomplete is ready as early
        # as possible. The heavy filesystem scan runs in a thread.
        self.run_worker(
            self._discover_skills(),
            exclusive=True,
            group="startup-skill-discovery",
        )

        self.run_worker(self._init_session_state, exclusive=True, group="session-init")

        # Server startup (model creation + server process)
        if self._server_kwargs is not None:
            self.run_worker(
                self._start_server_background,
                exclusive=True,
                group="server-startup",
            )

        # Background update check and what's-new banner
        # (opt-out via env var or config.toml [update].check)
        from deepagents_cli.update_check import is_update_check_enabled

        if is_update_check_enabled():
            self.run_worker(
                self._check_for_updates,
                exclusive=True,
                group="startup-update-check",
            )
            self.run_worker(
                self._show_whats_new,
                exclusive=True,
                group="startup-whats-new",
            )

        # Prewarm model discovery and profile caches unconditionally so
        # /model opens instantly even before the agent/server is ready.
        self.run_worker(
            self._prewarm_model_caches,
            exclusive=True,
            group="startup-model-prewarm",
        )

        # Prewarm thread message counts so /threads opens instantly.
        self.run_worker(
            self._prewarm_threads_cache,
            exclusive=True,
            group="startup-thread-prewarm",
        )

        # Optional tool warnings in a thread (shutil.which is sync I/O)
        self.run_worker(
            self._check_optional_tools_background,
            exclusive=True,
            group="startup-tool-check",
        )

        # Debug helper: inject sample notifications so the center UI can
        # be exercised without waiting for real conditions.
        from deepagents_cli._env_vars import DEBUG_NOTIFICATIONS

        if os.environ.get(DEBUG_NOTIFICATIONS):
            self.call_after_refresh(self._inject_debug_notifications)

        # Session-start sequence (history -> `--startup-cmd` -> initial prompt/
        # skill -> queue drain). When connecting, defer until
        # `on_deep_agents_app_server_ready` fires; otherwise run it now so the
        # non-connecting path (pre-built agent) also honors `--startup-cmd` and
        # serializes startup against user input.
        if not self._connecting:
            self.call_after_refresh(
                lambda: asyncio.create_task(self._run_session_start_sequence())
            )

    async def _init_session_state(self) -> None:
        """Create session state in a thread (imports deepagents_cli.sessions)."""

        def _create() -> TextualSessionState:
            return TextualSessionState(
                auto_approve=self._auto_approve,
                thread_id=self._lc_thread_id,
            )

        try:
            self._session_state = await asyncio.to_thread(_create)
        except Exception:
            logger.exception("Failed to create session state")
            self.notify(
                "Session initialization failed. Some features may be unavailable.",
                severity="error",
                timeout=10,
            )

    async def _check_optional_tools_background(self) -> None:
        """Check for optional tools and register actionable notices.

        Missing tools are added to the notifications registry. Toasts
        are posted only if no update modal is actually about to open;
        otherwise the modal's `clear_notifications` call would
        immediately drop them and cause visible flicker. Entries remain
        reachable via ctrl+n either way.
        """
        try:
            from deepagents_cli.main import (
                build_missing_tool_notification,
                check_optional_tools,
            )
            from deepagents_cli.update_check import is_update_check_enabled
        except ImportError:
            logger.warning(
                "Could not import optional tools checker",
                exc_info=True,
            )
            return

        try:
            missing = await asyncio.to_thread(check_optional_tools)
        except OSError:
            logger.debug("Failed to check for optional tools", exc_info=True)
            return
        except Exception:
            # Defensive: surface regressions (e.g. future refactors of
            # check_optional_tools raising an unexpected exception type)
            # instead of silently returning.
            logger.warning("Optional-tools check failed unexpectedly", exc_info=True)
            self.notify(
                "Could not check optional tools — see logs.",
                severity="warning",
                timeout=6,
                markup=False,
            )
            return

        if not missing:
            return

        # Wait for the update check so we know whether the update
        # modal is about to clear any toasts we post. Bounded by a
        # short timeout to avoid blocking indefinitely if PyPI hangs.
        if is_update_check_enabled():
            try:
                await asyncio.wait_for(self._update_check_done.wait(), timeout=5.0)
            except TimeoutError:
                logger.debug("Update check timed out; posting tool toasts anyway")

        # Suppress only when a modal is actually going to open — not
        # just when an update was detected. A detected-but-throttled
        # update (already notified within CACHE_TTL) does not open the
        # modal, so toasts must still fire or returning users never
        # see the warning.
        suppress_toasts = self._update_modal_pending.is_set()

        for tool in missing:
            notification = build_missing_tool_notification(tool)
            if suppress_toasts:
                # Register silently; the update modal's dismissal
                # leaves these reachable via ctrl+n (notification center).
                self._notice_registry.add(notification)
            else:
                self._notify_actionable(
                    notification,
                    severity="warning",
                    timeout=15,
                )

    async def _discover_skills(self) -> None:
        """Discover skills, cache metadata, and update autocomplete.

        Caches the full `ExtendedSkillMetadata` list and pre-resolved
        containment roots so that `/skill:<name>` invocations can skip
        re-walking every skill directory.

        Runs filesystem I/O in a thread to avoid blocking the event loop.
        """
        from deepagents_cli.command_registry import SLASH_COMMANDS, build_skill_commands

        try:
            skills, roots = await asyncio.to_thread(self._discover_skills_and_roots)
            self._discovered_skills = skills
            self._skill_allowed_roots = roots
            if skills:
                skill_commands = build_skill_commands(skills)
                if self._chat_input:
                    merged = list(SLASH_COMMANDS) + skill_commands
                    self._chat_input.update_slash_commands(merged)
                else:
                    logger.debug(
                        "Skill discovery completed (%d skills) but chat input "
                        "not yet mounted; autocomplete deferred",
                        len(skills),
                    )
        except OSError:
            # Clear stale cache so /reload failures don't silently
            # leave old data in place.
            self._discovered_skills = []
            self._skill_allowed_roots = []
            logger.warning(
                "Filesystem error during skill discovery",
                exc_info=True,
            )
            self.notify(
                "Could not scan skill directories. "
                "Some /skill: commands may be unavailable.",
                severity="warning",
                timeout=6,
                markup=False,
            )
        except Exception:
            self._discovered_skills = []
            self._skill_allowed_roots = []
            logger.exception("Unexpected error during skill discovery")
            self.notify(
                "Skill discovery failed unexpectedly. "
                "/skill: commands may not work. Check logs for details.",
                severity="warning",
                timeout=8,
                markup=False,
            )

    def _discover_skills_and_roots(
        self,
    ) -> tuple[list[ExtendedSkillMetadata], list[Path]]:
        """Discover skills and build pre-resolved containment roots.

        Shared by `_discover_skills` (startup/reload) and the cache-miss
        fallback in `_invoke_skill` to avoid duplicating the
        `list_skills` call and root-resolution logic.

        Returns:
            Tuple of `(skill metadata list, pre-resolved containment roots)`.
        """
        from deepagents_cli.skills.invocation import discover_skills_and_roots

        assistant_id = self._assistant_id or "agent"
        return discover_skills_and_roots(assistant_id)

    async def _resolve_resume_thread(self) -> None:
        """Resolve a `-r` resume intent into a concrete thread ID.

        Consumes `self._resume_thread_intent` and resolves it into a concrete
        thread ID. Mutates `self._lc_thread_id` and optionally
        `self._assistant_id` / `self._server_kwargs`. Falls back to a fresh
        thread on any DB error.
        """
        from deepagents_cli.sessions import (
            find_similar_threads,
            generate_thread_id,
            get_most_recent,
            get_thread_agent,
            thread_exists,
        )

        resume = self._resume_thread_intent
        self._resume_thread_intent = None  # consumed

        if not resume:
            return

        # Matches _DEFAULT_AGENT_NAME in main.py. Do NOT import it — main.py is
        # the CLI entry point and pulls in argparse, rich, etc. at module level.
        # Even a deferred import drags in the full dep tree for a single
        # string constant.
        default_agent = "agent"

        try:
            if resume == "__MOST_RECENT__":
                agent_filter = (
                    self._assistant_id if self._assistant_id != default_agent else None
                )
                thread_id = await get_most_recent(agent_filter)
                if thread_id:
                    agent_name = await get_thread_agent(thread_id)
                    if agent_name:
                        self._assistant_id = agent_name
                        if self._server_kwargs:
                            self._server_kwargs["assistant_id"] = agent_name
                    self._lc_thread_id = thread_id
                else:
                    self._lc_thread_id = generate_thread_id()
                    if agent_filter:
                        msg = f"No previous threads for '{agent_filter}', starting new."
                    else:
                        msg = "No previous threads, starting new."
                    self.notify(msg, severity="warning", markup=False)
            elif await thread_exists(resume):
                self._lc_thread_id = resume
                if self._assistant_id == default_agent:
                    agent_name = await get_thread_agent(resume)
                    if agent_name:
                        self._assistant_id = agent_name
                        if self._server_kwargs:
                            self._server_kwargs["assistant_id"] = agent_name
            else:
                # Thread not found — notify + fall back to new thread
                self._lc_thread_id = generate_thread_id()
                similar = await find_similar_threads(resume)
                hint = f"Thread '{resume}' not found."
                if similar:
                    hint += f" Did you mean: {', '.join(str(t) for t in similar)}?"
                self.notify(hint, severity="warning", timeout=6, markup=False)
        except Exception:
            logger.exception("Failed to resolve resume thread %r", resume)
            self._lc_thread_id = generate_thread_id()
            self.notify(
                "Could not look up thread history. Starting new session.",
                severity="warning",
            )

        # Update session state if ready (may still be initializing in a
        # concurrent worker)
        if self._session_state:
            self._session_state.thread_id = self._lc_thread_id

    async def _start_server_background(self) -> None:
        """Background worker: resolve resume-thread intent, start server + MCP preload.

        Also runs deferred model creation if `model_kwargs` was provided,
        so the langchain import + init doesn't block first paint.
        """
        # Phase 1: Resolve resume thread (if any) before server startup
        if self._resume_thread_intent:
            await self._resolve_resume_thread()

        # Run deferred model creation. settings.model_name / model_provider
        # are already set eagerly for the status bar display; this call
        # does the heavy langchain import + SDK init and may refine them
        # (e.g., context_limit from the model profile).
        if self._model_kwargs is not None:
            from deepagents_cli.config import create_model
            from deepagents_cli.model_config import ModelConfigError, save_recent_model

            try:
                result = create_model(**self._model_kwargs)
            except ModelConfigError as exc:
                self.post_message(self.ServerStartFailed(error=exc))
                return
            result.apply_to_settings()
            save_recent_model(f"{result.provider}:{result.model_name}")
            self._model_kwargs = None  # consumed

        # Persist the agent in use so a later bare `deepagents` relaunch
        # brings the user back to it (same pattern as `save_recent_model`).
        if self._assistant_id:
            from deepagents_cli.model_config import save_recent_agent

            saved = await asyncio.to_thread(save_recent_agent, self._assistant_id)
            if not saved:
                logger.warning(
                    "Could not persist recent agent %r to config at startup",
                    self._assistant_id,
                )

        from deepagents_cli.server_manager import start_server_and_get_agent

        coros: list[Any] = [start_server_and_get_agent(**self._server_kwargs)]  # type: ignore[arg-type]

        if self._mcp_preload_kwargs is not None:
            from deepagents_cli.main import _preload_session_mcp_server_info

            coros.append(_preload_session_mcp_server_info(**self._mcp_preload_kwargs))

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001  # defensive catch around gather
            self.post_message(self.ServerStartFailed(error=exc))
            return

        server_result = results[0]
        if isinstance(server_result, BaseException):
            self.post_message(
                self.ServerStartFailed(
                    error=server_result
                    if isinstance(server_result, Exception)
                    else RuntimeError(str(server_result)),
                )
            )
            return

        agent, server_proc, _ = server_result

        # Assign immediately so the finally block in run_textual_app can
        # clean up the server even if the ServerReady message is never
        # processed (e.g. user quits during startup).
        self._server_proc = server_proc

        mcp_info = None
        if len(results) > 1 and not isinstance(results[1], BaseException):
            mcp_info = results[1]
        elif len(results) > 1 and isinstance(results[1], BaseException):
            logger.warning(
                "MCP metadata preload failed: %s",
                results[1],
                exc_info=results[1],
            )

        self.post_message(
            self.ServerReady(
                agent=agent,
                server_proc=server_proc,
                mcp_server_info=mcp_info,
            )
        )

    def on_deep_agents_app_server_ready(self, event: ServerReady) -> None:
        """Handle successful background server startup."""
        self._connecting = False
        self._agent = event.agent
        self._server_proc = event.server_proc
        self._mcp_server_info = event.mcp_server_info
        self._mcp_tool_count = sum(len(s.tools) for s in (event.mcp_server_info or []))

        # Update welcome banner to show ready state
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.set_connected(self._mcp_tool_count)
        except NoMatches:
            logger.warning("Welcome banner not found during server ready transition")

        # Session-start sequence: load resumed history, run `--startup-cmd`
        # (if any), then dispatch the initial prompt/skill and drain
        # user-typed messages. Sequenced through a single task so the
        # startup command always resolves before the agent sees any user
        # input.
        self.call_after_refresh(
            lambda: asyncio.create_task(self._run_session_start_sequence())
        )

        # Drain deferred actions (e.g. model/thread switch queued during connection)
        # if the agent is not actively running. Wrapped in a helper so that
        # exceptions are logged rather than becoming unhandled task errors.
        if self._deferred_actions and not self._agent_running:

            async def _safe_drain() -> None:
                try:
                    await self._maybe_drain_deferred()
                except Exception:
                    logger.exception("Unhandled error while draining deferred actions")
                    with suppress(Exception):
                        await self._mount_message(
                            ErrorMessage(
                                "A deferred action failed during startup. "
                                "You may need to retry the operation."
                            )
                        )

            self.call_after_refresh(lambda: asyncio.create_task(_safe_drain()))

    def on_deep_agents_app_server_start_failed(self, event: ServerStartFailed) -> None:
        """Handle background server startup failure."""
        self._connecting = False
        self._server_startup_error = f"{type(event.error).__name__}: {event.error}"
        logger.error("Server startup failed: %s", event.error, exc_info=event.error)
        # Update banner to show persistent failure state
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.set_failed(self._server_startup_error)
        except NoMatches:
            logger.warning("Welcome banner not found during server failure transition")

        # Discard any messages queued while the server was starting
        if self._pending_messages:
            self._pending_messages.clear()
            for w in self._queued_widgets:
                w.remove()
            self._queued_widgets.clear()
        self._deferred_actions.clear()

    @staticmethod
    def _prewarm_deferred_imports() -> None:
        """Background-load modules deferred from the startup path.

        Populates `sys.modules` so the first user-triggered inline import
        is a cheap dict lookup instead of a cold module load.
        """
        # Internal modules moved from top-level to local imports — a failure
        # here indicates a packaging or code bug, not a missing optional dep, so
        # we let the exception propagate (the worker catches it and logs
        # at WARNING). textual_adapter and update_check are included so
        # _post_paint_init's inline imports are dict lookups.
        from deepagents_cli.clipboard import (
            copy_selection_to_clipboard,  # noqa: F401
        )
        from deepagents_cli.command_registry import ALWAYS_IMMEDIATE  # noqa: F401
        from deepagents_cli.config import settings  # noqa: F401
        from deepagents_cli.hooks import dispatch_hook  # noqa: F401
        from deepagents_cli.model_config import ModelSpec  # noqa: F401
        from deepagents_cli.textual_adapter import TextualUIAdapter  # noqa: F401
        from deepagents_cli.update_check import is_update_check_enabled  # noqa: F401

        try:
            # Heavy third-party deps deferred from textual_adapter /
            # tool_display — hit on first message send and first tool
            # approval. Best-effort: missing optional deps should not block the
            # TUI from rendering.
            from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT  # noqa: F401
            from langchain.agents.middleware.human_in_the_loop import (  # noqa: F401
                ApproveDecision,
            )
            from langchain_core.messages import AIMessage  # noqa: F401
            from langgraph.types import Command  # noqa: F401
        except Exception:
            logger.warning("Could not prewarm third-party imports", exc_info=True)

        # Markdown rendering stack — ~170 ms cold (textual._markdown pulls in
        # markdown_it, pygments, linkify_it — 438 modules).  Hit on first
        # SkillMessage compose() and first code-fence highlight.  Warming
        # here makes the first expand/Ctrl+O instant.
        import markdown_it  # noqa: F401
        from pygments.lexers import get_lexer_by_name as _get_lexer
        from textual.widgets import Markdown  # noqa: F401

        # Instantiate the Python lexer to populate Pygments' internal
        # lexer cache (~12 ms cold).  Python is the most common fence
        # language in skill bodies.
        _get_lexer("python")

        # Widgets deferred from app.py module level — a failure here indicates
        # a packaging or code bug (same as the block above), so we let
        # exceptions propagate.
        from deepagents_cli.widgets.approval import ApprovalMenu  # noqa: F401
        from deepagents_cli.widgets.ask_user import AskUserMenu  # noqa: F401
        from deepagents_cli.widgets.model_selector import (
            ModelSelectorScreen,  # noqa: F401
        )
        from deepagents_cli.widgets.thread_selector import (  # noqa: F401
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

    async def _prewarm_threads_cache(self) -> None:  # noqa: PLR6301  # Worker hook kept as instance method
        """Prewarm thread selector cache without blocking app startup."""
        from deepagents_cli.sessions import (
            get_thread_limit,
            prewarm_thread_message_counts,
        )

        await prewarm_thread_message_counts(limit=get_thread_limit())

    async def _prewarm_model_caches(self) -> None:
        """Prewarm model discovery and profile caches without blocking startup."""
        try:
            from deepagents_cli.model_config import (
                get_available_models,
                get_model_profiles,
            )

            await asyncio.to_thread(get_available_models)
            await asyncio.to_thread(
                get_model_profiles, cli_override=self._profile_override
            )
        except Exception:
            logger.warning("Could not prewarm model caches", exc_info=True)

    async def _check_for_updates(self) -> None:
        """Run the update check and signal completion for downstream waiters.

        Wraps `_check_for_updates_impl` so `_update_check_done.set()`
        always fires — lets `_check_optional_tools_background` unblock
        after the PyPI round-trip regardless of success, failure, or no-op.
        """
        try:
            await self._check_for_updates_impl()
        finally:
            # Always signal completion — the optional-tools worker
            # waits on this before deciding whether to post toasts.
            self._update_check_done.set()

    async def _check_for_updates_impl(self) -> None:
        """Check PyPI for a newer version and either auto-update or queue a modal.

        Phase 1 contacts PyPI and records the latest version on the app.
        Phase 2 either performs the auto-upgrade (when enabled), or
        registers the actionable notice and schedules the update modal.
        Phase 2 sets `_update_modal_pending` *only* when the modal is
        actually being scheduled; a detected-but-throttled update
        leaves the event clear so missing-dep toasts still fire.
        """
        # Phase 1: version check (benign failure)
        try:
            from deepagents_cli.update_check import (
                is_auto_update_enabled,
                is_update_available,
                upgrade_command,
            )

            available, latest = await asyncio.to_thread(is_update_available)
            if not available or latest is None:
                return

            self._update_available = (True, latest)
        except Exception:
            logger.debug("Background update check failed", exc_info=True)
            return

        # Phase 2: auto-update or register actionable notice
        try:
            from deepagents_cli._version import __version__ as cli_version

            if is_auto_update_enabled():
                from deepagents_cli.update_check import perform_upgrade

                self.notify(
                    f"Updating to v{latest}...",
                    severity="information",
                    timeout=5,
                )
                success, output = await perform_upgrade()
                if success:
                    self.notify(
                        f"Updated to v{latest}. Restart to use the new version.",
                        severity="information",
                        timeout=10,
                    )
                else:
                    logger.warning(
                        "Background auto-upgrade to v%s failed. Output:\n%s",
                        latest,
                        output,
                    )
                    cmd = upgrade_command()
                    snippet = _truncate(output, limit=160) if output else ""
                    message = f"Auto-update failed. Run manually: {cmd}"
                    if snippet:
                        message = f"{message}\n{snippet}"
                    self.notify(
                        message,
                        severity="warning",
                        timeout=15,
                        markup=False,
                    )
            else:
                from deepagents_cli.update_check import (
                    format_age_suffix,
                    mark_update_notified,
                    should_notify_update,
                )

                if not await asyncio.to_thread(should_notify_update, latest):
                    return

                cmd = upgrade_command()
                age_suffix = await asyncio.to_thread(format_age_suffix, latest)
                notification = self._build_update_notification(
                    latest=latest,
                    cli_version=cli_version,
                    age_suffix=age_suffix,
                    upgrade_cmd=cmd,
                )
                # Register without a toast: the dedicated modal is
                # the update's UI, so a parallel toast would be
                # redundant. Registration still makes the entry
                # reachable via ctrl+n if the modal is dismissed.
                self._notice_registry.add(notification)
                await asyncio.to_thread(mark_update_notified, latest)
                # Set *before* scheduling the modal: the optional-tools
                # worker may race with this path, and it gates toast
                # suppression on this event.
                self._update_modal_pending.set()
                self.call_after_refresh(self._open_update_available_modal, notification)
        except Exception:
            logger.warning("Update check/notify failed unexpectedly", exc_info=True)
            if is_auto_update_enabled():
                self.notify(
                    "Auto-update failed unexpectedly.",
                    severity="warning",
                    timeout=10,
                )

    @staticmethod
    def _build_update_notification(
        *,
        latest: str,
        cli_version: str,
        age_suffix: str,
        upgrade_cmd: str,
    ) -> PendingNotification:
        """Build the update-available registry entry.

        Args:
            latest: New version advertised by PyPI.
            cli_version: Currently installed version string.
            age_suffix: Pre-formatted "(released N days ago)" fragment.
            upgrade_cmd: Shell command to install the update.

        Returns:
            Registry entry ready to pass to `_notify_actionable`.
        """
        body = f"v{latest} is available (current: v{cli_version}{age_suffix})."
        return PendingNotification(
            key="update:available",
            title=f"Update available: v{latest}",
            body=body,
            actions=(
                NotificationAction(ActionId.INSTALL, "Install now", primary=True),
                NotificationAction(ActionId.SKIP_ONCE, "Remind me next launch"),
                NotificationAction(ActionId.SKIP_VERSION, "Skip this version"),
            ),
            payload=UpdateAvailablePayload(latest=latest, upgrade_cmd=upgrade_cmd),
        )

    async def _show_whats_new(self) -> None:
        """Show a 'what's new' banner on the first launch after an upgrade."""
        try:
            from deepagents_cli.update_check import should_show_whats_new

            if not await asyncio.to_thread(should_show_whats_new):
                return
        except Exception:
            logger.debug("What's new check failed", exc_info=True)
            return

        try:
            from deepagents_cli._version import __version__ as cli_version
            from deepagents_cli.config import _is_editable_install

            if await asyncio.to_thread(_is_editable_install):
                heading = f"Now running v{cli_version}"
            else:
                heading = f"Updated to v{cli_version}"

            await self._mount_message(
                AppMessage(f"{heading}\nSee what's new: {CHANGELOG_URL}")
            )
        except Exception:
            logger.debug("What's new banner display failed", exc_info=True)
            return

        try:
            from deepagents_cli._version import __version__ as cli_version
            from deepagents_cli.update_check import mark_version_seen

            await asyncio.to_thread(mark_version_seen, cli_version)
        except Exception:
            logger.warning("Failed to persist seen-version marker", exc_info=True)

    async def _handle_update_command(self) -> None:
        """Handle the `/update` slash command — check for and install updates."""
        await self._mount_message(UserMessage("/update"))
        try:
            from deepagents_cli._version import __version__ as cli_version
            from deepagents_cli.config import _is_editable_install
            from deepagents_cli.update_check import (
                format_age_suffix,
                is_update_available,
                perform_upgrade,
                upgrade_command,
            )

            if await asyncio.to_thread(_is_editable_install):
                age_suffix = await asyncio.to_thread(format_age_suffix, cli_version)
                await self._mount_message(
                    AppMessage(
                        "Updates are not available for editable installs. "
                        f"Currently on v{cli_version}{age_suffix}."
                    )
                )
                return

            await self._mount_message(AppMessage("Checking for updates..."))
            available, latest = await asyncio.to_thread(
                is_update_available, bypass_cache=True
            )
            if latest is None:
                await self._mount_message(
                    AppMessage(
                        "Could not determine the latest version. "
                        "Check your network and try again."
                    )
                )
                return
            if not available:
                age_suffix = await asyncio.to_thread(format_age_suffix, cli_version)
                await self._mount_message(
                    AppMessage(
                        f"Already on the latest version (v{cli_version}{age_suffix})."
                    )
                )
                return

            age_suffix = await asyncio.to_thread(format_age_suffix, latest)
            await self._mount_message(
                AppMessage(
                    f"Update available: v{latest} "
                    f"(current: v{cli_version}{age_suffix}). Upgrading..."
                )
            )
            success, output = await perform_upgrade()
            if success:
                self._update_available = (False, None)
                await self._mount_message(
                    AppMessage(f"Updated to v{latest}. Restart to use the new version.")
                )
            else:
                cmd = upgrade_command()
                detail = f": {output[:200]}" if output else ""
                await self._mount_message(
                    AppMessage(f"Auto-update failed{detail}\nRun manually: {cmd}")
                )
        except Exception as exc:
            logger.warning("/update command failed", exc_info=True)
            await self._mount_message(
                ErrorMessage(f"Update failed: {type(exc).__name__}: {exc}")
            )

    async def _handle_version_command(self) -> None:
        """Handle the `/version` slash command — show versions and update status.

        The CLI release age is served from the cache populated by the
        background update check. The SDK release age is served from its own
        cache; on the first call for a given SDK version (or on a cache
        miss) this triggers a one-off PyPI fetch bounded by a 3s timeout,
        then persists the result so subsequent calls stay local. The
        update-available hint reads `self._update_available`, which
        reflects the last completed background check.
        """
        from importlib.metadata import (
            PackageNotFoundError,
            version as _pkg_version,
        )

        lines: list[str] = []
        try:
            from deepagents_cli._version import __version__ as cli_version
            from deepagents_cli.update_check import format_age_suffix

            age_suffix = await asyncio.to_thread(format_age_suffix, cli_version)
            lines.append(f"deepagents-cli version: {cli_version}{age_suffix}")
        except ImportError:
            logger.debug("deepagents_cli._version module not found")
            lines.append("deepagents-cli version: unknown")
        except Exception:
            logger.warning("Unexpected error looking up CLI version", exc_info=True)
            lines.append("deepagents-cli version: unknown")

        try:
            from deepagents_cli.update_check import format_sdk_age_suffix

            sdk_version = _pkg_version("deepagents")
            sdk_age_suffix = await asyncio.to_thread(format_sdk_age_suffix, sdk_version)
            lines.append(f"deepagents (SDK) version: {sdk_version}{sdk_age_suffix}")
        except PackageNotFoundError:
            logger.debug("deepagents SDK package not found in environment")
            lines.append("deepagents (SDK) version: unknown")
        except Exception:
            logger.warning("Unexpected error looking up SDK version", exc_info=True)
            lines.append("deepagents (SDK) version: unknown")

        available, latest = self._update_available
        if available and latest:
            try:
                from deepagents_cli.update_check import upgrade_command

                cmd = upgrade_command()
            except Exception:
                logger.warning(
                    "Could not resolve upgrade command for /version; "
                    "falling back to generic pip hint",
                    exc_info=True,
                )
                from deepagents_cli.update_check import FALLBACK_UPGRADE_COMMAND

                cmd = FALLBACK_UPGRADE_COMMAND
            lines.extend(("", f"Update available: v{latest}. Run: {cmd}"))

        await self._mount_message(AppMessage("\n".join(lines)))

        try:
            from deepagents_cli.extras_info import (
                format_extras_status,
                get_extras_status,
            )

            extras_markdown = format_extras_status(get_extras_status())
        except Exception:
            logger.warning(
                "Failed to collect optional dependency status", exc_info=True
            )
            extras_markdown = ""
        if extras_markdown:
            await self._mount_message(AppMessage(extras_markdown, markdown=True))

    async def _handle_auto_update_toggle(self) -> None:
        """Handle the `/auto-update` slash command — persist toggle immediately."""
        try:
            from deepagents_cli.config import _is_editable_install
            from deepagents_cli.update_check import (
                is_auto_update_enabled,
                set_auto_update,
            )

            if await asyncio.to_thread(_is_editable_install):
                self.notify(
                    "Auto-updates are not available for editable installs.",
                    severity="warning",
                    timeout=5,
                )
                return

            currently_enabled = await asyncio.to_thread(is_auto_update_enabled)
            new_state = not currently_enabled
            await asyncio.to_thread(set_auto_update, new_state)
            label = "enabled" if new_state else "disabled"
            self.notify(
                f"Auto-updates {label}.",
                severity="information",
                timeout=5,
                markup=False,
            )
        except Exception as exc:
            logger.warning("/auto-update command failed", exc_info=True)
            self.notify(
                f"Auto-update toggle failed: {type(exc).__name__}: {exc}",
                severity="warning",
                timeout=5,
                markup=False,
            )

    def on_scroll_up(self, _event: ScrollUp) -> None:
        """Handle scroll up to check if we need to hydrate older messages."""
        self._check_hydration_needed()

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message."""
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def _update_tokens(self, count: int, *, approximate: bool = False) -> None:
        """Update the token count in the status bar.

        Low-level helper — only touches the UI.  Callers that also need to
        update the local cache should use `_on_tokens_update` instead.

        Args:
            count: Total context token count.
            approximate: Append "+" to signal a stale/interrupted count.
        """
        if self._status_bar:
            self._status_bar.set_tokens(count, approximate=approximate)

    def _on_tokens_update(self, count: int, *, approximate: bool = False) -> None:
        """Update the local cache *and* the status bar.

        This is the callback wired to the adapter's `_on_tokens_update`.

        Args:
            count: Total context token count to cache and display.
            approximate: Append "+" to signal a stale/interrupted count.
        """
        self._context_tokens = count
        self._tokens_approximate = approximate
        self._update_tokens(count, approximate=approximate)

    def _show_tokens(self, *, approximate: bool = False) -> None:
        """Restore the status bar to the cached token value.

        Args:
            approximate: Append "+" to signal a stale/interrupted count.

                This flag is sticky until `_on_tokens_update` receives a fresh
                count from the model.
        """
        self._tokens_approximate = self._tokens_approximate or approximate
        self._update_tokens(
            self._context_tokens,
            approximate=self._tokens_approximate,
        )

    def _hide_tokens(self) -> None:
        """Hide the token display during streaming."""
        if self._status_bar:
            self._status_bar.hide_tokens()

    def _check_hydration_needed(self) -> None:
        """Check if we need to hydrate messages from the store.

        Called when user scrolls up near the top of visible messages.
        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration check: #chat container not found")
            return

        scroll_y = chat.scroll_y
        viewport_height = chat.size.height

        if self._message_store.should_hydrate_above(scroll_y, viewport_height):
            self.call_later(self._hydrate_messages_above)

    async def _hydrate_messages_above(self) -> None:
        """Hydrate older messages when user scrolls near the top.

        This recreates widgets for archived messages and inserts them
        at the top of the messages container.
        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration: #chat not found")
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping hydration: #messages not found")
            return

        to_hydrate = self._message_store.get_messages_to_hydrate()
        if not to_hydrate:
            return

        old_scroll_y = chat.scroll_y
        first_child = (
            messages_container.children[0] if messages_container.children else None
        )

        # Build widgets in chronological order, then mount in reverse so
        # each is inserted before the previous first_child, resulting in
        # correct chronological order in the DOM.
        hydrated_count = 0
        hydrated_widgets: list[tuple] = []  # (widget, msg_data)
        for msg_data in to_hydrate:
            try:
                widget = msg_data.to_widget()
                hydrated_widgets.append((widget, msg_data))
            except Exception:
                logger.warning(
                    "Failed to create widget for message %s",
                    msg_data.id,
                    exc_info=True,
                )

        for widget, msg_data in reversed(hydrated_widgets):
            try:
                if first_child:
                    await messages_container.mount(widget, before=first_child)
                else:
                    await messages_container.mount(widget)
                first_child = widget
                hydrated_count += 1
                # Render Markdown content for hydrated assistant messages
                if isinstance(widget, AssistantMessage) and msg_data.content:
                    await widget.set_content(msg_data.content)
            except Exception:
                logger.warning(
                    "Failed to mount hydrated widget %s",
                    widget.id,
                    exc_info=True,
                )

        # Only update store for the number we actually mounted
        if hydrated_count > 0:
            self._message_store.mark_hydrated(hydrated_count)

        # Adjust scroll position to maintain the user's view.
        # Widget heights aren't known until after layout, so we use a
        # heuristic. A more accurate approach would measure actual heights
        # via call_after_refresh.
        estimated_height_per_message = 5  # terminal rows, rough estimate
        added_height = hydrated_count * estimated_height_per_message
        chat.scroll_y = old_scroll_y + added_height

    async def _mount_before_queued(self, container: Container, widget: Widget) -> None:
        """Mount a widget in the messages container, before any queued widgets.

        Queued-message widgets must stay at the bottom of the container so
        they remain visually anchored below the current agent response.
        This helper inserts `widget` just before the first queued widget,
        or appends at the end when the queue is empty.

        Args:
            container: The `#messages` container to mount into.
            widget: The widget to mount.
        """
        if not container.is_attached:
            return
        first_queued = self._queued_widgets[0] if self._queued_widgets else None
        if first_queued is not None and first_queued.parent is container:
            try:
                await container.mount(widget, before=first_queued)
            except Exception:
                logger.warning(
                    "Stale queued-widget reference; appending at end",
                    exc_info=True,
                )
            else:
                return
        await container.mount(widget)

    def _is_spinner_at_correct_position(self, container: Container) -> bool:
        """Check whether the loading spinner is already correctly positioned.

        The spinner should be immediately before the first queued widget, or
        at the very end of the container when the queue is empty.

        Args:
            container: The `#messages` container.

        Returns:
            `True` if the spinner is already in the correct position.
        """
        children = list(container.children)
        if not children or self._loading_widget not in children:
            return False

        if self._queued_widgets:
            first_queued = self._queued_widgets[0]
            if first_queued not in children:
                return False
            return children.index(self._loading_widget) == (
                children.index(first_queued) - 1
            )

        return children[-1] == self._loading_widget

    async def _set_spinner(self, status: SpinnerStatus) -> None:
        """Show, update, or hide the loading spinner.

        Args:
            status: The spinner status to display, or `None` to hide.
        """
        if status is None:
            # Hide
            if self._loading_widget:
                await self._loading_widget.remove()
                self._loading_widget = None
            return

        try:
            messages = self.query_one("#messages", Container)
        except NoMatches:
            # Container was torn down (e.g. shutdown mid-stream). Skip
            # silently so the streaming loop doesn't crash.
            return

        if self._loading_widget is None:
            # Create new
            self._loading_widget = LoadingWidget(status)
            await self._mount_before_queued(messages, self._loading_widget)
        else:
            # Update existing
            self._loading_widget.set_status(status)
            # Reposition via move_child so elapsed-time and animation state
            # carry through; remove + re-mount would reset both.
            if not self._is_spinner_at_correct_position(messages):
                self._reposition_spinner(messages)
        # NOTE: Don't call anchor() here - it would re-anchor and drag user back
        # to bottom if they've scrolled away during streaming

    def _reposition_spinner(self, container: Container) -> None:
        """Move the spinner to its correct position without resetting state.

        The spinner must sit immediately before the first queued widget, or
        at the very end of the container when no widgets are queued. Using
        `move_child` preserves the widget's internal state (elapsed time,
        animation frame) that a remove + re-mount would reset.

        Args:
            container: The messages container that hosts the spinner.
        """
        if self._loading_widget is None:
            return
        if self._loading_widget not in container.children:
            # The caller holds a spinner reference that isn't in this
            # container — the widget was reparented or removed by another
            # code path. Log so the desync is visible instead of silently
            # leaving the spinner in the wrong place.
            logger.debug(
                "Spinner widget not in container children; skipping reposition"
            )
            return
        first_queued = self._queued_widgets[0] if self._queued_widgets else None
        if first_queued is not None and first_queued.parent is container:
            container.move_child(self._loading_widget, before=first_queued)
            return
        non_spinner = [
            child for child in container.children if child is not self._loading_widget
        ]
        if non_spinner:
            container.move_child(self._loading_widget, after=non_spinner[-1])

    async def _request_approval(
        self,
        action_requests: Any,  # noqa: ANN401  # ActionRequest uses dynamic typing
        assistant_id: str | None,
    ) -> asyncio.Future:
        """Request user approval inline in the messages area.

        Mounts ApprovalMenu in the messages area (inline with chat).
        ChatInput stays visible - user can still see it.

        If another approval is already pending, queue this one.

        Auto-approves shell commands that are in the configured allow-list.

        Args:
            action_requests: List of action request dicts to approve
            assistant_id: The assistant ID for display purposes

        Returns:
            A Future that resolves to the user's decision.
        """
        from deepagents_cli.config import (
            SHELL_TOOL_NAMES,
            is_shell_command_allowed,
            settings,
        )

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()

        # Check if ALL actions in the batch are auto-approvable shell commands
        if settings.shell_allow_list and action_requests:
            all_auto_approved = True
            approved_commands = []

            for req in action_requests:
                if req.get("name") in SHELL_TOOL_NAMES:
                    command = req.get("args", {}).get("command", "")
                    if is_shell_command_allowed(command, settings.shell_allow_list):
                        approved_commands.append(command)
                    else:
                        all_auto_approved = False
                        break
                else:
                    # Non-shell commands need normal approval
                    all_auto_approved = False
                    break

            if all_auto_approved and approved_commands:
                # Auto-approve all commands in the batch
                result_future.set_result({"type": "approve"})

                # Mount system messages showing the auto-approvals
                try:
                    messages = self.query_one("#messages", Container)
                    for command in approved_commands:
                        auto_msg = AppMessage(
                            f"✓ Auto-approved shell command (allow-list): {command}"
                        )
                        await self._mount_before_queued(messages, auto_msg)
                    with suppress(NoMatches, ScreenStackError):
                        self.query_one("#chat", VerticalScroll).anchor()
                except Exception:  # noqa: S110, BLE001  # Resilient auto-message display
                    pass  # Don't fail if we can't show the message

                return result_future

        # If there's already a pending approval, wait for it to complete first
        if self._pending_approval_widget is not None:
            while self._pending_approval_widget is not None:  # noqa: ASYNC110  # Simple polling is sufficient here
                await asyncio.sleep(0.1)

        # Create menu with unique ID to avoid conflicts
        from deepagents_cli.widgets.approval import ApprovalMenu

        unique_id = f"approval-menu-{uuid.uuid4().hex[:8]}"
        menu = ApprovalMenu(action_requests, assistant_id, id=unique_id)
        menu.set_future(result_future)

        self._pending_approval_widget = menu

        if self._is_user_typing():
            # Show a placeholder until the user stops typing, then swap in the
            # real ApprovalMenu.  This prevents accidental key presses (e.g.
            # 'y', 'n') from triggering approval decisions mid-sentence.
            placeholder = Static(
                "Waiting for typing to finish...",
                classes="approval-placeholder",
            )
            self._approval_placeholder = placeholder
            try:
                messages = self.query_one("#messages", Container)
                await self._mount_before_queued(messages, placeholder)
                self.call_after_refresh(placeholder.scroll_visible)
            except Exception:
                logger.exception("Failed to mount approval placeholder")
                # Placeholder failed — fall back to showing the menu directly
                # so the future is always resolvable.
                self._approval_placeholder = None
                await self._mount_approval_widget(menu, result_future)
                return result_future

            self.run_worker(
                self._deferred_show_approval(placeholder, menu, result_future),
                exclusive=False,
            )
        else:
            await self._mount_approval_widget(menu, result_future)

        return result_future

    async def _mount_approval_widget(
        self,
        menu: ApprovalMenu,
        result_future: asyncio.Future[dict[str, str]],
    ) -> None:
        """Mount the approval menu widget inline in the messages area.

        If mounting fails, clears `_pending_approval_widget` and propagates
        the exception via `result_future`.

        Args:
            menu: The `ApprovalMenu` instance to mount.
            result_future: The future to resolve/reject for the caller.
        """
        try:
            messages = self.query_one("#messages", Container)
            await self._mount_before_queued(messages, menu)
            self.call_after_refresh(menu.scroll_visible)
            self.call_after_refresh(menu.focus)
        except Exception as e:
            logger.exception(
                "Failed to mount approval menu (id=%s) in messages container",
                menu.id,
            )
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.set_exception(e)

    async def _deferred_show_approval(
        self,
        placeholder: Static,
        menu: ApprovalMenu,
        result_future: asyncio.Future[dict[str, str]],
    ) -> None:
        """Wait until the user is idle, then swap the placeholder for the real menu.

        Exits early if the placeholder has already been detached (e.g. the
        approval was cancelled while waiting).  In that case the future is
        cancelled so the caller is not left hanging.

        Args:
            placeholder: The temporary placeholder widget currently mounted.
            menu: The `ApprovalMenu` to show once the user stops typing.
            result_future: The future backing this approval flow.
        """
        deadline = _monotonic() + _DEFERRED_APPROVAL_TIMEOUT_SECONDS
        while self._is_user_typing():  # Simple polling
            if _monotonic() > deadline:
                logger.warning(
                    "Timed out waiting for user to stop typing; showing approval now"
                )
                break
            await asyncio.sleep(0.2)

        # Guard: if the placeholder was already removed (e.g. agent cancelled
        # the approval while we were waiting), clean up and cancel the future.
        if not placeholder.is_attached:
            logger.warning(
                "Approval placeholder detached before menu shown (id=%s)",
                menu.id,
            )
            self._approval_placeholder = None
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.cancel()
            return

        self._approval_placeholder = None
        try:
            await placeholder.remove()
        except Exception:
            logger.warning(
                "Failed to remove approval placeholder during swap",
                exc_info=True,
            )
        await self._mount_approval_widget(menu, result_future)

    def _on_auto_approve_enabled(self) -> None:
        """Handle auto-approve being enabled via the HITL approval menu.

        Called when the user selects "Auto-approve all" from an approval
        dialog. Syncs the auto-approve state across the app flag, status
        bar indicator, and session state so subsequent tool calls skip
        the approval prompt.
        """
        self._auto_approve = True
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=True)
        if self._session_state:
            self._session_state.auto_approve = True

    async def _remove_ask_user_widget(  # noqa: PLR6301  # Shared helper used by ask_user event handlers
        self,
        widget: AskUserMenu,
        *,
        context: str,
    ) -> None:
        """Remove an ask_user widget without surfacing cleanup races.

        Args:
            widget: Ask-user widget instance to remove.
            context: Short context string for diagnostics.
        """
        try:
            await widget.remove()
        except Exception:
            logger.debug(
                "Failed to remove ask-user widget during %s",
                context,
                exc_info=True,
            )

    async def _request_ask_user(
        self,
        questions: list[Question],
    ) -> asyncio.Future[AskUserWidgetResult]:
        """Display the ask_user widget and return a Future with user response.

        Args:
            questions: List of question dicts, each with `question`, `type`,
                and optional `choices` and `required` keys.

        Returns:
            A Future that resolves to a dict with `'type'` (`'answered'` or
                `'cancelled'`) and, when answered, an `'answers'` list.
        """
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[AskUserWidgetResult] = loop.create_future()

        if self._pending_ask_user_widget is not None:
            deadline = _monotonic() + 30
            while self._pending_ask_user_widget is not None:
                if _monotonic() > deadline:
                    logger.error(
                        "Timed out waiting for previous ask-user widget to "
                        "clear. Forcefully cleaning up."
                    )
                    old_widget = self._pending_ask_user_widget
                    if old_widget is not None:
                        old_widget.action_cancel()
                        self._pending_ask_user_widget = None
                        await self._remove_ask_user_widget(
                            old_widget,
                            context="ask-user timeout cleanup",
                        )
                    break
                await asyncio.sleep(0.1)

        from deepagents_cli.widgets.ask_user import AskUserMenu

        unique_id = f"ask-user-menu-{uuid.uuid4().hex[:8]}"
        menu = AskUserMenu(questions, id=unique_id)
        menu.set_future(result_future)

        self._pending_ask_user_widget = menu

        try:
            messages = self.query_one("#messages", Container)
            await self._mount_before_queued(messages, menu)
            self.call_after_refresh(menu.scroll_visible)
            self.call_after_refresh(menu.focus_active)
        except Exception as e:
            logger.exception(
                "Failed to mount ask-user menu (id=%s)",
                unique_id,
            )
            self._pending_ask_user_widget = None
            if not result_future.done():
                result_future.set_exception(e)

        return result_future

    async def on_ask_user_menu_answered(
        self,
        event: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Handle ask_user menu answers - remove widget and refocus input."""
        if self._pending_ask_user_widget:
            widget = self._pending_ask_user_widget
            self._pending_ask_user_widget = None
            await self._remove_ask_user_widget(widget, context="ask-user answered")

        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def on_ask_user_menu_cancelled(
        self,
        event: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Handle ask_user menu cancellation - remove widget and refocus input."""
        if self._pending_ask_user_widget:
            widget = self._pending_ask_user_widget
            self._pending_ask_user_widget = None
            await self._remove_ask_user_widget(widget, context="ask-user cancelled")

        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def _process_message(self, value: str, mode: InputMode) -> None:
        """Route a message to the appropriate handler based on mode.

        Args:
            value: The message text to process.
            mode: The input mode that determines message routing.
        """
        if mode == "shell":
            await self._handle_shell_command(value.removeprefix("!"))
        elif mode == "command":
            await self._handle_command(value)
        elif mode == "normal":
            await self._handle_user_message(value)
        else:
            logger.warning("Unrecognized input mode %r, treating as normal", mode)
            await self._handle_user_message(value)

    def _has_initial_submission(self) -> bool:
        """Return whether startup should auto-submit a prompt or skill."""
        return self._initial_skill is not None or bool(
            self._initial_prompt and self._initial_prompt.strip()
        )

    async def _run_session_start_sequence(self) -> None:
        """Load history, run `--startup-cmd`, then dispatch initial work.

        Single entry point for the post-connect sequence. Sequencing the
        startup command before any user-facing agent work guarantees the
        agent never observes input until the command has completed.
        """
        self._startup_sequence_running = True
        try:
            should_load_history = bool(self._lc_thread_id and self._agent) and (
                self._resume_thread_intent is not None
                or not self._has_initial_submission()
            )
            if should_load_history:
                await self._load_thread_history()

            if self._startup_cmd:
                cmd = self._startup_cmd
                # One-shot: clear to avoid re-running on any subsequent server swap.
                self._startup_cmd = None
                await self._run_startup_command(cmd)

            if self._has_initial_submission():
                await self._submit_initial_submission()
                return
        finally:
            self._startup_sequence_running = False

        if self._agent_running or self._shell_running:
            return

        try:
            await self._maybe_drain_deferred()
        except Exception:
            logger.exception(
                "Failed to drain deferred actions after startup sequencing"
            )
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "A deferred action failed during startup. "
                        "You may need to retry the operation."
                    )
                )

        if self._pending_messages:
            await self._process_next_from_queue()

    async def _run_startup_command(self, command: str) -> None:
        """Execute the `--startup-cmd` and render its output in the transcript.

        Uses the same worker-backed subprocess path as the interactive `!`
        shell prefix, with an app-style header (since the user did not type
        the command). Non-zero exit is already rendered as an error by
        `_run_shell_task` but does not abort the session.

        Raises:
            CancelledError: If the worker is cancelled (e.g. Esc/Ctrl+C);
                re-raised so `_run_shell_task`'s finally can clean up.
        """
        try:
            await self._mount_message(
                AppMessage(
                    Content.from_markup("Running startup command: $cmd", cmd=command)
                )
            )
        except Exception:
            logger.warning("Failed to mount startup-command header", exc_info=True)

        self._shell_running = True
        if self._chat_input:
            self._chat_input.set_cursor_active(active=False)

        try:
            worker = self.run_worker(self._run_shell_task(command), exclusive=False)
        except Exception:
            # `run_worker` failed synchronously — `_run_shell_task`'s finally
            # never fires, so reset the busy flags here or the UI stays wedged.
            logger.exception("Failed to schedule startup-command worker")
            self._shell_running = False
            self._shell_worker = None
            if self._chat_input:
                self._chat_input.set_cursor_active(active=True)
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage("Failed to start startup command; continuing session.")
                )
            return

        self._shell_worker = worker
        try:
            await worker.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Startup command worker raised unexpectedly")

    async def _submit_initial_submission(self) -> None:
        """Submit the startup prompt or skill after the UI is ready."""
        try:
            if self._initial_skill is not None:
                await self._invoke_skill(
                    self._initial_skill, self._initial_prompt or ""
                )
                return
            if self._initial_prompt and self._initial_prompt.strip():
                await self._handle_user_message(self._initial_prompt)
        except Exception:
            logger.exception("Unhandled error during initial submission")
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "Failed to submit startup prompt. "
                        "Try running the command manually in the session."
                    )
                )

    def _can_bypass_queue(self, value: str) -> bool:
        """Check if a slash command can skip the message queue.

        Args:
            value: The lowered, stripped command string (e.g. `/model`).

        Returns:
            `True` if the command should bypass the busy-state queue.
        """
        from deepagents_cli.command_registry import (
            BYPASS_WHEN_CONNECTING,
            IMMEDIATE_UI,
            SIDE_EFFECT_FREE,
        )

        cmd = value.split(maxsplit=1)[0] if value else ""
        if cmd in BYPASS_WHEN_CONNECTING:
            return self._connecting and not (self._agent_running or self._shell_running)
        if cmd in IMMEDIATE_UI:
            # Only bare form (no args) bypasses — /model opens selector,
            # /model <name> does a direct switch that shouldn't race with agent.
            return value == cmd
        return cmd in SIDE_EFFECT_FREE

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle submitted input from ChatInput widget."""
        value = event.value
        mode: InputMode = event.mode  # type: ignore[assignment]  # Textual event mode is str at type level but InputMode at runtime

        # Reset quit pending state on any input
        self._quit_pending = False

        from deepagents_cli.hooks import dispatch_hook

        await dispatch_hook("user.prompt", {})

        # /quit and /q always execute immediately, even mid-thread-switch.
        from deepagents_cli.command_registry import ALWAYS_IMMEDIATE

        if mode == "command" and value.lower().strip() in ALWAYS_IMMEDIATE:
            self.exit()
            return

        # Prevent message handling while a thread switch is in-flight.
        if self._thread_switching:
            self.notify(
                "Thread switch in progress. Please wait.",
                severity="warning",
                timeout=3,
            )
            return

        # If the app is busy or still sequencing startup work, enqueue instead
        # of processing. Messages queued during startup are drained once the
        # session reaches its first stable idle/running state.
        if (
            self._agent_running
            or self._shell_running
            or self._connecting
            or self._startup_sequence_running
        ):
            if mode == "command" and self._can_bypass_queue(value.lower().strip()):
                await self._process_message(value, mode)
                return
            self._pending_messages.append(QueuedMessage(text=value, mode=mode))
            queued_widget = QueuedUserMessage(value)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)
            return

        await self._process_message(value, mode)

    def on_chat_input_mode_changed(self, event: ChatInput.ModeChanged) -> None:
        """Update status bar when input mode changes."""
        if self._status_bar:
            self._status_bar.set_mode(event.mode)

    def on_chat_input_typing(
        self,
        event: ChatInput.Typing,  # noqa: ARG002  # Textual event handler signature
    ) -> None:
        """Record the most recent keystroke time for typing-aware approval deferral."""
        self._last_typed_at = _monotonic()

    def _is_user_typing(self) -> bool:
        """Return whether the user typed recently (within the idle threshold).

        Returns:
            `True` if the last recorded typing event occurred within the last
                `_TYPING_IDLE_THRESHOLD_SECONDS` seconds, `False` otherwise.
        """
        if self._last_typed_at is None:
            return False
        return (_monotonic() - self._last_typed_at) < _TYPING_IDLE_THRESHOLD_SECONDS

    async def on_approval_menu_decided(
        self,
        event: Any,  # noqa: ARG002, ANN401  # Textual event handler signature
    ) -> None:
        """Handle approval menu decision - remove from messages and refocus input."""
        # Defensively remove any lingering placeholder (should already be gone
        # once the deferred worker swaps it, but guard against edge cases).
        if self._approval_placeholder is not None:
            if self._approval_placeholder.is_attached:
                try:
                    await self._approval_placeholder.remove()
                except Exception:
                    logger.warning(
                        "Failed to remove approval placeholder during cleanup",
                        exc_info=True,
                    )
            self._approval_placeholder = None

        # Remove ApprovalMenu using stored reference
        if self._pending_approval_widget:
            await self._pending_approval_widget.remove()
            self._pending_approval_widget = None

        # Refocus the chat input
        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def _handle_shell_command(self, command: str) -> None:
        """Handle a shell command (! prefix).

        Thin dispatcher that mounts the user message and spawns a worker
        so the event loop stays free for key events (Esc/Ctrl+C).

        Args:
            command: The shell command to execute.
        """
        await self._mount_message(UserMessage(f"!{command}"))
        self._shell_running = True

        if self._chat_input:
            self._chat_input.set_cursor_active(active=False)

        self._shell_worker = self.run_worker(
            self._run_shell_task(command),
            exclusive=False,
        )

    async def _run_shell_task(self, command: str) -> None:
        """Run a shell command in a background worker.

        This mirrors `_run_agent_task`: running in a worker keeps the event
        loop free so Esc/Ctrl+C can cancel the worker -> raise
        `CancelledError` -> kill the process.

        Args:
            command: The shell command to execute.

        Raises:
            CancelledError: If the command is interrupted by the user.
        """
        refresh_started = False
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                start_new_session=(sys.platform != "win32"),
            )
            self._shell_process = proc

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=60
                )
            except TimeoutError:
                await self._kill_shell_process()
                await self._mount_message(ErrorMessage("Command timed out (60s limit)"))
                return
            except asyncio.CancelledError:
                await self._kill_shell_process()
                raise

            # Start branch refresh as soon as the shell exits so it can overlap
            # with output rendering instead of trailing it.
            self._schedule_git_branch_refresh()
            refresh_started = True

            output = (stdout_bytes or b"").decode(errors="replace").strip()
            stderr_text = (stderr_bytes or b"").decode(errors="replace").strip()
            if stderr_text:
                output += f"\n[stderr]\n{stderr_text}"

            if output:
                msg = AssistantMessage(f"```\n{output}\n```")
                await self._mount_message(msg)
                await msg.write_initial_content()
            else:
                await self._mount_message(AppMessage("Command completed (no output)"))

            if proc.returncode and proc.returncode != 0:
                await self._mount_message(ErrorMessage(f"Exit code: {proc.returncode}"))

            # Anchor to bottom so shell output stays visible
            with suppress(NoMatches, ScreenStackError):
                self.query_one("#chat", VerticalScroll).anchor()

        except OSError as e:
            logger.exception("Failed to execute shell command: %s", command)
            err_msg = f"Failed to run command: {e}"
            await self._mount_message(ErrorMessage(err_msg))
        finally:
            await self._cleanup_shell_task(refresh_git_branch=not refresh_started)

    async def _cleanup_shell_task(self, *, refresh_git_branch: bool = True) -> None:
        """Clean up after shell command task completes or is cancelled.

        Args:
            refresh_git_branch: Whether to schedule a footer branch refresh
                during cleanup. Successful shell runs can launch this earlier
                so refresh overlaps with output rendering.
        """
        was_interrupted = self._shell_process is not None and (
            self._shell_worker is not None and self._shell_worker.is_cancelled
        )
        self._shell_process = None
        self._shell_running = False
        self._shell_worker = None
        if was_interrupted:
            await self._mount_message(AppMessage("Command interrupted"))
        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)
        if refresh_git_branch:
            # A `!` command may have changed git state (e.g. `git checkout`);
            # re-resolve so the footer reflects the new branch.
            self._schedule_git_branch_refresh()
        try:
            await self._maybe_drain_deferred()
        except Exception:
            logger.exception("Failed to drain deferred actions during shell cleanup")
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "A deferred action failed after task completion. "
                        "You may need to retry the operation."
                    )
                )
        if not self._startup_sequence_running:
            await self._process_next_from_queue()

    async def _kill_shell_process(self) -> None:
        """Terminate the running shell command process.

        On POSIX, sends SIGTERM to the entire process group (killing children).
        On Windows, terminates only the root process. No-op if the process has
        already exited. Waits up to 5s for clean shutdown, then escalates
        to SIGKILL.
        """
        proc = self._shell_process
        if proc is None or proc.returncode is not None:
            return

        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        except OSError:
            logger.warning(
                "Failed to terminate shell process (pid=%s)", proc.pid, exc_info=True
            )
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            logger.warning(
                "Shell process (pid=%s) did not exit after SIGTERM; sending SIGKILL",
                proc.pid,
            )
            with suppress(ProcessLookupError, OSError):
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            with suppress(ProcessLookupError, OSError):
                await proc.wait()
        except (ProcessLookupError, OSError):
            pass

    async def _open_url_command(self, command: str, cmd: str) -> None:
        """Open a URL in the browser and display a clickable link.

        The browser opens immediately regardless of busy state. When the app is
        busy, a queued indicator is shown and the real chat output (user echo
        + clickable link) replaces it after the current task finishes.

        Args:
            command: The raw command text (displayed as user message).
            cmd: The normalized slash command used to look up the URL.
        """
        url = _COMMAND_URLS[cmd]
        webbrowser.open(url)

        if self._agent_running or self._shell_running:
            queued_widget = QueuedUserMessage(command)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)

            async def _mount_output() -> None:
                # Remove the ephemeral queued widget, then mount real output.
                if queued_widget in self._queued_widgets:
                    self._queued_widgets.remove(queued_widget)
                with suppress(Exception):
                    await queued_widget.remove()
                await self._mount_message(UserMessage(command))
                link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
                await self._mount_message(AppMessage(link))

            # Append directly — no dedup; each URL command gets its own output.
            self._deferred_actions.append(
                DeferredAction(kind="chat_output", execute=_mount_output)
            )
            return

        await self._mount_message(UserMessage(command))
        link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
        await self._mount_message(AppMessage(link))

    @staticmethod
    async def _build_thread_message(prefix: str, thread_id: str) -> str | Content:
        """Build a thread status message, hyperlinking the ID when possible.

        Attempts to resolve the LangSmith thread URL with a short timeout.
        Falls back to plain text if tracing is not configured or resolution
        fails.

        Args:
            prefix: Label before the thread ID (e.g. `'Resumed thread'`).
            thread_id: The thread identifier.

        Returns:
            `Content` with a clickable thread ID, or a plain string.
        """
        from deepagents_cli.config import build_langsmith_thread_url

        try:
            url = await asyncio.wait_for(
                asyncio.to_thread(build_langsmith_thread_url, thread_id),
                timeout=2.0,
            )
        except (TimeoutError, Exception):  # noqa: BLE001  # Resilient non-interactive mode error handling
            url = None

        if url:
            return Content.assemble(
                f"{prefix}: ",
                (thread_id, TStyle(link=url)),
            )
        return f"{prefix}: {thread_id}"

    async def _handle_trace_command(self, command: str) -> None:
        """Open the current thread in LangSmith.

        Resolves the URL and opens the browser immediately regardless of busy
        state. When the app is busy, chat output (user echo + clickable link)
        is deferred until the current task finishes. Error conditions (no
        session, URL failure, tracing not configured) render immediately
        regardless of busy state.

        Args:
            command: The raw command text (displayed as user message).
        """
        from deepagents_cli.config import build_langsmith_thread_url

        if not self._session_state:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage("No active session."))
            return
        thread_id = self._session_state.thread_id
        try:
            url = await asyncio.to_thread(build_langsmith_thread_url, thread_id)
        except Exception:
            logger.exception("Failed to build LangSmith thread URL for %s", thread_id)
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage("Failed to resolve LangSmith thread URL.")
            )
            return
        if not url:
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                AppMessage(
                    "LangSmith tracing is not configured. "
                    "Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to enable."
                )
            )
            return

        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                logger.debug("Could not open browser for URL: %s", url, exc_info=True)

        asyncio.get_running_loop().run_in_executor(None, _open_browser)

        # Defer chat output while a turn is in progress — rendering the user
        # echo + link immediately would splice it into the middle of the
        # streaming assistant response
        if self._agent_running or self._shell_running:
            queued_widget = QueuedUserMessage(command)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)

            async def _mount_output() -> None:
                if queued_widget in self._queued_widgets:
                    self._queued_widgets.remove(queued_widget)
                with suppress(Exception):
                    await queued_widget.remove()
                await self._mount_message(UserMessage(command))
                link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
                await self._mount_message(AppMessage(link))

            # Append directly — no dedup; each /trace invocation gets its own output.
            self._deferred_actions.append(
                DeferredAction(kind="chat_output", execute=_mount_output)
            )
            return

        await self._mount_message(UserMessage(command))
        link = Content.styled(url, TStyle(dim=True, italic=True, link=url))
        await self._mount_message(AppMessage(link))

    async def _handle_command(self, command: str) -> None:
        """Handle a slash command.

        Args:
            command: The slash command (including /)
        """
        from deepagents_cli.config import newline_shortcut, settings

        cmd = command.lower().strip()

        if cmd in {"/quit", "/q"}:
            self.exit()
        elif cmd == "/help":
            await self._mount_message(UserMessage(command))
            help_body = (
                "Commands: /quit, /agents, /clear, /offload, /editor, /mcp, "
                "/model [--model-params JSON] [--default], /notifications, "
                "/reload, /skill:<name>, /remember, /skill-creator, /theme, "
                "/tokens, /threads, /trace, "
                "/update, /auto-update, /changelog, /docs, /feedback, /help\n\n"
                "Interactive Features:\n"
                "  Enter           Submit your message\n"
                f"  {newline_shortcut():<15} Insert newline\n"
                "  Ctrl+X          Open prompt in external editor\n"
                "  Ctrl+N          Review pending notifications\n"
                "  Shift+Tab       Toggle auto-approve mode\n"
                "  @filename       Auto-complete files and inject content\n"
                "  /command        Slash commands (/help, /clear, /quit)\n"
                "  !command        Run shell commands directly\n\n"
                "Docs: "
            )
            help_text = Content.assemble(
                (help_body, "dim italic"),
                (DOCS_URL, TStyle(dim=True, italic=True, link=DOCS_URL)),
            )
            await self._mount_message(AppMessage(help_text))

        elif cmd in {"/changelog", "/docs", "/feedback"}:
            await self._open_url_command(command, cmd)
        elif cmd == "/version":
            await self._mount_message(UserMessage(command))
            await self._handle_version_command()
        elif cmd == "/agents":
            await self._show_agent_selector()
        elif cmd == "/clear":
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            self._context_tokens = 0
            self._tokens_approximate = False
            self._update_tokens(0)
            # Clear status message (e.g., "Interrupted" from previous session)
            self._update_status("")
            # Reset thread to start fresh conversation
            if self._session_state:
                new_thread_id = self._session_state.reset_thread()
                try:
                    banner = self.query_one("#welcome-banner", WelcomeBanner)
                    banner.update_thread_id(new_thread_id)
                except NoMatches:
                    pass
                await self._mount_message(
                    AppMessage(f"Started new thread: {new_thread_id}")
                )
        elif cmd == "/editor":
            await self.action_open_editor()
        elif cmd in {"/offload", "/compact"}:
            await self._mount_message(UserMessage(command))
            await self._handle_offload()
        elif cmd == "/threads":
            await self._show_thread_selector()
        elif cmd == "/trace":
            await self._handle_trace_command(command)
        elif cmd == "/update":
            await self._handle_update_command()
        elif cmd == "/auto-update":
            await self._handle_auto_update_toggle()
        elif cmd == "/tokens":
            await self._mount_message(UserMessage(command))
            if self._context_tokens > 0:
                count = self._context_tokens
                formatted = format_token_count(count)

                model_name = settings.model_name
                context_limit = settings.model_context_limit

                if context_limit is not None:
                    limit_str = format_token_count(context_limit)
                    pct = count / context_limit * 100
                    usage = f"{formatted} / {limit_str} tokens ({pct:.0f}%)"
                else:
                    usage = f"{formatted} tokens used"

                msg = f"{usage} \u00b7 {model_name}" if model_name else usage

                conv_tokens = await self._get_conversation_token_count()
                if conv_tokens is not None:
                    overhead = max(0, count - conv_tokens)
                    overhead_str = format_token_count(overhead)
                    conv_str = format_token_count(conv_tokens)

                    overhead_unit = " tokens" if overhead < 1000 else ""  # noqa: PLR2004  # not bothersome, cosmetic
                    conv_unit = " tokens" if conv_tokens < 1000 else ""  # noqa: PLR2004  # not bothersome, cosmetic

                    msg += (
                        f"\n\u251c System prompt + tools: ~{overhead_str}{overhead_unit} (fixed)"  # noqa: E501
                        f"\n\u2514 Conversation: ~{conv_str}{conv_unit}"
                    )

                await self._mount_message(AppMessage(msg))
            else:
                model_name = settings.model_name
                context_limit = settings.model_context_limit

                parts: list[str] = ["No token usage yet"]
                if context_limit is not None:
                    limit_str = format_token_count(context_limit)
                    parts.append(f"{limit_str} token context window")
                if model_name:
                    parts.append(model_name)

                await self._mount_message(AppMessage(" · ".join(parts)))
        elif cmd == "/remember" or cmd.startswith("/remember "):
            # Convenience alias for /skill:remember — shorter and discoverable
            # before skill loading completes.
            if not await self._has_conversation_messages():
                await self._mount_message(UserMessage(command))
                await self._mount_message(
                    AppMessage(
                        "Nothing to remember yet. Start a conversation first,"
                        " then use /remember to capture learnings."
                    )
                )
                return
            args = command.strip()[len("/remember") :].strip()
            rewritten = f"/skill:remember {args}" if args else "/skill:remember"
            await self._handle_skill_command(rewritten)
        elif cmd == "/skill-creator" or cmd.startswith("/skill-creator "):
            # Convenience alias for /skill:skill-creator — shorter and
            # discoverable before skill loading completes.
            args = command.strip()[len("/skill-creator") :].strip()
            rewritten = (
                f"/skill:skill-creator {args}" if args else "/skill:skill-creator"
            )
            await self._handle_skill_command(rewritten)
        elif cmd == "/mcp":
            await self._show_mcp_viewer()
        elif cmd == "/theme":
            await self._show_theme_selector()
        elif cmd == "/notifications":
            await self._show_notification_settings()
        elif cmd == "/model" or cmd.startswith("/model "):
            model_arg = None
            set_default = False
            extra_kwargs: dict[str, Any] | None = None
            if cmd.startswith("/model "):
                raw_arg = command.strip()[len("/model ") :].strip()
                try:
                    raw_arg, extra_kwargs = _extract_model_params_flag(raw_arg)
                except (ValueError, TypeError) as exc:
                    await self._mount_message(UserMessage(command))
                    await self._mount_message(ErrorMessage(str(exc)))
                    return
                if raw_arg.startswith("--default"):
                    set_default = True
                    model_arg = raw_arg[len("--default") :].strip() or None
                else:
                    model_arg = raw_arg or None

            if set_default:
                await self._mount_message(UserMessage(command))
                if extra_kwargs:
                    await self._mount_message(
                        ErrorMessage(
                            "--model-params cannot be used with --default. "
                            "Model params are applied per-session, not "
                            "persisted."
                        )
                    )
                elif model_arg == "--clear":
                    await self._clear_default_model()
                elif model_arg:
                    await self._set_default_model(model_arg)
                else:
                    await self._mount_message(
                        AppMessage(
                            "Usage: /model --default provider:model\n"
                            "       /model --default --clear"
                        )
                    )
            elif model_arg:
                # Direct switch: /model claude-sonnet-4-5
                await self._mount_message(UserMessage(command))
                await self._switch_model(model_arg, extra_kwargs=extra_kwargs)
            else:
                await self._show_model_selector(extra_kwargs=extra_kwargs)
        elif cmd == "/reload":
            await self._mount_message(UserMessage(command))
            try:
                changes = settings.reload_from_environment()

                from deepagents_cli.model_config import clear_caches

                clear_caches()
            except (OSError, ValueError):
                logger.exception("Failed to reload configuration")
                await self._mount_message(
                    AppMessage(
                        "Failed to reload configuration. Check your .env "
                        "file and environment variables for syntax errors, "
                        "then try again."
                    )
                )
                return

            # Reload user themes from config.toml and re-register with Textual
            theme_reload_ok = True
            try:
                theme.reload_registry()
                self._register_custom_themes()
            except Exception:
                theme_reload_ok = False
                logger.warning("Failed to reload user themes", exc_info=True)

            if changes:
                report = "Configuration reloaded. Changes:\n" + "\n".join(
                    f"  - {change}" for change in changes
                )
            else:
                report = "Configuration reloaded. No changes detected."
            report += "\nModel config caches cleared."
            if theme_reload_ok:
                report += "\nTheme registry reloaded."
            else:
                report += (
                    "\nTheme registry reload failed. Check config.toml for errors."
                )
            await self._mount_message(AppMessage(report))

            # Re-discover skills so autocomplete reflects any new/removed skills
            self.run_worker(
                self._discover_skills(),
                exclusive=True,
                group="startup-skill-discovery",
            )
        elif cmd.startswith("/skill:"):
            await self._handle_skill_command(command)
        # -- Hidden debug commands (not in COMMANDS / autocomplete) -----------
        elif cmd == "/debug-error":
            await self._mount_message(
                ErrorMessage(
                    "Server failed to start: RuntimeError: Server process"
                    " exited with code 3"
                )
            )
        else:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(f"Unknown command: {cmd}"))

        # Anchor to bottom so command output stays visible
        with suppress(NoMatches, ScreenStackError):
            self.query_one("#chat", VerticalScroll).anchor()

    async def _invoke_skill(
        self,
        skill_name: str,
        args: str = "",
        *,
        command: str | None = None,
    ) -> None:
        """Load a skill, render its widget, and send its prompt to the agent.

        Looks up the skill from cached metadata (populated at startup), falling
        back to a fresh filesystem walk on cache miss. Reads the `SKILL.md`
        body, wraps it in a prompt envelope with any user-provided arguments,
        and sends the composed message to the agent.

        Args:
            skill_name: Skill name to invoke.
            args: Optional user request to append after the skill body.
            command: Original slash command text for UI echo, if any.
        """
        from deepagents_cli.skills.invocation import build_skill_invocation_envelope
        from deepagents_cli.skills.load import load_skill_content

        normalized_name = skill_name.strip().lower()

        async def _mount_error(message: str) -> None:
            if command is not None:
                await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(message))

        if not normalized_name:
            if command is not None:
                await self._mount_message(UserMessage(command))
                await self._mount_message(AppMessage("Usage: /skill:<name> [args]"))
            else:
                await self._mount_message(AppMessage("Skill name is required."))
            return

        # Fast path: look up from the cached discovery results
        cached = next(
            (s for s in self._discovered_skills if s["name"] == normalized_name),
            None,
        )
        allowed_roots = self._skill_allowed_roots

        # Cache miss — fall back to fresh discovery (offloaded to thread)
        if cached is None:
            try:
                skills, allowed_roots = await asyncio.to_thread(
                    self._discover_skills_and_roots
                )
                # Backfill cache so subsequent invocations are fast
                self._discovered_skills = skills
                self._skill_allowed_roots = allowed_roots
                cached = next((s for s in skills if s["name"] == normalized_name), None)
            except OSError as exc:
                logger.warning(
                    "Filesystem error loading skill %r", normalized_name, exc_info=True
                )
                await _mount_error(
                    f"Could not load skill: {normalized_name}. Filesystem error: {exc}"
                )
                return
            except Exception as exc:
                logger.warning(
                    "Error searching for skill %r", normalized_name, exc_info=True
                )
                await _mount_error(
                    f"Error loading skill: {normalized_name}. "
                    f"Unexpected error: {type(exc).__name__}: {exc}"
                )
                return

        if cached is None:
            logger.warning("Skill not found: %r", normalized_name)
            await _mount_error(f"Skill not found: {normalized_name}")
            return

        # Load SKILL.md content (filesystem I/O offloaded to thread)
        skill_path = cached["path"]

        def _load() -> str | None:
            return load_skill_content(str(skill_path), allowed_roots=allowed_roots)

        try:
            content = await asyncio.to_thread(_load)
        except PermissionError as exc:
            logger.warning(
                "Containment check failed for skill %r",
                normalized_name,
                exc_info=True,
            )
            await _mount_error(str(exc))
            return
        except OSError as exc:
            logger.warning(
                "Filesystem error loading skill %r", normalized_name, exc_info=True
            )
            await _mount_error(
                f"Could not load skill: {normalized_name}. Filesystem error: {exc}"
            )
            return
        except Exception as exc:
            logger.warning("Error reading skill %r", normalized_name, exc_info=True)
            await _mount_error(
                f"Error loading skill: {normalized_name}. "
                f"Unexpected error: {type(exc).__name__}: {exc}"
            )
            return

        if content is None:
            await _mount_error(
                f"Could not read content for skill: {normalized_name}. "
                "Check that the SKILL.md file exists, is readable, "
                "and is saved as UTF-8."
            )
            return

        if not content.strip():
            await _mount_error(
                f"Skill '{normalized_name}' has an empty SKILL.md file. "
                "Add instructions to the file before invoking."
            )
            return

        envelope = build_skill_invocation_envelope(cached, content, args)

        await self._mount_message(
            SkillMessage(
                skill_name=cached["name"],
                description=str(cached.get("description", "")),
                source=str(cached.get("source", "")),
                body=content,
                args=args,
            )
        )
        await self._send_to_agent(
            envelope.prompt,
            message_kwargs=envelope.message_kwargs,
        )

    async def _handle_skill_command(self, command: str) -> None:
        """Handle a `/skill:<name>` command by loading and invoking a skill.

        Args:
            command: The full command string (e.g., `/skill:web-research find X`).
        """
        from deepagents_cli.command_registry import parse_skill_command

        skill_name, args = parse_skill_command(command)
        await self._invoke_skill(skill_name, args, command=command)

    async def _has_conversation_messages(self) -> bool:
        """Check whether the current thread has at least one human message.

        Returns:
            `True` if the conversation contains a `HumanMessage`, `False`
            otherwise. On transient errors (network, corrupt state) returns
            `True` so that `/remember` is not blocked with a misleading
            "nothing to remember" message.
        """
        if not self._agent:
            return False
        try:
            from langchain_core.messages import HumanMessage

            config: RunnableConfig = {
                "configurable": {"thread_id": self._lc_thread_id},
            }
            state = await self._agent.aget_state(config)
            if not state or not state.values:
                return False
            messages = state.values.get("messages", [])
            return any(isinstance(m, HumanMessage) for m in messages)
        except Exception:
            logger.warning(
                "Failed to check conversation messages; allowing /remember to proceed",
                exc_info=True,
            )
            return True

    async def _get_conversation_token_count(self) -> int | None:
        """Return the approximate conversation-only token count.

        Returns:
            Token count as an integer, or `None` if state is unavailable.
        """
        if not self._agent:
            return None
        try:
            from langchain_core.messages.utils import (
                count_tokens_approximately,
            )

            config: RunnableConfig = {
                "configurable": {"thread_id": self._lc_thread_id},
            }
            state = await self._agent.aget_state(config)
            if not state or not state.values:
                return None
            messages = state.values.get("messages", [])
            if not messages:
                return None
            return count_tokens_approximately(messages)
        except Exception:  # best-effort for /tokens display
            logger.debug("Failed to retrieve conversation token count", exc_info=True)
            return None

    def _resolve_offload_budget_str(self) -> str | None:
        """Resolve the offload retention budget as a human-readable string.

        Instantiates a model and computes summarization defaults, so this is
        not a trivial accessor.

        Returns:
            A string like `"20.0K (10% of 200.0K)"` or
            `"last 6 messages"`, or `None` if the budget cannot be determined.
        """
        from deepagents_cli.config import create_model, settings

        try:
            from deepagents.middleware.summarization import (
                compute_summarization_defaults,
            )

            model_spec = f"{settings.model_provider}:{settings.model_name}"
            result = create_model(
                model_spec,
                profile_overrides=self._profile_override,
            )
            defaults = compute_summarization_defaults(result.model)
            from deepagents_cli.offload import format_offload_limit

            return format_offload_limit(
                defaults["keep"],
                settings.model_context_limit,
            )
        except Exception:  # best-effort for /tokens display
            logger.debug("Failed to compute offload budget string", exc_info=True)
            return None

    async def _handle_offload(self) -> None:
        """Offload older messages to free context window space."""
        from deepagents_cli.config import settings
        from deepagents_cli.offload import (
            OffloadModelError,
            OffloadThresholdNotMet,
            perform_offload,
        )

        if not self._agent or not self._lc_thread_id:
            await self._mount_message(
                AppMessage("Nothing to offload \u2014 start a conversation first")
            )
            return

        if self._agent_running:
            await self._mount_message(
                AppMessage("Cannot offload while agent is running")
            )
            return

        config: RunnableConfig = {"configurable": {"thread_id": self._lc_thread_id}}

        try:
            state_values = await self._get_thread_state_values(self._lc_thread_id)
        except Exception as exc:  # noqa: BLE001
            await self._mount_message(ErrorMessage(f"Failed to read state: {exc}"))
            return

        if not state_values:
            await self._mount_message(
                AppMessage("Nothing to offload \u2014 start a conversation first")
            )
            return

        # Prevent concurrent user input while offload modifies state
        self._agent_running = True
        try:
            from deepagents_cli.hooks import dispatch_hook

            await dispatch_hook("context.offload", {})
            # Keep old hook name for backward compatibility
            await dispatch_hook("context.compact", {})
            await self._set_spinner("Offloading")

            result = await perform_offload(
                messages=state_values.get("messages", []),
                prior_event=state_values.get("_summarization_event"),
                thread_id=self._lc_thread_id,
                model_spec=(f"{settings.model_provider}:{settings.model_name}"),
                profile_overrides=self._profile_override,
                context_limit=settings.model_context_limit,
                total_context_tokens=self._context_tokens,
                backend=self._backend,
            )

            if isinstance(result, OffloadThresholdNotMet):
                conv_str = format_token_count(result.conversation_tokens)
                if (
                    result.total_context_tokens > 0
                    and result.context_limit is not None
                    and result.total_context_tokens > result.context_limit
                ):
                    total_str = format_token_count(
                        result.total_context_tokens,
                    )
                    await self._mount_message(
                        AppMessage(
                            f"Offload threshold not met \u2014 conversation "
                            f"is only ~{conv_str} tokens.\n\n"
                            f"The remaining context "
                            f"({total_str} tokens) is system overhead "
                            f"that can't be offloaded.\n\n"
                            f"Use /tokens for a full breakdown."
                        )
                    )
                else:
                    await self._mount_message(
                        AppMessage(
                            f"Offload threshold not met \u2014 conversation "
                            f"(~{conv_str} tokens) is within the "
                            f"retention budget "
                            f"({result.budget_str}).\n\n"
                            f"Use /tokens for a full breakdown."
                        )
                    )
                return

            # OffloadResult — success
            if result.offload_warning:
                await self._mount_message(ErrorMessage(result.offload_warning))

            if remote := self._remote_agent():
                await remote.aensure_thread(config)  # ty: ignore[invalid-argument-type]

            await self._agent.aupdate_state(
                config, {"_summarization_event": result.new_event}
            )

            before = format_token_count(result.tokens_before)
            after = format_token_count(result.tokens_after)
            await self._mount_message(
                AppMessage(
                    f"Offloaded {result.messages_offloaded} older messages, "
                    f"freeing up context window space.\n"
                    f"Context: {before} \u2192 {after} tokens "
                    f"({result.pct_decrease}% decrease), "
                    f"{result.messages_kept} messages kept."
                )
            )

            self._on_tokens_update(result.tokens_after)
            from deepagents_cli.textual_adapter import _persist_context_tokens

            await _persist_context_tokens(self._agent, config, result.tokens_after)

        except OffloadModelError as exc:
            logger.warning("Offload model creation failed: %s", exc, exc_info=True)
            await self._mount_message(ErrorMessage(str(exc)))
        except Exception as exc:  # surface offload errors to user
            logger.exception("Offload failed")
            await self._mount_message(ErrorMessage(f"Offload failed: {exc}"))
        finally:
            self._agent_running = False
            try:
                await self._set_spinner(None)
            except Exception:  # best-effort spinner cleanup
                logger.exception("Failed to dismiss spinner after offload")

    async def _handle_user_message(self, message: str) -> None:
        """Handle a user message to send to the agent.

        Args:
            message: The user's message
        """
        # Mount the user message
        await self._mount_message(UserMessage(message))
        await self._send_to_agent(message)

    async def _send_to_agent(
        self,
        message: str,
        *,
        message_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Send a message to the agent and start execution.

        This is the low-level send path. It does NOT mount any widget — the
        caller is responsible for mounting the appropriate visual representation
        (e.g., `UserMessage`, `SkillMessage`) before calling this method.

        Args:
            message: The prompt to send to the agent.
            message_kwargs: Extra fields merged into the stream input message
                dict (e.g., `additional_kwargs` for skill metadata).
        """
        # Anchor to bottom so streaming response stays visible
        with suppress(NoMatches, ScreenStackError):
            self.query_one("#chat", VerticalScroll).anchor()

        # Check if agent is available
        if self._agent and self._ui_adapter and self._session_state:
            self._agent_running = True

            if self._chat_input:
                self._chat_input.set_cursor_active(active=False)

            # Use run_worker to avoid blocking the main event loop
            # This allows the UI to remain responsive during agent execution
            self._agent_worker = self.run_worker(
                self._run_agent_task(message, message_kwargs=message_kwargs),
                exclusive=False,
            )
        elif self._server_startup_error:
            await self._mount_message(
                ErrorMessage(f"Server failed to start: {self._server_startup_error}")
            )
        else:
            await self._mount_message(
                AppMessage("Agent not configured for this session.")
            )

    async def _run_agent_task(
        self,
        message: str,
        *,
        message_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Run the agent task in a background worker.

        This runs in a Textual worker so the main event loop stays responsive.

        Args:
            message: The prompt to send to the agent.
            message_kwargs: Extra fields merged into the stream input message
                dict (e.g., `additional_kwargs` for skill metadata).
        """
        # Caller ensures _ui_adapter is set (checked in _handle_user_message)
        if self._ui_adapter is None:
            return
        from deepagents_cli.textual_adapter import execute_task_textual

        # Create the stats object up-front and store on the app so
        # exit() can merge it synchronously if the worker is cancelled
        # before this method can return (e.g. Ctrl+D during HITL).
        turn_stats = SessionStats()
        self._inflight_turn_stats = turn_stats
        self._inflight_turn_start = time.monotonic()
        try:
            await execute_task_textual(
                user_input=message,
                agent=self._agent,
                assistant_id=self._assistant_id,
                session_state=self._session_state,
                adapter=self._ui_adapter,
                backend=self._backend,
                image_tracker=self._image_tracker,
                sandbox_type=self._sandbox_type,
                message_kwargs=message_kwargs,
                context=CLIContext(
                    model=self._model_override,
                    model_params=self._model_params_override or {},
                ),
                turn_stats=turn_stats,
            )
        except Exception as e:  # Resilient tool rendering
            logger.exception("Agent execution failed")
            # Ensure any in-flight tool calls don't remain stuck in "Running..."
            # when streaming aborts before tool results arrive.
            if self._ui_adapter:
                self._ui_adapter.finalize_pending_tools_with_error(f"Agent error: {e}")
            try:
                await self._mount_message(ErrorMessage(f"Agent error: {e}"))
            except Exception:
                logger.debug(
                    "Could not mount error message (app closing?)", exc_info=True
                )
        finally:
            # Merge turn stats before cleanup — _cleanup_agent_task may raise
            # during teardown (widget removal on a torn-down DOM), and stats
            # should ideally be captured regardless.
            # exit() clears _inflight_turn_stats when it merges, so
            # checking for None prevents double-counting.
            if self._inflight_turn_stats is not None:
                self._session_stats.merge(turn_stats)
                self._inflight_turn_stats = None
            await self._cleanup_agent_task()

    async def _process_next_from_queue(self) -> None:
        """Process the next message from the queue if any exist.

        Dequeues and processes the next pending message in FIFO order.
        Uses the `_processing_pending` flag to prevent reentrant execution.
        """
        if self._processing_pending or not self._pending_messages or self._exit:
            return

        self._processing_pending = True
        try:
            msg = self._pending_messages.popleft()

            # Remove the ephemeral queued-message widget
            if self._queued_widgets:
                widget = self._queued_widgets.popleft()
                await widget.remove()

            await self._process_message(msg.text, msg.mode)
        except Exception:
            logger.exception("Failed to process queued message")
            await self._mount_message(
                ErrorMessage(f"Failed to process queued message: {msg.text[:60]}")
            )
        finally:
            self._processing_pending = False

        # Command mode messages complete synchronously without spawning
        # a worker, so cleanup won't fire again. Continue draining the
        # queue if no worker was started.
        busy = self._agent_running or self._shell_running
        if not busy and self._pending_messages:
            await self._process_next_from_queue()

    async def _cleanup_agent_task(self) -> None:
        """Clean up after agent task completes or is cancelled."""
        self._agent_running = False
        self._agent_worker = None

        # Remove spinner if present
        await self._set_spinner(None)

        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)

        # Ensure token display is restored (in case of early cancellation).
        # Pass the cached approximate flag so an interrupted "+" isn't clobbered.
        self._show_tokens(approximate=self._tokens_approximate)

        # Agent-executed commands and tools can mutate repo state (e.g. git
        # checkout inside an execute call), so refresh the footer on turn end.
        self._schedule_git_branch_refresh()

        try:
            await self._maybe_drain_deferred()
        except Exception:
            logger.exception("Failed to drain deferred actions during agent cleanup")
            with suppress(Exception):
                await self._mount_message(
                    ErrorMessage(
                        "A deferred action failed after task completion. "
                        "You may need to retry the operation."
                    )
                )

        # Process next message from queue if any
        if not self._startup_sequence_running:
            await self._process_next_from_queue()

    @staticmethod
    def _convert_messages_to_data(messages: list[Any]) -> list[MessageData]:
        """Convert LangChain messages into lightweight `MessageData` objects.

        This is a pure function with zero DOM operations. Tool call matching
        happens here: `ToolMessage` results are matched by `tool_call_id` and
        stored directly on the corresponding `MessageData`.

        Args:
            messages: LangChain message objects from a thread checkpoint.

        Returns:
            Ordered list of `MessageData` ready for `MessageStore.bulk_load`.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result: list[MessageData] = []
        # Maps tool_call_id -> index into result list
        pending_tool_indices: dict[str, int] = {}

        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if content.startswith("[SYSTEM]"):
                    continue

                # Detect skill invocations persisted via additional_kwargs
                skill_meta = (msg.additional_kwargs or {}).get("__skill")
                if isinstance(skill_meta, dict) and skill_meta.get("name"):
                    result.append(
                        MessageData(
                            type=MessageType.SKILL,
                            content="",
                            skill_name=skill_meta["name"],
                            skill_description=str(skill_meta.get("description", "")),
                            skill_source=str(skill_meta.get("source", "")),
                            skill_args=str(skill_meta.get("args", "")),
                            skill_body=content,
                        )
                    )
                else:
                    result.append(MessageData(type=MessageType.USER, content=content))

            elif isinstance(msg, AIMessage):
                # Extract text content
                content = msg.content
                text = ""
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                    text = text.strip()

                if text:
                    result.append(MessageData(type=MessageType.ASSISTANT, content=text))

                # Track tool calls for later matching
                for tc in getattr(msg, "tool_calls", []):
                    tc_id = tc.get("id")
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    data = MessageData(
                        type=MessageType.TOOL,
                        content="",
                        tool_name=name,
                        tool_args=args,
                        tool_status=ToolStatus.PENDING,
                    )
                    result.append(data)
                    if tc_id:
                        pending_tool_indices[tc_id] = len(result) - 1
                    else:
                        data.tool_status = ToolStatus.REJECTED

            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in pending_tool_indices:
                    idx = pending_tool_indices.pop(tc_id)
                    data = result[idx]
                    status = getattr(msg, "status", "success")
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    if status == "success":
                        data.tool_status = ToolStatus.SUCCESS
                    else:
                        data.tool_status = ToolStatus.ERROR
                    data.tool_output = content
                else:
                    logger.debug(
                        "ToolMessage with tool_call_id=%r could not be "
                        "matched to a pending tool call",
                        tc_id,
                    )

            else:
                logger.debug(
                    "Skipping unsupported message type %s during history conversion",
                    type(msg).__name__,
                )

        # Mark unmatched tool calls as rejected
        for idx in pending_tool_indices.values():
            result[idx].tool_status = ToolStatus.REJECTED

        return result

    async def _get_thread_state_values(self, thread_id: str) -> dict[str, Any]:
        """Fetch thread state values, with remote checkpointer fallback.

        In server mode the LangGraph dev server can report an empty thread state
        after a restart even when checkpoints exist on disk. When that happens,
        read the latest checkpoint directly so resumed threads can still load
        history and offload correctly.

        Args:
            thread_id: Thread ID to fetch from checkpoint storage.

        Returns:
            Thread state values keyed by channel name. Returns an empty dict
                when no checkpointed values are available.
        """
        if not self._agent:
            return {}

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = await self._agent.aget_state(config)

        values: dict[str, Any] = {}
        if state and state.values:
            values = dict(state.values)

        messages = values.get("messages")
        if isinstance(messages, list) and messages:
            return values
        if not self._remote_agent():
            return values

        logger.debug(
            "Remote state empty for thread %s; falling back to local checkpointer",
            thread_id,
        )
        fallback_values = await self._read_channel_values_from_checkpointer(thread_id)
        fallback_messages = fallback_values.get("messages")
        if isinstance(fallback_messages, list) and fallback_messages:
            values["messages"] = fallback_messages
        if (
            values.get("_summarization_event") is None
            and "_summarization_event" in fallback_values
        ):
            values["_summarization_event"] = fallback_values["_summarization_event"]
        if (
            values.get("_context_tokens") is None
            and "_context_tokens" in fallback_values
        ):
            values["_context_tokens"] = fallback_values["_context_tokens"]
        return values

    async def _fetch_thread_history_data(self, thread_id: str) -> _ThreadHistoryPayload:
        """Fetch and convert stored messages for a thread.

        In server mode the LangGraph dev server starts with an empty thread
        store, so `aget_state` via the HTTP API returns no messages even when
        checkpoints exist on disk. We fall back to reading the SQLite
        checkpointer directly to guarantee resumed threads load their history.

        Args:
            thread_id: Thread ID to fetch from checkpoint storage.

        Returns:
            Payload containing converted message data and the persisted
            context-token count.
        """
        state_values = await self._get_thread_state_values(thread_id)
        raw_tokens = state_values.get("_context_tokens")
        context_tokens = (
            raw_tokens if isinstance(raw_tokens, int) and raw_tokens >= 0 else 0
        )
        messages = state_values.get("messages", [])

        if not messages:
            return _ThreadHistoryPayload([], context_tokens)

        # Server mode / direct checkpointer may return dicts; convert to
        # LangChain message objects so _convert_messages_to_data works.
        # `any(...)` guards against heterogeneous lists where only some
        # elements are serialized.
        if any(isinstance(m, dict) for m in messages):
            from langchain_core.messages.utils import convert_to_messages

            messages = convert_to_messages(messages)

        # Offload conversion so large histories don't block the UI loop.
        data = await asyncio.to_thread(self._convert_messages_to_data, messages)
        return _ThreadHistoryPayload(data, context_tokens)

    @staticmethod
    async def _read_channel_values_from_checkpointer(thread_id: str) -> dict[str, Any]:
        """Read checkpoint channel values directly from the SQLite checkpointer.

        Args:
            thread_id: Thread ID to look up.

        Returns:
            Channel values from the latest checkpoint, or an empty dict on
                failure.
        """
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            from deepagents_cli.sessions import get_db_path

            db_path = str(get_db_path())
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
                tup = await saver.aget_tuple(config)
                if tup and tup.checkpoint:
                    channel_values = tup.checkpoint.get("channel_values", {})
                    if isinstance(channel_values, dict):
                        return dict(channel_values)
        except (ImportError, OSError) as exc:
            logger.warning(
                "Failed to read checkpointer directly for %s: %s",
                thread_id,
                exc,
            )
        except Exception:
            logger.warning(
                "Unexpected error reading checkpointer for %s",
                thread_id,
                exc_info=True,
            )
        return {}

    async def _upgrade_thread_message_link(
        self,
        widget: AppMessage,
        *,
        prefix: str,
        thread_id: str,
    ) -> None:
        """Upgrade a plain thread message to a linked one when URL resolves.

        Args:
            widget: The already-mounted app message.
            prefix: Text prefix before thread ID.
            thread_id: Thread ID to resolve.
        """
        try:
            thread_msg = await self._build_thread_message(prefix, thread_id)
            if not isinstance(thread_msg, Content):
                logger.debug(
                    "Skipping thread link upgrade for %s: URL did not resolve",
                    thread_id,
                )
                return
            if widget.parent is None:
                logger.debug(
                    "Skipping thread link upgrade for %s: widget no longer mounted",
                    thread_id,
                )
                return
            # Keep serialized content in sync with the rendered content.
            widget._content = thread_msg
            widget.update(thread_msg)
        except Exception:
            logger.warning(
                "Failed to upgrade thread message link for %s",
                thread_id,
                exc_info=True,
            )

    def _schedule_thread_message_link(
        self,
        widget: AppMessage,
        *,
        prefix: str,
        thread_id: str,
    ) -> None:
        """Schedule thread URL link resolution and apply updates in the background.

        Args:
            widget: The message widget to update.
            prefix: Text prefix before thread ID.
            thread_id: Thread ID to resolve.
        """
        self.run_worker(
            self._upgrade_thread_message_link(
                widget,
                prefix=prefix,
                thread_id=thread_id,
            ),
            exclusive=False,
        )

    async def _load_thread_history(
        self,
        *,
        thread_id: str | None = None,
        preloaded_payload: _ThreadHistoryPayload | None = None,
    ) -> None:
        """Load and render message history when resuming a thread.

        When `preloaded_payload` is provided (e.g., from `_resume_thread`),
        this reuses that data. Otherwise, it fetches checkpoint state from the
        agent and converts stored messages into lightweight `MessageData`
        objects. The method then bulk-loads into the `MessageStore` and mounts
        only the last `WINDOW_SIZE` widgets to reduce DOM operations on large
        threads.

        Args:
            thread_id: Optional explicit thread ID to load.

                Defaults to current.
            preloaded_payload: Optional pre-fetched history payload for the
                thread.
        """
        history_thread_id = thread_id or self._lc_thread_id
        if not history_thread_id:
            logger.debug("Skipping history load: no thread ID available")
            return
        if preloaded_payload is None and not self._agent:
            logger.debug(
                "Skipping history load for %s: no active agent and no preloaded data",
                history_thread_id,
            )
            return

        try:
            # Fetch + convert, or reuse preloaded payload on thread switch.
            payload = (
                preloaded_payload
                if preloaded_payload is not None
                else await self._fetch_thread_history_data(history_thread_id)
            )
            if not payload.messages:
                return

            # Seed token cache from persisted state
            if payload.context_tokens > 0:
                self._on_tokens_update(payload.context_tokens)

            # 3. Bulk load into store (sets visible window)
            _archived, visible = self._message_store.bulk_load(payload.messages)

            # 5. Cache container ref (single query)
            try:
                messages_container = self.query_one("#messages", Container)
            except NoMatches:
                return

            # 6-7. Create and mount only visible widgets (max WINDOW_SIZE)
            widgets = [msg_data.to_widget() for msg_data in visible]
            if widgets:
                await messages_container.mount(*widgets)

            # 8. Render content for AssistantMessage after mount
            assistant_updates = [
                widget.set_content(msg_data.content)
                for widget, msg_data in zip(widgets, visible, strict=False)
                if isinstance(widget, AssistantMessage) and msg_data.content
            ]
            if assistant_updates:
                assistant_results = await asyncio.gather(
                    *assistant_updates,
                    return_exceptions=True,
                )
                for error in assistant_results:
                    if isinstance(error, Exception):
                        logger.warning(
                            "Failed to render assistant history message for %s: %s",
                            history_thread_id,
                            error,
                        )

            # 9. Add footer immediately and resolve link asynchronously
            thread_msg_widget = AppMessage(f"Resumed thread: {history_thread_id}")
            await self._mount_message(thread_msg_widget)
            self._schedule_thread_message_link(
                thread_msg_widget,
                prefix="Resumed thread",
                thread_id=history_thread_id,
            )

            # 10. Scroll once to bottom after history loads
            def scroll_to_end() -> None:
                with suppress(NoMatches):
                    chat = self.query_one("#chat", VerticalScroll)
                    chat.scroll_end(animate=False, immediate=True)

            self.set_timer(0.1, scroll_to_end)

        except Exception as e:  # Resilient history loading
            logger.exception(
                "Failed to load thread history for %s",
                history_thread_id,
            )
            await self._mount_message(AppMessage(f"Could not load history: {e}"))

    async def _mount_message(
        self, widget: Static | AssistantMessage | ToolCallMessage | SkillMessage
    ) -> None:
        """Mount a message widget to the messages area.

        This method also stores the message data and handles pruning
        when the widget count exceeds the maximum.

        If the ``#messages`` container is not present (e.g. the screen has
        been torn down during an interruption), the call is silently skipped
        to avoid cascading `NoMatches` errors.

        Args:
            widget: The message widget to mount
        """
        try:
            messages = self.query_one("#messages", Container)
        except NoMatches:
            return

        # During shutdown (e.g. Ctrl+D mid-stream) the container may still
        # be in the DOM tree but already detached, so mount() would raise
        # MountError. Bail out silently — the app is exiting anyway.
        if not messages.is_attached:
            return

        # Store message data for virtualization
        message_data = MessageData.from_widget(widget)
        # Ensure the widget's DOM id matches the store id so that
        # features like click-to-show-timestamp can look it up.
        if not widget.id:
            widget.id = message_data.id
        self._message_store.append(message_data)

        # Queued-message widgets must always stay at the bottom so they
        # remain visually anchored below the current agent response.
        if isinstance(widget, QueuedUserMessage):
            await messages.mount(widget)
        else:
            await self._mount_before_queued(messages, widget)

        # Prune old widgets if window exceeded
        await self._prune_old_messages()

        # Scroll to keep input bar visible
        try:
            input_container = self.query_one("#bottom-app-container", Container)
            input_container.scroll_visible()
        except NoMatches:
            pass

    async def _prune_old_messages(self) -> None:
        """Prune oldest message widgets if we exceed the window size.

        This removes widgets from the DOM but keeps data in MessageStore
        for potential re-hydration when scrolling up.
        """
        if not self._message_store.window_exceeded():
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping pruning: #messages container not found")
            return

        to_prune = self._message_store.get_messages_to_prune()
        if not to_prune:
            return

        pruned_ids: list[str] = []
        for msg_data in to_prune:
            try:
                widget = messages_container.query_one(f"#{msg_data.id}")
                await widget.remove()
                pruned_ids.append(msg_data.id)
            except NoMatches:
                # Widget not found -- do NOT mark as pruned to avoid
                # desyncing the store from the actual DOM state
                logger.debug(
                    "Widget %s not found during pruning, skipping",
                    msg_data.id,
                )

        if pruned_ids:
            self._message_store.mark_pruned(pruned_ids)

    def _set_active_message(self, message_id: str | None) -> None:
        """Set the active streaming message (won't be pruned).

        Args:
            message_id: The ID of the active message, or None to clear.
        """
        self._message_store.set_active_message(message_id)

    def _sync_message_content(self, message_id: str, content: str) -> None:
        """Sync final message content back to the store after streaming.

        Called when streaming finishes so the store holds the full text
        instead of the empty string captured at mount time.

        Args:
            message_id: The ID of the message to update.
            content: The final content after streaming.
        """
        self._message_store.update_message(
            message_id,
            content=content,
            is_streaming=False,
        )

    async def _clear_messages(self) -> None:
        """Clear the messages area and message store."""
        # Clear the message store first
        self._message_store.clear()
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            logger.warning(
                "Messages container (#messages) not found during clear; "
                "UI may be out of sync with message store"
            )

    def _pop_last_queued_message(self) -> None:
        """Remove the most recently queued message (LIFO).

        If the chat input is empty the evicted text is restored there so the
        user can edit and re-submit. Otherwise the message is discarded. The
        toast message distinguishes between the two outcomes.

        Caller must ensure `_pending_messages` is non-empty. A defensive guard
        is included in case of async TOCTOU races.
        """
        if not self._pending_messages:
            return
        msg = self._pending_messages.pop()
        if self._queued_widgets:
            widget = self._queued_widgets.pop()
            widget.remove()
        else:
            logger.warning(
                "Queued-widget deque empty while pending-messages was not; "
                "widget/message tracking may be out of sync"
            )

        if not self._chat_input:
            logger.warning(
                "Chat input unavailable during queue pop; "
                "message text cannot be restored: %s",
                msg.text[:60],
            )
            self.notify("Queued message discarded", timeout=2)
            return

        if not self._chat_input.value.strip():
            self._chat_input.value = msg.text
            self.notify("Queued message moved to input", timeout=2)
        else:
            self.notify("Queued message discarded (input not empty)", timeout=3)

    def _discard_queue(self) -> None:
        """Clear pending messages, deferred actions, and queued widgets."""
        self._pending_messages.clear()
        for w in self._queued_widgets:
            w.remove()
        self._queued_widgets.clear()
        self._deferred_actions.clear()

    def _defer_action(self, action: DeferredAction) -> None:
        """Queue a deferred action, replacing any existing action of the same kind.

        Last-write-wins: if the user selects a model twice while busy, only the
        final selection runs.

        Args:
            action: The deferred action to queue.
        """
        self._deferred_actions = [
            a for a in self._deferred_actions if a.kind != action.kind
        ]
        self._deferred_actions.append(action)

    async def _maybe_drain_deferred(self) -> None:
        """Drain deferred actions unless startup sequencing is still in progress."""
        if not self._connecting and not self._startup_sequence_running:
            await self._drain_deferred_actions()

    async def _drain_deferred_actions(self) -> None:
        """Execute deferred actions queued while busy (e.g. model/thread switch)."""
        while self._deferred_actions:
            action = self._deferred_actions.pop(0)
            try:
                await action.execute()
            except Exception:
                logger.exception(
                    "Failed to execute deferred action %r (callable=%r)",
                    action.kind,
                    action.execute,
                )
                label = action.kind.replace("_", " ")
                with suppress(Exception):
                    await self._mount_message(
                        ErrorMessage(
                            f"Deferred {label} failed unexpectedly. "
                            "You may need to retry the operation."
                        )
                    )

    def _cancel_worker(self, worker: Worker[None] | None) -> None:
        """Discard the message queue and cancel an active worker.

        Args:
            worker: The worker to cancel.
        """
        self._discard_queue()
        if worker is not None:
            worker.cancel()

    def action_quit_or_interrupt(self) -> None:
        """Handle Ctrl+C - interrupt agent, reject approval, or quit on double press.

        Priority order:
        1. If shell command is running, kill it
        2. If approval menu is active, reject it
        3. If agent is running, interrupt it (preserve input)
        4. If double press (quit_pending), quit
        5. Otherwise show quit hint
        """
        # If shell command is running, cancel the worker
        if self._shell_running and self._shell_worker:
            self._cancel_worker(self._shell_worker)
            self._quit_pending = False
            return

        # If approval menu is active, reject it before cancelling the agent worker.
        # During HITL the agent worker remains active while awaiting approval,
        # so this must be checked before the worker cancellation branch to
        # avoid leaving a stale approval widget interactive after interruption.
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            self._quit_pending = False
            return

        # If ask_user menu is active, cancel it before cancelling the agent
        # worker, following the same pattern as the approval widget above.
        if self._pending_ask_user_widget:
            self._pending_ask_user_widget.action_cancel()
            self._quit_pending = False
            return

        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._cancel_worker(self._agent_worker)
            self._quit_pending = False
            return

        # Double Ctrl+C to quit
        if self._quit_pending:
            self.exit()
        else:
            self._arm_quit_pending("Ctrl+C")

    def _arm_quit_pending(self, shortcut: str) -> None:
        """Set the pending-quit flag and show a matching hint.

        Args:
            shortcut: The key chord to show in the quit hint.
        """
        self._quit_pending = True
        quit_timeout = 3
        self.notify(
            f"Press {shortcut} again to quit", timeout=quit_timeout, markup=False
        )
        self.set_timer(quit_timeout, lambda: setattr(self, "_quit_pending", False))

    def action_interrupt(self) -> None:
        """Handle escape key.

        Priority order:
        1. If modal screen is active, dismiss it
        2. If completion popup is open, dismiss it
        3. If input is in command/shell mode, exit to normal mode
        4. If shell command is running, kill it
        5. If approval menu is active, reject it
        6. If ask-user menu is active, cancel it
        7. If queued messages exist, pop the last one (LIFO)
        8. If agent is running, interrupt it
        """
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        if (
            isinstance(self.screen, ThreadSelectorScreen)
            and self.screen.is_delete_confirmation_open
        ):
            self.screen.action_cancel()
            return

        # If a modal screen is active, let it cancel itself (so it can
        # restore state, e.g. the theme selector reverts the previewed theme).
        # Fall back to a plain dismiss for modals without action_cancel.
        if isinstance(self.screen, ModalScreen):
            cancel = getattr(self.screen, "action_cancel", None)
            if cancel is not None:
                cancel()
            else:
                self.screen.dismiss(None)
            return

        # Close completion popup or exit slash/shell command mode
        if self._chat_input:
            if self._chat_input.dismiss_completion():
                return
            if self._chat_input.exit_mode():
                return

        # If shell command is running, cancel the worker
        if self._shell_running and self._shell_worker:
            self._cancel_worker(self._shell_worker)
            return

        # If approval menu is active, reject it before cancelling the agent worker.
        # During HITL the agent worker remains active while awaiting approval,
        # so this must be checked before the worker cancellation branch to
        # avoid leaving a stale approval widget interactive after interruption.
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            return

        # If ask_user menu is active, cancel it before cancelling the agent
        # worker, following the same pattern as the approval widget above.
        if self._pending_ask_user_widget:
            self._pending_ask_user_widget.action_cancel()
            return

        # If queued messages exist, pop the last one (LIFO) instead of
        # interrupting the agent.  This lets the user retract queued messages
        # one at a time; once the queue is empty the next ESC will interrupt.
        if self._pending_messages:
            self._pop_last_queued_message()
            return

        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._cancel_worker(self._agent_worker)
            return

    def action_quit_app(self) -> None:
        """Handle quit action (Ctrl+D)."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        if isinstance(self.screen, ThreadSelectorScreen):
            self.screen.action_delete_thread()
            return
        if isinstance(self.screen, DeleteThreadConfirmScreen):
            if self._quit_pending:
                self.exit()
                return
            self._arm_quit_pending("Ctrl+D")
            return
        self.exit()

    def exit(
        self,
        result: Any = None,  # noqa: ANN401  # Dynamic LangGraph stream result type
        return_code: int = 0,
        message: Any = None,  # noqa: ANN401  # Dynamic LangGraph message type
    ) -> None:
        """Exit the app, restoring iTerm2 cursor guide if applicable.

        Overrides parent to restore iTerm2's cursor guide before Textual's
        cleanup. The atexit handler serves as a fallback for abnormal
        termination.

        Args:
            result: Return value passed to the app runner.
            return_code: Exit code (non-zero for errors).
            message: Optional message to display on exit.
        """
        # Merge in-flight turn stats before any cleanup that might raise.
        # When the agent worker is cancelled (e.g. Ctrl+D during a pending tool
        # call), the worker's finally block will see _inflight_turn_stats is
        # already None and skip the merge.
        inflight = self._inflight_turn_stats
        if inflight is not None:
            self._inflight_turn_stats = None
            if not inflight.wall_time_seconds:
                inflight.wall_time_seconds = (
                    time.monotonic() - self._inflight_turn_start
                )
            self._session_stats.merge(inflight)

        # Discard queued messages so _cleanup_agent_task won't try to
        # process them after the event loop is torn down, and cancel
        # active workers so their subprocesses are terminated
        # (SIGTERM → SIGKILL) instead of being orphaned.
        self._discard_queue()

        if self._shell_running and self._shell_worker:
            self._shell_worker.cancel()
        if self._agent_running and self._agent_worker:
            self._agent_worker.cancel()
        if self._git_branch_refresh_task is not None:
            self._git_branch_refresh_task.cancel()

        # Dispatch synchronously — the event loop is about to be torn down by
        # super().exit(), so an async task would never complete.
        from deepagents_cli.hooks import _dispatch_hook_sync, _load_hooks

        hooks = _load_hooks()
        if hooks:
            payload = json.dumps(
                {
                    "event": "session.end",
                    "thread_id": getattr(self, "_lc_thread_id", ""),
                }
            ).encode()
            _dispatch_hook_sync("session.end", payload, hooks)

        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
        super().exit(result=result, return_code=return_code, message=message)

    def action_toggle_auto_approve(self) -> None:
        """Toggle auto-approve mode for the current session.

        When enabled, all tool calls (shell execution, file writes/edits,
        web search, URL fetch) run without prompting. Updates the status
        bar indicator and session state.
        """
        from deepagents_cli.widgets.agent_selector import AgentSelectorScreen
        from deepagents_cli.widgets.notification_settings import (
            NotificationSettingsScreen,
        )
        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen
        from deepagents_cli.widgets.update_available import UpdateAvailableScreen

        if isinstance(self.screen, ThreadSelectorScreen):
            self.screen.action_focus_previous_filter()
            return
        if isinstance(self.screen, (ThemeSelectorScreen, AgentSelectorScreen)):
            self.screen.action_cursor_up()
            return
        if isinstance(self.screen, NotificationSettingsScreen):
            self.screen.focus_previous()
            return
        if isinstance(self.screen, UpdateAvailableScreen):
            self.screen.action_move_up()
            return
        # shift+tab is reused for navigation inside modal screens (e.g.
        # ModelSelectorScreen); skip the toggle so it doesn't fire through.
        if isinstance(self.screen, ModalScreen):
            return
        # Delegate shift+tab to ask_user navigation when interview is active.
        if self._pending_ask_user_widget is not None:
            self._pending_ask_user_widget.action_previous_question()
            return
        self._auto_approve = not self._auto_approve
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)
        if self._session_state:
            self._session_state.auto_approve = self._auto_approve

    def action_toggle_tool_output(self) -> None:
        """Toggle expand/collapse of the most recent tool output or skill body."""
        # Try skill messages first (most recent collapsible content)
        with suppress(NoMatches):
            skill_messages = list(self.query(SkillMessage))
            for skill_msg in reversed(skill_messages):
                if skill_msg._stripped_body.strip():
                    skill_msg.toggle_body()
                    return
        # Fall back to tool messages with output
        with suppress(NoMatches):
            tool_messages = list(self.query(ToolCallMessage))
            for tool_msg in reversed(tool_messages):
                if tool_msg.has_output:
                    tool_msg.toggle_output()
                    return

    # Approval menu action handlers (delegated from App-level bindings)
    # NOTE: These only activate when approval widget is pending
    # AND input is not focused
    def action_approval_up(self) -> None:
        """Handle up arrow in approval menu."""
        # Only handle if approval is active
        # (input handles its own up for history/completion)
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_up()

    def action_approval_down(self) -> None:
        """Handle down arrow in approval menu."""
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_down()

    def action_approval_select(self) -> None:
        """Handle enter in approval menu."""
        # Only handle if approval is active AND input is not focused
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_select()

    def _is_input_focused(self) -> bool:
        """Check if the chat input (or its text area) has focus.

        Returns:
            True if the input widget has focus, False otherwise.
        """
        if not self._chat_input:
            return False
        focused = self.focused
        if focused is None:
            return False
        # Check if focused widget is the text area inside chat input
        return focused.id == "chat-input" or focused in self._chat_input.walk_children()

    def action_approval_yes(self) -> None:
        """Handle yes/1 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_approve()

    def action_approval_auto(self) -> None:
        """Handle auto/2 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_auto()

    def action_approval_no(self) -> None:
        """Handle no/3 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_approval_escape(self) -> None:
        """Handle escape in approval menu - reject."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    async def action_open_editor(self) -> None:
        """Open the current prompt text in an external editor ($VISUAL/$EDITOR)."""
        from deepagents_cli.editor import open_in_editor

        chat_input = self._chat_input
        if not chat_input or not chat_input._text_area:
            return

        current_text = chat_input._text_area.text or ""

        edited: str | None = None
        try:
            with self.suspend():
                edited = open_in_editor(current_text)
        except Exception:
            logger.warning("External editor failed", exc_info=True)
            self.notify(
                "External editor failed. Check $VISUAL/$EDITOR.",
                severity="error",
                timeout=5,
            )
            chat_input.focus_input()
            return

        if edited is not None:
            chat_input._text_area.text = edited
            lines = edited.split("\n")
            chat_input._text_area.move_cursor((len(lines) - 1, len(lines[-1])))
        chat_input.focus_input()

    def on_paste(self, event: Paste) -> None:
        """Route unfocused paste events to chat input for drag/drop reliability."""
        if not self._chat_input:
            return
        if (
            self._pending_approval_widget
            or self._pending_ask_user_widget
            or self._is_input_focused()
        ):
            return
        if self._chat_input.handle_external_paste(event.text):
            event.prevent_default()
            event.stop()

    def on_app_focus(self) -> None:
        """Restore chat input focus when the terminal regains OS focus.

        When the user opens a link via `webbrowser.open`, OS focus shifts to
        the browser. On returning to the terminal, Textual fires `AppFocus`
        (requires a terminal that supports FocusIn events). Re-focusing the chat
        input here keeps it ready for typing.
        """
        if not self._chat_input:
            return
        if isinstance(self.screen, ModalScreen):
            return
        if self._pending_approval_widget or self._pending_ask_user_widget:
            return
        self._chat_input.focus_input()

    def on_click(self, event: Click) -> None:
        """Handle clicks anywhere in the terminal.

        Clicks on registered actionable toasts open the notification
        center. The toast itself dismisses as normal; we only piggyback
        on the click. Other clicks restore focus to the chat input.
        """
        widget = event.widget
        if isinstance(widget, _Toast):
            identity = _toast_identity(widget, app=self)
            if identity is not None and self._notice_registry.is_actionable_toast(
                identity
            ):
                self.call_after_refresh(self._open_notification_center)
            return

        if not self._chat_input:
            return
        # Don't steal focus from approval or ask_user widgets
        if self._pending_approval_widget or self._pending_ask_user_widget:
            return
        self.call_after_refresh(self._chat_input.focus_input)

    def on_mouse_up(self, event: MouseUp) -> None:  # noqa: ARG002  # Textual event handler signature
        """Copy selection to clipboard on mouse release."""
        from deepagents_cli.clipboard import copy_selection_to_clipboard

        copy_selection_to_clipboard(self)

    # =========================================================================
    # Model Switching
    # =========================================================================

    async def _show_model_selector(
        self,
        *,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Show interactive model selector as a modal screen.

        Args:
            extra_kwargs: Extra constructor kwargs from `--model-params`.
        """
        from functools import partial

        from deepagents_cli.config import settings
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        def handle_result(result: tuple[str, str] | None) -> None:
            """Handle the model selector result."""
            if result is not None:
                model_spec, _ = result
                if self._agent_running or self._shell_running or self._connecting:
                    self._defer_action(
                        DeferredAction(
                            kind="model_switch",
                            execute=partial(
                                self._switch_model,
                                model_spec,
                                extra_kwargs=extra_kwargs,
                            ),
                        )
                    )
                    self.notify(
                        "Model will switch after current task completes.", timeout=3
                    )
                else:
                    self.call_later(
                        partial(
                            self._switch_model,
                            model_spec,
                            extra_kwargs=extra_kwargs,
                        )
                    )
            # Refocus input after modal closes
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ModelSelectorScreen(
            current_model=settings.model_name,
            current_provider=settings.model_provider,
            cli_profile_override=self._profile_override,
        )
        self.push_screen(screen, handle_result)

    def _register_custom_themes(self) -> None:
        """Register all custom themes (built-in LC + user-defined) with Textual."""
        for name, entry in theme.ThemeEntry.REGISTRY.items():
            if entry.custom:
                c = entry.colors
                try:
                    self.register_theme(
                        Theme(
                            name=name,
                            primary=c.primary,
                            secondary=c.secondary,
                            accent=c.accent,
                            foreground=c.foreground,
                            background=c.background,
                            surface=c.surface,
                            panel=c.panel,
                            warning=c.warning,
                            error=c.error,
                            success=c.success,
                            dark=entry.dark,
                            variables={
                                "footer-key-foreground": c.primary,
                            },
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to register theme '%s'; skipping",
                        name,
                        exc_info=True,
                    )

    async def _show_theme_selector(self) -> None:
        """Show interactive theme selector as a modal screen."""
        from deepagents_cli.widgets.theme_selector import ThemeSelectorScreen

        # Capture scroll state.  The submit handler may have already caused
        # a reflow that re-anchored to the bottom, so we save the *current*
        # offset and release the anchor to prevent further drift while the
        # modal is open.
        chat = self.query_one("#chat", VerticalScroll)
        saved_y = chat.scroll_y
        was_anchored = chat.is_anchored
        chat.release_anchor()

        def handle_result(result: str | None) -> None:
            """Handle the theme selector result."""
            if result is not None:
                self.theme = result
                self.refresh_css(animate=False)

                async def _persist() -> None:
                    try:
                        ok = await asyncio.to_thread(save_theme_preference, result)
                        if not ok:
                            self.notify(
                                "Theme applied for this session but could not"
                                " be saved. Check logs for details.",
                                severity="warning",
                                timeout=6,
                                markup=False,
                            )
                    except Exception:
                        logger.warning(
                            "Failed to persist theme preference",
                            exc_info=True,
                        )
                        self.notify(
                            "Theme applied for this session but could not"
                            " be saved. Check logs for details.",
                            severity="warning",
                            timeout=6,
                            markup=False,
                        )

                self.call_later(_persist)
            # Restore scroll position, then re-anchor if it was anchored.
            chat.scroll_to(y=saved_y, animate=False)
            if was_anchored:
                chat.anchor()
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ThemeSelectorScreen(current_theme=self.theme)
        self.push_screen(screen, handle_result)

    async def _show_agent_selector(self) -> None:
        """Show the interactive agent selector modal."""
        from deepagents_cli.agent import get_available_agent_names
        from deepagents_cli.widgets.agent_selector import AgentSelectorScreen

        agent_names = await asyncio.to_thread(get_available_agent_names)

        def handle_result(result: str | None) -> None:
            """Handle the agent selector result."""
            if result is not None and result != self._assistant_id:
                self._switch_agent(result)
            if self._chat_input:
                self._chat_input.focus_input()

        screen = AgentSelectorScreen(
            current_agent=self._assistant_id,
            agent_names=agent_names,
        )
        self.push_screen(screen, handle_result)

    def _switch_agent(self, agent_name: str) -> None:
        """Switch to a different agent and hot-restart the backing server.

        Runs guard checks (remote-server mode, mid-run, re-entry, missing
        agent directory), then kicks off `_restart_server_for_agent_swap` as
        a worker. That worker restarts the langgraph subprocess with the new
        `assistant_id` so the new agent's `AGENTS.md` is actually loaded —
        memory, skills, thread, and system prompt all align.

        Args:
            agent_name: The name of the agent to switch to.
        """
        from deepagents_cli.config import settings

        if agent_name == self._assistant_id:
            return

        if self._server_kwargs is None:
            # Remote-server mode: we don't own the subprocess, so we can't
            # restart it. Changing identity locally would leave the running
            # server's system prompt pointing at a different agent.
            self.notify(
                "Cannot switch agents against a remote server. "
                "Relaunch the CLI with -a <name> instead.",
                severity="warning",
                markup=False,
            )
            return

        if self._server_proc is None:
            if self._connecting:

                async def _deferred_switch() -> None:  # noqa: RUF029  # DeferredAction requires an awaitable; the UI mutation must stay on the main thread.
                    self._switch_agent(agent_name)

                self._defer_action(
                    DeferredAction(
                        kind="agent_switch",
                        execute=_deferred_switch,
                    )
                )
                self.notify(
                    "Agent will switch after connection completes.",
                    timeout=3,
                    markup=False,
                )
                return

            self.notify(
                "Cannot switch agents until the local server is ready.",
                severity="warning",
                markup=False,
            )
            return

        if self._agent_running or self._shell_running:
            self.notify(
                "Cannot switch agents while a task is running. "
                "Interrupt or wait for it to finish first.",
                severity="warning",
                markup=False,
            )
            return

        if self._agent_switching:
            self.notify(
                "Agent switch already in progress.",
                severity="warning",
                markup=False,
            )
            return

        try:
            agent_dir_exists = (settings.user_deepagents_dir / agent_name).is_dir()
        except OSError:
            logger.warning(
                "Could not stat agent directory for %r", agent_name, exc_info=True
            )
            agent_dir_exists = False

        if not agent_dir_exists:
            self.notify(
                f"Agent {agent_name!r} is no longer available.",
                severity="warning",
                markup=False,
            )
            return

        self._agent_switching = True
        self.run_worker(
            self._restart_server_for_agent_swap(agent_name),
            exclusive=True,
            group="agent-switch-restart",
        )

    async def _restart_server_for_agent_swap(self, agent_name: str) -> None:
        """Restart the langgraph server with a new `assistant_id`.

        Runs in three phases so failures are attributable:

        1. **UI teardown** — flip banner to connecting, clear chat, reject
            pending HITL widgets, reset the thread. Failures here notify the
            user and return early; the previous server is still alive and
            identity is untouched.
        2. **Server restart** — mutate `_assistant_id`, stage the new
            `DEEPAGENTS_CLI_SERVER_ASSISTANT_ID` env var, call
            `ServerProcess.restart()`, and rebuild the `RemoteAgent` against
            the (possibly new) server URL. A failure rolls back identity and
            posts `ServerStartFailed` because the old subprocess is dead.
        3. **Confirmation** — show "Switched to X", optional resume hint,
            persist the recent agent, and drain any messages queued during
            the swap.

        Args:
            agent_name: The name of the agent to switch to.
        """
        from deepagents_cli._env_vars import SERVER_ENV_PREFIX
        from deepagents_cli.remote_client import RemoteAgent as _RemoteAgent

        def _build_agent(url: str) -> Any:  # noqa: ANN401  # see docstring
            """Build a new `RemoteAgent` typed as `Any`.

            Returns `Any` so `self._agent`'s attribute type stays aligned
            with the permissive type the startup path assigns, avoiding a
            union that would trip call-site type checks on
            `aget_state(config)` et al.

            Args:
                url: Server base URL to point the new client at.

            Returns:
                A fresh `RemoteAgent`, exposed as `Any`.
            """
            return _RemoteAgent(url=url, graph_name="agent")

        previous_agent = self._assistant_id
        previous_thread_id = self._lc_thread_id
        server_proc = self._server_proc
        if server_proc is None:
            # Guarded in _switch_agent, but the worker runs in the next tick
            # so re-check to keep the type narrow.
            self._agent_switching = False
            return

        try:
            # Phase 1: UI teardown. A failure here does NOT mean the server
            # is gone — we notify the user and bail out with the previous
            # agent still live. Only Phase 2 escalates to ServerStartFailed.
            try:
                self._connecting = True
                self._agent = None
                try:
                    banner = self.query_one("#welcome-banner", WelcomeBanner)
                    banner.set_connecting()
                except NoMatches:
                    pass

                if self._chat_input:
                    self._chat_input.set_cursor_active(active=False)

                # Reject pending HITL prompts — they're bound to the old
                # server's in-flight request and won't be resolved after
                # restart. Wrap each call narrowly so a widget-cleanup bug
                # can't abort the swap.
                if self._pending_approval_widget is not None:
                    try:
                        self._pending_approval_widget.action_select_reject()
                    except Exception:
                        logger.debug(
                            "Failed to reject pending approval during agent swap",
                            exc_info=True,
                        )
                if self._pending_ask_user_widget is not None:
                    try:
                        await self._pending_ask_user_widget.remove()
                    except Exception:
                        logger.debug(
                            "Failed to remove pending ask_user during agent swap",
                            exc_info=True,
                        )
                    self._pending_ask_user_widget = None

                self._pending_messages.clear()
                for widget in self._queued_widgets:
                    try:
                        await widget.remove()
                    except Exception:
                        logger.debug(
                            "Failed to remove queued widget during agent swap",
                            exc_info=True,
                        )
                self._queued_widgets.clear()
                self._deferred_actions.clear()

                await self._clear_messages()
                self._context_tokens = 0
                self._tokens_approximate = False
                self._update_tokens(0)
                self._update_status("")

                if self._session_state:
                    new_thread_id = self._session_state.reset_thread()
                    self._lc_thread_id = new_thread_id
                    self._update_welcome_banner(
                        new_thread_id,
                        missing_message=(
                            "Welcome banner not found during agent switch to %s"
                        ),
                        warn_if_missing=False,
                    )
            except Exception:
                logger.exception(
                    "UI teardown failed during agent swap to %r", agent_name
                )
                # Restore the previous-agent UI state so the user isn't
                # stuck on a permanent "Connecting..." banner.
                self._connecting = False
                try:
                    banner = self.query_one("#welcome-banner", WelcomeBanner)
                    banner.set_connected(self._mcp_tool_count)
                except NoMatches:
                    pass
                self.notify(
                    f"Could not prepare to switch to {agent_name!r}. "
                    "Staying on current agent.",
                    severity="error",
                    markup=False,
                )
                return

            # Phase 2: server restart. Identity is mutated BEFORE
            # `restart()` so the subprocess picks up the new assistant_id
            # from the staged env override; on failure, both are rolled
            # back and the old server is confirmed dead (ServerStartFailed).
            self._assistant_id = agent_name
            if self._server_kwargs is not None:
                self._server_kwargs["assistant_id"] = agent_name

            try:
                server_proc.update_env(
                    **{f"{SERVER_ENV_PREFIX}ASSISTANT_ID": agent_name}
                )
                await server_proc.restart()
                # `ServerProcess.restart()` may rebind to a different port
                # if the original is still in TIME_WAIT, so rebuild the
                # client against the current URL rather than reusing it.
                self._agent = _build_agent(server_proc.url)
            except Exception as exc:
                self._assistant_id = previous_agent
                if self._server_kwargs is not None:
                    self._server_kwargs["assistant_id"] = previous_agent
                self._agent = None
                self._connecting = False
                logger.exception(
                    "Server restart failed during agent swap to %r", agent_name
                )
                self.post_message(self.ServerStartFailed(error=exc))
                return

            # Phase 3: confirmation. Past here all failures are
            # cosmetic — the new server is healthy.
            self._connecting = False
            try:
                banner = self.query_one("#welcome-banner", WelcomeBanner)
                banner.set_connected(self._mcp_tool_count)
            except NoMatches:
                pass

            # Refresh skills so /skill: autocomplete reflects the new agent's
            # SKILL.md files.
            self.run_worker(
                self._discover_skills(),
                exclusive=True,
                group="agent-switch-skill-discovery",
            )

            # Persist the swap so a bare `deepagents` relaunch brings the
            # user back to this agent (same pattern as `save_recent_model`).
            # Offloaded to a thread to avoid blocking the event loop on disk I/O.
            from deepagents_cli.model_config import save_recent_agent

            saved = await asyncio.to_thread(save_recent_agent, agent_name)
            if not saved:
                logger.warning(
                    "Could not persist recent agent %r to config; "
                    "next bare launch will not return to it",
                    agent_name,
                )

            confirmation = Content.from_markup(
                "Switched to $name. New thread started.",
                name=agent_name,
            )
            await self._mount_message(AppMessage(confirmation))

            # Surface a resume command for the previous session so the
            # previous thread isn't stranded out of reach. `-r <thread>`
            # alone is enough: `_resolve_resume_thread` infers the owning
            # agent from persisted thread metadata via `get_thread_agent`.
            # Build via `from_markup` so a thread ID with stray brackets
            # can't corrupt rendering.
            if previous_thread_id:
                resume_hint = Content.from_markup(
                    "[dim]Relaunch with[/dim] deepagents -r $thread "
                    "[dim]to resume the previous thread.[/dim]",
                    thread=previous_thread_id,
                )
                await self._mount_message(AppMessage(resume_hint))

            # Drain any messages the user typed after we cleared the queue
            # but before the new server was ready.
            if self._pending_messages and not self._agent_running:
                self.call_after_refresh(
                    lambda: asyncio.create_task(self._process_next_from_queue())
                )
        finally:
            self._agent_switching = False
            if self._chat_input:
                self._chat_input.set_cursor_active(active=not self._agent_running)

    async def _show_notification_settings(self) -> None:
        """Show notification settings modal."""
        from deepagents_cli.model_config import is_warning_suppressed
        from deepagents_cli.widgets.notification_settings import (
            WARNING_TOGGLES,
            NotificationSettingsScreen,
        )

        suppressed: set[str] = set()
        try:
            for key, _ in WARNING_TOGGLES:
                if await asyncio.to_thread(is_warning_suppressed, key):
                    suppressed.add(key)
        except Exception:
            logger.warning("Failed to read notification settings", exc_info=True)
            suppressed = set()
            self.notify(
                "Could not read notification preferences. Showing defaults.",
                severity="warning",
                timeout=6,
                markup=False,
            )

        def handle_result(_result: None) -> None:
            if self._chat_input:
                self._chat_input.focus_input()

        screen = NotificationSettingsScreen(suppressed=suppressed)
        self.push_screen(screen, handle_result)

    def _notify_actionable(
        self,
        notification: PendingNotification,
        *,
        severity: Literal["information", "warning", "error"] = "information",
        timeout: float | None = None,
    ) -> None:
        """Register *notification* and post its actionable toast.

        Posts the toast as a raw `Notification` so the identity can be
        captured and bound to the registry entry for click routing.

        Args:
            notification: Registry entry to register and surface.
            severity: Toast severity banner color.
            timeout: Seconds the toast stays on screen (defaults to
                `App.NOTIFICATION_TIMEOUT`).
        """
        self._notice_registry.add(notification)

        toast_body = f"{notification.body}\n\nctrl+n for options"
        effective_timeout = (
            timeout if timeout is not None else self.NOTIFICATION_TIMEOUT
        )
        # `markup=False` is load-bearing: `notification.body` can carry
        # dynamic content (tool names, versions, URLs, exception text)
        # with square brackets that would crash Textual's toast
        # renderer if parsed as Rich markup.
        toast = _Notification(
            message=toast_body,
            title=notification.title,
            severity=severity,
            timeout=effective_timeout,
            markup=False,
        )
        self._notice_registry.bind_toast(notification.key, toast.identity)
        self.post_message(_Notify(toast))

    def _inject_debug_notifications(self) -> None:
        """Register sample notifications so the center UI can be exercised.

        Gated by `DEEPAGENTS_CLI_DEBUG_NOTIFICATIONS`; no-op without it.
        Covers every payload type and action, and auto-opens the
        dedicated update modal so the startup flow can be exercised
        without waiting for a real PyPI release.

        Mirrors the real startup flow: when an update is being
        surfaced, missing-dep entries register silently (no toast)
        so they don't flicker before the modal's
        `clear_notifications` fires.
        """
        try:
            from deepagents_cli.main import build_missing_tool_notification
        except Exception:
            logger.warning(
                "Could not inject debug notifications; main import failed",
                exc_info=True,
            )
            return

        self._notice_registry.add(build_missing_tool_notification("ripgrep"))
        self._notice_registry.add(build_missing_tool_notification("tavily"))
        update_notification = self._build_update_notification(
            latest="9.9.9",
            cli_version="0.0.1",
            age_suffix=", released 2 days ago",
            upgrade_cmd="uv tool upgrade deepagents-cli",
        )
        self._notice_registry.add(update_notification)
        self._update_modal_pending.set()
        self.call_after_refresh(self._open_update_available_modal, update_notification)

    def action_open_notifications(self) -> None:
        """Open the notification center via the `ctrl+n` keybind."""
        self._open_notification_center()

    def _open_notification_center(self) -> None:
        """Push the notification center modal, or toast when empty."""
        from deepagents_cli.widgets.notification_center import (
            NotificationActionResult,
            NotificationCenterScreen,
        )

        if isinstance(self.screen, ModalScreen):
            # Don't stack on top of another modal (e.g. approval, model
            # selector). Surface feedback so the user knows why ctrl+n
            # appeared to do nothing.
            self.notify(
                "Close the current dialog to view notifications.",
                severity="information",
                timeout=3,
                markup=False,
            )
            return

        pending = self._notice_registry.list_all()
        if not pending:
            self.notify(
                "No pending notifications.",
                severity="information",
                timeout=2,
                markup=False,
            )
            return

        def handle_result(result: NotificationActionResult | None) -> None:
            if result is not None:
                self.run_worker(
                    self._dispatch_notification_action(result.key, result.action_id),
                    exclusive=False,
                    group=f"notification-action-{result.key}",
                )
            elif self._chat_input:
                self._chat_input.focus_input()

        self.push_screen(NotificationCenterScreen(pending), handle_result)

    def _open_update_available_modal(self, entry: PendingNotification) -> None:
        """Push the dedicated update-available modal for *entry*.

        When another modal is already open the entry stays registered
        and a toast hint points the user at `ctrl+n` once the blocking
        modal closes. Also clears `_update_modal_pending` so
        missing-dep toasts stop suppressing themselves.
        """
        from deepagents_cli.widgets.update_available import UpdateAvailableScreen

        if isinstance(self.screen, ModalScreen):
            # We can't stack; leave the entry in the registry and tell
            # the user how to reach it.
            self._update_modal_pending.clear()
            self.notify(
                "Update available. Close the current dialog, "
                "then press ctrl+n to review it.",
                severity="information",
                timeout=8,
                markup=False,
            )
            return

        # Textual layers are per-screen, so base-screen toasts visually
        # bleed through the modal's dim. Drop them before opening so
        # the modal reads cleanly; underlying notification entries
        # stay in the registry and remain reachable via ctrl+n.
        self.clear_notifications()

        def handle_result(result: ActionId | None) -> None:
            if result is not None:
                self.run_worker(
                    self._dispatch_notification_action(entry.key, result),
                    exclusive=False,
                    group=f"notification-action-{entry.key}",
                )
            elif self._chat_input:
                self._chat_input.focus_input()

        self.push_screen(UpdateAvailableScreen(entry), handle_result)

    async def _dispatch_notification_action(
        self, key: str, action_id: ActionId
    ) -> None:
        """Execute the side effect for a notification action.

        Resolves the registry entry and runs the payload-specific
        handler. Catches `Exception` broadly (not just `OSError`) so
        browser-launch failures, clipboard backend errors, TOML write
        errors, and any unhandled-payload `TypeError` surface as a
        user-visible warning toast instead of vanishing into the
        background worker's logs.

        Args:
            key: Registry key of the notification.
            action_id: The action the user selected.
        """
        entry = self._notice_registry.get(key)
        if entry is None:
            return

        action_label = _action_label(entry, action_id)
        try:
            await self._route_payload_action(entry, action_id)
        except Exception as exc:  # every failure surfaces to the user
            logger.warning(
                "Action %r on %r failed: %s", action_id, key, exc, exc_info=True
            )
            self.notify(
                f"{action_label} failed: {type(exc).__name__}: {exc}",
                severity="warning",
                timeout=8,
                markup=False,
            )

        if self._chat_input:
            self._chat_input.focus_input()

    async def _route_payload_action(
        self, entry: PendingNotification, action_id: ActionId
    ) -> None:
        """Dispatch *action_id* to the payload-specific handler.

        Raises:
            TypeError: When `entry.payload` has no registered handler —
                programming error; the caller converts it into a
                user-facing warning toast.
        """
        if isinstance(entry.payload, MissingDepPayload):
            await self._handle_missing_dep_action(entry, entry.payload, action_id)
            return
        if isinstance(entry.payload, UpdateAvailablePayload):
            await self._handle_update_action(entry, entry.payload, action_id)
            return
        msg = f"unhandled payload type {type(entry.payload).__name__}"
        raise TypeError(msg)

    async def _handle_missing_dep_action(
        self,
        entry: PendingNotification,
        payload: MissingDepPayload,
        action_id: ActionId,
    ) -> None:
        """Complete a missing-dependency action.

        Args:
            entry: The notification entry for the affected tool.
            payload: Typed payload (tool name + install hint or URL).
            action_id: The specific action the user selected.
                Unknown ids are logged and treated as a no-op.
        """
        if action_id == ActionId.SUPPRESS:
            from deepagents_cli.model_config import suppress_warning

            if await asyncio.to_thread(suppress_warning, payload.tool):
                self._notice_registry.remove(entry.key)
                self.notify(
                    f"Won't warn about {payload.tool} again.",
                    severity="information",
                    timeout=4,
                    markup=False,
                )
            else:
                self.notify(
                    "Could not save notification preference. "
                    "Check file permissions for ~/.deepagents/config.toml.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )
            return
        if action_id == ActionId.COPY_INSTALL:
            if payload.install_command is None:
                logger.warning(
                    "COPY_INSTALL action fired without install_command on %r",
                    entry.key,
                )
                self.notify(
                    "No install command recorded for this notification.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )
                return
            self.copy_to_clipboard(payload.install_command)
            self.notify(
                f"Copied: {payload.install_command}",
                severity="information",
                timeout=4,
                markup=False,
            )
            return
        if action_id == ActionId.OPEN_WEBSITE:
            if payload.url is None:
                logger.warning("OPEN_WEBSITE action fired without url on %r", entry.key)
                self.notify(
                    "No URL recorded for this notification.",
                    severity="warning",
                    timeout=6,
                    markup=False,
                )
                return
            try:
                opened = await asyncio.to_thread(webbrowser.open, payload.url)
            except (webbrowser.Error, OSError) as exc:
                logger.warning(
                    "webbrowser.open failed for %s: %s",
                    payload.url,
                    exc,
                    exc_info=True,
                )
                opened = False
            if opened:
                self.notify(
                    f"Opened {payload.url}",
                    severity="information",
                    timeout=3,
                    markup=False,
                )
            else:
                self.notify(
                    f"Could not open a browser. URL: {payload.url}",
                    severity="warning",
                    timeout=8,
                    markup=False,
                )
            return
        logger.warning(
            "Unknown action_id %r for missing-dep entry %s", action_id, entry.key
        )

    async def _handle_update_action(
        self,
        entry: PendingNotification,
        payload: UpdateAvailablePayload,
        action_id: ActionId,
    ) -> None:
        """Complete an update-available action.

        Args:
            entry: The update notification entry.
            payload: Typed payload (target version + upgrade command).
            action_id: The specific action the user selected.
                Unknown ids are logged and treated as a no-op.
        """
        from deepagents_cli.update_check import (
            clear_update_notified,
            mark_update_notified,
            perform_upgrade,
            upgrade_command,
        )

        if action_id == ActionId.INSTALL:
            self.notify(
                f"Updating to v{payload.latest}...",
                severity="information",
                timeout=5,
                markup=False,
            )
            success, output = await perform_upgrade()
            if success:
                self._notice_registry.remove(entry.key)
                self.notify(
                    f"Updated to v{payload.latest}. Restart to use the new version.",
                    severity="information",
                    timeout=10,
                    markup=False,
                )
                return
            logger.warning(
                "Auto-upgrade failed for v%s. Output:\n%s", payload.latest, output
            )
            self._notice_registry.remove(entry.key)
            cmd = upgrade_command()
            snippet = _truncate(output, limit=160) if output else ""
            message = f"Auto-update failed. Run manually: {cmd}"
            if snippet:
                message = f"{message}\n{snippet}"
            self.notify(
                message,
                severity="warning",
                timeout=15,
                markup=False,
            )
            return
        if action_id == ActionId.SKIP_VERSION:
            await asyncio.to_thread(mark_update_notified, payload.latest)
            self._notice_registry.remove(entry.key)
            return
        if action_id == ActionId.SKIP_ONCE:
            await asyncio.to_thread(clear_update_notified)
            self._notice_registry.remove(entry.key)
            return
        logger.warning("Unknown action_id %r for update entry %s", action_id, entry.key)

    async def _show_mcp_viewer(self) -> None:
        """Show read-only MCP server/tool viewer as a modal screen."""
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        screen = MCPViewerScreen(server_info=self._mcp_server_info or [])

        def handle_result(result: None) -> None:  # noqa: ARG001
            if self._chat_input:
                self._chat_input.focus_input()

        self.push_screen(screen, handle_result)

    async def _show_thread_selector(self) -> None:
        """Show interactive thread selector as a modal screen."""
        from functools import partial

        from deepagents_cli.sessions import get_cached_threads, get_thread_limit
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        current = self._session_state.thread_id if self._session_state else None
        thread_limit = get_thread_limit()

        initial_threads = get_cached_threads(limit=thread_limit)

        def handle_result(result: str | None) -> None:
            """Handle the thread selector result."""
            if result is not None:
                if self._agent_running or self._shell_running or self._connecting:
                    self._defer_action(
                        DeferredAction(
                            kind="thread_switch",
                            execute=partial(self._resume_thread, result),
                        )
                    )
                    self.notify(
                        "Thread will switch after current task completes.", timeout=3
                    )
                else:
                    self.call_later(self._resume_thread, result)
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ThreadSelectorScreen(
            current_thread=current,
            thread_limit=thread_limit,
            initial_threads=initial_threads,
        )
        self.push_screen(screen, handle_result)

    def _update_welcome_banner(
        self,
        thread_id: str,
        *,
        missing_message: str,
        warn_if_missing: bool,
    ) -> None:
        """Update the welcome banner thread ID when the banner is mounted.

        Args:
            thread_id: Thread ID to display on the banner.
            missing_message: Log message template when banner is missing.
            warn_if_missing: Whether to log missing-banner cases at warning level.
        """
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.update_thread_id(thread_id)
        except NoMatches:
            if warn_if_missing:
                logger.warning(missing_message, thread_id)
            else:
                logger.debug(missing_message, thread_id)

    async def _resume_thread(self, thread_id: str) -> None:
        """Resume a previously saved thread.

        Fetches the selected thread history, then atomically switches UI state.
        Prefetching first avoids clearing the active chat when history loading
        fails.

        Args:
            thread_id: The thread ID to resume.
        """
        if not self._agent:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active agent")
            )
            return

        if not self._session_state:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active session")
            )
            return

        # Skip if already on this thread
        if self._session_state.thread_id == thread_id:
            await self._mount_message(AppMessage(f"Already on thread: {thread_id}"))
            return

        if self._thread_switching:
            await self._mount_message(AppMessage("Thread switch already in progress."))
            return

        # Save previous state for rollback on failure
        prev_thread_id = self._lc_thread_id
        prev_session_thread = self._session_state.thread_id
        self._thread_switching = True
        if self._chat_input:
            self._chat_input.set_cursor_active(active=False)

        prefetched_payload: _ThreadHistoryPayload | None = None
        try:
            self._update_status(f"Loading thread: {thread_id}")
            prefetched_payload = await self._fetch_thread_history_data(thread_id)

            # Clear conversation (similar to /clear, without creating a new thread)
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            self._context_tokens = 0
            self._tokens_approximate = False
            self._update_tokens(0)
            self._update_status("")

            # Switch to the selected thread
            self._session_state.thread_id = thread_id
            self._lc_thread_id = thread_id

            self._update_welcome_banner(
                thread_id,
                missing_message="Welcome banner not found during thread switch to %s",
                warn_if_missing=False,
            )

            # Load thread history
            await self._load_thread_history(
                thread_id=thread_id,
                preloaded_payload=prefetched_payload,
            )
        except Exception as exc:
            if prefetched_payload is None:
                logger.exception("Failed to prefetch history for thread %s", thread_id)
                await self._mount_message(
                    AppMessage(
                        f"Failed to switch to thread {thread_id}: {exc}. "
                        "Use /threads to try again."
                    )
                )
                return
            logger.exception("Failed to switch to thread %s", thread_id)
            # Restore previous thread IDs so the user can retry
            self._session_state.thread_id = prev_session_thread
            self._lc_thread_id = prev_thread_id
            self._update_welcome_banner(
                prev_session_thread,
                missing_message=(
                    "Welcome banner not found during rollback to thread %s; "
                    "banner may display stale thread ID"
                ),
                warn_if_missing=True,
            )
            rollback_restore_failed = False
            # Attempt to restore the previous thread's visible history
            try:
                await self._clear_messages()
                await self._load_thread_history(thread_id=prev_session_thread)
            except Exception:  # Resilient session state saving
                rollback_restore_failed = True
                msg = (
                    "Could not restore previous thread history after failed "
                    "switch to %s"
                )
                logger.warning(msg, thread_id, exc_info=True)
            error_message = f"Failed to switch to thread {thread_id}: {exc}."
            if rollback_restore_failed:
                error_message += " Previous thread history could not be restored."
            error_message += " Use /threads to try again."
            await self._mount_message(AppMessage(error_message))
        finally:
            self._thread_switching = False
            self._update_status("")
            if self._chat_input:
                self._chat_input.set_cursor_active(active=not self._agent_running)

    async def _switch_model(
        self,
        model_spec: str,
        *,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Switch to a new model, preserving conversation history.

        This requires a server-backed interactive session. It sets a model
        override that `ConfigurableModelMiddleware` picks up on the next
        invocation, so the conversation thread stays intact and no server
        restart is required.

        Args:
            model_spec: The model specification to switch to.

                Can be in `provider:model` format
                (e.g., `'anthropic:claude-sonnet-4-5'`) or just the model name
                for auto-detection.
            extra_kwargs: Extra constructor kwargs from `--model-params`.
        """
        from deepagents_cli.config import create_model, detect_provider, settings
        from deepagents_cli.model_config import (
            ModelSpec,
            get_credential_env_var,
            has_provider_credentials,
            save_recent_model,
        )

        logger.info("Switching model to %s", model_spec)

        if self._model_switching:
            await self._mount_message(AppMessage("Model switch already in progress."))
            return

        self._model_switching = True
        try:
            # Defensively strip leading colon in case of empty provider,
            # treat ":claude-opus-4-6" as "claude-opus-4-6"
            model_spec = model_spec.removeprefix(":")

            if not self._remote_agent():
                await self._mount_message(
                    ErrorMessage("Model switching requires a server-backed session.")
                )
                return

            parsed = ModelSpec.try_parse(model_spec)
            if parsed:
                provider: str | None = parsed.provider
                model_name = parsed.model
            else:
                model_name = model_spec
                provider = detect_provider(model_spec)

            # Check credentials
            has_creds = has_provider_credentials(provider) if provider else None
            if has_creds is False and provider is not None:
                env_var = get_credential_env_var(provider)
                detail = (
                    f"{env_var} is not set or is empty"
                    if env_var
                    else (
                        f"provider '{provider}' is not recognized. "
                        "Add it to ~/.deepagents/config.toml with an "
                        "api_key_env field"
                    )
                )
                await self._mount_message(
                    ErrorMessage(f"Missing credentials: {detail}")
                )
                return
            if has_creds is None and provider:
                logger.debug(
                    "Credentials for provider '%s' cannot be verified;"
                    " proceeding anyway",
                    provider,
                )

            # Check if already using this exact model
            if model_name == settings.model_name and (
                not provider or provider == settings.model_provider
            ):
                current = f"{settings.model_provider}:{settings.model_name}"
                await self._mount_message(AppMessage(f"Already using {current}"))
                return

            # Build the provider:model spec for the configurable middleware.
            display = model_spec
            if provider and not parsed:
                display = f"{provider}:{model_name}"

            try:
                create_model(
                    display,
                    extra_kwargs=extra_kwargs,
                    profile_overrides=self._profile_override,
                ).apply_to_settings()
            except Exception as exc:
                logger.exception("Failed to resolve model metadata for %s", display)
                await self._mount_message(
                    ErrorMessage(f"Failed to switch model: {exc}")
                )
                return

            # Set the model override for ConfigurableModelMiddleware.
            # The next stream call passes CLIContext via context= and the
            # middleware swaps the model per-invocation — no graph recreation.
            self._model_override = display
            self._model_params_override = extra_kwargs

            if self._status_bar:
                self._status_bar.set_model(
                    provider=settings.model_provider or "",
                    model=settings.model_name or "",
                )

            if not await asyncio.to_thread(save_recent_model, display):
                await self._mount_message(
                    ErrorMessage(
                        "Model switched for this session, but could not save "
                        "preference. Check permissions for ~/.deepagents/"
                    )
                )
            else:
                await self._mount_message(AppMessage(f"Switched to {display}"))
            logger.info("Model switched to %s (via configurable middleware)", display)

            # Anchor to bottom so the confirmation message is visible
            with suppress(NoMatches, ScreenStackError):
                self.query_one("#chat", VerticalScroll).anchor()
        finally:
            self._model_switching = False

    async def _set_default_model(self, model_spec: str) -> None:
        """Set the default model in config without switching the current session.

        Updates `[models].default` in `~/.deepagents/config.toml` so that
        future CLI launches use this model. Does not affect the running session.

        Args:
            model_spec: The model specification (e.g., `'anthropic:claude-opus-4-6'`).
        """
        from deepagents_cli.config import detect_provider
        from deepagents_cli.model_config import ModelSpec, save_default_model

        model_spec = model_spec.removeprefix(":")

        parsed = ModelSpec.try_parse(model_spec)
        if not parsed:
            provider = detect_provider(model_spec)
            if provider:
                model_spec = f"{provider}:{model_spec}"

        if await asyncio.to_thread(save_default_model, model_spec):
            await self._mount_message(AppMessage(f"Default model set to {model_spec}"))
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not save default model. Check permissions for ~/.deepagents/"
                )
            )

    async def _clear_default_model(self) -> None:
        """Remove the default model from config.

        After clearing, future launches fall back to `[models].recent` or
        environment auto-detection.
        """
        from deepagents_cli.model_config import clear_default_model

        if await asyncio.to_thread(clear_default_model):
            await self._mount_message(
                AppMessage(
                    "Default model cleared. "
                    "Future launches will use recent model or auto-detect."
                )
            )
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not clear default model. "
                    "Check permissions for ~/.deepagents/"
                )
            )


@dataclass(frozen=True)
class AppResult:
    """Result from running the Textual application."""

    return_code: int
    """Exit code (0 for success, non-zero for error)."""

    thread_id: str | None
    """The final thread ID at shutdown. May differ from the initial thread ID if
    the user switched threads via `/threads`."""

    session_stats: SessionStats = field(default_factory=SessionStats)
    """Cumulative usage stats across all turns in the session."""

    update_available: tuple[bool, str | None] = (False, None)
    """`(is_available, latest_version)` for post-exit update warning."""


async def run_textual_app(
    *,
    agent: Any = None,  # noqa: ANN401
    assistant_id: str | None = None,
    backend: CompositeBackend | None = None,
    auto_approve: bool = False,
    cwd: str | Path | None = None,
    thread_id: str | None = None,
    resume_thread: str | None = None,
    initial_prompt: str | None = None,
    initial_skill: str | None = None,
    startup_cmd: str | None = None,
    mcp_server_info: list[MCPServerInfo] | None = None,
    profile_override: dict[str, Any] | None = None,
    server_proc: ServerProcess | None = None,
    server_kwargs: dict[str, Any] | None = None,
    mcp_preload_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> AppResult:
    """Run the Textual application.

    When `server_kwargs` is provided (and `agent` is `None`), the app starts
    immediately with a "Connecting..." banner and launches the server in the
    background.  Server cleanup is handled automatically after the app exits.

    Args:
        agent: Pre-configured LangGraph agent (optional).
        assistant_id: Agent identifier for memory storage.
        backend: Backend for file operations.
        auto_approve: Whether to start with auto-approve enabled.
        cwd: Current working directory to display.
        thread_id: Thread ID for the session.

            `None` when `resume_thread` is provided (the TUI resolves the final
            ID asynchronously).
        resume_thread: Raw resume intent from `-r` flag. `'__MOST_RECENT__'` for
            bare `-r`, a thread ID string for `-r <id>`, or `None` for new
            sessions.

            Resolved asynchronously during TUI startup.
        initial_prompt: Optional prompt to auto-submit when session starts.
        initial_skill: Optional skill name to invoke when session starts.
        startup_cmd: Optional shell command to run at startup before the first
            prompt is accepted. Output is rendered in the transcript and
            non-zero exits warn but do not abort the session.
        mcp_server_info: MCP server metadata for the `/mcp` viewer.
        profile_override: Extra profile fields from `--profile-override`,
            retained so later profile-aware behavior stays consistent with
            the CLI override, including model selection details, offload
            budget display, and on-demand `create_model()` calls such
            as `/offload`.
        server_proc: LangGraph server process for the interactive session.
        server_kwargs: Kwargs for deferred `start_server_and_get_agent` call.
        mcp_preload_kwargs: Kwargs for concurrent MCP metadata preload.
        model_kwargs: Kwargs for deferred `create_model()` call.

            When provided, model creation runs in a background worker after
            first paint so the splash screen appears immediately.

    Returns:
        An `AppResult` with the return code and final thread ID.
    """
    app = DeepAgentsApp(
        agent=agent,
        assistant_id=assistant_id,
        backend=backend,
        auto_approve=auto_approve,
        cwd=cwd,
        thread_id=thread_id,
        resume_thread=resume_thread,
        initial_prompt=initial_prompt,
        initial_skill=initial_skill,
        startup_cmd=startup_cmd,
        mcp_server_info=mcp_server_info,
        profile_override=profile_override,
        server_proc=server_proc,
        server_kwargs=server_kwargs,
        mcp_preload_kwargs=mcp_preload_kwargs,
        model_kwargs=model_kwargs,
    )
    try:
        await app.run_async()
    finally:
        # Guarantee server cleanup regardless of how the app exits.
        # Covers both the pre-started server_proc path and the deferred
        # server_kwargs path (where the background worker sets _server_proc).
        if app._server_proc is not None:
            app._server_proc.stop()

    return AppResult(
        return_code=app.return_code or 0,
        thread_id=app._lc_thread_id,
        session_stats=app._session_stats,
        update_available=app._update_available,
    )
