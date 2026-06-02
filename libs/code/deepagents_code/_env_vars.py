"""Canonical registry of `DEEPAGENTS_CODE_*` environment variables.

Every env var the app reads whose name starts with `DEEPAGENTS_CODE_` must
be defined here as a module-level constant.  A drift-detection test
(`tests/unit_tests/test_env_vars.py`) fails when a bare string literal
like `"DEEPAGENTS_CODE_FOO"` appears in source code instead of a constant
imported from this module.

Import the short-name constants (e.g. `AUTO_UPDATE`, `DEBUG`) and pass them
to `os.environ.get()` instead of using raw string literals. If the env var is
ever renamed, only the value here changes.

!!! note

    `resolve_env_var` also supports a dynamic prefix override for API keys
    and provider credentials: setting `DEEPAGENTS_CODE_{NAME}` takes priority
    over `{NAME}`.  For example, `DEEPAGENTS_CODE_OPENAI_API_KEY` overrides
    `OPENAI_API_KEY`. Only call sites that use `resolve_env_var` benefit from
    this -- direct `os.environ.get` lookups (like the constants below) do not.
    Dynamic overrides are not listed here because they mirror third-party
    variable names.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Constants — import these instead of bare string literals.
# Keep alphabetically sorted by constant name.
# ---------------------------------------------------------------------------

AUTO_UPDATE = "DEEPAGENTS_CODE_AUTO_UPDATE"
"""Enable automatic app updates ('1', 'true', or 'yes')."""

DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER = (
    "DEEPAGENTS_CODE_DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER"
)
"""Override the startup splash subheader text when set."""

DEBUG = "DEEPAGENTS_CODE_DEBUG"
"""Enable verbose debug logging and preserve the server subprocess log.

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` (case-insensitive)
as enabled, and `0`, `false`, `no`, `off`, empty string, or unset as disabled.
"""

DEBUG_FILE = "DEEPAGENTS_CODE_DEBUG_FILE"
"""Path for the debug log file (default: `/tmp/deepagents_debug.log`)."""

DEBUG_MCP_PROJECT_TRUST = "DEEPAGENTS_CODE_DEBUG_MCP_PROJECT_TRUST"
"""Force the project MCP approval prompt for manual UI testing.

Set to a truthy value when launching the interactive TUI to render the
project-level MCP trust prompt without relying on an untrusted config state. If
project MCP servers are discovered, the prompt shows those real servers;
otherwise it shows a sample server. The TUI exits after the prompt response so
the debug run does not continue into TUI or server startup, and it does not
persist trust decisions.

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` as enabled.
"""

DEBUG_NOTIFICATIONS = "DEEPAGENTS_CODE_DEBUG_NOTIFICATIONS"
"""Inject sample missing-dependency notifications at launch so the notification
center UI can be exercised without waiting for real conditions.

Does not auto-open the update modal (use `DEEPAGENTS_CODE_DEBUG_UPDATE` for that).

Any non-empty value enables the flag (including `"0"` or `"false"`).
"""

DEBUG_ONBOARDING = "DEEPAGENTS_CODE_DEBUG_ONBOARDING"
"""Force the onboarding flow to open on every interactive startup.

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` as enabled.
"""

DEBUG_UPDATE = "DEEPAGENTS_CODE_DEBUG_UPDATE"
"""Inject a sample update-available notification and auto-open the update modal
at launch so the update-available flow can be exercised without waiting for a
real PyPI release.

Any non-empty value enables the flag (including `"0"` or `"false"`).
"""

EXTERNAL_EVENT_SOCKET = "DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET"
"""Enable the local Unix-socket external event listener.

Parsed by `is_env_truthy`; off by default. Wire format and behavior are
considered experimental until the listener is documented in the README.
"""

EXTERNAL_EVENT_SOCKET_PATH = "DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET_PATH"
"""Override the default Unix-socket path for the external event listener."""

EXTRA_SKILLS_DIRS = "DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS"
"""Colon-separated paths added to the skill containment allowlist."""

HIDE_CWD = "DEEPAGENTS_CODE_HIDE_CWD"
"""Hide local path displays in the TUI footer and startup splash when enabled."""

HIDE_GIT_BRANCH = "DEEPAGENTS_CODE_HIDE_GIT_BRANCH"
"""Hide the current git branch in the TUI footer when enabled."""

HIDE_LANGSMITH_TRACING = "DEEPAGENTS_CODE_HIDE_LANGSMITH_TRACING"
"""Hide LangSmith tracing project/thread info in the startup splash when enabled."""

HIDE_SPLASH_TIPS = "DEEPAGENTS_CODE_HIDE_SPLASH_TIPS"
"""Hide rotating tips in the startup splash when enabled."""

HIDE_SPLASH_VERSION = "DEEPAGENTS_CODE_HIDE_SPLASH_VERSION"
"""Hide version and local-install details in the splash screen when enabled."""

KITTY_KEYBOARD = "DEEPAGENTS_CODE_KITTY_KEYBOARD"
"""Override kitty-keyboard detection (`1` forces on, `0` forces off)."""

LANGSMITH_PROJECT = "DEEPAGENTS_CODE_LANGSMITH_PROJECT"
"""Override LangSmith project name for agent traces."""

NO_TERMINAL_ESCAPE = "DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE"
"""Disable all terminal escape/control sequence output when enabled."""

NO_UPDATE_CHECK = "DEEPAGENTS_CODE_NO_UPDATE_CHECK"
"""Disable automatic update checking when set."""

OLLAMA_DISCOVERY = "DEEPAGENTS_CODE_OLLAMA_DISCOVERY"
"""Toggle Ollama model and profile discovery probes.

Defaults to enabled. Suppress the probe when the daemon is intentionally
offline or the probe latency is undesirable. The probe is lazy and never
runs on the startup hot path. When enabled, discovery may call `/api/tags`
and `/api/show`. See `_ollama_discovery_enabled` for accepted truthy/falsy
values.
"""

SERVER_ENV_PREFIX = "DEEPAGENTS_CODE_SERVER_"
"""Environment variable prefix used to pass CLI config to the server subprocess."""

SHELL_ALLOW_LIST = "DEEPAGENTS_CODE_SHELL_ALLOW_LIST"
"""Comma-separated shell commands to allow (or 'recommended'/'all')."""

SHOW_HEADER = "DEEPAGENTS_CODE_SHOW_HEADER"
"""Show Textual's native header bar at the top of the TUI when enabled."""

THEME = "DEEPAGENTS_CODE_THEME"
"""Force the CLI to launch with this theme name when set."""

USER_ID = "DEEPAGENTS_CODE_USER_ID"
"""Attach a user identifier to LangSmith trace metadata."""


_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", ""})


def is_env_truthy(name: str, *, default: bool = False) -> bool:
    """Return whether env var *name* is set to a recognizably truthy value.

    Unlike `bool(os.environ.get(name))`, this does not treat `"0"` or
    `"false"` as enabled. Use this for on/off flags where the user would
    reasonably expect `VAR=0` to mean "disabled".

    Args:
        name: Environment variable name (typically a `DEEPAGENTS_CODE_*`
            constant from this module).
        default: Value returned when the variable is unset OR set to a
            value that is neither recognizably truthy nor falsy.

    Returns:
        `True` for `1`/`true`/`yes`/`on` (case-insensitive), `False` for
        `0`/`false`/`no`/`off`/empty string, or `default` otherwise.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUTHY_VALUES:
        return True
    if lowered in _FALSY_VALUES:
        return False
    return default
