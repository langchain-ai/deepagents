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
"""Toggle automatic app updates. Enabled by default; set to a falsy value
('0', 'false', 'no', 'off', or empty) to opt out."""

DEBUG = "DEEPAGENTS_CODE_DEBUG"
"""Enable verbose debug logging and preserve the server subprocess log.

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` (case-insensitive)
as enabled, and `0`, `false`, `no`, `off`, empty string, or unset as disabled.
"""

DEBUG_FILE = "DEEPAGENTS_CODE_DEBUG_FILE"
"""Path for the debug log file (default: `DEFAULT_DEBUG_FILE`)."""

DEFAULT_DEBUG_FILE = "/tmp/deepagents_debug.log"  # noqa: S108  # opt-in debug log
"""Default path for the debug log when `DEBUG_FILE` is unset."""

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
"""Hide local path displays in the TUI footer and the editable-install path in
the startup splash when enabled.

Does not control the splash working-directory row, which is gated solely by
`SPLASH_SHOW_CWD`.
"""

HIDE_GIT_BRANCH = "DEEPAGENTS_CODE_HIDE_GIT_BRANCH"
"""Hide the current git branch in the TUI footer when enabled."""

HIDE_LANGSMITH_TRACING = "DEEPAGENTS_CODE_HIDE_LANGSMITH_TRACING"
"""Hide LangSmith tracing project/thread info in the startup splash when enabled."""

HIDE_SPLASH_VERSION = "DEEPAGENTS_CODE_HIDE_SPLASH_VERSION"
"""Hide version and local-install details in the splash screen when enabled."""

KITTY_KEYBOARD = "DEEPAGENTS_CODE_KITTY_KEYBOARD"
"""Override kitty-keyboard detection (`1` forces on, `0` forces off)."""

LANGSMITH_PROJECT = "DEEPAGENTS_CODE_LANGSMITH_PROJECT"
"""Override LangSmith project name for agent traces."""

LANGSMITH_REDACT = "DEEPAGENTS_CODE_LANGSMITH_REDACT"
"""Toggle LangSmith secret redaction for agent traces (defaults to on)."""

LANGSMITH_REPLICA_PROJECTS = "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS"
"""Comma-separated LangSmith project names to *also* write agent traces to.

When set (and tracing is active), each agent run is dual-written to the primary
deepagents-code project *and* one extra project via LangSmith write replicas.

Only the first listed project is used: the LangGraph server mirrors a run to a
single extra project, so any additional entries are dropped (with a warning).
The value is comma-separated for forward-compatibility, not because multiple
destinations are written today.
"""

NO_TERMINAL_ESCAPE = "DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE"
"""Disable all terminal escape/control sequence output when enabled."""

NO_UPDATE_CHECK = "DEEPAGENTS_CODE_NO_UPDATE_CHECK"
"""Disable automatic update checking when set."""

OFFLINE = "DEEPAGENTS_CODE_OFFLINE"
"""Disable network downloads of managed binaries (e.g. ripgrep).

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` as enabled. When
truthy, `managed_tools.ensure_ripgrep` will not attempt to download a binary
and falls back to the existing missing-tool notification + slow Python regex
path."""

OLLAMA_DISCOVERY = "DEEPAGENTS_CODE_OLLAMA_DISCOVERY"
"""Toggle Ollama model and profile discovery probes.

Defaults to enabled. Suppress the probe when the daemon is intentionally
offline or the probe latency is undesirable. The probe is lazy and never
runs on the startup hot path. When enabled, discovery may call `/api/tags`
and `/api/show`. See `_ollama_discovery_enabled` for accepted truthy/falsy
values.
"""

ONBOARDING_INTEGRATIONS_SCREEN = "DEEPAGENTS_CODE_ONBOARDING_INTEGRATIONS_SCREEN"
"""Show the "Installed Integrations" summary screen during first-run onboarding.

Off by default: onboarding goes straight from the name prompt to the model
selector, which already surfaces (and installs) uninstalled model providers.
Set to a truthy value to bring the standalone integrations screen back into the
flow. Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` as enabled.
"""

RESTARTED_AFTER_UPDATE = "DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE"
"""Internal sentinel recording the target version immediately before the
startup auto-update re-execs the process.

Not user-facing. The re-exec'd process consumes it and, if that same version
still reports as available (a no-op upgrade that did not change the running
version), skips auto-updating to break out of an otherwise endless
upgrade/restart loop. Set and read internally across `os.execv`.
"""

RIPGREP_INSTALLER = "DEEPAGENTS_CODE_RIPGREP_INSTALLER"
"""Select how ripgrep is provisioned: `managed` (default) or `system`.

`managed` downloads the pinned, SHA-256-verified upstream binary into
`~/.deepagents/bin` (no sudo). `system` skips that download so power users can
rely on their distro package / existing toolchain instead; the install script's
`system` mode keeps the brew/apt/cargo path. A system `rg` already on `PATH` is
reused under either setting. Unrecognized values fall back to `managed`. See
`managed_tools.ripgrep_installer`."""

SERVER_ENV_PREFIX = "DEEPAGENTS_CODE_SERVER_"
"""Environment variable prefix used to pass CLI config to the server subprocess."""

SHELL_ALLOW_LIST = "DEEPAGENTS_CODE_SHELL_ALLOW_LIST"
"""Comma-separated shell commands to allow (or 'recommended'/'all')."""

SHOW_HEADER = "DEEPAGENTS_CODE_SHOW_HEADER"
"""Show Textual's native header bar at the top of the TUI when enabled."""

SHOW_LANGSMITH_REPLICA_TRACING = "DEEPAGENTS_CODE_SHOW_LANGSMITH_REPLICA_TRACING"
"""Show LangSmith replica project info in the startup splash when enabled.

Defaults to enabled; set to a falsy value (`0`, `false`, `no`, `off`, or empty)
to hide replica tracing details from the splash while leaving tracing active.
"""

SHOW_SCROLLBAR = "DEEPAGENTS_CODE_SHOW_SCROLLBAR"
"""Show the vertical scrollbar in the chat area when enabled.

Off by default; use the `/scrollbar` slash command or `[ui].show_scrollbar` in
config.toml to toggle. Parsed by `classify_env_bool` (an unrecognized or empty
value falls through to the config value rather than forcing the default).

When set, this env var takes precedence over the persisted `[ui].show_scrollbar`
config value on launch, so a `/scrollbar` toggle will not appear to "stick"
across restarts while the env var remains set.
"""

SHOW_URL_OPEN_TOAST = "DEEPAGENTS_CODE_SHOW_URL_OPEN_TOAST"
"""Show a confirmation toast after clicking a URL that opens in a browser.

Defaults to enabled; set to a falsy value (`0`, `false`, `no`, `off`, or empty)
to suppress the success toast while still opening URLs normally.
"""

SPLASH_SHOW_CWD = "DEEPAGENTS_CODE_SPLASH_SHOW_CWD"
"""Show the working-directory row in the startup welcome banner when enabled.

Off by default and independent of the status bar's `HIDE_CWD`.
"""

SPLASH_SHOW_MODEL = "DEEPAGENTS_CODE_SPLASH_SHOW_MODEL"
"""Show the active model row in the startup welcome banner when enabled.

Off by default; the model is always visible in the status bar, so the banner
row is opt-in to avoid duplicating it.
"""

SUPPRESS_ENV_OVERRIDE_WARNING = "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING"
"""Silence the startup warning emitted when a `DEEPAGENTS_CODE_`-prefixed
LangSmith variable overrides its canonical counterpart (e.g. both
`LANGSMITH_API_KEY` and `DEEPAGENTS_CODE_LANGSMITH_API_KEY` are set to
different values).

The override is intentional: the prefixed value overwrites the canonical
variable inside the Deep Agents Code process (so the LangSmith SDK, which
only reads canonical names, picks it up). The value you exported in your own
shell is unaffected, since a process cannot change its parent's environment.
Off by default; set to a truthy value (`1`, `true`, `yes`, `on`) to suppress
the warning when this coexistence is expected. Parsed by `is_env_truthy`.
"""

THEME = "DEEPAGENTS_CODE_THEME"
"""Force the CLI to launch with this theme name when set."""

USER_ID = "DEEPAGENTS_CODE_USER_ID"
"""Attach a user identifier to LangSmith trace metadata."""

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", ""})


def classify_env_bool(raw: str) -> bool | None:
    """Classify a raw env-var string as a truthy, falsy, or unrecognized token.

    The single source of truth for which strings count as boolean on/off
    values; `is_env_truthy` and the config resolver both build on it so they
    agree on what "recognizably boolean" means.

    Args:
        raw: The raw (unstripped) environment-variable value.

    Returns:
        `True` for `1`/`true`/`yes`/`on`, `False` for `0`/`false`/`no`/`off`/
            empty string (case-insensitive), or `None` when the value
            is neither.
    """
    lowered = raw.strip().lower()
    if lowered in _TRUTHY_VALUES:
        return True
    if lowered in _FALSY_VALUES:
        return False
    return None


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
    classified = classify_env_bool(raw)
    return default if classified is None else classified
