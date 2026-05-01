"""Canonical registry of `DEEPAGENTS_CLI_*` environment variables.

Every env var the CLI reads whose name starts with `DEEPAGENTS_CLI_` must
be defined here as a module-level constant.  A drift-detection test
(`tests/unit_tests/test_env_vars.py`) fails when a bare string literal
like `"DEEPAGENTS_CLI_FOO"` appears in source code instead of a constant
imported from this module.

Import the short-name constants (e.g. `AUTO_UPDATE`, `DEBUG`) and pass them
to `os.environ.get()` instead of using raw string literals. If the env var is
ever renamed, only the value here changes.

!!! note

    `resolve_env_var` also supports a dynamic prefix override for API keys
    and provider credentials: setting `DEEPAGENTS_CLI_{NAME}` takes priority
    over `{NAME}`.  For example, `DEEPAGENTS_CLI_OPENAI_API_KEY` overrides
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

AUTO_UPDATE = "DEEPAGENTS_CLI_AUTO_UPDATE"
"""Enable automatic CLI updates ('1', 'true', or 'yes')."""

DEBUG = "DEEPAGENTS_CLI_DEBUG"
"""Enable verbose debug logging and preserve the server subprocess log.

Parsed by `is_env_truthy`: accepts `1`, `true`, `yes`, `on` (case-insensitive)
as enabled, and `0`, `false`, `no`, `off`, empty string, or unset as disabled."""

DEBUG_FILE = "DEEPAGENTS_CLI_DEBUG_FILE"
"""Path for the debug log file (default: `/tmp/deepagents_debug.log`)."""

DEBUG_NOTIFICATIONS = "DEEPAGENTS_CLI_DEBUG_NOTIFICATIONS"
"""Inject sample missing-dependency notifications at launch so the notification
center UI can be exercised without waiting for real conditions. Does not
auto-open the update modal (use `DEEPAGENTS_CLI_DEBUG_UPDATE` for that). Any
non-empty value enables the flag (including `"0"` or `"false"`)."""

DEBUG_UPDATE = "DEEPAGENTS_CLI_DEBUG_UPDATE"
"""Inject a sample update-available notification and auto-open the update modal
at launch so the update-available flow can be exercised without waiting for a
real PyPI release. Any non-empty value enables the flag (including `"0"` or
`"false"`)."""

EXTRA_SKILLS_DIRS = "DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS"
"""Colon-separated paths added to the skill containment allowlist."""

KITTY_KEYBOARD = "DEEPAGENTS_CLI_KITTY_KEYBOARD"
"""Override kitty-keyboard detection (`1` forces on, `0` forces off)."""

LANGSMITH_PROJECT = "DEEPAGENTS_CLI_LANGSMITH_PROJECT"
"""Override LangSmith project name for agent traces."""

NO_UPDATE_CHECK = "DEEPAGENTS_CLI_NO_UPDATE_CHECK"
"""Disable automatic update checking when set."""

SERVER_ENV_PREFIX = "DEEPAGENTS_CLI_SERVER_"
"""Environment variable prefix used to pass CLI config to the server subprocess."""

SHELL_ALLOW_LIST = "DEEPAGENTS_CLI_SHELL_ALLOW_LIST"
"""Comma-separated shell commands to allow (or 'recommended'/'all')."""

USER_ID = "DEEPAGENTS_CLI_USER_ID"
"""Attach a user identifier to LangSmith trace metadata."""


_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", ""})


def is_env_truthy(name: str, *, default: bool = False) -> bool:
    """Return whether env var *name* is set to a recognizably truthy value.

    Unlike `bool(os.environ.get(name))`, this does not treat `"0"` or
    `"false"` as enabled. Use this for on/off flags where the user would
    reasonably expect `VAR=0` to mean "disabled".

    Args:
        name: Environment variable name (typically a `DEEPAGENTS_CLI_*`
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
