"""Terminal capability detection.

Detect optional terminal features without reading from `stdin`.

The CLI only uses kitty-keyboard-protocol support to choose a user-facing
newline shortcut label. To keep startup safe on remote or high-latency PTYs,
detection is conservative and relies on side-effect-free terminal identity
signals plus an explicit environment-variable override.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import cache
from typing import TYPE_CHECKING

from deepagents_cli._env_vars import KITTY_KEYBOARD

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_KNOWN_KITTY_KEYBOARD_TERMS = frozenset({"xterm-ghostty", "xterm-kitty"})


def _override_supports_kitty_keyboard_protocol(
    env: Mapping[str, str],
) -> bool | None:
    """Return an explicit kitty-keyboard override from `env`, if present.

    Accepted truthy values are `'1'`, `'true'`, `'yes'`, and `'on'`.
    Accepted falsy values are `'0'`, `'false'`, `'no'`, and `'off'`.
    `'auto'`, the empty string, and invalid values fall back to heuristic
    detection.
    """
    raw = env.get(KITTY_KEYBOARD)
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    logger.warning(
        "%s=%r ignored; expected one of: %s, or 'auto' to defer to detection.",
        KITTY_KEYBOARD,
        raw,
        ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES)),
    )
    return None


def _terminal_identity_supports_kitty_keyboard_protocol(
    env: Mapping[str, str],
) -> bool:
    """Return whether `env` identifies a terminal with built-in kitty support.

    This intentionally only recognizes terminals whose environment markers
    imply kitty-keyboard support is part of the terminal's default identity.
    Configurable terminals such as iTerm2 and WezTerm are intentionally not
    auto-detected because protocol support can be disabled in user settings.
    """
    if env.get("KITTY_WINDOW_ID"):
        return True

    term = env.get("TERM", "")
    return term in _KNOWN_KITTY_KEYBOARD_TERMS


@cache
def supports_kitty_keyboard_protocol() -> bool:
    """Return whether the attached terminal should be treated as kitty-aware.

    Detection is side-effect free: it never writes escape sequences or reads
    queued input bytes. That means it may under-detect some configurable
    terminals, but it will not interfere with Textual's input stream.

    Set `DEEPAGENTS_CLI_KITTY_KEYBOARD` to an accepted truthy value (`1`,
    `true`, `yes`, `on`) to force-enable the label, a falsy value (`0`,
    `false`, `no`, `off`) to force-disable it, or `auto`/unset to use
    heuristic detection.

    Returns:
        `True` when the terminal is known to support the kitty keyboard
        protocol, `False` otherwise.
    """
    if sys.platform == "win32":
        logger.debug("kitty kbd detection: False (win32 unsupported)")
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        logger.debug("kitty kbd detection: False (stdin/stdout not a tty)")
        return False

    override = _override_supports_kitty_keyboard_protocol(os.environ)
    if override is not None:
        logger.debug("kitty kbd detection: %s (explicit override)", override)
        return override

    detected = _terminal_identity_supports_kitty_keyboard_protocol(os.environ)
    logger.debug(
        "kitty kbd detection: %s (terminal identity TERM=%r KITTY_WINDOW_ID=%r)",
        detected,
        os.environ.get("TERM", ""),
        os.environ.get("KITTY_WINDOW_ID"),
    )
    return detected
