"""iTerm2 cursor guide workaround for Textual alternate-screen rendering."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from deepagents_code.terminal_capabilities import is_iterm2

logger = logging.getLogger(__name__)

# iTerm2's cursor guide (highlight cursor line) causes visual artifacts when
# Textual takes over the terminal in alternate screen mode. We disable it at
# module load and restore it on exit only if the active/default iTerm2 profile
# had cursor guide enabled before launch.

_IS_ITERM = is_iterm2()

_CURSOR_GUIDE_OSC = "1337"
"""iTerm2's proprietary OSC command number."""

# Payloads for `OSC 1337 ; HighlightCursorLine=<yes|no> ST`.
_CURSOR_GUIDE_OFF = "HighlightCursorLine=no"
_CURSOR_GUIDE_ON = "HighlightCursorLine=yes"
_ITERM_PREFS_PATH = Path("~/Library/Preferences/com.googlecode.iterm2.plist")


def _write_cursor_guide(payload: str) -> None:
    """Send a cursor-guide `OSC 1337` command to the terminal.

    Delegates to the shared escape writer, which prefers `/dev/tty` and wraps
    the sequence in tmux's passthrough envelope. tmux only forwards the
    handful of operating-system commands it understands and drops the rest, so
    writing `OSC 1337` straight to stderr is a silent no-op in every pane —
    the outer terminal is still iTerm2, but nothing reaches it.

    Failures are swallowed by the shared writer: this is cosmetic, so it must
    never crash the app.

    Args:
        payload: The `OSC 1337` command body, such as `HighlightCursorLine=no`.
    """
    if not _IS_ITERM:
        return

    from deepagents_code.terminal_escape import write_osc

    write_osc(_CURSOR_GUIDE_OSC, payload, st=True)


def _plist_bool(value: object) -> bool | None:
    """Return a plist boolean/int value as `bool`, or `None` if not boolean-like."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


def _profile_uses_cursor_guide(profile: dict[str, object]) -> bool:
    """Return whether an iTerm2 profile has cursor guide enabled."""
    enabled = _plist_bool(profile.get("Use Cursor Guide"))
    if enabled is not None:
        return enabled

    # Newer iTerm2 profiles may carry separate light/dark values. If the shared
    # value is absent, restoring when either variant is enabled preserves the
    # user's visible default better than losing the guide for both appearances.
    return any(
        _plist_bool(profile.get(key)) is True
        for key in ("Use Cursor Guide (Dark)", "Use Cursor Guide (Light)")
    )


def _coerce_profile(raw: object) -> dict[str, object] | None:
    """Return a string-keyed profile dictionary from raw plist data."""
    if not isinstance(raw, dict):
        return None
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _find_iterm_profile(
    profiles: list[object], *, name: str, guid: str
) -> dict[str, object] | None:
    """Find the current iTerm2 profile by name, then by default profile GUID.

    Args:
        profiles: Profile entries from iTerm2 preferences.
        name: Active profile name from `ITERM_PROFILE`.
        guid: Default profile GUID from iTerm2 preferences.

    Returns:
        The matching profile dictionary, or `None` when no match is found.
    """
    for raw in profiles:
        profile = _coerce_profile(raw)
        if profile is None:
            continue
        if profile.get("Name") == name:
            return profile
    for raw in profiles:
        profile = _coerce_profile(raw)
        if profile is None:
            continue
        if profile.get("Guid") == guid:
            return profile
    return None


def _iterm_profile_cursor_guide_enabled() -> bool:
    """Infer whether iTerm2 cursor guide was enabled before startup.

    iTerm2's OSC 1337 `HighlightCursorLine` command can set the guide to yes/no
    but does not report the current state. The best cheap signal available at
    startup is the active profile preference, exposed in the iTerm2 plist. The
    `ITERM_PROFILE` environment variable is set by iTerm2; when it is missing,
    fall back to the default profile GUID in preferences.

    Returns:
        `True` if the matched iTerm2 profile has cursor guide enabled.
    """
    if not _IS_ITERM:
        return False

    import plistlib

    try:
        with _ITERM_PREFS_PATH.expanduser().open("rb") as f:
            prefs = plistlib.load(f)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return False

    if not isinstance(prefs, dict):
        return False

    profiles = prefs.get("New Bookmarks")
    if not isinstance(profiles, list):
        return False

    name = os.environ.get("ITERM_PROFILE", "").strip()
    guid = str(prefs.get("Default Bookmark Guid", ""))
    profile = _find_iterm_profile(profiles, name=name, guid=guid)
    # Inside tmux `ITERM_PROFILE` is only as current as the server's start,
    # so record what was matched; `dcode doctor` reports the same hazard.
    logger.debug(
        "cursor guide: ITERM_PROFILE=%r matched profile %r",
        name,
        profile.get("Name") if profile else None,
    )
    if profile is None:
        return False
    return _profile_uses_cursor_guide(profile)


_RESTORE_ITERM_CURSOR_GUIDE = _iterm_profile_cursor_guide_enabled()
_ITERM_CURSOR_GUIDE_RESTORED = False


def restore_iterm_cursor_guide() -> None:
    """Restore iTerm2 cursor guide when launch-time profile state required it."""
    global _ITERM_CURSOR_GUIDE_RESTORED  # noqa: PLW0603  # atexit/exit idempotence

    if not _RESTORE_ITERM_CURSOR_GUIDE or _ITERM_CURSOR_GUIDE_RESTORED:
        return
    _ITERM_CURSOR_GUIDE_RESTORED = True
    _write_cursor_guide(_CURSOR_GUIDE_ON)


def _disable_iterm_cursor_guide() -> None:
    """Disable iTerm2 cursor guide only when the module has a restore path."""
    if not _RESTORE_ITERM_CURSOR_GUIDE:
        return
    _write_cursor_guide(_CURSOR_GUIDE_OFF)


# Disable cursor guide at module load (before Textual takes over), but only
# when launch-time state detection confirmed that exit cleanup will restore it.
_disable_iterm_cursor_guide()

if _RESTORE_ITERM_CURSOR_GUIDE:
    import atexit

    atexit.register(restore_iterm_cursor_guide)
