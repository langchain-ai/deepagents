"""iTerm2 cursor guide workaround for Textual alternate-screen rendering."""

from __future__ import annotations

import os
from pathlib import Path

# iTerm2's cursor guide (highlight cursor line) causes visual artifacts when
# Textual takes over the terminal in alternate screen mode. We disable it at
# module load and restore it on exit only if the active/default iTerm2 profile
# had cursor guide enabled before launch.

# Detection: check env vars AND that stderr is a TTY (avoids false positives
# when env vars are inherited but running in non-TTY context like CI).
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
_ITERM_PREFS_PATH = Path("~/Library/Preferences/com.googlecode.iterm2.plist")


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
        # Terminal may be unavailable (redirected, closed, broken pipe).
        pass


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
    """Infer whether iTerm2 cursor guide was enabled before CLI startup.

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
    _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)


def _disable_iterm_cursor_guide() -> None:
    """Disable iTerm2 cursor guide only when the module has a restore path."""
    if not _RESTORE_ITERM_CURSOR_GUIDE:
        return
    _write_iterm_escape(_ITERM_CURSOR_GUIDE_OFF)


# Disable cursor guide at module load (before Textual takes over), but only
# when launch-time state detection confirmed that exit cleanup will restore it.
_disable_iterm_cursor_guide()

if _RESTORE_ITERM_CURSOR_GUIDE:
    import atexit

    atexit.register(restore_iterm_cursor_guide)
