r"""Registering and unregistering dcode as the `dcode://` handler.

Registration is always an explicit user action (`dcode url install`). Installing
dcode does not claim the scheme, because taking over a URL scheme changes how
the user's browser behaves and is not a side effect an install should have.

Each platform gets the artifact its desktop stack actually reads, all of them
user-scoped so no step needs administrator rights and uninstall is a file or key
removal:

- macOS: an AppleScript applet in `~/Applications` whose `Info.plist` declares
    `CFBundleURLTypes`, registered with Launch Services.
- Linux: a `.desktop` entry in the XDG data directory declaring
    `x-scheme-handler/dcode`, made the default through `xdg-mime`.
- Windows: the `HKCU\Software\Classes\dcode` protocol key.

The handler command every backend writes is the same shape — the absolute path
of this dcode's console script, then `url open`, then the link — so the parsing,
confirmation, and launch rules in `request` and `handler` apply no matter which
desktop stack delivered the link.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.url_scheme.request import URL_SCHEME

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_SUPPORTED_PLATFORMS = "macOS, Linux, and Windows"


class RegistrationError(RuntimeError):
    """Registration could not be completed.

    The message reaches the user and says what could not be done, so it names
    the tool or path that failed rather than only reporting failure.
    """


class TerminalChoice(StrEnum):
    """Terminal a macOS link opens the session in.

    A link arrives with no terminal attached, so the macOS applet has to ask one
    to run the session. Both supported values are scriptable terminals that ship
    an AppleScript `do script` command; `AUTO` reads `TERM_PROGRAM` at install
    time so the choice matches the terminal the user was in when they installed.
    """

    AUTO = "auto"
    TERMINAL = "terminal"
    ITERM = "iterm"


@dataclass(frozen=True)
class HandlerStatus:
    """What the operating system currently knows about the `dcode://` scheme.

    Attributes:
        scheme: The scheme this status describes.
        platform: `sys.platform` value the status was collected on.
        supported: Whether registration is implemented for this platform.
        installed: Whether dcode's own handler artifact is present.
        handler_path: The artifact — app bundle path, desktop entry path, or
            registry key — or `None` when nothing is installed. A string because
            not every platform's artifact is a filesystem path.
        launcher: Absolute path of the dcode console script the handler runs, or
            `None` when it could not be resolved.
        default_handler: What the desktop currently opens the scheme with, where
            the platform can answer that; `None` when it cannot be queried.
        detail: One-line human-readable summary.
    """

    scheme: str
    platform: str
    supported: bool
    installed: bool
    handler_path: str | None
    launcher: Path | None
    default_handler: str | None
    detail: str


def resolve_launcher() -> Path:
    """Resolve the absolute path of the dcode console script to register.

    A link is delivered by the browser or desktop launcher, whose environment
    has little to do with the user's shell, so a bare command name in a
    registered handler is the standard way this breaks. Everything written to
    disk gets an absolute path instead.

    The path is deliberately *not* symlink-resolved. A `uv tool` or `pipx`
    install exposes `~/.local/bin/dcode` as a symlink into a versioned
    environment; recording the symlink keeps the handler pointing at whatever
    dcode the user has installed, while recording its target would pin the
    handler to today's version and break it on the next upgrade.

    Returns:
        Absolute path of the console script.

    Raises:
        RegistrationError: No dcode console script could be located.
    """
    from deepagents_code._invocation import STANDARD_INVOKED_NAMES, invoked_name

    argv0 = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if (
        argv0 is not None
        and argv0.is_absolute()
        and argv0.suffix.lower() != ".py"
        and argv0.is_file()
    ):
        return argv0

    for name in (invoked_name(), *sorted(STANDARD_INVOKED_NAMES)):
        found = shutil.which(name)
        if found:
            return Path(found)

    msg = (
        "Could not find the dcode command to register. Install dcode so that "
        "`dcode` is on PATH (for example `uv tool install deepagents-code`), "
        "then run this command again."
    )
    raise RegistrationError(msg)


def install_handler(*, terminal: TerminalChoice = TerminalChoice.AUTO) -> HandlerStatus:
    """Register dcode as the operating system's `dcode://` handler.

    Idempotent: an existing registration is replaced with one built from the
    current dcode path and options.

    Args:
        terminal: Terminal a link should open the session in. macOS only; other
            platforms have a desktop-level or shell-level answer already.

    Returns:
        Status collected after registering.

    Raises:
        RegistrationError: The platform is unsupported, the dcode command could
            not be located, or a registration step failed.
    """
    launcher = resolve_launcher()
    if sys.platform == "darwin":
        from deepagents_code.url_scheme import _macos

        _macos.install(launcher, terminal=terminal)
    elif sys.platform.startswith("linux"):
        from deepagents_code.url_scheme import _linux

        _linux.install(launcher)
    elif sys.platform == "win32":
        from deepagents_code.url_scheme import _windows

        _windows.install(launcher)
    else:
        raise RegistrationError(_unsupported_message())
    return handler_status()


def uninstall_handler() -> tuple[HandlerStatus, Sequence[str]]:
    """Remove dcode's `dcode://` handler.

    Idempotent: removing a handler that is not installed succeeds and reports
    that nothing was removed.

    Returns:
        The status collected after removal, and the artifacts that were removed.

    Raises:
        RegistrationError: The platform is unsupported, or an artifact exists but
            could not be removed.
    """
    if sys.platform == "darwin":
        from deepagents_code.url_scheme import _macos

        removed = _macos.uninstall()
    elif sys.platform.startswith("linux"):
        from deepagents_code.url_scheme import _linux

        removed = _linux.uninstall()
    elif sys.platform == "win32":
        from deepagents_code.url_scheme import _windows

        removed = _windows.uninstall()
    else:
        raise RegistrationError(_unsupported_message())
    return handler_status(), removed


def handler_status() -> HandlerStatus:
    """Report what the operating system knows about the `dcode://` scheme.

    Never raises: status is the command a user runs when something is wrong, so
    an unresolvable launcher or an unsupported platform is part of the report
    rather than an error.

    Returns:
        The current status.
    """
    if sys.platform == "darwin":
        from deepagents_code.url_scheme import _macos

        return _macos.status()
    if sys.platform.startswith("linux"):
        from deepagents_code.url_scheme import _linux

        return _linux.status()
    if sys.platform == "win32":
        from deepagents_code.url_scheme import _windows

        return _windows.status()
    return HandlerStatus(
        scheme=URL_SCHEME,
        platform=sys.platform,
        supported=False,
        installed=False,
        handler_path=None,
        launcher=None,
        default_handler=None,
        detail=_unsupported_message(),
    )


def build_status(
    *,
    installed: bool,
    handler_path: str | None,
    default_handler: str | None,
    detail: str,
) -> HandlerStatus:
    """Assemble a `HandlerStatus` for the running platform.

    Shared by the platform backends so every status carries the same scheme,
    platform tag, and best-effort launcher path.

    Args:
        installed: Whether dcode's handler artifact is present.
        handler_path: The artifact's path or registry key, or `None`.
        default_handler: What the desktop opens the scheme with, where known.
        detail: One-line human-readable summary.

    Returns:
        The assembled status.
    """
    try:
        launcher: Path | None = resolve_launcher()
    except RegistrationError:
        launcher = None
    return HandlerStatus(
        scheme=URL_SCHEME,
        platform=sys.platform,
        supported=True,
        installed=installed,
        handler_path=handler_path,
        launcher=launcher,
        default_handler=default_handler,
        detail=detail,
    )


def _unsupported_message() -> str:
    """Return the message for a platform with no registration backend.

    Returns:
        The message.
    """
    return (
        f"Registering the {URL_SCHEME}:// scheme is supported on "
        f"{_SUPPORTED_PLATFORMS}, not on {sys.platform!r}."
    )
