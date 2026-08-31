"""macOS `dcode://` registration through an AppleScript applet.

Launch Services only dispatches a URL scheme to an application bundle, so a
console script cannot claim one on its own. The bundle here is the smallest
thing that can: an AppleScript applet whose `on open location` handler asks a
terminal to run `dcode url open <link>`, with `CFBundleURLTypes` in its
`Info.plist` declaring the scheme.

The applet is the only place in this feature where a link is interpolated into a
shell command, so it is built to make that safe by construction. The dcode path
is quoted with `shlex.quote` when the applet is generated, and the link is quoted
at dispatch time by AppleScript's `quoted form of`, which is the language's own
POSIX-shell quoter. No other part of the link reaches the command line: the
applet does not inspect, split, or reassemble it.

The bundle lives in `~/Applications`, so installing needs no administrator
rights and uninstalling is a directory removal.
"""

from __future__ import annotations

import logging
import os
import plistlib
import shlex
import shutil
import subprocess  # noqa: S404  # fixed-argv macOS system tools
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from deepagents_code.url_scheme.registration import (
    HandlerStatus,
    RegistrationError,
    TerminalChoice,
    build_status,
)
from deepagents_code.url_scheme.request import URL_SCHEME

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

BUNDLE_ID: Final = "com.langchain.deepagents-code.url-handler"
"""Bundle identifier, also the marker that a bundle is ours to replace."""

APP_NAME: Final = "dcode.app"
"""Bundle name.

Browsers label their "open this link?" prompt with the handler's display name,
so the bundle is named after the command it runs: the prompt reads "Open dcode?"
rather than naming some helper the user has never heard of.
"""

_LAUNCHER_KEY: Final = "DcodeUrlHandlerLauncher"
_TERMINAL_KEY: Final = "DcodeUrlHandlerTerminal"
"""Custom `Info.plist` keys recording what the applet was built to run.

`dcode url status` reads them back to report which dcode an installed handler
launches, which is how it notices a handler left behind pointing at a dcode that
has since moved.
"""

_OSACOMPILE: Final = Path("/usr/bin/osacompile")
_OSASCRIPT: Final = Path("/usr/bin/osascript")
_LSREGISTER: Final = Path(
    "/System/Library/Frameworks/CoreServices.framework/Frameworks"
    "/LaunchServices.framework/Support/lsregister"
)
_COMMAND_TIMEOUT: Final = 60.0


def app_path() -> Path:
    """Return the bundle location.

    Returns:
        Path of the applet bundle, whether or not it exists.
    """
    return Path.home() / "Applications" / APP_NAME


def install(launcher: Path, *, terminal: TerminalChoice) -> Path:
    """Build and register the applet bundle.

    Args:
        launcher: Absolute path of the dcode console script to run.
        terminal: Terminal the applet should open the session in.

    Returns:
        Path of the installed bundle.

    Raises:
        RegistrationError: `osacompile` is unavailable, compilation failed, the
            bundle could not be written, or an unrelated application already
            occupies the bundle path.
    """
    resolved_terminal = _resolve_terminal(terminal)
    bundle = app_path()
    _clear_existing_bundle(bundle)

    if not _OSACOMPILE.is_file():
        msg = (
            f"Cannot build the URL handler: {_OSACOMPILE} is missing. It ships "
            "with macOS, so this may be a stripped-down system image."
        )
        raise RegistrationError(msg)

    source = _applet_source(launcher, resolved_terminal)
    with tempfile.TemporaryDirectory(prefix="dcode-url-scheme-") as tmp:
        script = Path(tmp) / "handler.applescript"
        script.write_text(source, encoding="utf-8")
        _run(
            [str(_OSACOMPILE), "-o", str(bundle), str(script)],
            what="compile the URL handler applet",
        )

    _write_bundle_metadata(bundle, launcher=launcher, terminal=resolved_terminal)
    _register_with_launch_services(bundle)
    return bundle


def uninstall() -> Sequence[str]:
    """Unregister and delete the applet bundle.

    Returns:
        The bundle path when one was removed, otherwise an empty sequence.

    Raises:
        RegistrationError: The bundle exists but could not be removed.
    """
    bundle = app_path()
    if not bundle.exists():
        return ()
    if _bundle_identifier(bundle) != BUNDLE_ID:
        msg = (
            f"Refusing to remove {bundle}: it is not dcode's URL handler. Remove "
            "it yourself if that is what you intended."
        )
        raise RegistrationError(msg)

    if _LSREGISTER.is_file():
        # Best-effort: Launch Services also drops handlers whose bundle is gone.
        _run(
            [str(_LSREGISTER), "-u", str(bundle)],
            what="unregister the URL handler",
            required=False,
        )
    try:
        shutil.rmtree(bundle)
    except OSError as exc:
        msg = f"Could not remove {bundle}: {exc}"
        raise RegistrationError(msg) from exc
    return (str(bundle),)


def status() -> HandlerStatus:
    """Report the applet bundle's state.

    Returns:
        Current status. `default_handler` is always `None`: macOS exposes no
        first-party command that reports the application bound to a scheme, so
        the report covers dcode's own artifact and leaves the binding to the
        browser's own prompt.
    """
    bundle = app_path()
    plist = _read_plist(bundle)
    installed = plist is not None and plist.get("CFBundleIdentifier") == BUNDLE_ID
    if not installed:
        return build_status(
            installed=False,
            handler_path=None,
            default_handler=None,
            detail=(
                f"No {URL_SCHEME}:// handler installed. Run `dcode url install` "
                "to register one."
            ),
        )

    assert plist is not None  # noqa: S101  # `installed` implies a parsed plist
    recorded = plist.get(_LAUNCHER_KEY)
    terminal = plist.get(_TERMINAL_KEY, "unknown")
    detail = f"Handler installed; links open in {terminal}."
    if isinstance(recorded, str) and recorded:
        detail += f" It runs {recorded}."
    return build_status(
        installed=True,
        handler_path=str(bundle),
        default_handler=None,
        detail=detail,
    )


def _resolve_terminal(terminal: TerminalChoice) -> TerminalChoice:
    """Resolve `AUTO` to a concrete terminal, and check an explicit one.

    `osacompile` resolves an application's AppleScript terminology at build
    time, so a dispatch to a terminal that is not installed cannot be compiled.
    Terminal.app is always present; iTerm has to be checked. `AUTO` degrades to
    Terminal.app so it cannot fail, while an explicit `--terminal iterm` says
    what is wrong instead of surfacing a compiler error.

    Args:
        terminal: Requested terminal.

    Returns:
        The terminal to build the applet for. Terminals with no scriptable
            `do script` equivalent are not offered, and resolve to Terminal.app.

    Raises:
        RegistrationError: `ITERM` was requested explicitly but iTerm is not
            installed.
    """
    if terminal is TerminalChoice.ITERM:
        if _iterm_available():
            return TerminalChoice.ITERM
        msg = (
            "Cannot build an iTerm handler: iTerm does not appear to be "
            "installed, so its AppleScript terminology cannot be resolved. "
            "Install iTerm, or use `--terminal terminal`."
        )
        raise RegistrationError(msg)
    if terminal is TerminalChoice.TERMINAL:
        return TerminalChoice.TERMINAL

    from deepagents_code._env_vars import LAUNCH_TERM_PROGRAM

    # The snapshot `cli_main` takes at entry, so a project `.env` cannot decide
    # which terminal gets baked into an installed handler.
    launch_term = os.environ.get(LAUNCH_TERM_PROGRAM, "").strip().lower()
    if launch_term.startswith("iterm") and _iterm_available():
        return TerminalChoice.ITERM
    return TerminalChoice.TERMINAL


def _iterm_available() -> bool:
    """Report whether iTerm's AppleScript terminology can be resolved.

    Asks for the application's bundle id, which is the same lookup
    `osacompile` performs for a `tell application "iTerm"` block.

    Returns:
        Whether iTerm was found.
    """
    if not _OSASCRIPT.is_file():
        return False
    try:
        result = subprocess.run(  # noqa: S603  # fixed argv, absolute system path
            [str(_OSASCRIPT), "-e", 'id of application "iTerm"'],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Could not probe for iTerm", exc_info=True)
        return False
    return result.returncode == 0


def _applet_source(launcher: Path, terminal: TerminalChoice) -> str:
    """Build the applet's AppleScript source.

    Args:
        launcher: Absolute path of the dcode console script.
        terminal: Terminal to open the session in.

    Returns:
        AppleScript source text.
    """
    command = _applescript_string(f"exec {shlex.quote(str(launcher))} url open ")
    dispatch = (
        _ITERM_DISPATCH if terminal is TerminalChoice.ITERM else _TERMINAL_DISPATCH
    )
    return _APPLET_TEMPLATE.format(
        command=command,
        dispatch=dispatch,
        run_message=_applescript_string(_RUN_MESSAGE),
    )


def _applescript_string(value: str) -> str:
    """Quote `value` as an AppleScript string literal.

    Args:
        value: Text to embed in generated source.

    Returns:
        The quoted literal, backslashes and double quotes escaped.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _clear_existing_bundle(bundle: Path) -> None:
    """Remove a previous dcode bundle so `osacompile` can write a fresh one.

    Args:
        bundle: Bundle path to clear.

    Raises:
        RegistrationError: The path holds something that is not dcode's handler,
            or it could not be removed.
    """
    if not bundle.exists():
        try:
            bundle.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"Could not create {bundle.parent}: {exc}"
            raise RegistrationError(msg) from exc
        return

    identifier = _bundle_identifier(bundle)
    if identifier is not None and identifier != BUNDLE_ID:
        msg = (
            f"Refusing to replace {bundle}: it belongs to another application "
            f"({identifier}). Move it aside and run this command again."
        )
        raise RegistrationError(msg)
    try:
        shutil.rmtree(bundle)
    except OSError as exc:
        msg = f"Could not replace {bundle}: {exc}"
        raise RegistrationError(msg) from exc


def _plist_path(bundle: Path) -> Path:
    """Return a bundle's `Info.plist` path.

    Args:
        bundle: Bundle path.

    Returns:
        Path of the bundle's `Info.plist`.
    """
    return bundle / "Contents" / "Info.plist"


def _read_plist(bundle: Path) -> dict[str, object] | None:
    """Read a bundle's `Info.plist`.

    Args:
        bundle: Bundle path.

    Returns:
        The parsed plist, or `None` when it is absent or unreadable.
    """
    try:
        with _plist_path(bundle).open("rb") as handle:
            parsed = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError):
        logger.debug("Could not read %s", _plist_path(bundle), exc_info=True)
        return None
    return parsed if isinstance(parsed, dict) else None


def _bundle_identifier(bundle: Path) -> str | None:
    """Read a bundle's identifier.

    Args:
        bundle: Bundle path.

    Returns:
        The `CFBundleIdentifier` value, or `None` when it cannot be read.
    """
    plist = _read_plist(bundle)
    identifier = None if plist is None else plist.get("CFBundleIdentifier")
    return identifier if isinstance(identifier, str) else None


def _write_bundle_metadata(
    bundle: Path, *, launcher: Path, terminal: TerminalChoice
) -> None:
    """Declare the URL scheme and identity in the compiled bundle's plist.

    `osacompile` writes a generic applet plist, so the scheme declaration, the
    identifier, and the display name browsers show in their prompt are added
    afterwards.

    Args:
        bundle: Compiled bundle to update.
        launcher: dcode path the applet runs, recorded for `dcode url status`.
        terminal: Terminal the applet opens, recorded for `dcode url status`.

    Raises:
        RegistrationError: The plist could not be read or written.
    """
    plist = _read_plist(bundle)
    if plist is None:
        msg = f"Compiled bundle at {bundle} has no readable Info.plist."
        raise RegistrationError(msg)

    plist.update(
        {
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleName": "dcode",
            "CFBundleDisplayName": "dcode",
            # The applet has no interface of its own; it hands off to a terminal
            # and exits, so it stays out of the Dock and the app switcher.
            "LSUIElement": True,
            "CFBundleURLTypes": [
                {
                    "CFBundleURLName": "deepagents code session",
                    "CFBundleTypeRole": "Viewer",
                    "CFBundleURLSchemes": [URL_SCHEME],
                }
            ],
            _LAUNCHER_KEY: str(launcher),
            _TERMINAL_KEY: str(terminal),
        }
    )
    try:
        with _plist_path(bundle).open("wb") as handle:
            plistlib.dump(plist, handle)
    except OSError as exc:
        msg = f"Could not write {_plist_path(bundle)}: {exc}"
        raise RegistrationError(msg) from exc


def _register_with_launch_services(bundle: Path) -> None:
    """Tell Launch Services about the bundle now, rather than on first scan.

    Best-effort: Launch Services also picks up a bundle in `~/Applications` on
    its own schedule, so a failure here delays the registration instead of
    breaking it.

    Args:
        bundle: Bundle to register.
    """
    if not _LSREGISTER.is_file():
        logger.debug("lsregister not found at %s; skipping", _LSREGISTER)
        return
    _run(
        [str(_LSREGISTER), "-f", str(bundle)],
        what="register the URL handler with Launch Services",
        required=False,
    )


def _run(argv: list[str], *, what: str, required: bool = True) -> None:
    """Run a fixed-path macOS command.

    Args:
        argv: Argument vector. Every element is either an absolute path this
            module owns or a path dcode resolved, never link-derived text.
        what: Phrase naming the step, used in the error message.
        required: When `False`, a failure is logged instead of raised.

    Raises:
        RegistrationError: The command failed, timed out, or could not be run
            while `required` is `True`.
    """
    try:
        result = subprocess.run(  # noqa: S603  # fixed argv, absolute system paths
            argv,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if not required:
            logger.warning("Could not %s: %s", what, exc)
            return
        msg = f"Could not {what}: {exc}"
        raise RegistrationError(msg) from exc

    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    reason = detail[-1] if detail else f"exit code {result.returncode}"
    if not required:
        logger.warning("Could not %s: %s", what, reason)
        return
    msg = f"Could not {what}: {reason}"
    raise RegistrationError(msg)


_TERMINAL_DISPATCH: Final = """\
	tell application "Terminal"
		activate
		do script sessionCommand
	end tell"""

_ITERM_DISPATCH: Final = """\
	tell application "iTerm"
		activate
		create window with default profile command sessionCommand
	end tell"""

_RUN_MESSAGE: Final = (
    "This helper opens dcode:// links in a terminal. There is nothing to open "
    "here - follow a dcode:// link, or run dcode from your terminal."
)

_APPLET_TEMPLATE: Final = """\
-- deepagents code URL handler.
-- Generated by `dcode url install`; re-run that command to rebuild it.
--
-- `quoted form of` is AppleScript's POSIX-shell quoter: the link becomes one
-- quoted argument, so no part of it can be read as shell syntax. The dcode
-- path was quoted when this applet was generated.
--
-- Kept to ASCII on purpose: this source is handed to `osacompile`.

on open location this_URL
	set sessionCommand to {command} & quoted form of this_URL
{dispatch}
end open location

on run
	display alert "dcode URL handler" message {run_message} as informational
end run
"""
