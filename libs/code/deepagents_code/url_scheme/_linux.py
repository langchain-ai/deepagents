"""Linux `dcode://` registration through an XDG desktop entry.

A desktop entry declaring `x-scheme-handler/dcode` is what browsers and portals
consult, so that is the whole artifact. `Terminal=true` asks the desktop to run
the command in the user's terminal, which is how a TUI gets a usable window
without dcode having to guess which terminal emulator is installed.

Nothing on this path goes through a shell: the desktop launcher expands `%u` to
the link as a single argument vector element, so the link cannot be read as
shell syntax no matter what it contains.

`xdg-mime` sets the default when it is available. When it is not — a minimal
container, a system without `xdg-utils` — the same association is written
directly to `mimeapps.list`, editing only dcode's own keys so a hand-tuned file
survives.
"""

from __future__ import annotations

import configparser
import logging
import os
import shutil
import subprocess  # noqa: S404  # fixed-argv desktop integration tools
from pathlib import Path
from typing import TYPE_CHECKING, Final

from deepagents_code.url_scheme.registration import (
    HandlerStatus,
    RegistrationError,
    build_status,
)
from deepagents_code.url_scheme.request import URL_SCHEME

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DESKTOP_FILE_NAME: Final = "dcode-url-handler.desktop"
"""Desktop entry filename, also the id `xdg-mime` associates with the scheme."""

MIME_TYPE: Final = f"x-scheme-handler/{URL_SCHEME}"
"""The pseudo-MIME type XDG uses to name a URL scheme."""

_DEFAULT_SECTION: Final = "Default Applications"
_ADDED_SECTION: Final = "Added Associations"
_COMMAND_TIMEOUT: Final = 30.0


def desktop_file_path() -> Path:
    """Return the desktop entry location.

    Returns:
        Path of the entry under the XDG data directory, whether or not it
            exists.
    """
    return _data_home() / "applications" / DESKTOP_FILE_NAME


def install(launcher: Path) -> Path:
    """Write the desktop entry and make it the scheme's default.

    Args:
        launcher: Absolute path of the dcode console script to run.

    Returns:
        Path of the installed desktop entry.

    Raises:
        RegistrationError: The dcode path cannot be expressed in a desktop entry,
            the entry could not be written, or the default could not be set.
    """
    entry = desktop_file_path()
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(_desktop_entry(launcher), encoding="utf-8")
    except OSError as exc:
        msg = f"Could not write {entry}: {exc}"
        raise RegistrationError(msg) from exc

    _update_desktop_database(entry.parent)
    if not _set_default_with_xdg_mime():
        _write_mimeapps_default()
    return entry


def uninstall() -> Sequence[str]:
    """Remove the desktop entry and dcode's scheme associations.

    Returns:
        The artifacts that were changed or removed.

    Raises:
        RegistrationError: The desktop entry exists but could not be removed.
    """
    removed: list[str] = []
    entry = desktop_file_path()
    if entry.exists():
        try:
            entry.unlink()
        except OSError as exc:
            msg = f"Could not remove {entry}: {exc}"
            raise RegistrationError(msg) from exc
        removed.append(str(entry))
        _update_desktop_database(entry.parent)

    if _clear_mimeapps_default():
        removed.append(str(_mimeapps_path()))
    return removed


def status() -> HandlerStatus:
    """Report the desktop entry's state and the scheme's current default.

    Returns:
        Current status, including what `xdg-mime` reports as the scheme's
            default handler when it can be queried.
    """
    entry = desktop_file_path()
    installed = entry.is_file()
    default = _query_default_with_xdg_mime()
    if default is None:
        default = _mimeapps_default()

    if not installed:
        detail = (
            f"No {URL_SCHEME}:// handler installed. Run `dcode url install` to "
            "register one."
        )
        if default:
            detail += f" The scheme is currently associated with {default}."
        return build_status(
            installed=False,
            handler_path=None,
            default_handler=default,
            detail=detail,
        )

    if default == DESKTOP_FILE_NAME:
        detail = "Handler installed and set as the default for the scheme."
    elif default:
        detail = (
            f"Handler installed, but the desktop opens the scheme with "
            f"{default}. Re-run `dcode url install` to take it back."
        )
    else:
        detail = (
            "Handler installed. The scheme's default could not be read; your "
            "desktop may ask which application to use."
        )
    return build_status(
        installed=True,
        handler_path=str(entry),
        default_handler=default,
        detail=detail,
    )


def _data_home() -> Path:
    """Return `$XDG_DATA_HOME`, defaulting per the XDG base directory spec.

    Returns:
        The data home directory.
    """
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw.startswith("/"):
        return Path(raw)
    return Path.home() / ".local" / "share"


def _config_home() -> Path:
    """Return `$XDG_CONFIG_HOME`, defaulting per the XDG base directory spec.

    Returns:
        The config home directory.
    """
    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if raw.startswith("/"):
        return Path(raw)
    return Path.home() / ".config"


def _mimeapps_path() -> Path:
    """Return the user's `mimeapps.list` location.

    Returns:
        Path of the file, whether or not it exists.
    """
    return _config_home() / "mimeapps.list"


def _desktop_entry(launcher: Path) -> str:
    """Build the desktop entry text.

    Propagates `RegistrationError` from `_exec_value` when the dcode path holds
    a character the `Exec` grammar cannot carry.

    Args:
        launcher: Absolute path of the dcode console script.

    Returns:
        Desktop entry file contents.
    """
    return "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.0",
            # Browsers show this name when they ask whether to open the link.
            "Name=dcode",
            "GenericName=deepagents code",
            f"Comment=Open a deepagents code session from a {URL_SCHEME}:// link",
            f"Exec={_exec_value(launcher)}",
            # The desktop supplies the terminal window a TUI needs.
            "Terminal=true",
            "StartupNotify=false",
            "NoDisplay=false",
            "Categories=Development;",
            f"MimeType={MIME_TYPE};",
            "",
        )
    )


def _exec_value(launcher: Path) -> str:
    """Build the entry's `Exec` value.

    `%u` hands the link to dcode as one argument, so the desktop launcher never
    builds a shell command line out of it.

    Args:
        launcher: Absolute path of the dcode console script.

    Returns:
        The `Exec` value.

    Raises:
        RegistrationError: The path holds a character the `Exec` grammar reserves
            and cannot represent safely.
    """
    path = str(launcher)
    # The desktop entry spec escapes these inside a quoted argument, but a path
    # holding one is so unlikely that refusing beats emitting an entry whose
    # correctness nobody can check.
    forbidden = {'"', "\\", "`", "$", "\n", "\r"}
    found = sorted(forbidden.intersection(path))
    if found:
        msg = (
            f"Cannot register {path}: the path contains {''.join(found)!r}, which "
            "a desktop entry's Exec line cannot carry. Install dcode at a path "
            "without it."
        )
        raise RegistrationError(msg)
    return f'"{path}" url open %u'


def _update_desktop_database(applications_dir: Path) -> None:
    """Refresh the desktop database so the new association is visible.

    Best-effort: the file itself is authoritative, and desktops re-read it on
    their own schedule.

    Args:
        applications_dir: Directory holding the desktop entry.
    """
    tool = shutil.which("update-desktop-database")
    if tool is None:
        logger.debug("update-desktop-database not found; skipping")
        return
    _run([tool, str(applications_dir)], what="refresh the desktop database")


def _set_default_with_xdg_mime() -> bool:
    """Make dcode's entry the scheme's default via `xdg-mime`.

    Returns:
        Whether `xdg-mime` was available and succeeded.
    """
    tool = shutil.which("xdg-mime")
    if tool is None:
        logger.debug("xdg-mime not found; writing mimeapps.list directly")
        return False
    return _run(
        [tool, "default", DESKTOP_FILE_NAME, MIME_TYPE],
        what="set the scheme's default handler",
    )


def _query_default_with_xdg_mime() -> str | None:
    """Ask `xdg-mime` which entry currently handles the scheme.

    Returns:
        The desktop entry id, or `None` when `xdg-mime` is unavailable, failed,
            or reported no association.
    """
    tool = shutil.which("xdg-mime")
    if tool is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # resolved tool path, fixed argv
            [tool, "query", "default", MIME_TYPE],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("xdg-mime query failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _load_mimeapps() -> configparser.RawConfigParser:
    """Read `mimeapps.list` into a parser that preserves its keys verbatim.

    Returns:
        The parser, empty when the file is absent or unreadable.
    """
    parser = configparser.RawConfigParser(delimiters=("=",))
    # MIME types and desktop ids are case-sensitive; the default lower-casing
    # would rewrite every key in the user's file.
    parser.optionxform = str  # ty: ignore[invalid-assignment]
    path = _mimeapps_path()
    if not path.is_file():
        return parser
    try:
        parser.read(path, encoding="utf-8")
    except (OSError, configparser.Error):
        logger.warning("Could not parse %s; leaving it alone", path, exc_info=True)
        return configparser.RawConfigParser(delimiters=("=",))
    return parser


def _save_mimeapps(parser: configparser.RawConfigParser) -> None:
    """Write `mimeapps.list` back.

    Args:
        parser: Parser holding the file's new contents.

    Raises:
        RegistrationError: The file could not be written.
    """
    path = _mimeapps_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            parser.write(handle, space_around_delimiters=False)
    except OSError as exc:
        msg = f"Could not write {path}: {exc}"
        raise RegistrationError(msg) from exc


def _write_mimeapps_default() -> None:
    """Associate the scheme with dcode's entry directly in `mimeapps.list`.

    The fallback for systems without `xdg-utils`. Only dcode's own key is
    touched. Propagates `RegistrationError` from `_save_mimeapps` when the file
    cannot be written.
    """
    parser = _load_mimeapps()
    if not parser.has_section(_DEFAULT_SECTION):
        parser.add_section(_DEFAULT_SECTION)
    parser.set(_DEFAULT_SECTION, MIME_TYPE, DESKTOP_FILE_NAME)
    _save_mimeapps(parser)


def _clear_mimeapps_default() -> bool:
    """Remove dcode's scheme associations from `mimeapps.list`.

    Only entries naming dcode's own desktop file are removed, so an association
    the user pointed somewhere else is left as they set it. Propagates
    `RegistrationError` from `_save_mimeapps` when the file cannot be written.

    Returns:
        Whether the file was changed.
    """
    parser = _load_mimeapps()
    changed = False
    for section in (_DEFAULT_SECTION, _ADDED_SECTION):
        if not parser.has_section(section):
            continue
        value = parser.get(section, MIME_TYPE, fallback="")
        entries = [item for item in value.split(";") if item]
        remaining = [item for item in entries if item != DESKTOP_FILE_NAME]
        if remaining == entries:
            continue
        if remaining:
            parser.set(section, MIME_TYPE, ";".join(remaining) + ";")
        else:
            parser.remove_option(section, MIME_TYPE)
        changed = True

    if changed:
        _save_mimeapps(parser)
    return changed


def _mimeapps_default() -> str | None:
    """Read the scheme's default from `mimeapps.list`.

    The fallback for reporting status where `xdg-mime` is unavailable.

    Returns:
        The first desktop entry id associated with the scheme, or `None`.
    """
    parser = _load_mimeapps()
    for section in (_DEFAULT_SECTION, _ADDED_SECTION):
        value = parser.get(section, MIME_TYPE, fallback="")
        for item in value.split(";"):
            if item:
                return item
    return None


def _run(argv: list[str], *, what: str) -> bool:
    """Run a resolved desktop-integration tool.

    Args:
        argv: Argument vector. Every element is a resolved tool path or a
            constant this module owns, never link-derived text.
        what: Phrase naming the step, used in the log message.

    Returns:
        Whether the command succeeded.
    """
    try:
        result = subprocess.run(  # noqa: S603  # resolved tool path, fixed argv
            argv,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not %s: %s", what, exc)
        return False
    if result.returncode == 0:
        return True
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    logger.warning(
        "Could not %s: %s", what, detail[-1] if detail else result.returncode
    )
    return False
