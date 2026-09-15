r"""Windows `dcode://` registration through the per-user protocol key.

Windows resolves a URL scheme through a `URL Protocol` key under
`Software\Classes`. dcode writes the `HKEY_CURRENT_USER` copy, which needs no
elevation and takes precedence over a machine-wide entry for this user, so
uninstalling is a key deletion rather than a repair.

`dcode` is a console application, so the shell gives the launch its own console
window and the TUI has somewhere to draw. No terminal has to be chosen or
installed, and no shell sits between the browser and dcode: the registered
command is dcode itself, with `"%1"` expanded into its argument vector.

`request` refuses every parameter that could weaken a session, and `handler`
requires an approval before launching, which is what keeps that last point from
mattering: even where the shell's quoting rules let a crafted link push an extra
token into the command line, there is no dangerous token to push.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from deepagents_code.url_scheme.registration import (
    HandlerStatus,
    RegistrationError,
    build_status,
)
from deepagents_code.url_scheme.request import URL_SCHEME

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

KEY_PATH: Final = f"Software\\Classes\\{URL_SCHEME}"
"""Per-user protocol key dcode owns."""

_COMMAND_SUBKEY: Final = "shell\\open\\command"
_MARKER_NAME: Final = "DcodeUrlHandler"
"""Value marking the key as dcode's, so uninstall cannot delete someone else's.

Windows has no per-key ownership, and the scheme name is short enough that
another tool could plausibly have claimed it. The marker is what makes removal
safe to automate.
"""


def install(launcher: Path) -> str:
    """Write the protocol key.

    Args:
        launcher: Absolute path of the dcode console script to run.

    Returns:
        The registry key path that was written.

    Raises:
        RegistrationError: The registry could not be written, or the key already
            belongs to another application.
    """
    import winreg

    existing_owner = _key_owner()
    if existing_owner is False:
        msg = (
            f"Refusing to replace HKCU\\{KEY_PATH}: another application already "
            f"handles {URL_SCHEME}://. Remove that registration first."
        )
        raise RegistrationError(msg)

    command = f'"{launcher}" url open "%1"'
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, KEY_PATH) as key:
            winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f"URL:{URL_SCHEME} protocol")
            # Presence, not content, is what marks a scheme key as a protocol.
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
            winreg.SetValueEx(key, _MARKER_NAME, 0, winreg.REG_SZ, str(launcher))
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, f"{KEY_PATH}\\{_COMMAND_SUBKEY}"
        ) as key:
            winreg.SetValueEx(key, None, 0, winreg.REG_SZ, command)
    except OSError as exc:
        msg = f"Could not write HKCU\\{KEY_PATH}: {exc}"
        raise RegistrationError(msg) from exc
    return f"HKCU\\{KEY_PATH}"


def uninstall() -> Sequence[str]:
    """Delete the protocol key.

    Returns:
        A one-element sequence naming the removed key, or an empty sequence when
            nothing was installed.

    Raises:
        RegistrationError: The key exists but is not dcode's, or it could not be
            removed.
    """
    import winreg

    owner = _key_owner()
    if owner is None:
        return ()
    if owner is False:
        msg = f"Refusing to remove HKCU\\{KEY_PATH}: it belongs to another application."
        raise RegistrationError(msg)

    # `DeleteKey` only removes a key with no subkeys, so unwind depth-first.
    subkeys = ("shell\\open\\command", "shell\\open", "shell")
    try:
        for subkey in subkeys:
            _delete_key_if_present(f"{KEY_PATH}\\{subkey}")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, KEY_PATH)
    except OSError as exc:
        msg = f"Could not remove HKCU\\{KEY_PATH}: {exc}"
        raise RegistrationError(msg) from exc
    return (f"HKCU\\{KEY_PATH}",)


def status() -> HandlerStatus:
    """Report the protocol key's state.

    Returns:
        Current status. `default_handler` names the application the key points
            at, which on Windows is the same question as which handler is
            registered.
    """
    owner = _key_owner()
    if owner is None:
        return build_status(
            installed=False,
            handler_path=None,
            default_handler=None,
            detail=(
                f"No {URL_SCHEME}:// handler installed. Run `dcode url install` "
                "to register one."
            ),
        )
    if owner is False:
        return build_status(
            installed=False,
            handler_path=None,
            default_handler=_command_value(),
            detail=(
                f"HKCU\\{KEY_PATH} belongs to another application, so "
                f"{URL_SCHEME}:// links do not reach dcode."
            ),
        )
    return build_status(
        installed=True,
        handler_path=f"HKCU\\{KEY_PATH}",
        default_handler=_command_value(),
        detail=f"Handler installed at HKCU\\{KEY_PATH}.",
    )


def _key_owner() -> bool | None:
    """Report who owns the protocol key.

    Returns:
        `True` when the key is dcode's, `False` when it exists but belongs to
            another application, and `None` when it does not exist.
    """
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY_PATH) as key:
            try:
                winreg.QueryValueEx(key, _MARKER_NAME)
            except OSError:
                return False
            return True
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Could not read HKCU\\%s", KEY_PATH, exc_info=True)
        return None


def _command_value() -> str | None:
    """Read the command the protocol key runs.

    Returns:
        The command string, or `None` when it cannot be read.
    """
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, f"{KEY_PATH}\\{_COMMAND_SUBKEY}"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None
    return value if isinstance(value, str) else None


def _delete_key_if_present(path: str) -> None:
    """Delete a registry key, ignoring one that is already gone.

    Propagates `OSError` when the key exists but cannot be deleted; the caller
    turns that into a `RegistrationError` naming the whole key.

    Args:
        path: Key path under `HKEY_CURRENT_USER`.
    """
    import winreg

    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
    except FileNotFoundError:
        return
