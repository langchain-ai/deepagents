"""Fixed operating-system paths for managed configuration."""

import logging
import sys
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

_PROGRAM_DATA_DEFAULT = "C:/ProgramData"


class _PathState:
    """Whether the last Windows path resolution fell back to the default."""

    registry_fallback: str | None = None


_path_state = _PathState()


def managed_path_fallback() -> str | None:
    """Return why the managed path is a guess, or `None` when it is authoritative.

    A failed registry query leaves the lookup pointing at the hardcoded
    default. On a host whose ProgramData is relocated, the guessed path holds
    no file, which reads as "no administrator deployed policy" — the same state
    as a machine with no policy at all. Callers that report health surface this
    so the two are distinguishable.

    Returns:
        A short reason, or `None` when the path came from the registry or from
        a platform that needs no registry lookup.
    """
    return _path_state.registry_fallback


def _program_data_from_registry() -> str | None:
    """Read ProgramData from the Windows registry, if available.

    Returns:
        The registry-reported ProgramData path, or `None` off-Windows or when
        the registry query fails.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "Common AppData")
    except (ImportError, OSError) as exc:
        _path_state.registry_fallback = (
            f"ProgramData could not be read from the registry ({type(exc).__name__}); "
            f"looked under {_PROGRAM_DATA_DEFAULT}"
        )
        logger.warning(
            "Could not read ProgramData from the registry (%s); falling back to "
            "%s. Managed policy stored elsewhere will not be found.",
            type(exc).__name__,
            _PROGRAM_DATA_DEFAULT,
        )
        return None
    if isinstance(value, str) and value:
        _path_state.registry_fallback = None
        return value
    _path_state.registry_fallback = (
        f"registry ProgramData value is unusable; looked under {_PROGRAM_DATA_DEFAULT}"
    )
    logger.warning(
        "Registry ProgramData value is unusable (%r); falling back to %s. "
        "Managed policy stored elsewhere will not be found.",
        value,
        _PROGRAM_DATA_DEFAULT,
    )
    return None


def _windows_program_data(environ: Mapping[str, str] | None) -> str:
    r"""Resolve the real ProgramData directory, ignoring process env vars.

    `%ProgramData%` can be redefined by any unprivileged user in their own
    shell, which would redirect the managed-config lookup to a user-controlled
    path and silently drop (or replace) administrator policy. Read the value
    from the registry (`HKLM\...\Shell Folders\Common AppData`) and fall
    back to the hardcoded default only if the registry query fails.

    Returns:
        The administrator-owned ProgramData directory.
    """
    if environ is not None:
        # Tests inject a fake environ to exercise platform logic off-Windows.
        return (
            environ.get("ProgramData")
            or environ.get("PROGRAMDATA")
            or _PROGRAM_DATA_DEFAULT
        )
    return _program_data_from_registry() or _PROGRAM_DATA_DEFAULT


def managed_config_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Return the fixed managed-config path for the current operating system.

    Args:
        platform: Override the detected platform; intended for tests.
        environ: Test-only injection point for the Windows branch. Production
            passes `None` so the ProgramData directory comes from the
            registry. Passing a real environment restores the redirection this
            module exists to prevent.

    Returns:
        The managed-config path for the platform.
    """
    active_platform = sys.platform if platform is None else platform
    if active_platform == "darwin":
        return Path("/Library/Application Support/dcode/managed_config.toml")
    if active_platform == "win32":
        return Path(_windows_program_data(environ)) / "dcode" / "managed_config.toml"
    return Path("/etc/dcode/managed_config.toml")
