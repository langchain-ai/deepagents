"""Fixed operating-system paths for managed configuration."""

import sys
from collections.abc import Mapping
from pathlib import Path

_PROGRAM_DATA_DEFAULT = "C:/ProgramData"


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
    except (ImportError, OSError):
        return None
    return value if isinstance(value, str) and value else None


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
