"""Fixed operating-system paths for managed configuration."""

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROGRAM_DATA_DEFAULT = "C:/ProgramData"


@dataclass(frozen=True, slots=True)
class ResolvedManagedPath:
    """Where managed policy is read from, and whether that location is certain.

    `fallback` holds why the path is a guess, or `None` when the path is
    authoritative. A failed registry query leaves the lookup pointing at the
    hardcoded default. On a host whose ProgramData is relocated, the guessed
    path holds no file, which reads as "no administrator deployed policy" — the
    same state as a machine with no policy at all. The reason travels with the
    path so the two can never be confused: a guessed path yields an
    `INDETERMINATE` snapshot rather than a clean `MISSING` one.
    """

    path: Path
    fallback: str | None = None


def _program_data_from_registry() -> tuple[str | None, str | None]:
    """Read ProgramData from the Windows registry, if available.

    Returns:
        The registry-reported ProgramData path and `None`, or `None` and the
        reason the lookup failed. Off-Windows both are `None`: that platform
        needs no registry lookup, so the path is not a guess.
    """
    if sys.platform != "win32":
        return None, None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "Common AppData")
    except (ImportError, OSError) as exc:
        logger.warning(
            "Could not read ProgramData from the registry (%s); falling back to "
            "%s. Managed policy stored elsewhere will not be found.",
            type(exc).__name__,
            _PROGRAM_DATA_DEFAULT,
        )
        return None, (
            f"ProgramData could not be read from the registry ({type(exc).__name__}); "
            f"looked under {_PROGRAM_DATA_DEFAULT}"
        )
    if isinstance(value, str) and value:
        return value, None
    logger.warning(
        "Registry ProgramData value is unusable (%r); falling back to %s. "
        "Managed policy stored elsewhere will not be found.",
        value,
        _PROGRAM_DATA_DEFAULT,
    )
    return None, (
        f"registry ProgramData value is unusable; looked under {_PROGRAM_DATA_DEFAULT}"
    )


def _windows_program_data(
    environ: Mapping[str, str] | None,
) -> tuple[str, str | None]:
    r"""Resolve the real ProgramData directory, ignoring process env vars.

    `%ProgramData%` can be redefined by any unprivileged user in their own
    shell, which would redirect the managed-config lookup to a user-controlled
    path and silently drop (or replace) administrator policy. Read the value
    from the registry (`HKLM\...\Shell Folders\Common AppData`) and fall
    back to the hardcoded default only if the registry query fails.

    Returns:
        The administrator-owned ProgramData directory, and why it is a guess
        when the registry query failed.
    """
    if environ is not None:
        # Tests inject a fake environ to exercise platform logic off-Windows.
        return (
            environ.get("ProgramData")
            or environ.get("PROGRAMDATA")
            or _PROGRAM_DATA_DEFAULT
        ), None
    value, fallback = _program_data_from_registry()
    return value or _PROGRAM_DATA_DEFAULT, fallback


def _resolve(
    platform: str | None, environ: Mapping[str, str] | None
) -> ResolvedManagedPath:
    """Map one platform to its fixed managed-config path.

    Both public entry points delegate here rather than to each other, so a test
    that redirects one of them cannot change what the other computes.

    Returns:
        The path for the platform, paired with the reason it is a guess.
    """
    active_platform = sys.platform if platform is None else platform
    if active_platform == "darwin":
        return ResolvedManagedPath(
            Path("/Library/Application Support/dcode/managed_config.toml")
        )
    if active_platform == "win32":
        root, fallback = _windows_program_data(environ)
        return ResolvedManagedPath(
            Path(root) / "dcode" / "managed_config.toml", fallback
        )
    return ResolvedManagedPath(Path("/etc/dcode/managed_config.toml"))


def resolve_managed_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> ResolvedManagedPath:
    """Return the fixed managed-config path and whether it is authoritative.

    What the snapshot loader reads. Callers that report health need this rather
    than `managed_config_path`, so a guessed path is never mistaken for an
    authoritative one.

    Args:
        platform: Override the detected platform; intended for tests.
        environ: Test-only injection point for the Windows branch. Production
            passes `None` so the ProgramData directory comes from the registry.
            Passing a real environment restores the redirection this module
            exists to prevent.

    Returns:
        The path for the platform, paired with the reason it is a guess.
    """
    return _resolve(platform, environ)


def managed_config_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Return the fixed managed-config path for the current operating system.

    For display and error messages. Anything that decides whether policy is
    enforceable wants `resolve_managed_path`.

    Args:
        platform: Override the detected platform; intended for tests.
        environ: Test-only injection point for the Windows branch. See
            `resolve_managed_path`.

    Returns:
        The managed-config path for the platform.
    """
    return _resolve(platform, environ).path
