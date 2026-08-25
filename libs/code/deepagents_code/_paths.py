"""Immutable filesystem paths captured before configuration is loaded.

`DEEPAGENTS_HOME` selects the user's profile and therefore a trust boundary. It
must come from the inherited launch environment, not from a project or global
dotenv file, and it must not move when the process changes directory or reloads
settings. `PATHS` is the single launch-time snapshot used by both processes.

This module also owns `classify_path`. `Path.exists()` returns `False` for some
permission errors, which makes an unreadable configured path indistinguishable
from one that has not been created yet. Diagnostics need that distinction, so
they probe with `Path.stat()` and retain an explicit `UNREADABLE` state.

Keep this module limited to the standard library: it is imported on the CLI
startup path and by the server subprocess before heavier packages are needed.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from deepagents_code._home_error import DeepAgentsHomeError

logger = logging.getLogger(__name__)

_MISSING_ERRNOS = {errno.ENOENT, errno.ENOTDIR}

__all__ = [
    "PATHS",
    "DeepAgentsHomeError",
    "DeepAgentsPathSnapshot",
    "InstallationPaths",
    "PathState",
    "ProfilePaths",
    "ProjectPaths",
    "classify_path",
    "get_deepagents_home",
    "project_paths",
]


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    """Paths whose contents belong to one user profile and trust root."""

    root: Path
    config_file: Path
    dotenv_file: Path
    mcp_config_file: Path
    agent_profiles_dir: Path
    default_skills_dir: Path
    hooks_file: Path
    plugins_dir: Path
    state_dir: Path
    auth_file: Path
    mcp_tokens_dir: Path
    sessions_file: Path
    history_file: Path
    offload_dir: Path
    bin_dir: Path
    """Per-profile fallback for managed binaries.

    Preferred location is `InstallationPaths.managed_bin_dir`, so profiles can
    share one verified download. This is used when that directory is not
    writable — a root-owned or system install prefix — because sharing is a
    convenience and a working `rg` is not.
    """

    locks_dir: Path
    """Per-profile fallback for install/update locks.

    Twin of `bin_dir`: preferred location is
    `InstallationPaths.locks_dir`. Falling back keeps self-upgrades serialized
    rather than silently fail-open when the install prefix is unwritable.
    """

    def agent_dir(self, name: str) -> Path:
        """Return the profile directory for an agent name."""
        return self.agent_profiles_dir / name

    def agent_skills_dir(self, name: str) -> Path:
        """Return the user-skill directory for an agent name."""
        return self.agent_dir(name) / "skills"


@dataclass(frozen=True, slots=True)
class InstallationPaths:
    """Paths owned by the installed tool rather than a selected profile."""

    root: Path
    managed_bin_dir: Path
    locks_dir: Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Project-controlled paths derived from an explicit repository root."""

    root: Path
    config_dir: Path
    root_mcp_config_file: Path
    config_mcp_config_file: Path
    skills_dir: Path
    agents_dir: Path
    hooks_file: Path


@dataclass(frozen=True, slots=True)
class DeepAgentsPathSnapshot:
    """Frozen launch-time profile and installation paths."""

    profile: ProfilePaths
    installation: InstallationPaths
    launch_home: Path | None
    uses_default_profile: bool

    def display(self, path: Path) -> str:
        """Abbreviate a path for display.

        Paths under the default profile render with a leading `~`. A configured
        profile renders literally, because abbreviating it would hide which
        profile is in use. A path outside the profile root (an installation
        path, say) also renders literally.

        Returns:
            A concise user-facing path.
        """
        if not self.uses_default_profile:
            return str(path)
        try:
            relative = path.relative_to(self.profile.root)
        except ValueError:
            return str(path)
        # `Path("~/.deepagents") / Path(".")` is `~/.deepagents`, so the
        # profile root itself needs no special case.
        return str(Path("~/.deepagents") / relative)


def project_paths(root: Path) -> ProjectPaths:
    """Return project-controlled paths for an explicit repository root.

    Args:
        root: Absolute project root.

    Returns:
        Paths rooted at the normalized project directory.

    Note:
        `_normalize_absolute` raises `ValueError` for a relative `root`. That is
        a caller bug rather than a `DEEPAGENTS_HOME` misconfiguration, so it is
        deliberately not a `DeepAgentsHomeError`.
    """
    normalized = _normalize_absolute(root, what="Project root")
    config_dir = normalized / ".deepagents"
    return ProjectPaths(
        root=normalized,
        config_dir=config_dir,
        root_mcp_config_file=normalized / ".mcp.json",
        config_mcp_config_file=config_dir / ".mcp.json",
        skills_dir=config_dir / "skills",
        agents_dir=config_dir / "agents",
        hooks_file=config_dir / "hooks.json",
    )


class PathState(StrEnum):
    """Whether a probed path exists, is absent, or could not be read.

    A `StrEnum` so the value serializes directly to JSON without a custom
    encoder.
    """

    EXISTS = "exists"
    """The path is present on disk."""

    MISSING = "missing"
    """The path is absent (and its parents are readable)."""

    UNREADABLE = "unreadable"
    """Existence could not be determined because `Path.stat()` raised.

    Typically EACCES when a parent directory denies traversal. Kept distinct
    from `MISSING` so diagnostics can flag it as a genuine problem rather than
    a not-yet-created path.
    """


def classify_path(path: Path) -> PathState:
    """Classify a path as existing, missing, or unreadable.

    Args:
        path: Filesystem path to probe.

    Returns:
        `PathState.EXISTS` for a present path, `PathState.MISSING` for expected
            absent-path errors, and `PathState.UNREADABLE` when `Path.stat()`
            raises another `OSError` (e.g. a parent directory denies traversal).
            The error is logged at debug level so an unreadable path is never
            silently indistinguishable from a missing one.
    """
    try:
        path.stat()
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            return PathState.MISSING
        logger.debug("Could not stat %s", path, exc_info=True)
        return PathState.UNREADABLE
    else:
        return PathState.EXISTS


def _normalize_absolute(path: Path, *, what: str = "Path") -> Path:
    """Normalize an already-absolute path without touching the filesystem.

    Args:
        path: Path to normalize.
        what: Noun used in the error message, so a failure names the input that
            was actually wrong.

    Returns:
        The lexically normalized absolute path.

    Raises:
        ValueError: If `path` is relative.
    """
    if not path.is_absolute():
        msg = f"{what} must be absolute: {path}"
        raise ValueError(msg)
    return Path(os.path.normpath(str(path)))


def _resolve_launch_home(launch_home: Path | None) -> Path:
    """Return the explicit or OS-resolved launch home as an absolute path.

    Returns:
        The normalized launch home.

    Raises:
        DeepAgentsHomeError: If the home directory cannot be determined or is
            not absolute. Both are reported against `DEEPAGENTS_HOME` because
            setting it to an absolute path is the way out of either.
    """
    if launch_home is None:
        try:
            launch_home = Path.home()
        except RuntimeError as exc:
            # `Path.home()` raises when $HOME is unset and the uid has no passwd
            # entry: a bare container, or a cleared-environment service unit.
            msg = (
                "Could not determine the home directory: set $HOME, or set "
                "DEEPAGENTS_HOME to an absolute profile path."
            )
            raise DeepAgentsHomeError(msg) from exc
    try:
        return _normalize_absolute(launch_home, what="Home directory")
    except ValueError as exc:
        msg = (
            f"Home directory is not absolute: {launch_home}. Set $HOME to an "
            "absolute path, or set DEEPAGENTS_HOME to an absolute profile path."
        )
        raise DeepAgentsHomeError(msg) from exc


def _reject_degenerate_root(root: Path, launch_home: Path | None) -> None:
    """Reject a resolved profile root that would scatter state.

    A profile root is a trust boundary that owns everything beneath it, so it
    must be a directory of its own. The rejected cases all resolve to something
    the user did not mean:

    - The filesystem root, from `DEEPAGENTS_HOME=/` or a `..` chain that walks
      past it, would put credentials in `/.state/auth.json`.
    - The home directory itself, from a `DEEPAGENTS_HOME=~/` typo, would make
      the profile dotenv the user's generic `~/.env` and load it as trusted
      configuration.
    - An existing non-directory cannot hold a profile at all.

    Raises:
        DeepAgentsHomeError: If the root is one of those cases.
    """
    if root.parent == root:
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: the filesystem root cannot "
            "be a profile. Use a dedicated directory."
        )
        raise DeepAgentsHomeError(msg)
    if launch_home is not None and root == launch_home:
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: the home directory itself "
            "cannot be a profile, because its '.env' would be loaded as "
            "trusted configuration. Use a subdirectory such as "
            "'~/.deepagents'."
        )
        raise DeepAgentsHomeError(msg)
    if classify_path(root) is PathState.EXISTS and not root.is_dir():
        msg = f"Invalid DEEPAGENTS_HOME {str(root)!r}: exists but is not a directory."
        raise DeepAgentsHomeError(msg)


def _resolve_profile_root(
    configured: str | None, launch_home: Path | None
) -> tuple[Path, bool, Path | None]:
    """Resolve a launch value using only the captured launch-user home.

    An absolute `DEEPAGENTS_HOME` never consults the home directory, so a host
    with no resolvable home can still run by setting it.

    Returns:
        The normalized root, whether it is the default profile, and the
        captured launch home when resolution required one.

    Note:
        `DeepAgentsHomeError` propagates from the resolution and validation
        helpers when the configured value is not supported.
    """
    root, uses_default, home = _resolve_profile_root_unchecked(configured, launch_home)
    # An absolute value resolves without a home, but the "profile is the home
    # directory" hazard applies to `/Users/me` exactly as much as to `~/`. Look
    # the home up best-effort for that comparison only, so a host with no
    # resolvable home still launches.
    _reject_degenerate_root(root, home if home is not None else _best_effort_home())
    return root, uses_default, home


def _best_effort_home() -> Path | None:
    """Return the launch home if it can be determined, else `None`.

    Used only for validation, never to build a path, so an unresolvable home
    must degrade to "cannot check" rather than fail the launch.

    Returns:
        The normalized home directory, or `None` when it cannot be resolved.
    """
    try:
        return _resolve_launch_home(None)
    except DeepAgentsHomeError:
        logger.debug("Could not resolve the home directory for validation")
        return None


def _resolve_profile_root_unchecked(
    configured: str | None, launch_home: Path | None
) -> tuple[Path, bool, Path | None]:
    """Apply the precedence rules without the degenerate-root checks.

    Returns:
        The normalized root, whether it is the default profile, and the
        captured launch home when resolution required one.

    Raises:
        DeepAgentsHomeError: If the configured value is not supported.
    """
    if not configured:
        home = _resolve_launch_home(launch_home)
        return home / ".deepagents", True, home
    if configured.startswith("~/"):
        home = _resolve_launch_home(launch_home)
        return _normalize_absolute(home / configured[2:]), False, home
    if configured.startswith("~"):
        msg = (
            "Invalid DEEPAGENTS_HOME: only an absolute path or a leading '~/' "
            "path is supported; '~user' forms are not allowed."
        )
        raise DeepAgentsHomeError(msg)

    path = Path(configured)
    if not path.is_absolute():
        msg = (
            f"Invalid DEEPAGENTS_HOME {configured!r}: use an absolute path or "
            "a path beginning with '~/'."
        )
        raise DeepAgentsHomeError(msg)
    # An absolute profile does not need a home directory; only report a home
    # that was handed to us explicitly.
    home = _normalize_absolute(launch_home) if launch_home is not None else None
    return _normalize_absolute(path), False, home


def _profile_paths(root: Path) -> ProfilePaths:
    """Build the profile-owned portion of the immutable snapshot.

    Returns:
        All paths owned by the selected user profile.
    """
    state_dir = root / ".state"
    return ProfilePaths(
        root=root,
        config_file=root / "config.toml",
        dotenv_file=root / ".env",
        mcp_config_file=root / ".mcp.json",
        agent_profiles_dir=root,
        default_skills_dir=root / "agent" / "skills",
        hooks_file=root / "hooks.json",
        plugins_dir=root / "plugins",
        state_dir=state_dir,
        auth_file=state_dir / "auth.json",
        mcp_tokens_dir=state_dir / "mcp-tokens",
        sessions_file=state_dir / "sessions.db",
        history_file=state_dir / "history.jsonl",
        offload_dir=root / "conversation_history",
        bin_dir=root / "bin",
        locks_dir=state_dir / "locks",
    )


def _installation_paths() -> InstallationPaths:
    """Build paths tied to this interpreter/tool environment.

    Returns:
        Resource and lock paths for the current installation.
    """
    root = _normalize_absolute(Path(sys.prefix), what="sys.prefix")
    resources = root / "share" / "deepagents-code"
    locks = root.parent / f".{root.name}.deepagents-code-locks"
    return InstallationPaths(
        root=root,
        managed_bin_dir=resources / "bin",
        locks_dir=locks,
    )


def _capture_paths(
    configured: str | None, *, launch_home: Path | None = None
) -> DeepAgentsPathSnapshot:
    """Construct a snapshot; exposed privately for deterministic unit tests.

    Returns:
        An immutable path snapshot.
    """
    profile_root, uses_default, home = _resolve_profile_root(configured, launch_home)
    return DeepAgentsPathSnapshot(
        profile=_profile_paths(profile_root),
        installation=_installation_paths(),
        launch_home=home,
        uses_default_profile=uses_default,
    )


PATHS = _capture_paths(os.environ.get("DEEPAGENTS_HOME"))
"""Process-wide path snapshot captured before any dotenv loader can run."""

# Normalize the inherited value for every descendant process. Callers that
# build an explicit child environment should still assign from `PATHS` so a
# later accidental `os.environ` mutation cannot create client/server split-brain.
os.environ["DEEPAGENTS_HOME"] = str(PATHS.profile.root)


def get_deepagents_home() -> Path:
    """Return the immutable launch-time user profile root."""
    return PATHS.profile.root
