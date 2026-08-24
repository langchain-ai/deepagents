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

logger = logging.getLogger(__name__)

_MISSING_ERRNOS = {errno.ENOENT, errno.ENOTDIR}


class DeepAgentsHomeError(ValueError):
    """Raised when the launch-time `DEEPAGENTS_HOME` value is invalid."""


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
    """Frozen launch-time profile/install paths plus project-path construction."""

    profile: ProfilePaths
    installation: InstallationPaths
    launch_home: Path
    uses_default_profile: bool

    @staticmethod
    def for_project(root: Path) -> ProjectPaths:
        """Return project-controlled paths for an explicit root.

        Returns:
            Paths rooted at the normalized project directory.
        """
        normalized = _normalize_absolute(root)
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

    def display(self, path: Path) -> str:
        """Render default profile paths with `~` and configured paths literally.

        Returns:
            A concise user-facing path.
        """
        if not self.uses_default_profile:
            return str(path)
        try:
            relative = path.relative_to(self.profile.root)
        except ValueError:
            return str(path)
        default_root = Path("~/.deepagents")
        return str(default_root if relative == Path() else default_root / relative)


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


def _normalize_absolute(path: Path) -> Path:
    """Normalize an already-absolute path without touching the filesystem.

    Returns:
        The lexically normalized absolute path.

    Raises:
        DeepAgentsHomeError: If `path` is relative.
    """
    if not path.is_absolute():
        msg = f"Path must be absolute: {path}"
        raise DeepAgentsHomeError(msg)
    return Path(os.path.normpath(str(path)))


def _resolve_profile_root(
    configured: str | None, launch_home: Path
) -> tuple[Path, bool]:
    """Resolve a launch value using only the captured launch-user home.

    Returns:
        The normalized root and whether it is the default profile.

    Raises:
        DeepAgentsHomeError: If the configured value is not supported.
    """
    if not configured:
        return _normalize_absolute(launch_home / ".deepagents"), True
    if configured.startswith("~/"):
        return _normalize_absolute(launch_home / configured[2:]), False
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
    return _normalize_absolute(path), False


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
    )


def _installation_paths() -> InstallationPaths:
    """Build paths tied to this interpreter/tool environment.

    Returns:
        Resource and lock paths for the current installation.
    """
    root = _normalize_absolute(Path(sys.prefix))
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
    home = _normalize_absolute(launch_home or Path.home())
    profile_root, uses_default = _resolve_profile_root(configured, home)
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
