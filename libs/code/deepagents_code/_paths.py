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
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code._home_error import DeepAgentsHomeError

if TYPE_CHECKING:
    from collections.abc import MutableMapping

logger = logging.getLogger(__name__)

_MISSING_ERRNOS = {errno.ENOENT, errno.ENOTDIR}

__all__ = [
    "DEEPAGENTS_HOME_ENV",
    "DEFAULT_PROFILE_DIR_NAME",
    "DEFAULT_PROFILE_MARKER_ENV",
    "PATHS",
    "DeepAgentsHomeError",
    "DeepAgentsPathSnapshot",
    "InstallationPaths",
    "PathState",
    "ProfilePaths",
    "ProjectPaths",
    "classify_path",
    "get_deepagents_home",
    "harden_state_dir",
    "probe_writable",
    "project_paths",
]

DEEPAGENTS_HOME_ENV = "DEEPAGENTS_HOME"
"""Name of the variable that selects the user profile and trust root."""

DEFAULT_PROFILE_MARKER_ENV = "DEEPAGENTS_HOME_IS_DEFAULT"
"""Internal marker that records "the profile was defaulted, not configured".

`DEEPAGENTS_HOME` is re-exported for every descendant process, so a child
cannot tell a defaulted profile from one the user selected by simply reading
the variable. Without this marker every child concludes the profile was
configured: the server subprocess renders absolute paths (leaking the OS
username into the system prompt) and a post-upgrade re-exec announces a profile
the user never set.

Set by the parent only, never by a user. It is a display hint, never a trust
input: `_honors_default_marker` re-derives the default location and ignores the
marker unless the resolved root matches, so a forged value cannot change which
directory is used.
"""

DEFAULT_PROFILE_DIR_NAME = ".deepagents"
"""Directory under the home directory used when no profile is configured."""


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
    home_check_skipped: bool = False
    """Whether the "profile is the home directory" check could not run.

    True when the home directory could not be resolved. That silently disables
    a security check, and this module is imported before any log handler
    exists, so a log line alone would be invisible even under `--debug`.
    `dcode doctor` reports this field instead.
    """

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
        return str(Path("~") / DEFAULT_PROFILE_DIR_NAME / relative)


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


def probe_writable(directory: Path, *, mode: int = 0o777) -> None:
    """Create *directory* and prove the process can write files inside it.

    `mkdir(exist_ok=True)` only proves the directory exists; it succeeds on a
    pre-existing root-owned directory. Creating a file is the only check that
    distinguishes "present" from "usable".

    The probe uses `tempfile.mkstemp`, not a PID-named file. These directories
    are deliberately shared across profiles and processes, and PIDs are not
    unique across containers or PID namespaces: a colliding name would make a
    writable directory look unusable and would delete a peer's live probe.
    Removal failures are suppressed separately, so a directory that accepts
    files but refuses unlinks is still reported as writable.

    Args:
        directory: Directory to create and probe.
        mode: Permission bits for directories this call creates.

    Note:
        `OSError` propagates from `mkdir` or `mkstemp` when the directory
        cannot be created or cannot accept a file. Callers select a fallback
        location on that error.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    handle, name = tempfile.mkstemp(prefix=".deepagents-probe-", dir=directory)
    os.close(handle)
    try:
        Path(name).unlink()
    except OSError:
        logger.debug("Could not remove the write probe %s", name, exc_info=True)


def harden_state_dir(state_dir: Path | None = None) -> bool:
    """Create a state directory and restrict it to its owner.

    The state directory holds `sessions.db` and `history.jsonl`, which hold
    full conversation content and are written with the default file mode. The
    directory permissions are the only thing that keeps another local user out,
    so every creator must go through this function rather than a bare `mkdir`.

    Args:
        state_dir: Directory to create, defaulting to the active profile's.
            `state_migration` passes its own, since it migrates a profile that
            is not necessarily the active one.

    Returns:
        Whether the directory now exists. A caller that must not write into a
        missing directory checks this; one that only wants the hardening
        applied can ignore it.

    Note:
        Failures are logged, never raised. `mkdir` can fail on a read-only or
        full filesystem, and `chmod` is routinely refused on CIFS/exFAT mounts.
        Neither is a reason to abort the launch that needed the directory.
    """
    if state_dir is None:
        state_dir = PATHS.profile.state_dir
    try:
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        logger.warning("Could not create %s", state_dir, exc_info=True)
        return False
    if os.name == "nt":
        return True
    try:
        state_dir.chmod(0o700)
    except OSError:
        # `mkdir(mode=...)` applies only when this call creates the directory,
        # and umask can still clear bits, so an existing directory needs the
        # explicit chmod. A refusal leaves the directory usable, so keep going.
        logger.warning("Could not restrict permissions on %s", state_dir, exc_info=True)
    return True


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


def _same_directory(left: Path, right: Path) -> bool:
    """Report whether two paths name the same directory.

    Path construction stays lexical on purpose, so `..` chains resolve without
    touching the filesystem. Identity is a different question: a lexical `==`
    misses a symlinked spelling of the target, and misses a case difference on
    the case-insensitive filesystems that are the default on macOS and Windows.
    Both are ordinary ways to spell the home directory, so both must compare
    equal here. `os.path.samefile` compares device and inode, which settles
    every spelling at once; `os.path.normcase` would not, because it is a
    no-op on POSIX.

    Args:
        left: First path to compare.
        right: Second path to compare.

    Returns:
        `True` when the two paths name one directory.
    """
    if str(left) == str(right):
        return True
    try:
        return Path(left).samefile(right)
    except OSError:
        # One of them does not exist or cannot be read. A profile root that is
        # not there yet cannot be the home directory, and the lexical
        # comparison above has already ruled out the spelling-only case.
        logger.debug("Could not compare %s with %s", left, right)
        return False


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

    Comparisons go through `_same_directory`, so a symlinked or differently
    cased spelling of `/` or of the home directory is rejected too.

    Raises:
        DeepAgentsHomeError: If the root is one of those cases.
    """
    if root.parent == root or _same_directory(root, Path(root.anchor or "/")):
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: the filesystem root cannot "
            "be a profile. Use a dedicated directory."
        )
        raise DeepAgentsHomeError(msg)
    if launch_home is not None and _same_directory(root, launch_home):
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: the home directory itself "
            "cannot be a profile, because its '.env' would be loaded as "
            "trusted configuration. Use a subdirectory such as "
            "'~/.deepagents'."
        )
        raise DeepAgentsHomeError(msg)
    state = classify_path(root)
    if state is not PathState.EXISTS and root.is_symlink():
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: is a symlink whose target "
            "is missing or cannot be resolved."
        )
        raise DeepAgentsHomeError(msg)
    if state is PathState.EXISTS and not root.is_dir():
        msg = f"Invalid DEEPAGENTS_HOME {str(root)!r}: exists but is not a directory."
        raise DeepAgentsHomeError(msg)


def _resolve_profile_root(
    configured: str | None, launch_home: Path | None
) -> tuple[Path, bool, Path | None, bool]:
    """Resolve a launch value, preferring the captured launch-user home.

    An absolute `DEEPAGENTS_HOME` never consults the home directory, so a host
    with no resolvable home can still run by setting it.

    Returns:
        The normalized root, whether it is the default profile, the captured
        launch home when resolution required one, and whether the home
        comparison had to be skipped.

    Note:
        `DeepAgentsHomeError` propagates from the resolution and validation
        helpers when the configured value is not supported.
    """
    root, uses_default, home = _resolve_profile_root_unchecked(configured, launch_home)
    # An absolute value resolves without a home, but the "profile is the home
    # directory" hazard applies to `/Users/me` exactly as much as to `~/`. Look
    # the home up best-effort for that comparison only, so a host with no
    # resolvable home still launches.
    comparison_home = home if home is not None else _best_effort_home()
    _reject_degenerate_root(root, comparison_home)
    return root, uses_default, home, comparison_home is None


def _best_effort_home() -> Path | None:
    """Return the launch home if it can be determined, else `None`.

    Used only for validation, never to build a path, so an unresolvable home
    must degrade to "cannot check" rather than fail the launch. A `None` return
    is recorded on the snapshot, because a check that quietly stops running is
    worse than one that never existed.

    Returns:
        The normalized home directory, or `None` when it cannot be resolved.
    """
    try:
        return _resolve_launch_home(None)
    except DeepAgentsHomeError:
        logger.warning(
            "Could not resolve the home directory; skipping the check that "
            "DEEPAGENTS_HOME is not the home directory itself. Set $HOME to "
            "restore it."
        )
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
        return home / DEFAULT_PROFILE_DIR_NAME, True, home
    if configured.startswith("~/"):
        home = _resolve_launch_home(launch_home)
        relative = configured[2:].lstrip("/")
        return _normalize_absolute(home / relative), False, home
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


def _honors_default_marker(root: Path, home: Path | None) -> bool:
    """Report whether `root` really is this user's default profile location.

    The parent process re-exports `DEEPAGENTS_HOME` unconditionally, so a child
    needs the marker to recover "defaulted" from "configured". Trusting the
    marker alone would let a forged value relabel any profile, so re-derive the
    default location and honor the marker only when the two agree. The marker
    can then change how a path is displayed but never which path is used.

    Args:
        root: The already-resolved profile root.
        home: The launch home when resolution produced one.

    Returns:
        `True` when `root` is the default profile directory for the launch home.
    """
    resolved_home = home if home is not None else _best_effort_home()
    if resolved_home is None:
        return False
    return root == resolved_home / DEFAULT_PROFILE_DIR_NAME


def _capture_paths(
    configured: str | None,
    *,
    launch_home: Path | None = None,
    default_marker: bool = False,
) -> DeepAgentsPathSnapshot:
    """Construct a snapshot; exposed privately for deterministic unit tests.

    Args:
        configured: Raw `DEEPAGENTS_HOME` value, or `None`/empty when unset.
        launch_home: Explicit launch home, for tests that must not read `$HOME`.
        default_marker: Whether the parent process recorded that it defaulted
            the profile. Only honored when the resolved root matches the
            default location; see `_honors_default_marker`.

    Returns:
        An immutable path snapshot.
    """
    profile_root, uses_default, home, home_check_skipped = _resolve_profile_root(
        configured, launch_home
    )
    if not uses_default and default_marker:
        uses_default = _honors_default_marker(profile_root, home)
    return DeepAgentsPathSnapshot(
        profile=_profile_paths(profile_root),
        installation=_installation_paths(),
        launch_home=home,
        uses_default_profile=uses_default,
        home_check_skipped=home_check_skipped,
    )


PATHS = _capture_paths(
    os.environ.get(DEEPAGENTS_HOME_ENV),
    default_marker=os.environ.get(DEFAULT_PROFILE_MARKER_ENV) == "1",
)
"""Process-wide path snapshot captured before any dotenv loader can run."""


def export_profile_env(env: MutableMapping[str, str]) -> None:
    """Pin the resolved profile selection into a child environment.

    Writes both the resolved root and the defaulted/configured marker, so a
    child reconstructs the same snapshot the parent captured. Always assigns
    both keys — a stale inherited marker must be cleared rather than left to
    describe a profile it no longer applies to.

    Args:
        env: Environment mapping to update in place.
    """
    env[DEEPAGENTS_HOME_ENV] = str(PATHS.profile.root)
    if PATHS.uses_default_profile:
        env[DEFAULT_PROFILE_MARKER_ENV] = "1"
    else:
        env.pop(DEFAULT_PROFILE_MARKER_ENV, None)


# Normalize the inherited value for every descendant process. Callers that
# build an explicit child environment should still assign from `PATHS` so a
# later accidental `os.environ` mutation cannot create client/server split-brain.
export_profile_env(os.environ)


def get_deepagents_home() -> Path:
    """Return the immutable launch-time user profile root."""
    return PATHS.profile.root
