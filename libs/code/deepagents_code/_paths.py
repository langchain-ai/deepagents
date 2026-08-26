"""Immutable filesystem paths captured before configuration is loaded.

`DEEPAGENTS_HOME` selects the user's profile and therefore a trust boundary. It
must come from the inherited launch environment, not from a project or global
dotenv file, and it must not move when the process changes directory or reloads
settings. `PATHS` is the single launch-time snapshot used by both the client
and the server subprocess.

This module also owns `classify_path`. `Path.exists()` returns `False` for some
permission errors, which makes an unreadable configured path indistinguishable
from one that has not been created yet. Diagnostics need that distinction, so
they probe with `Path.stat()` and retain an explicit `UNREADABLE` state.

Keep this module to the standard library plus `_home_error`, which has no
imports of its own. This module is imported on the CLI startup path and by the
server subprocess before heavier packages are needed.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code._home_error import DeepAgentsHomeError

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

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
    "ensure_agent_dir",
    "ensure_project_skills_dir",
    "ensure_user_skills_dir",
    "export_profile_env",
    "first_writable",
    "get_agent_dir",
    "get_built_in_skills_dir",
    "get_deepagents_home",
    "get_project_agent_md_path",
    "get_project_agent_skills_dir",
    "get_project_agents_dir",
    "get_project_claude_skills_dir",
    "get_project_skills_dir",
    "get_user_agent_md_path",
    "get_user_agent_skills_dir",
    "get_user_agents_dir",
    "get_user_claude_skills_dir",
    "get_user_skills_dir",
    "harden_state_dir",
    "probe_writable",
    "project_paths",
    "user_agents_dir",
    "user_deepagents_dir",
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
"""Directory name used under the home directory when no profile is configured."""


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    """Paths whose contents belong to one user profile and trust root."""

    root: Path
    config_file: Path
    dotenv_file: Path
    mcp_config_file: Path
    agent_profiles_dir: Path
    """The profile root itself.

    Agent profiles are direct children of the root, not of a dedicated
    subdirectory, so an agent name can collide with an app-owned directory.
    `_reserved_names.reserved_agent_dir_names` is what stops that.
    """

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
    """Resolved user home for optional home-based integrations, if available."""

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


def user_deepagents_dir() -> Path:
    """Return the immutable launch-time user profile root."""
    return PATHS.profile.root


def _validate_agent_name(agent_name: str) -> None:
    """Raise when an agent name cannot safely identify a profile directory.

    Raises:
        ValueError: If the name is empty, unsafe, or reserved by dcode.
    """
    if (
        not agent_name
        or not agent_name.strip()
        or not re.fullmatch(r"[a-zA-Z0-9_\-\s]+", agent_name)
    ):
        msg = (
            f"Invalid agent name: {agent_name!r}. Agent names can only "
            "contain letters, numbers, hyphens, underscores, and spaces."
        )
        raise ValueError(msg)
    from deepagents_code._reserved_names import is_reserved_agent_dir_name

    if is_reserved_agent_dir_name(agent_name):
        msg = f"Invalid agent name: {agent_name!r} is reserved for dcode's own state."
        raise ValueError(msg)


def get_agent_dir(agent_name: str) -> Path:
    """Return the validated profile directory for an agent name.

    Args:
        agent_name: Agent profile name.

    Returns:
        Path to the agent's profile directory.

    """
    _validate_agent_name(agent_name)
    return PATHS.profile.agent_dir(agent_name)


def ensure_agent_dir(agent_name: str) -> Path:
    """Create the validated profile directory for an agent.

    Returns:
        Path to the agent's profile directory.
    """
    agent_dir = get_agent_dir(agent_name)
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir


def get_user_agent_md_path(agent_name: str) -> Path:
    """Return the user-level `AGENTS.md` path for an agent profile."""
    return get_agent_dir(agent_name) / "AGENTS.md"


def get_project_agent_md_path(project_root: Path | None) -> list[Path]:
    """Return existing project-level `AGENTS.md` paths."""
    if project_root is None:
        return []
    from deepagents_code.project_utils import find_project_agent_md

    return find_project_agent_md(project_root)


def get_user_skills_dir(agent_name: str) -> Path:
    """Return the user-level skills directory for an agent profile."""
    return get_agent_dir(agent_name) / "skills"


def ensure_user_skills_dir(agent_name: str) -> Path:
    """Create the user-level skills directory for an agent.

    Returns:
        Path to the agent's user-level skills directory.
    """
    skills_dir = get_user_skills_dir(agent_name)
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def get_project_skills_dir(project_root: Path | None) -> Path | None:
    """Return the project-level dcode skills directory, when in a project."""
    return None if project_root is None else project_paths(project_root).skills_dir


def ensure_project_skills_dir(project_root: Path | None) -> Path | None:
    """Create the project-level dcode skills directory.

    Returns:
        Path to the project skills directory, or `None` outside a project.
    """
    skills_dir = get_project_skills_dir(project_root)
    if skills_dir is not None:
        skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def get_user_agents_dir(agent_name: str) -> Path:
    """Return the custom-subagent directory for an agent profile."""
    return get_agent_dir(agent_name) / "agents"


def get_project_agents_dir(project_root: Path | None) -> Path | None:
    """Return the project-level custom-subagent directory, when available."""
    return None if project_root is None else project_paths(project_root).agents_dir


def user_agents_dir() -> Path | None:
    """Return the launch user's tool-agnostic `~/.agents` directory."""
    return None if PATHS.launch_home is None else PATHS.launch_home / ".agents"


def get_user_agent_skills_dir() -> Path | None:
    """Return the launch user's tool-agnostic `~/.agents/skills` directory."""
    base = user_agents_dir()
    return None if base is None else base / "skills"


def get_project_agent_skills_dir(project_root: Path | None) -> Path | None:
    """Return the project-level tool-agnostic `.agents/skills` directory."""
    return None if project_root is None else project_root / ".agents" / "skills"


def get_user_claude_skills_dir() -> Path | None:
    """Return the launch user's experimental `~/.claude/skills` directory."""
    if PATHS.launch_home is None:
        return None
    return PATHS.launch_home / ".claude" / "skills"


def get_project_claude_skills_dir(project_root: Path | None) -> Path | None:
    """Return the project's experimental `.claude/skills` directory."""
    return None if project_root is None else project_root / ".claude" / "skills"


def get_built_in_skills_dir() -> Path:
    """Return the directory containing skills bundled with dcode."""
    return Path(__file__).parent / "built_in_skills"


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
    are shared across processes, and in the installation-scoped case across
    profiles too. PIDs are not unique across containers or PID namespaces: a
    colliding name would make a writable directory look unusable and would
    delete a peer's live probe. Removal failures are suppressed separately, so
    a directory that accepts files but refuses unlinks is still reported as
    writable.

    Args:
        directory: Directory to create and probe.
        mode: Permission bits for directories this call creates. The default
            `0o777` defers to the process umask, which is what an ordinary
            tool directory wants. The state directory is the one that must not,
            so `harden_state_dir` passes `0o700` and chmods afterwards.

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
        # This runs on every launch, so a directory that never accepts an
        # unlink (an NFS or SMB mount with delete denied, for example) grows an
        # unbounded pile of `.deepagents-probe-*` files. Warn on the first leak
        # per directory and suppress repeats in this process.
        key = str(directory)
        if key in _LEAKED_PROBE_DIRS:
            return
        _LEAKED_PROBE_DIRS.add(key)
        logger.warning(
            "Cannot remove write probes in %s, so they are accumulating. "
            "Delete the '.deepagents-probe-*' files there and check the "
            "directory's delete permissions.",
            directory,
        )


_LEAKED_PROBE_DIRS: set[str] = set()
"""Directories that have already refused to remove a write probe."""


def first_writable(
    candidates: Sequence[Path], *, mode: int = 0o777, what: str
) -> Path | None:
    """Return the first candidate directory that accepts a file.

    Shared by the managed-bin and update-lock resolvers, which both walk a
    preferred-then-fallback pair. Callers keep their own "fell back" warning,
    because the consequence of falling back differs between them.

    Args:
        candidates: Directories to try, most preferred first.
        mode: Permission bits for directories this call creates.
        what: Noun phrase naming the directory's purpose, used in the log.

    Returns:
        The first usable directory, or `None` when none of them is.
    """
    for directory in candidates:
        try:
            probe_writable(directory, mode=mode)
        except OSError:
            logger.info(
                "%s directory %s is unusable; trying the next location",
                what,
                directory,
                exc_info=True,
            )
            continue
        return directory
    return None


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

        On Windows the `chmod` step is skipped. POSIX mode bits do not restrict
        access there, so the directory keeps the ACL it inherits from its
        parent and this function adds no protection of its own.
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
    equal here. `Path.samefile` compares device and inode. That settles every
    spelling at once. `os.path.normcase` does not, because it is a no-op on
    POSIX.

    A missing path is a real answer: it cannot be the directory it is compared
    against. Any other `OSError` is not an answer, and this function refuses to
    guess. Callers use it to reject a profile root, so a wrong `False` accepts
    the alias the caller meant to reject.

    Args:
        left: First path to compare.
        right: Second path to compare.

    Returns:
        `True` when the two paths name one directory.

    Raises:
        DeepAgentsHomeError: If identity cannot be determined, such as when a
            parent directory denies traversal or a symlink chain loops.
    """
    if str(left) == str(right):
        return True
    try:
        return Path(left).samefile(right)
    except FileNotFoundError:
        # A profile root that is not there yet cannot be the home directory,
        # and the lexical comparison above has already ruled out the
        # spelling-only case.
        logger.debug("Could not compare %s with %s: one is missing", left, right)
        return False
    except OSError as exc:
        # EACCES, ELOOP, EIO, ESTALE: the answer is unknown, not "different".
        # The whole point of `samefile` here is to catch the non-lexical
        # spellings the comparison above misses, so returning `False` would
        # accept exactly the aliases this guard exists to reject.
        msg = (
            f"Cannot determine whether {str(left)!r} is {str(right)!r}: "
            f"{exc.strerror or exc}. Fix the permissions on those paths, or "
            "set DEEPAGENTS_HOME to a path that can be read."
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
    - A root that exists but cannot be read. Every later access fails one file
      at a time, and each failure looks like a first run, so reject it once
      here with the real cause.
    - A symlink whose target is missing. The profile root is created lazily, so
      it would otherwise be created through a link the user cannot see.

    Comparisons go through `_same_directory`, so a symlinked or differently
    cased spelling of `/` or of the home directory is rejected too.

    The readability check runs first. `_same_directory` cannot compare a path
    it may not read, so checking state first reports the permission problem
    itself instead of the comparison that failed because of it.

    Raises:
        DeepAgentsHomeError: If the root is one of those cases.
    """
    state = classify_path(root)
    if state is PathState.UNREADABLE:
        # Checked before the symlink branch too: `Path.is_symlink` swallows the
        # `OSError` and reports `False` under EACCES, so an unreadable root
        # would otherwise fall through every check and be accepted.
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: exists but cannot be read. "
            "Check the permissions on it and on its parent directories."
        )
        raise DeepAgentsHomeError(msg)
    if state is PathState.EXISTS and not root.is_dir():
        msg = f"Invalid DEEPAGENTS_HOME {str(root)!r}: exists but is not a directory."
        raise DeepAgentsHomeError(msg)
    if state is PathState.EXISTS and not os.access(root, os.R_OK | os.X_OK):
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: exists but cannot be read "
            "or searched. Check the permissions on it and on its parent "
            "directories."
        )
        raise DeepAgentsHomeError(msg)
    # `root` is normalized-absolute, so `anchor` is always set.
    if root.parent == root or _same_directory(root, Path(root.anchor)):
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
    if state is PathState.MISSING and root.is_symlink():
        msg = (
            f"Invalid DEEPAGENTS_HOME {str(root)!r}: is a symlink whose target "
            "is missing."
        )
        raise DeepAgentsHomeError(msg)


def _resolve_profile_root(
    configured: str | None, launch_home: Path | None
) -> tuple[Path, bool, Path | None, Path | None]:
    """Resolve a launch value, preferring the captured launch-user home.

    An absolute `DEEPAGENTS_HOME` never consults the home directory, so a host
    with no resolvable home can still run by setting it.

    Returns:
        The normalized root, whether it is the default profile, the captured
        launch home when resolution required one, and the home used for the
        degenerate-root comparison. The last is `None` when the home could not
        be resolved, which means that comparison did not run.

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
    # Returned rather than reduced to a boolean: `_capture_paths` needs the
    # same value for the default-marker check, and resolving it twice would
    # warn twice on a host with no resolvable home.
    return root, uses_default, home, comparison_home


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
        home: The home already resolved for the degenerate-root comparison, or
            `None` when it could not be resolved. Passed in rather than looked
            up again, so an unresolvable home warns once per launch.

    Returns:
        `True` when `root` is the default profile directory for the launch home.
    """
    if home is None:
        return False
    return root == home / DEFAULT_PROFILE_DIR_NAME


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
    profile_root, uses_default, _resolution_home, comparison_home = (
        _resolve_profile_root(configured, launch_home)
    )
    if not uses_default and default_marker:
        uses_default = _honors_default_marker(profile_root, comparison_home)
    return DeepAgentsPathSnapshot(
        profile=_profile_paths(profile_root),
        installation=_installation_paths(),
        launch_home=comparison_home,
        uses_default_profile=uses_default,
        home_check_skipped=comparison_home is None,
    )


PATHS = _capture_paths(
    os.environ.get(DEEPAGENTS_HOME_ENV),
    default_marker=os.environ.get(DEFAULT_PROFILE_MARKER_ENV) == "1",
)
"""Process-wide path snapshot captured before any dotenv loader can run."""


def export_profile_env(env: MutableMapping[str, str]) -> None:
    """Pin the resolved profile selection into a child environment.

    Writes the resolved root, then sets or clears the defaulted/configured
    marker, so a child reconstructs the same snapshot the parent captured.
    Both keys are always written. The marker is removed rather than left in
    place when the profile is configured, because a stale inherited marker
    would describe a profile it no longer applies to.

    Args:
        env: Environment mapping to update in place.
    """
    env[DEEPAGENTS_HOME_ENV] = str(PATHS.profile.root)
    if PATHS.uses_default_profile:
        env[DEFAULT_PROFILE_MARKER_ENV] = "1"
    else:
        env.pop(DEFAULT_PROFILE_MARKER_ENV, None)


# Normalize the inherited value for every descendant process. Callers that
# build an explicit child environment should still assign from `PATHS`. Then a
# later accidental `os.environ` mutation cannot give the client and the server
# different profile roots.
export_profile_env(os.environ)


def get_deepagents_home() -> Path:
    """Return the immutable launch-time user profile root."""
    return PATHS.profile.root
