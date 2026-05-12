"""Lightweight git metadata helpers for CLI state detection."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GIT_DIR_POINTER_PREFIX = "gitdir: "
"""Prefix used by worktree-style `.git` files to point at the real git dir."""

_GIT_HEAD_REF_PREFIX = "ref: "
"""Prefix used by `HEAD` when it points at a named ref instead of a commit."""

_GIT_REF_PREFIXES = ("refs/heads/", "refs/remotes/", "refs/tags/", "refs/")
"""Known git ref prefixes stripped when formatting a branch-like display name."""

_git_dir_cache: dict[str, Path] = {}
"""Positive-only cache of resolved git metadata directories keyed by lookup path."""


def _abbreviate_git_ref(ref: str) -> str:
    """Convert a full git ref into a short display name.

    Args:
        ref: Full git ref name from repository metadata.

    Returns:
        The abbreviated ref name suitable for display.
    """
    for prefix in _GIT_REF_PREFIXES:
        if ref.startswith(prefix):
            return ref.removeprefix(prefix)
    return ref


def _parse_git_dir_pointer(git_entry: Path) -> Path | None:
    """Resolve a `.git` file containing a `gitdir:` pointer.

    Args:
        git_entry: `.git` file to parse.

    Returns:
        The resolved git directory path, or `None` if the file is not a valid
        gitdir pointer.
    """
    try:
        raw = git_entry.read_text(encoding="utf-8").strip()
    except OSError:
        logger.debug("Failed to read gitdir pointer from %s", git_entry, exc_info=True)
        return None

    if not raw.startswith(_GIT_DIR_POINTER_PREFIX):
        return None

    pointer = raw.removeprefix(_GIT_DIR_POINTER_PREFIX).strip()
    if not pointer:
        return None

    git_dir = Path(pointer)
    if not git_dir.is_absolute():
        git_dir = git_entry.parent / git_dir
    return git_dir.resolve(strict=False)


def _normalize_lookup_path(path: str | Path) -> Path:
    """Normalize a lookup path for git metadata discovery.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        A normalized absolute path when possible, or the expanded path if full
        resolution fails.
    """
    try:
        return Path(path).expanduser().resolve(strict=False)
    except OSError:
        return Path(path).expanduser()


def _find_git_dir_uncached(path: Path) -> Path | None:
    """Locate the effective git metadata directory without using caches.

    Args:
        path: Normalized directory or file path inside a repository.

    Returns:
        The git metadata directory for the repository containing `path`, or
        `None` when no repository can be identified.
    """
    current = path
    if not current.is_dir():
        current = current.parent

    for directory in (current, *current.parents):
        git_entry = directory / ".git"
        if git_entry.is_dir():
            return git_entry
        if git_entry.is_file():
            git_dir = _parse_git_dir_pointer(git_entry)
            if git_dir is not None and git_dir.is_dir():
                return git_dir
            return None

    return None


def find_git_dir(path: str | Path) -> Path | None:
    """Locate the effective git metadata directory for a path.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The git metadata directory for the repository containing `path`, or
        `None` when no repository can be identified.
    """
    current = _normalize_lookup_path(path)
    key = str(current)
    cached = _git_dir_cache.get(key)
    if cached is not None:
        return cached

    git_dir = _find_git_dir_uncached(current)
    if git_dir is not None:
        _git_dir_cache[key] = git_dir
    return git_dir


def find_git_root(path: str | Path) -> Path | None:
    """Locate the repository root for a path.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The repository root containing `path`, or `None` when no repository can
        be identified.
    """
    current = _normalize_lookup_path(path)
    if not current.is_dir():
        current = current.parent

    for directory in (current, *current.parents):
        git_entry = directory / ".git"
        if git_entry.is_dir():
            return directory
        if git_entry.is_file():
            git_dir = _parse_git_dir_pointer(git_entry)
            if git_dir is not None and git_dir.is_dir():
                return directory
            return None

    return None


def read_git_branch_from_filesystem(path: str | Path) -> str | None:
    """Read the current git branch from repository metadata.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The abbreviated branch name, `HEAD` for detached HEAD, an empty string
        when `path` is not inside a git repository, or `None` when metadata
        exists but cannot be parsed confidently.
    """
    git_dir = find_git_dir(path)
    if git_dir is None:
        return ""

    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.debug("Git HEAD file not found in %s", git_dir)
        return None
    except OSError:
        logger.debug("Failed to read git HEAD from %s", git_dir, exc_info=True)
        return None

    if not head:
        return ""
    if head.startswith(_GIT_HEAD_REF_PREFIX):
        ref = head.removeprefix(_GIT_HEAD_REF_PREFIX).strip()
        return _abbreviate_git_ref(ref) if ref else None
    return "HEAD"


def read_git_branch_via_subprocess(path: str | Path) -> str:
    """Fall back to `git rev-parse` for unusual repository layouts.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The branch name reported by git, or an empty string on failure.
    """
    import subprocess  # noqa: S404  # stdlib subprocess fallback

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            cwd=path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass  # git not installed
    except subprocess.TimeoutExpired:
        logger.debug("Git branch detection timed out")
    except OSError:
        logger.debug("Git branch detection failed", exc_info=True)
    return ""


def resolve_git_branch(path: str | Path) -> str:
    """Resolve the current git branch with a filesystem-first strategy.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The current branch name, `HEAD` for detached HEAD, or an empty string
        when no branch can be determined.
    """
    branch = read_git_branch_from_filesystem(path)
    if branch is not None:
        return branch
    return read_git_branch_via_subprocess(path)
