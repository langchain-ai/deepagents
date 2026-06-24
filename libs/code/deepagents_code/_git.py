"""Lightweight git metadata helpers for state detection."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

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


_GIT_SHA_RE = re.compile(r"\A[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
"""Matches a full 40-char SHA-1 (or 64-char SHA-256) git object id."""


def read_git_commit_sha_from_filesystem(path: str | Path) -> str | None:
    """Read the current `HEAD` commit SHA from repository metadata.

    Resolves a symbolic `HEAD` (`ref: refs/heads/<branch>`) by reading the
    loose ref file, then `packed-refs`. A detached `HEAD` already contains the
    raw SHA.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The full commit SHA, an empty string when `path` is not inside a git
        repository, or `None` when metadata exists but cannot be resolved.
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
        return None
    if not head.startswith(_GIT_HEAD_REF_PREFIX):
        # Detached HEAD: the file holds the commit SHA directly.
        return head if _GIT_SHA_RE.match(head) else None

    ref = head.removeprefix(_GIT_HEAD_REF_PREFIX).strip()
    if not ref:
        return None

    loose_ref = git_dir / Path(ref)
    try:
        sha = loose_ref.read_text(encoding="utf-8").strip()
        if _GIT_SHA_RE.match(sha):
            return sha
    except FileNotFoundError:
        pass  # ref may be packed
    except OSError:
        logger.debug("Failed to read loose ref %s", loose_ref, exc_info=True)
        return None

    return _read_packed_ref(git_dir, ref)


def _read_packed_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve a ref to its SHA from `packed-refs`.

    Args:
        git_dir: Resolved git metadata directory.
        ref: Full ref name (e.g. `refs/heads/main`).

    Returns:
        The full commit SHA, or `None` when the ref is absent or unreadable.
    """
    try:
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Failed to read packed-refs in %s", git_dir, exc_info=True)
        return None

    for line in packed.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref and _GIT_SHA_RE.match(sha):
            return sha
    return None


def read_git_commit_sha_via_subprocess(path: str | Path) -> str:
    """Fall back to `git rev-parse HEAD` for unusual repository layouts.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The full commit SHA reported by git, or an empty string on failure.
    """
    import subprocess  # noqa: S404  # stdlib subprocess fallback

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
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
        logger.debug("Git commit detection timed out")
    except OSError:
        logger.debug("Git commit detection failed", exc_info=True)
    return ""


def resolve_git_commit_sha(path: str | Path) -> str:
    """Resolve the current `HEAD` commit SHA, filesystem-first.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The full commit SHA, or an empty string when none can be determined.
    """
    sha = read_git_commit_sha_from_filesystem(path)
    if sha is not None:
        return sha
    return read_git_commit_sha_via_subprocess(path)


def read_git_remote_url_from_filesystem(path: str | Path) -> str | None:
    """Read the `origin` remote URL from the repository `config` file.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The `origin` remote URL, an empty string when `path` is not inside a
        git repository, or `None` when no `origin` URL is configured.
    """
    git_dir = find_git_dir(path)
    if git_dir is None:
        return ""

    try:
        raw = (git_dir / "config").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Failed to read git config in %s", git_dir, exc_info=True)
        return None

    in_origin = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            # Section header, e.g. [remote "origin"]. Match case-insensitively
            # on the remote name to mirror git's own behavior.
            in_origin = stripped.replace(" ", "").lower() == '[remote"origin"]'
            continue
        if in_origin and stripped.lower().startswith("url"):
            _, _, value = stripped.partition("=")
            url = value.strip()
            if url:
                return url
    return None


def read_git_remote_url_via_subprocess(path: str | Path) -> str:
    """Fall back to `git config --get remote.origin.url`.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The `origin` remote URL reported by git, or an empty string on failure.
    """
    import subprocess  # noqa: S404  # stdlib subprocess fallback

    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],  # noqa: S607
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
        logger.debug("Git remote detection timed out")
    except OSError:
        logger.debug("Git remote detection failed", exc_info=True)
    return ""


def resolve_git_remote_url(path: str | Path) -> str:
    """Resolve the `origin` remote URL, filesystem-first.

    Args:
        path: Directory or file path inside a repository.

    Returns:
        The `origin` remote URL, or an empty string when none can be determined.
    """
    url = read_git_remote_url_from_filesystem(path)
    if url is not None:
        return url
    return read_git_remote_url_via_subprocess(path)


_REPO_PROVIDERS: dict[str, str] = {
    "github.com": "github",
    "gitlab.com": "gitlab",
    "bitbucket.org": "bitbucket",
}
"""Maps a known git host to its `repository_provider` slug."""


def parse_repository_metadata(remote_url: str) -> tuple[str, str, str] | None:
    """Derive `(repository_url, repository_provider, repository_name)` from a remote.

    Handles both HTTPS (`https://github.com/org/repo.git`) and scp-style SSH
    (`git@github.com:org/repo.git`) remotes, normalizing the URL to its
    canonical `https://<host>/<org>/<repo>` form and stripping any embedded
    credentials.

    Args:
        remote_url: The raw `origin` remote URL.

    Returns:
        A tuple of the normalized URL, provider slug (`github`/`gitlab`/
        `bitbucket`/`other`), and `org/repo` name, or `None` when the URL
        cannot be parsed.
    """
    url = (remote_url or "").strip()
    if not url:
        return None

    host: str
    repo_path: str
    if "://" in url:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        repo_path = parsed.path.lstrip("/")
    else:
        # scp-style: [user@]host:org/repo(.git)
        userhost, sep, repo_path = url.partition(":")
        if not sep:
            return None
        host = userhost.rsplit("@", 1)[-1].lower()
        repo_path = repo_path.lstrip("/")

    repo_path = repo_path.removesuffix(".git").strip("/")
    if not host or not repo_path:
        return None

    provider = _REPO_PROVIDERS.get(host, "other")
    normalized_url = f"https://{host}/{repo_path}"
    return normalized_url, provider, repo_path
