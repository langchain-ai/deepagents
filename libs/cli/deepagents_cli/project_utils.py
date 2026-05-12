"""Utilities for project root detection and project-specific configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_cli._env_vars import SERVER_ENV_PREFIX
from deepagents_cli._git import find_git_root

if TYPE_CHECKING:
    from collections.abc import Mapping

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectContext:
    """Explicit user/project path context for project-sensitive behavior.

    Attributes:
        user_cwd: Authoritative working directory from the CLI invocation.
        project_root: Resolved project root for `user_cwd`, if one exists.
    """

    user_cwd: Path
    project_root: Path | None = None

    def __post_init__(self) -> None:
        """Validate that path fields are absolute.

        Raises:
            ValueError: If `user_cwd` or `project_root` is not absolute.
        """
        if not self.user_cwd.is_absolute():
            msg = f"user_cwd must be absolute, got {self.user_cwd!r}"
            raise ValueError(msg)
        if self.project_root is not None and not self.project_root.is_absolute():
            msg = f"project_root must be absolute, got {self.project_root!r}"
            raise ValueError(msg)

    @classmethod
    def from_user_cwd(cls, user_cwd: str | Path) -> ProjectContext:
        """Build a project context from an explicit user working directory.

        Args:
            user_cwd: User invocation directory.

        Returns:
            Resolved project context.
        """
        resolved_cwd = Path(user_cwd).expanduser().resolve()
        return cls(
            user_cwd=resolved_cwd,
            project_root=find_project_root(resolved_cwd),
        )

    def resolve_user_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the explicit user working directory.

        Args:
            path: Absolute or relative user-facing path.

        Returns:
            Absolute resolved path.
        """
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.user_cwd / candidate).resolve()

    def project_agent_md_paths(self) -> list[Path]:
        """Return project-level `AGENTS.md` files for this context."""
        if self.project_root is None:
            return []
        return find_project_agent_md(self.project_root)

    def project_skills_dir(self) -> Path | None:
        """Return the project `.deepagents/skills` directory, if any."""
        if self.project_root is None:
            return None
        return self.project_root / ".deepagents" / "skills"

    def project_agents_dir(self) -> Path | None:
        """Return the project `.deepagents/agents` directory, if any."""
        if self.project_root is None:
            return None
        return self.project_root / ".deepagents" / "agents"

    def project_agent_skills_dir(self) -> Path | None:
        """Return the project `.agents/skills` directory, if any."""
        if self.project_root is None:
            return None
        return self.project_root / ".agents" / "skills"


def get_server_project_context(
    env: Mapping[str, str] | None = None,
) -> ProjectContext | None:
    """Read the server project context from environment transport data.

    Args:
        env: Environment mapping to read from.

    Returns:
        Reconstructed project context, or `None` if no server context exists.
    """
    environment = os.environ if env is None else env
    raw_cwd = environment.get(f"{SERVER_ENV_PREFIX}CWD")
    if not raw_cwd:
        return None

    try:
        user_cwd = Path(raw_cwd).expanduser().resolve()
        raw_project_root = environment.get(f"{SERVER_ENV_PREFIX}PROJECT_ROOT")
        project_root = (
            Path(raw_project_root).expanduser().resolve()
            if raw_project_root
            else find_project_root(user_cwd)
        )
    except OSError:
        logger.warning(
            "Could not resolve server project context from CWD=%s",
            raw_cwd,
            exc_info=True,
        )
        return None

    return ProjectContext(user_cwd=user_cwd, project_root=project_root)


def find_project_root(start_path: str | Path | None = None) -> Path | None:
    """Find the project root by looking for git metadata.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).expanduser().resolve()
    return find_git_root(current)


def find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific AGENTS.md file(s).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/AGENTS.md
    2. project_root/AGENTS.md

    Both files will be loaded and combined if both exist.

    Candidates with symlinked path components are followed only when the
    resolved target stays inside `project_root`. The returned `Path` is the
    resolved target when any symlink component was traversed, so
    `FilesystemBackend.download_files` opens a regular file rather than
    tripping `O_NOFOLLOW` on the link itself. Symlinks pointing outside the
    project root, symlink loops, and unreadable parents are skipped with a
    warning. Broken symlinks are treated as missing files (no warning),
    matching the pre-existing behavior for absent candidates.

    Why: project AGENTS.md is auto-discovered and loaded into the system
    prompt before the first model call. Without the in-tree check, a
    malicious clone could ship `AGENTS.md -> ~/.ssh/config` (or any other
    locally-readable file) and have its contents injected as agent
    instructions on first run.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Existing AGENTS.md paths, with in-tree symlinked path components
            pre-resolved to their targets. Empty if neither file exists, one
            entry if only one is present, or two entries if both locations
            have the file.
    """
    # Resolve the root once so the candidate-equality check below works even
    # when the caller passes a non-canonical `project_root` (e.g., macOS
    # `/var` -> `/private/var`, or a path with symlinked ancestors).
    project_root_resolved = project_root.resolve()
    candidates = [
        project_root_resolved / ".deepagents" / "AGENTS.md",
        project_root_resolved / "AGENTS.md",
    ]
    paths: list[Path] = []
    for candidate in candidates:
        try:
            # Single syscall handles existence, broken symlinks, loops, and
            # canonicalization. `strict=True` raises rather than returning
            # the unresolved path.
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            # Absent file or broken symlink — matches the pre-existing
            # silent-skip behavior for missing candidates. `Path.exists()`
            # also returns False for symlink loops on some Python versions,
            # so loops do NOT come through here; see the OSError branch.
            continue
        except (OSError, RuntimeError) as exc:
            # `OSError(ELOOP)` on Python 3.13+, `RuntimeError("Symlink loop
            # ...")` on 3.11-3.12; bare `OSError` for permission/unreadable
            # parent. Security-relevant — warn and skip.
            logger.warning(
                "Skipping AGENTS.md candidate %s: %s",
                candidate,
                exc,
            )
            continue

        try:
            resolved.relative_to(project_root_resolved)
        except ValueError:
            logger.warning(
                "Skipping AGENTS.md symlink %s: target %s is outside "
                "the project root %s",
                candidate,
                resolved,
                project_root_resolved,
            )
            continue

        if candidate.absolute() == resolved:
            paths.append(candidate)
        else:
            paths.append(resolved)
    return paths
