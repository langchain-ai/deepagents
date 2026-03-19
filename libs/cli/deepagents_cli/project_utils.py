"""Utilities for project root detection and project-specific configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_cli._server_constants import ENV_PREFIX as _ENV_PREFIX

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
        """Return project-level `AGENTS.md` files with hierarchical discovery.

        Supports ancestor walk from project_root to user_cwd and subdir
        auto-discovery.
        """
        if self.project_root is None:
            return []
        return find_project_agent_md(self.project_root, cwd=self.user_cwd)

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
    raw_cwd = environment.get(f"{_ENV_PREFIX}CWD")
    if not raw_cwd:
        return None

    try:
        user_cwd = Path(raw_cwd).expanduser().resolve()
        raw_project_root = environment.get(f"{_ENV_PREFIX}PROJECT_ROOT")
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
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).expanduser().resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path, cwd: Path | None = None) -> list[Path]:
    """Find project-specific AGENTS.md file(s) with hierarchical discovery.

    Supports three discovery mechanisms:
    1. Ancestor walk: Walk from project_root to cwd, collecting AGENTS.md files
    2. Subdir auto-discovery: Scan subdirectories for AGENTS.md files
    3. Traditional: Check .deepagents/AGENTS.md and AGENTS.md at each level

    Results are ordered so files closer to cwd/leaf come last (override earlier ones).

    Args:
        project_root: Path to the project root directory.
        cwd: Current working directory for ancestor walk. Defaults to project_root.
            When provided, performs ancestor walk from project_root to cwd.

    Returns:
        Existing AGENTS.md paths in override order (root → leaf).

    Raises:
        ValueError: If cwd is not a subdirectory of project_root.
    """
    if cwd is None:
        cwd = project_root

    # Validate cwd is under project_root
    try:
        cwd.relative_to(project_root)
    except ValueError as e:
        msg = f"cwd ({cwd}) must be under project_root ({project_root})"
        raise ValueError(msg) from e

    paths: list[Path] = []
    seen: set[Path] = set()

    def add_if_exists(path: Path) -> None:
        """Add AGENTS.md path if it exists and hasn't been seen."""
        try:
            if path.exists() and path not in seen:
                seen.add(path)
                paths.append(path)
        except OSError:
            pass

    # Phase 1: Ancestor walk from project_root to cwd
    # Walk down from project_root to cwd, collecting AGENTS.md at each level
    current = project_root
    while str(current) != str(cwd):
        add_if_exists(current / ".deepagents" / "AGENTS.md")
        add_if_exists(current / "AGENTS.md")

        # Move closer to cwd
        if str(current) == str(cwd):
            break

        # Find the next child towards cwd
        cwd_parts = cwd.parts
        current_parts = current.parts
        if len(cwd_parts) > len(current_parts):
            current /= cwd_parts[len(current_parts)]
        else:
            break

    # Also add the final cwd level
    add_if_exists(cwd / ".deepagents" / "AGENTS.md")
    add_if_exists(cwd / "AGENTS.md")

    # Phase 2: Subdir auto-discovery (bounded scan under cwd)
    # Scan immediate subdirectories of cwd for AGENTS.md files
    try:
        for subdir in cwd.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                add_if_exists(subdir / "AGENTS.md")
                add_if_exists(subdir / ".deepagents" / "AGENTS.md")
    except OSError:
        pass

    return paths
