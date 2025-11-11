"""Utilities for project root detection and project-specific configuration."""

from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from. Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path) -> Path | None:
    """Find project-specific agent.md file.

    Checks two locations in order:
    1. project_root/.deepagents/agent.md
    2. project_root/agent.md

    Args:
        project_root: Path to the project root directory.

    Returns:
        Path to the project agent.md if found, None otherwise.
    """
    # Check .deepagents/agent.md first
    deepagents_md = project_root / ".deepagents" / "agent.md"
    if deepagents_md.exists():
        return deepagents_md

    # Check root agent.md
    root_md = project_root / "agent.md"
    if root_md.exists():
        return root_md

    return None
