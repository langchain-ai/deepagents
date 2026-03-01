"""Utilities for project root detection and project-specific configuration."""

from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific AGENTS.md file(s).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/AGENTS.md
    2. project_root/AGENTS.md

    Both files will be loaded and combined if both exist.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Existing AGENTS.md paths.

            Empty if neither file exists, one entry if only one is present, or
            two entries if both locations have the file.
    """
    candidates = [
        project_root / ".deepagents" / "AGENTS.md",
        project_root / "AGENTS.md",
    ]
    paths: list[Path] = []
    for candidate in candidates:
        try:
            if candidate.exists():
                paths.append(candidate)
        except OSError:
            pass
    return paths
