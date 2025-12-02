"""Middleware for injecting local context into system prompt."""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


class LocalContextState(AgentState):
    """State for local context middleware."""

    local_context: NotRequired[str]
    """Formatted local context: git, cwd, files, tree."""


class LocalContextStateUpdate(TypedDict):
    """State update for local context middleware."""

    local_context: str
    """Formatted local context: git, cwd, files, tree."""


class LocalContextMiddleware(AgentMiddleware):
    """Middleware for injecting local context into system prompt.

    This middleware:
    1. Detects current git branch (if in a git repo)
    2. Checks if main/master branches exist locally
    3. Lists files in current directory (max 20)
    4. Shows directory tree structure (max 3 levels, 20 entries)
    5. Appends local context to system prompt
    """

    state_schema = LocalContextState

    def _get_git_info(self) -> dict[str, str | list[str]]:
        """Gather git state information.

        Returns:
            Dict with 'branch' (current branch) and 'main_branches' (list of main/master if they exist).
            Returns empty dict if not in git repo.
        """
        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=Path.cwd(),
                check=False,
            )
            if result.returncode != 0:
                return {}

            current_branch = result.stdout.strip()

            # Get local branches to check for main/master
            main_branches = []
            result = subprocess.run(
                ["git", "branch"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=Path.cwd(),
                check=False,
            )
            if result.returncode == 0:
                branches = set()
                for line in result.stdout.strip().split("\n"):
                    branch = line.strip().lstrip("*").strip()
                    if branch:
                        branches.add(branch)

                if "main" in branches:
                    main_branches.append("main")
                if "master" in branches:
                    main_branches.append("master")

            return {"branch": current_branch, "main_branches": main_branches}

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return {}

    def _get_file_list(self, max_files: int = 20) -> list[str]:
        """Get list of files in current directory (non-recursive).

        Args:
            max_files: Maximum number of files to show (default 20).

        Returns:
            List of file paths (sorted), truncated to max_files.
        """
        cwd = Path.cwd()

        # Ignore patterns
        ignore_patterns = {
            ".git",
            "node_modules",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".coverage",
            ".eggs",
            "dist",
            "build",
        }

        files = []
        try:
            for item in sorted(cwd.iterdir()):
                # Skip hidden files (except .deepagents)
                if item.name.startswith(".") and item.name != ".deepagents":
                    continue

                # Skip ignored patterns
                if item.name in ignore_patterns:
                    continue

                # Add files and dirs
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    files.append(f"{item.name}/")

                if len(files) >= max_files:
                    break

        except (OSError, PermissionError):
            return []

        return files

    def _get_directory_tree(self, max_depth: int = 3, max_entries: int = 20) -> str:
        """Get directory tree structure.

        Args:
            max_depth: Maximum depth to traverse (default 3).
            max_entries: Maximum total entries to show (default 20).

        Returns:
            Formatted tree string or empty if error.
        """
        cwd = Path.cwd()

        ignore_patterns = {
            ".git",
            "node_modules",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".coverage",
            ".eggs",
            "dist",
            "build",
        }

        lines = []
        entry_count = [0]  # Mutable for closure

        def _build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            """Recursive tree builder."""
            if depth >= max_depth or entry_count[0] >= max_entries:
                return

            try:
                items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except (OSError, PermissionError):
                return

            for i, item in enumerate(items):
                if entry_count[0] >= max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    return

                # Skip hidden and ignored
                if item.name.startswith(".") and item.name != ".deepagents":
                    continue
                if item.name in ignore_patterns:
                    continue

                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "

                display_name = f"{item.name}/" if item.is_dir() else item.name
                lines.append(f"{prefix}{connector}{display_name}")
                entry_count[0] += 1

                # Recurse into directories
                if item.is_dir() and depth + 1 < max_depth:
                    extension = "    " if is_last else "│   "
                    _build_tree(item, prefix + extension, depth + 1)

        try:
            lines.append(f"{cwd.name}/")
            _build_tree(cwd)
        except Exception:
            return ""

        return "\n".join(lines)

    def before_agent(
        self,
        state: LocalContextState,
        runtime: Runtime,
    ) -> LocalContextStateUpdate | None:
        """Load local context before agent execution.

        Runs once at session start.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with local_context populated, or None if no context available.
        """
        cwd = Path.cwd()
        sections = ["## Local Context", ""]

        # Current directory
        sections.append(f"**Current Directory**: `{cwd}`")
        sections.append("")

        # Git info
        git_info = self._get_git_info()
        if git_info:
            git_text = f"**Git**: Current branch `{git_info['branch']}`"
            if git_info.get("main_branches"):
                main_branches = ", ".join(f"`{b}`" for b in git_info["main_branches"])
                git_text += f", main branch available: {main_branches}"
            sections.append(git_text)
            sections.append("")

        # File list
        files = self._get_file_list()
        if files:
            total_items = len(list(Path.cwd().iterdir()))
            sections.append(f"**Files** ({len(files)} shown):")
            for file in files:
                sections.append(f"- {file}")
            if len(files) < total_items:
                remaining = total_items - len(files)
                sections.append(f"... ({remaining} more files)")
            sections.append("")

        # Directory tree
        tree = self._get_directory_tree()
        if tree:
            sections.append("**Tree** (3 levels):")
            sections.append(tree)

        local_context = "\n".join(sections)
        return LocalContextStateUpdate(local_context=local_context)

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """Get modified request with local context injected, or None if no context.

        Args:
            request: The original model request.

        Returns:
            Modified request with local context appended, or None if no local context.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")

        if not local_context:
            return None

        # Append local context to system prompt
        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + local_context

        return request.override(system_prompt=new_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request if modified_request else request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request if modified_request else request)


__all__ = ["LocalContextMiddleware"]
