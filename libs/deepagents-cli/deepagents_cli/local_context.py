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

# Directories to ignore in file listings and tree views
IGNORE_PATTERNS = frozenset(
    {
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
)


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

        files = []
        try:
            for item in sorted(cwd.iterdir()):
                # Skip hidden files (except .deepagents)
                if item.name.startswith(".") and item.name != ".deepagents":
                    continue

                # Skip ignored patterns
                if item.name in IGNORE_PATTERNS:
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

        lines: list[str] = []
        entry_count = [0]  # Mutable for closure

        def _should_include(item: Path) -> bool:
            """Check if item should be included in tree."""
            # Skip hidden files (except .deepagents)
            if item.name.startswith(".") and item.name != ".deepagents":
                return False
            # Skip ignored patterns
            return item.name not in IGNORE_PATTERNS

        def _build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            """Recursive tree builder."""
            if depth >= max_depth or entry_count[0] >= max_entries:
                return

            try:
                all_items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
                # Pre-filter to get correct is_last determination
                items = [item for item in all_items if _should_include(item)]
            except (OSError, PermissionError):
                return

            for i, item in enumerate(items):
                if entry_count[0] >= max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    return

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
        except (OSError, PermissionError):
            return ""

        return "\n".join(lines)

    def _detect_package_manager(self) -> str | None:
        """Detect Python package manager in use.

        Checks for lock files and config files to determine the package manager.

        Uses priority order: `uv > poetry > pipenv > pip`. First match wins if multiple
        indicators are present.

        Returns:
            Package manager name (uv, poetry, pipenv, pip) or `None` if not detected.
        """
        cwd = Path.cwd()

        # Check for uv (uv.lock or pyproject.toml with [tool.uv])
        if (cwd / "uv.lock").exists():
            return "uv"

        # Check for poetry (poetry.lock or pyproject.toml with [tool.poetry])
        if (cwd / "poetry.lock").exists():
            return "poetry"

        # Check for pipenv
        if (cwd / "Pipfile.lock").exists() or (cwd / "Pipfile").exists():
            return "pipenv"

        # Check pyproject.toml for tool sections
        pyproject = cwd / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                if "[tool.uv]" in content:
                    return "uv"
                if "[tool.poetry]" in content:
                    return "poetry"
                # Has pyproject.toml but no specific tool - likely pip/setuptools
                return "pip"
            except (OSError, PermissionError, UnicodeDecodeError):
                pass

        # Check for requirements.txt
        if (cwd / "requirements.txt").exists():
            return "pip"

        return None

    def _detect_node_package_manager(self) -> str | None:
        """Detect Node.js package manager in use.

        Uses priority order: `bun > pnpm > yarn > npm`.

        First match wins if multiple lock files are present.

        Returns:
            Package manager name (bun, pnpm, yarn, npm) or `None` if not detected.
        """
        cwd = Path.cwd()

        if (cwd / "bun.lockb").exists() or (cwd / "bun.lock").exists():
            return "bun"
        if (cwd / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (cwd / "yarn.lock").exists():
            return "yarn"
        if (cwd / "package-lock.json").exists() or (cwd / "package.json").exists():
            return "npm"

        return None

    def _get_makefile_preview(self, max_lines: int = 20) -> str | None:
        """Get first N lines of `Makefile` if present.

        Args:
            max_lines: Maximum lines to show.

        Returns:
            `Makefile` preview or `None` if not found.
        """
        cwd = Path.cwd()
        makefile = cwd / "Makefile"

        if not makefile.exists():
            return None

        try:
            content = makefile.read_text()
            lines = content.split("\n")[:max_lines]
            preview = "\n".join(lines)
            if len(content.split("\n")) > max_lines:
                preview += "\n... (truncated)"
            return preview
        except (OSError, PermissionError, UnicodeDecodeError):
            return None

    def _detect_project_info(self) -> dict[str, str | bool | None]:
        """Detect project type, language, and structure.

        Returns:
            Dict with `language`, `is_monorepo`, `project_root`, `has_venv`, `has_node_modules`.
        """
        cwd = Path.cwd()
        info: dict[str, str | bool | None] = {
            "language": None,
            "is_monorepo": False,
            "project_root": None,
            "has_venv": False,
            "has_node_modules": False,
        }

        # Check for virtual environments
        info["has_venv"] = (cwd / ".venv").exists() or (cwd / "venv").exists()
        info["has_node_modules"] = (cwd / "node_modules").exists()

        # Detect primary language
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            info["language"] = "python"
        elif (cwd / "package.json").exists():
            info["language"] = "javascript/typescript"
        elif (cwd / "Cargo.toml").exists():
            info["language"] = "rust"
        elif (cwd / "go.mod").exists():
            info["language"] = "go"
        elif (cwd / "pom.xml").exists() or (cwd / "build.gradle").exists():
            info["language"] = "java"

        # Detect monorepo patterns
        # Check for common monorepo indicators
        monorepo_indicators = [
            (cwd / "lerna.json").exists(),
            (cwd / "pnpm-workspace.yaml").exists(),
            (cwd / "packages").is_dir(),
            (cwd / "libs").is_dir() and (cwd / "apps").is_dir(),
            (cwd / "workspaces").is_dir(),
        ]
        info["is_monorepo"] = any(monorepo_indicators)

        # Try to find project root (look for .git or pyproject.toml up the tree)
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=cwd,
                check=False,
            )
            if result.returncode == 0:
                info["project_root"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return info

    def _detect_test_command(self) -> str | None:
        """Detect how to run tests based on project structure.

        Returns:
            Suggested test command or `None` if not detected.
        """
        cwd = Path.cwd()

        # Check Makefile for test target
        makefile = cwd / "Makefile"
        if makefile.exists():
            try:
                content = makefile.read_text()
                if "test:" in content or "tests:" in content:
                    return "make test"
            except (OSError, PermissionError, UnicodeDecodeError):
                pass

        # Python projects
        if (cwd / "pyproject.toml").exists():
            pyproject = cwd / "pyproject.toml"
            try:
                content = pyproject.read_text()
                if "[tool.pytest" in content or (cwd / "pytest.ini").exists():
                    return "pytest"
            except (OSError, PermissionError, UnicodeDecodeError):
                pass
            if (cwd / "tests").is_dir() or (cwd / "test").is_dir():
                return "pytest"

        # Node projects
        if (cwd / "package.json").exists():
            try:
                import json

                pkg = json.loads((cwd / "package.json").read_text())
                if "scripts" in pkg and "test" in pkg["scripts"]:
                    return "npm test"
            except (OSError, PermissionError, UnicodeDecodeError, json.JSONDecodeError):
                pass

        return None

    def before_agent(
        self,
        state: LocalContextState,
        runtime: Runtime,
    ) -> LocalContextStateUpdate | None:
        """Load local context before agent execution.

        Runs once at session start to preserve prompt caching.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with local_context populated, or None if already set.
        """
        # Only compute context on first interaction to preserve prompt caching
        if state.get("local_context"):
            return None

        cwd = Path.cwd()
        sections = ["## Local Context", ""]

        # Current directory
        sections.append(f"**Current Directory**: `{cwd}`")
        sections.append("")

        # Project info (language, monorepo, root, environments)
        project_info = self._detect_project_info()
        project_lines = []
        if project_info.get("language"):
            project_lines.append(f"Language: {project_info['language']}")
        if project_info.get("project_root") and str(project_info["project_root"]) != str(cwd):
            project_lines.append(f"Project root: `{project_info['project_root']}`")
        if project_info.get("is_monorepo"):
            project_lines.append("Monorepo: yes")
        env_indicators = []
        if project_info.get("has_venv"):
            env_indicators.append(".venv")
        if project_info.get("has_node_modules"):
            env_indicators.append("node_modules")
        if env_indicators:
            project_lines.append(f"Environments: {', '.join(env_indicators)}")
        if project_lines:
            sections.append("**Project**:")
            sections.extend(f"- {line}" for line in project_lines)
            sections.append("")

        # Package managers
        pkg_managers = []
        python_pkg = self._detect_package_manager()
        if python_pkg:
            pkg_managers.append(f"Python: {python_pkg}")
        node_pkg = self._detect_node_package_manager()
        if node_pkg:
            pkg_managers.append(f"Node: {node_pkg}")
        if pkg_managers:
            sections.append(f"**Package Manager**: {', '.join(pkg_managers)}")
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

        # Test command
        test_cmd = self._detect_test_command()
        if test_cmd:
            sections.append(f"**Run Tests**: `{test_cmd}`")
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
            sections.append("")

        # Makefile preview
        makefile_preview = self._get_makefile_preview()
        if makefile_preview:
            sections.append("**Makefile** (first 20 lines):")
            sections.append("```makefile")
            sections.append(makefile_preview)
            sections.append("```")

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
