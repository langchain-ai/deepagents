"""Middleware for injecting git state and directory context into system prompt."""

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


class GitContextState(AgentState):
    """State for git context middleware."""

    git_context: NotRequired[str]
    """Formatted git state and directory information."""


class GitContextStateUpdate(TypedDict):
    """State update for git context middleware."""

    git_context: str
    """Formatted git state and directory information."""


class GitContextMiddleware(AgentMiddleware):
    """Middleware for injecting git state into system prompt.

    This middleware:
    1. Detects current git branch (if in a git repo)
    2. Checks if main/master branches exist locally
    3. Appends git context to system prompt
    """

    state_schema = GitContextState

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

    def before_agent(
        self,
        state: GitContextState,
        runtime: Runtime,
    ) -> GitContextStateUpdate | None:
        """Load git context before agent execution.

        Runs once at session start.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with git_context populated, or None if not in git repo.
        """
        git_info = self._get_git_info()
        if not git_info:
            return None

        # Build git context section
        git_lines = ["## Git Context", "", f"Current branch: `{git_info['branch']}`"]
        for main_branch in git_info.get("main_branches", []):
            git_lines.append(f"Main branch available: `{main_branch}`")

        git_context = "\n".join(git_lines)
        return GitContextStateUpdate(git_context=git_context)

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """Get modified request with git context injected, or None if no context.

        Args:
            request: The original model request.

        Returns:
            Modified request with git context appended, or None if no git context.
        """
        state = cast("GitContextState", request.state)
        git_context = state.get("git_context", "")

        if not git_context:
            return None

        # Append git context to system prompt
        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + git_context

        return request.override(system_prompt=new_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject git context into system prompt.

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
        """(async) Inject git context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request if modified_request else request)


__all__ = ["GitContextMiddleware"]
