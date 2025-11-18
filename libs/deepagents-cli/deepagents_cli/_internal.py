"""Shell tool middleware that survives HITL pauses.

This is temporary implementation of ResumableShellToolMiddleware until
the patch is released in langchain.
"""

from __future__ import annotations

from langchain.agents.middleware.shell_tool import (
    ShellToolMiddleware,
)

ResumableShellToolMiddleware = ShellToolMiddleware

__all__ = ["ResumableShellToolMiddleware"]
