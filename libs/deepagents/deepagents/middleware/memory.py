"""Middleware for loading agent memory/context from AGENTS.md files.

This module implements support for the AGENTS.md specification (https://agents.md/),
loading memory/context from configurable sources and injecting into the system prompt.

## Overview

AGENTS.md files provide project-specific context and instructions to help AI agents
work effectively. Unlike skills (which are on-demand workflows), memory is always
loaded and provides persistent context.

## Usage

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        {"path": "~/.deepagents/AGENTS.md", "name": "user"},
        {"path": "./.deepagents/AGENTS.md", "name": "project"},
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## Memory Sources

Sources are loaded in order and combined. Each source has:
- `path`: Path to the AGENTS.md file (resolved via backend)
- `name`: Display name for the source (e.g., "user", "project")

Multiple sources are combined in order, with all content included.
Later sources appear after earlier ones in the combined prompt.

## File Format

AGENTS.md files are standard Markdown with no required structure.
Common sections include:
- Project overview
- Build/test commands
- Code style guidelines
- Architecture notes
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class MemorySource(TypedDict):
    """Configuration for a memory source.

    Attributes:
        path: Path to the AGENTS.md file.
        name: Display name for this source (e.g., "user", "project").
    """

    path: str
    name: str


class MemoryState(AgentState):
    """State schema for MemoryMiddleware.

    Attributes:
        memory_contents: Dict mapping source names to their loaded content.
    """

    memory_contents: NotRequired[dict[str, str]]


class MemoryStateUpdate(TypedDict):
    """State update for MemoryMiddleware."""

    memory_contents: dict[str, str]


# Default system prompt template for memory
MEMORY_SYSTEM_PROMPT = """
## Agent Memory

You have access to persistent memory that provides context and instructions.

{memory_locations}

{memory_contents}

**Memory Guidelines:**
- Memory content above provides project-specific context and instructions
- Follow any guidelines, conventions, or patterns described in memory
- Memory is read-only during this session (loaded at startup)
- If you need to update memory, use the appropriate file editing tools
"""

# Type alias for backend or factory function
BackendOrFactory = "BackendProtocol | Callable[[Runtime], BackendProtocol]"


class MemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent memory from AGENTS.md files.

    Loads memory content from configured sources and injects into the system prompt.
    Supports multiple sources that are combined together.

    Example:
        >>> from deepagents.backends.filesystem import FilesystemBackend
        >>> backend = FilesystemBackend(root_dir="/")
        >>> middleware = MemoryMiddleware(
        ...     backend=backend,
        ...     sources=[
        ...         {"path": "/home/user/.deepagents/AGENTS.md", "name": "user"},
        ...         {"path": "/project/.deepagents/AGENTS.md", "name": "project"},
        ...     ],
        ... )

    Args:
        backend: Backend instance or factory function for file operations.
        sources: List of MemorySource configurations specifying paths and names.
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BackendOrFactory,
        sources: list[MemorySource],
    ) -> None:
        """Initialize the memory middleware.

        Args:
            backend: Backend instance or factory function that takes runtime
                     and returns a backend. Use a factory for StateBackend.
            sources: List of memory sources to load. Each source specifies
                     a path and display name. Sources are loaded in order.
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = MEMORY_SYSTEM_PROMPT

    def _get_backend(self, runtime: Runtime) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            runtime: Runtime context for factory functions.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            return self._backend(runtime)
        return self._backend

    def _format_memory_locations(self) -> str:
        """Format memory source locations for display."""
        if not self.sources:
            return "**Memory Sources:** None configured"

        lines = ["**Memory Sources:**"]
        for source in self.sources:
            lines.append(f"- **{source['name'].capitalize()}**: `{source['path']}`")
        return "\n".join(lines)

    def _format_memory_contents(self, contents: dict[str, str]) -> str:
        """Format loaded memory contents for injection into prompt.

        Args:
            contents: Dict mapping source names to content.

        Returns:
            Formatted string with all memory contents.
        """
        if not contents:
            return "(No memory loaded)"

        sections = []
        for source in self.sources:
            name = source["name"]
            if contents.get(name):
                sections.append(f"<{name}_memory>\n{contents[name]}\n</{name}_memory>")

        if not sections:
            return "(No memory loaded)"

        return "\n\n".join(sections)

    def _load_memory_from_backend(
        self,
        backend: BackendProtocol,
        path: str,
    ) -> str | None:
        """Load memory content from a backend path.

        Args:
            backend: Backend to load from.
            path: Path to the AGENTS.md file.

        Returns:
            File content if found, None otherwise.
        """
        try:
            results = backend.download_files([path])
            # download_files returns a list of FileDownloadResponse objects
            for response in results:
                if response.path == path and response.content is not None:
                    # Content may be bytes or string
                    if isinstance(response.content, bytes):
                        return response.content.decode("utf-8")
                    if isinstance(response.content, list):
                        return "\n".join(response.content)
                    return str(response.content)
        except Exception as e:
            logger.debug(f"Could not load memory from {path}: {e}")
        return None

    def before_agent(self, state: MemoryState, runtime: Runtime) -> MemoryStateUpdate | None:
        """Load memory content before agent execution.

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with memory_contents populated.
        """
        # Skip if already loaded
        if "memory_contents" in state:
            return None

        backend = self._get_backend(runtime)
        contents: dict[str, str] = {}

        for source in self.sources:
            content = self._load_memory_from_backend(backend, source["path"])
            if content:
                contents[source["name"]] = content
                logger.debug(f"Loaded memory from {source['name']}: {source['path']}")

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject memory content into the system prompt.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with memory injected into system prompt.
        """
        contents = request.state.get("memory_contents", {})
        memory_locations = self._format_memory_locations()
        memory_contents = self._format_memory_contents(contents)

        memory_section = self.system_prompt_template.format(
            memory_locations=memory_locations,
            memory_contents=memory_contents,
        )

        if request.system_prompt:
            system_prompt = memory_section + "\n\n" + request.system_prompt
        else:
            system_prompt = memory_section

        return request.override(system_prompt=system_prompt)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
