"""Sandbox provider interface used by the deepagents CLI."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol


class SandboxError(Exception):
    """Base error for sandbox provider operations."""

    @property
    def original_exc(self) -> BaseException | None:
        """Return the original exception that caused this error, if any."""
        return self.__cause__


class SandboxNotFoundError(SandboxError):
    """Raised when the requested sandbox cannot be found."""


class SandboxProvider(ABC):
    """Interface for creating and deleting sandbox backends."""

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox, or create one if needed."""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox by id."""
        raise NotImplementedError

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Async wrapper around get_or_create.

        Returns:
            The created or existing sandbox backend.
        """
        return await asyncio.to_thread(
            self.get_or_create, sandbox_id=sandbox_id, **kwargs
        )

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Async wrapper around delete."""
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)
