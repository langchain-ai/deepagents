"""Sandbox provider interface used by Deep Agents Code."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol


@dataclass(frozen=True)
class SandboxInstallHint:
    """How to install the package that provides a sandbox backend.

    Built-in providers ship as `deepagents-code` extras (`kind="extra"`);
    third-party providers install as arbitrary packages (`kind="package"`).
    The distinction lets error messages emit the correct install command
    (`/install daytona` vs. `/install acme-dcode-sandbox --package`).
    """

    kind: Literal["extra", "package"]
    name: str

    def command(self, *, in_app: bool) -> str:
        """Render the install command for this hint.

        Args:
            in_app: Whether to render the in-app slash command (`/install`)
                rather than the CLI command (`dcode --install`).

        Returns:
            The install command string.
        """
        prefix = "/install" if in_app else "dcode --install"
        suffix = " --package" if self.kind == "package" else ""
        return f"{prefix} {self.name}{suffix}"


@dataclass(frozen=True)
class SandboxProviderMetadata:
    """Static description of a sandbox provider used by the registry.

    Lets the CLI and registry describe built-in and config providers without
    instantiating them (which may require credentials or optional
    dependencies). Entry-point providers expose their own instance via the
    `SandboxProvider.metadata` property, which the registry reads only when it
    already needs to construct the provider.
    """

    name: str
    working_dir: str
    install: SandboxInstallHint | None = None
    supports_sandbox_id: bool = True
    supports_snapshot_name: bool = False
    backend_module: str | None = None
    """Importable backend module checked by the pre-flight dependency probe.

    `None` skips the probe (e.g. bundled providers, or third-party providers
    whose package is only resolved when the provider is constructed).
    """


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

    @property
    def metadata(self) -> SandboxProviderMetadata | None:
        """Static metadata describing this provider.

        Third-party providers published under the
        `deepagents_code.sandbox_providers` entry-point group override this so
        the registry can surface their working directory and capability flags
        (snapshot/sandbox-id support) instead of falling back to a generic
        placeholder. Returns `None` by default; the registry then synthesizes a
        minimal default.
        """
        return None

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
