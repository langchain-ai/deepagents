"""Structural contract for ranked configuration providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.types import ProviderStatus


@runtime_checkable
class ConfigProvider(Protocol):
    """A ranked source of typed configuration values."""

    @property
    def name(self) -> str:
        """Provider display label."""
        ...

    @property
    def rank(self) -> int:
        """Numeric precedence rank."""
        ...

    @property
    def durable(self) -> bool:
        """Whether the source survives the process."""
        ...

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read and coerce one manifest option."""
        ...

    def status(self) -> ProviderStatus:
        """Return current provider health and display metadata."""
        ...

    def reload(self) -> None:
        """Refresh provider state when the source supports it."""
        ...
