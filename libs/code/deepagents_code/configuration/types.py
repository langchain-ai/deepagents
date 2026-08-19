"""Typed configuration-provider results and health metadata."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ProviderHealth(StrEnum):
    """Health of one configuration source."""

    OK = "OK"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"
    CORRUPT = "CORRUPT"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Health and safe diagnostic detail for one provider."""

    name: str
    path: Path | None
    health: ProviderHealth
    detail: str | None = None

    @property
    def usable(self) -> bool:
        """Whether the provider can safely participate in resolution."""
        return self.health in {ProviderHealth.OK, ProviderHealth.MISSING}


@dataclass(frozen=True, slots=True)
class TomlSnapshot:
    """One parsed TOML source and its health."""

    data: dict[str, Any]
    status: ProviderStatus
