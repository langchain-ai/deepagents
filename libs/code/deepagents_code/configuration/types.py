"""Typed configuration-provider results and health metadata."""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from pathlib import Path
from typing import Any

type TomlScalar = str | int | float | bool | date | datetime | time
type TomlValue = TomlScalar | list[TomlValue] | dict[str, TomlValue]
type Provenance = dict[str, str]


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
class Found[T]:
    """A provider supplied a valid value, including an explicit empty value."""

    value: T
    source: str


@dataclass(frozen=True, slots=True)
class Unset:
    """A provider did not define a value."""


@dataclass(frozen=True, slots=True)
class Invalid:
    """A provider defined a value that failed coercion."""

    source: str
    detail: str


type ProviderResult[T] = Found[T] | Unset | Invalid


@dataclass(frozen=True, slots=True)
class ResolvedValue[T]:
    """Effective value plus source and rejected-layer metadata."""

    value: T
    source: str
    provenance: Provenance = field(default_factory=dict)
    invalid: tuple[Invalid, ...] = ()


@dataclass(frozen=True, slots=True)
class TomlSnapshot:
    """One parsed TOML source and its health."""

    data: dict[str, Any]
    status: ProviderStatus
