"""Typed configuration-provider results and health metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Found[T]:
    """A provider declared a value and coerced it successfully."""

    value: T


@dataclass(frozen=True, slots=True)
class Unset:
    """A provider made no declaration for an option."""


@dataclass(frozen=True, slots=True)
class Invalid:
    """A provider declared a value that its input domain could not coerce."""

    reason: str


type ProviderResult[T] = Found[T] | Unset | Invalid
"""Complete result of reading one option from one provider."""


class ProviderHealth(StrEnum):
    """Health of one configuration source."""

    OK = "OK"
    MISSING = "MISSING"
    INDETERMINATE = "INDETERMINATE"
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
        """Whether the provider can safely participate in resolution.

        `MISSING` is usable because no file at an authoritative path means the
        administrator deployed no policy. `INDETERMINATE` is not: the path
        itself is a guess, so an empty read proves nothing about what policy
        the administrator deployed.
        """
        return self.health in {ProviderHealth.OK, ProviderHealth.MISSING}


@dataclass(frozen=True, slots=True)
class TomlSnapshot:
    """One parsed TOML source and its health.

    `data` is empty whenever `status.health` is not `OK`, so it must always be
    read together with `status`: an empty table alone cannot distinguish "this
    source declares nothing" from "this source could not be read".
    """

    data: dict[str, Any]
    status: ProviderStatus

    def __post_init__(self) -> None:
        """Reject a snapshot that carries data it could not have read.

        `data` is a `dict`, not a `Mapping`, and deliberately not copied
        behind a `MappingProxyType` the way `ResolvedValue.__post_init__`
        copies its own mappings. It is a raw TOML table, and the coercers test
        nested values with `isinstance(value, dict)` to tell a table from a
        scalar -- so any `Mapping` that is not a `dict` fails that test and
        every option under it falls back to its next source, silently. The
        narrower annotation makes a `mappingproxy` (or any other `Mapping`) a
        type error at the call site rather than a fall-through at runtime.

        The immutability this type promises is therefore a convention, kept by
        the providers that build the snapshot and by callers handed one.

        Raises:
            ValueError: If an unhealthy snapshot carries a non-empty table.
        """
        if self.status.health is not ProviderHealth.OK and self.data:
            msg = (
                f"a {self.status.health.value} snapshot must carry no data; an "
                "empty table is what every reader reads as 'nothing declared'"
            )
            raise ValueError(msg)

    @classmethod
    def from_table(cls, name: str, data: dict[str, Any]) -> TomlSnapshot:
        """Build a readable snapshot around an already-parsed table.

        For a caller that holds the parsed data and no health metadata of its
        own. A caller that has real health passes both halves directly.

        Args:
            name: Human-readable source label.
            data: Parsed TOML table.

        Returns:
            An `OK` snapshot carrying `data`.
        """
        return cls(data, ProviderStatus(name, None, ProviderHealth.OK))

    @classmethod
    def declaring_nothing(cls, name: str) -> TomlSnapshot:
        """Build a readable snapshot for a source that declares nothing.

        `ProviderStatus.usable` gates whether a source takes part in
        resolution, so the difference between this and `absent` or
        `unknown_origin` is a real decision. Naming it keeps that decision from
        being made by copy-paste: three of the call sites this replaced picked
        three different healths for a literally empty user table.

        Args:
            name: Human-readable source label.

        Returns:
            An `OK` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.OK))

    @classmethod
    def absent(cls, name: str) -> TomlSnapshot:
        """Build a snapshot for a source that is not there at all.

        Args:
            name: Human-readable source label.

        Returns:
            A `MISSING` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.MISSING))

    @classmethod
    def unknown_origin(cls, name: str) -> TomlSnapshot:
        """Build a snapshot for a source whose state could not be determined.

        Args:
            name: Human-readable source label.

        Returns:
            An `INDETERMINATE` snapshot carrying no data.
        """
        return cls({}, ProviderStatus(name, None, ProviderHealth.INDETERMINATE))
