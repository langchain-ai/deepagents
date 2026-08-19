"""Typed configuration-provider results and health metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


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

    data: Mapping[str, Any]
    status: ProviderStatus

    def __post_init__(self) -> None:
        """Reject a snapshot that carries data it could not have read.

        Raises:
            ValueError: If an unhealthy snapshot carries a non-empty table.
        """
        if self.status.health is not ProviderHealth.OK and self.data:
            msg = (
                f"a {self.status.health.value} snapshot must carry no data; an "
                "empty table is what every reader reads as 'nothing declared'"
            )
            raise ValueError(msg)
