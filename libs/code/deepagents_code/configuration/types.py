"""Typed configuration-provider results and health metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
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


REMOTE_SOURCE_MAX_CHARS = 2048
"""Longest remote source URL retained in operator-facing diagnostics."""


def _validate_remote_source_url(source: str) -> str:
    """Validate and normalize one configured remote source URL.

    Returns:
        The normalized absolute HTTPS URL.

    Raises:
        ValueError: If the source is unsafe or is not an absolute HTTPS URL.
    """
    from urllib.parse import urlsplit

    if len(source) > REMOTE_SOURCE_MAX_CHARS:
        msg = "remote source is too long"
        raise ValueError(msg)
    if any(char.isspace() or not char.isprintable() for char in source):
        msg = "remote source must not contain whitespace or control characters"
        raise ValueError(msg)
    try:
        parsed = urlsplit(source)
    except ValueError as exc:
        msg = "remote source is not a valid URL"
        raise ValueError(msg) from exc
    if not source.isascii():
        msg = "remote source must contain only ASCII URI characters"
        raise ValueError(msg)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        msg = "remote source must be an absolute HTTPS URL"
        raise ValueError(msg)
    if parsed.username is not None or parsed.password is not None:
        msg = "remote source must not contain credentials"
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        msg = "remote source must not contain a query string or fragment"
        raise ValueError(msg)
    try:
        port = parsed.port
    except ValueError as exc:
        msg = "remote source has an invalid port"
        raise ValueError(msg) from exc
    host = parsed.hostname.rstrip(".")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return parsed._replace(scheme="https", netloc=netloc).geturl()


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Health and safe diagnostic detail for one provider."""

    name: str
    path: Path | None
    health: ProviderHealth
    detail: str | None = None
    remote_source: str | None = field(default=None, kw_only=True)
    """Validated URL this status came from, when `path` is only a trust anchor.

    A remote managed policy reports the *local* descriptor file as its `path`,
    because that is the file an operator can edit. Without this field an error
    renders as "repair or remove" that file, which is not broken -- and
    removing it drops all policy. Set only after `_validate_remote_url`
    accepts the URL, so a rejected source is never echoed back --
    `__post_init__` enforces that rather than trusting every construction site.
    """

    def __post_init__(self) -> None:
        """Reject a remote source that never passed URL validation.

        Every surface interpolates this field straight into operator-facing
        text, so "only a validated URL reaches here" is a security invariant.
        It held by convention -- one construction site in each of two modules
        -- and `ProviderStatus` is public, so a third site is a rejected
        source, credentials and all, in a `doctor` row. This restates the
        canonical remote-source validator and also requires its normalized
        output: the point is that an unvalidated string cannot get in.

        Raises:
            ValueError: If `remote_source` is not the normalized output of the
                remote-source validator.
        """
        source = self.remote_source
        if source is None:
            return
        msg = "remote_source must be a validated absolute HTTPS URL"
        try:
            normalized = _validate_remote_source_url(source)
        except ValueError as exc:
            raise ValueError(msg) from exc
        if normalized != source:
            raise ValueError(msg)

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
