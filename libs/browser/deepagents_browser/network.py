"""Outbound network policy for browser traffic."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import SplitResult, urlsplit

from deepagents_browser.errors import BrowserErrorCode, NetworkPolicyError

Resolver = Callable[[str, int], Awaitable[Iterable[str]]]
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_MAX_URL_CHARS = 8_192
_DEFAULT_DNS_TIMEOUT_SECONDS = 5.0
_MAX_DNS_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    """A URL that passed scheme, host, port, and address validation."""

    value: str
    host: str
    addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]


class RequestLike(Protocol):
    """Minimal Playwright request interface used by interception."""

    @property
    def url(self) -> str:
        """Return the current request URL, including redirect targets."""


class RouteLike(Protocol):
    """Minimal Playwright route interface used by interception."""

    @property
    def request(self) -> RequestLike:
        """Return the intercepted request."""

    async def continue_(self) -> None:
        """Continue an allowed request."""

    async def abort(self, error_code: str = "failed") -> None:
        """Abort a denied request."""


async def _default_resolver(host: str, port: int) -> Iterable[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return {str(result[4][0]) for result in results}


def _blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or address in _CGNAT
    )


def _port(parts: SplitResult) -> int:
    try:
        explicit = parts.port
    except ValueError as exc:
        msg = "URL contains an invalid port"
        raise NetworkPolicyError(msg) from exc
    return explicit if explicit is not None else (443 if parts.scheme == "https" else 80)


class NetworkPolicy:
    """Validate HTTP(S) URLs and reject non-public destination addresses.

    Args:
        resolver: Asynchronous hostname resolver. Primarily injectable for tests.
        dns_timeout_seconds: Hard timeout for each DNS resolution.
    """

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        dns_timeout_seconds: float = _DEFAULT_DNS_TIMEOUT_SECONDS,
    ) -> None:
        """Create a network policy without performing DNS or network I/O."""
        if not 0 < dns_timeout_seconds <= _MAX_DNS_TIMEOUT_SECONDS:
            msg = "dns_timeout_seconds must be greater than 0 and at most 30"
            raise ValueError(msg)
        self._resolver = resolver or _default_resolver
        self._dns_timeout_seconds = dns_timeout_seconds

    async def _resolve_addresses(
        self,
        host: str,
        port: int,
    ) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        try:
            async with asyncio.timeout(self._dns_timeout_seconds):
                raw_addresses = tuple(await self._resolver(host, port))
        except (OSError, UnicodeError) as exc:
            msg = "Hostname resolution failed"
            raise NetworkPolicyError(msg) from exc
        if not raw_addresses:
            msg = "Hostname did not resolve to any address"
            raise NetworkPolicyError(msg)
        try:
            return tuple(ipaddress.ip_address(item) for item in raw_addresses)
        except ValueError as exc:
            msg = "Resolver returned an invalid IP address"
            raise NetworkPolicyError(msg) from exc

    async def validate_url(self, url: str) -> ValidatedURL:
        """Validate a URL and every DNS answer for its host.

        Args:
            url: Absolute URL to validate.

        Returns:
            The normalized host and resolved addresses.

        Raises:
            NetworkPolicyError: If the URL is malformed, uses a disallowed scheme,
                has credentials, fails resolution, or any answer is non-public.
        """
        if len(url) > _MAX_URL_CHARS:
            msg = "URL exceeds the 8192-character limit"
            raise NetworkPolicyError(msg)
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            msg = "Only http and https URLs are allowed"
            raise NetworkPolicyError(msg)
        if parts.username is not None or parts.password is not None:
            msg = "URLs containing credentials are not allowed"
            raise NetworkPolicyError(msg)
        host = parts.hostname
        if host is None:
            msg = "URL must include a hostname"
            raise NetworkPolicyError(msg)
        port = _port(parts)
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            addresses = await self._resolve_addresses(host, port)
        else:
            addresses = (literal,)
        if any(_blocked_address(address) for address in addresses):
            msg = "Destination resolves to a blocked address"
            raise NetworkPolicyError(msg, code=BrowserErrorCode.BLOCKED_URL)
        return ValidatedURL(value=url, host=host, addresses=addresses)

    async def handle_route(self, route: RouteLike) -> None:
        """Validate every intercepted request URL and fail closed on denial."""
        try:
            await self.validate_url(route.request.url)
        except Exception:  # noqa: BLE001  # fail closed at the network boundary
            await route.abort("blockedbyclient")
            return
        await route.continue_()
