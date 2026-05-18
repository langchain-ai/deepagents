"""Tests for the `fetch_url` tool."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

import pytest
import requests
import responses

from deepagents_code.tools import _UrlValidationError, _validate_url, fetch_url

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    *ips: str,
) -> tuple[Callable[..., list[tuple[Any, ...]]], list[str]]:
    """Return a `socket.getaddrinfo` stand-in plus the list of hostnames it saw.

    Each call returns a single-entry result so callers can simulate multi-hop
    redirect resolution by passing more than one IP. The returned `calls`
    list grows as the stub is invoked, allowing tests to assert that
    `_validate_url` ran on every redirect hop.
    """
    iterator = iter(ips)
    calls: list[str] = []

    def _impl(
        host: str,
        port: int | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        calls.append(host)
        ip = next(iterator)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    return _impl, calls


def _make_multi_record_resolver(
    *ips: str,
) -> Callable[..., list[tuple[Any, ...]]]:
    """Return a resolver that yields *all* `ips` in a single call.

    Simulates real DNS responses that include multiple A/AAAA records for
    one hostname.
    """

    def _impl(
        _host: str,
        port: int | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0)) for ip in ips
        ]

    return _impl


@pytest.fixture
def resolve_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `socket.getaddrinfo` to always return a public IP."""
    public_ip = "93.184.216.34"

    def _impl(
        _host: str,
        port: int | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (public_ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _impl)


@responses.activate
@pytest.mark.usefixtures("resolve_public")
def test_fetch_url_success() -> None:
    """Successful fetch converts HTML to markdown."""
    responses.add(
        responses.GET,
        "http://example.com",
        body="<html><body><h1>Test</h1><p>Content</p></body></html>",
        status=200,
    )

    result = fetch_url("http://example.com")

    assert result["status_code"] == 200
    assert "Test" in result["markdown_content"]
    assert result["url"].startswith("http://example.com")
    assert result["content_length"] > 0


@responses.activate
@pytest.mark.usefixtures("resolve_public")
def test_fetch_url_http_error() -> None:
    """4xx responses surface as structured errors."""
    responses.add(
        responses.GET,
        "http://example.com/notfound",
        status=404,
    )

    result = fetch_url("http://example.com/notfound")

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/notfound"
    assert result["category"] == "network"


@responses.activate
@pytest.mark.usefixtures("resolve_public")
def test_fetch_url_timeout() -> None:
    """Timeouts surface as structured errors."""
    responses.add(
        responses.GET,
        "http://example.com/slow",
        body=requests.exceptions.Timeout(),
    )

    result = fetch_url("http://example.com/slow", timeout=1)

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/slow"
    assert result["category"] == "network"


@responses.activate
@pytest.mark.usefixtures("resolve_public")
def test_fetch_url_connection_error() -> None:
    """Connection errors surface as structured errors."""
    responses.add(
        responses.GET,
        "http://example.com/error",
        body=requests.exceptions.ConnectionError(),
    )

    result = fetch_url("http://example.com/error")

    assert "error" in result
    assert "Fetch URL error" in result["error"]
    assert result["url"] == "http://example.com/error"
    assert result["category"] == "network"


def test_fetch_url_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned direct connection must not be replaced by an env proxy."""
    resolver, _ = _make_resolver("93.184.216.34")
    trust_env_values: list[bool] = []
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    class RecordingSession:
        """Minimal `requests.Session` stand-in that records proxy trust."""

        def __init__(self) -> None:
            self.trust_env = True

        def get(self, url: str, **_kwargs: Any) -> requests.Response:
            trust_env_values.append(self.trust_env)
            response = requests.Response()
            response.status_code = 200
            response.url = url
            response._content = b"<html><body><h1>Proxied</h1></body></html>"
            return response

    monkeypatch.setattr(requests, "Session", RecordingSession)

    result = fetch_url("http://example.com/proxy")

    assert result["status_code"] == 200
    assert trust_env_values == [False]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/",
        "javascript:alert(1)",
    ],
)
def test_validate_url_rejects_disallowed_schemes(url: str) -> None:
    """Only http(s) is allowed."""
    with pytest.raises(_UrlValidationError, match="scheme not allowed"):
        _validate_url(url)


def test_validate_url_rejects_missing_hostname() -> None:
    """URLs without a hostname are rejected."""
    with pytest.raises(_UrlValidationError, match="missing a hostname"):
        _validate_url("http:///path")


@pytest.mark.parametrize(
    "blocked_ip",
    [
        # IPv4 loopback / RFC1918 / link-local (IMDS) / unspecified
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.5",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        # IPv6 loopback / link-local / unique-local
        "::1",
        "fe80::1",
        "fc00::1",
        # IPv4-mapped IPv6 — loopback and IMDS encoded inside ::ffff:/96
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "::ffff:10.0.0.1",
        # 6to4 wrapping private v4 (2002::/16 prefix + private v4 in next 32 bits)
        "2002:a9fe:a9fe::1",
    ],
)
def test_validate_url_rejects_private_addresses(
    monkeypatch: pytest.MonkeyPatch, blocked_ip: str
) -> None:
    """Hostnames that resolve to private/internal IPs are rejected."""
    resolver, _ = _make_resolver(blocked_ip)
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    with pytest.raises(_UrlValidationError, match="blocked address"):
        _validate_url("http://evil.example.com/")


def test_validate_url_allows_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public IP passes validation and is returned."""
    resolver, _ = _make_resolver("93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    validated = _validate_url("https://example.com/")
    assert validated == ["93.184.216.34"]


def test_validate_url_rejects_when_any_record_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual-stack DNS where one record is private must still be rejected."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _make_multi_record_resolver("93.184.216.34", "127.0.0.1"),
    )

    with pytest.raises(_UrlValidationError, match="blocked address"):
        _validate_url("http://example.com/")


def test_validate_url_handles_dns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unresolvable hostnames raise `_UrlValidationError`."""

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        msg = "nodename nor servname provided"
        raise socket.gaierror(8, msg)

    monkeypatch.setattr(socket, "getaddrinfo", _raise)

    with pytest.raises(_UrlValidationError, match="Could not resolve hostname"):
        _validate_url("https://does-not-exist.invalid/")


def test_validate_url_strips_ipv6_scope_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope-suffixed sockaddrs (`fe80::1%eth0`) parse cleanly and are blocked."""

    def _impl(
        _host: str,
        port: int | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        sockaddr = ("fe80::1%eth0", port or 0, 0, 2)
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", _impl)

    with pytest.raises(_UrlValidationError, match="blocked address"):
        _validate_url("http://router.example/")


def test_fetch_url_blocks_imds(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMDS endpoint via hostname is blocked before any HTTP request."""
    resolver, _ = _make_resolver("169.254.169.254")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    result = fetch_url("http://metadata.example/latest/meta-data/")

    assert "error" in result
    assert "blocked address" in result["error"]
    assert result["url"] == "http://metadata.example/latest/meta-data/"
    assert result["category"] == "validation"


def test_fetch_url_blocks_file_scheme() -> None:
    """`file://` URLs are rejected without a network call."""
    result = fetch_url("file:///etc/passwd")

    assert "error" in result
    assert "scheme not allowed" in result["error"]
    assert result["category"] == "validation"


def test_fetch_url_blocks_ipv4_mapped_ipv6_imds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv4-mapped IPv6 wrapping IMDS (`::ffff:169.254.169.254`) is blocked."""
    resolver, _ = _make_resolver("::ffff:169.254.169.254")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    result = fetch_url("http://metadata.example/latest/meta-data/")

    assert "error" in result
    assert "blocked address" in result["error"]
    assert result["category"] == "validation"


def test_fetch_url_blocks_numeric_ip_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decimal-encoded loopback (`2130706433` = `127.0.0.1`) is blocked.

    `getaddrinfo` would normally resolve the numeric form to `127.0.0.1`;
    we simulate that here so the test does not depend on platform DNS.
    """
    resolver, _ = _make_resolver("127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    result = fetch_url("http://2130706433/")

    assert "error" in result
    assert "blocked address" in result["error"]
    assert result["category"] == "validation"


@responses.activate
def test_fetch_url_rejects_redirect_to_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public URL that redirects to a private host is blocked on the hop."""
    resolver, calls = _make_resolver("93.184.216.34", "169.254.169.254")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    responses.add(
        responses.GET,
        "http://example.com/redir",
        status=302,
        headers={"Location": "http://internal.example/secrets"},
    )

    result = fetch_url("http://example.com/redir")

    assert "error" in result
    assert "blocked address" in result["error"]
    assert result["category"] == "validation"
    # Both hops were validated — initial public host AND the redirect target.
    assert len(calls) == 2  # two hops: initial + one redirect


@responses.activate
def test_fetch_url_rejects_redirect_scheme_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect to a non-http(s) scheme (e.g., `file://`) is rejected."""
    resolver, _ = _make_resolver("93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    responses.add(
        responses.GET,
        "http://example.com/redir",
        status=302,
        headers={"Location": "file:///etc/passwd"},
    )

    result = fetch_url("http://example.com/redir")

    assert "error" in result
    assert "scheme not allowed" in result["error"]
    assert result["category"] == "validation"


@responses.activate
def test_fetch_url_follows_relative_redirect_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative `Location` resolves against the current URL and succeeds."""
    resolver, calls = _make_resolver("93.184.216.34", "93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    responses.add(
        responses.GET,
        "http://example.com/redir",
        status=302,
        headers={"Location": "/landing"},
    )
    responses.add(
        responses.GET,
        "http://example.com/landing",
        body="<html><body><h1>Landed</h1></body></html>",
        status=200,
    )

    result = fetch_url("http://example.com/redir")

    assert result["status_code"] == 200
    assert "Landed" in result["markdown_content"]
    assert result["url"].endswith("/landing")
    assert len(calls) == 2  # initial + one redirect hop


@responses.activate
def test_fetch_url_rejects_redirect_without_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3xx without Location surfaces a structured error, not a redirect-cap message."""
    resolver, _ = _make_resolver("93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    responses.add(
        responses.GET,
        "http://example.com/broken",
        status=302,
    )

    result = fetch_url("http://example.com/broken")

    assert "error" in result
    assert "missing a Location" in result["error"]
    assert result["category"] == "validation"


@responses.activate
def test_fetch_url_redirect_cap_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceeding the redirect cap raises a structured `redirects` error."""
    # `_MAX_FETCH_REDIRECTS + 1` total hops are allowed (initial + 5 redirects).
    # Make every hop a redirect so the loop falls through.
    public_ip = "93.184.216.34"
    resolver, _ = _make_resolver(*([public_ip] * 10))
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    for i in range(8):
        responses.add(
            responses.GET,
            f"http://example.com/hop/{i}",
            status=302,
            headers={"Location": f"/hop/{i + 1}"},
        )

    result = fetch_url("http://example.com/hop/0")

    assert "error" in result
    assert "Exceeded" in result["error"]
    assert result["category"] == "redirects"
