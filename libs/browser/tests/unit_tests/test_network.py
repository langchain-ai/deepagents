import asyncio
import ipaddress

import pytest

from deepagents_browser.network import NetworkPolicy, NetworkPolicyError


async def _resolve_public(host: str, port: int):
    assert host == "example.com"
    assert port in {80, 443}
    return ["93.184.216.34"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https://user:password@example.com/",
        "https:///missing-host",
        "https://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://224.0.0.1/",
        "http://192.0.2.1/",
        "http://0.0.0.0/",
        "http://100.64.0.1/",
    ],
)
async def test_policy_rejects_unsafe_urls(url):
    with pytest.raises(NetworkPolicyError):
        await NetworkPolicy(resolver=_resolve_public).validate_url(url)


async def test_policy_accepts_public_http_and_https():
    result = await NetworkPolicy(resolver=_resolve_public).validate_url("https://example.com/path")
    assert result.host == "example.com"
    assert result.addresses == (ipaddress.ip_address("93.184.216.34"),)


async def test_policy_rejects_mixed_dns_answers():
    async def resolve_mixed(host: str, port: int):
        return ["93.184.216.34", "127.0.0.1"]

    with pytest.raises(NetworkPolicyError, match="blocked"):
        await NetworkPolicy(resolver=resolve_mixed).validate_url("https://example.com")


async def test_policy_rejects_empty_invalid_and_timed_out_dns():
    async def empty(host: str, port: int):
        return []

    async def invalid(host: str, port: int):
        return ["not-an-address"]

    async def slow(host: str, port: int):
        await asyncio.sleep(1)
        return ["93.184.216.34"]

    with pytest.raises(NetworkPolicyError, match="did not resolve"):
        await NetworkPolicy(resolver=empty).validate_url("https://example.com")
    with pytest.raises(NetworkPolicyError, match="invalid IP"):
        await NetworkPolicy(resolver=invalid).validate_url("https://example.com")
    with pytest.raises(NetworkPolicyError, match="resolution failed"):
        await NetworkPolicy(resolver=slow, dns_timeout_seconds=0.001).validate_url(
            "https://example.com"
        )


class _Request:
    def __init__(self, url: str) -> None:
        self.url = url


class _Route:
    def __init__(self, url: str) -> None:
        self.request = _Request(url)
        self.continued = 0
        self.aborted = []

    async def continue_(self) -> None:
        self.continued += 1

    async def abort(self, error_code: str = "failed") -> None:
        self.aborted.append(error_code)


async def test_route_interception_validates_redirect_targets():
    policy = NetworkPolicy(resolver=_resolve_public)
    initial = _Route("https://example.com/start")
    redirect = _Route("http://127.0.0.1/secret")
    await policy.handle_route(initial)
    await policy.handle_route(redirect)
    assert initial.continued == 1
    assert redirect.aborted == ["blockedbyclient"]
