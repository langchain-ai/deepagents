from __future__ import annotations

import pytest

from deepagents_talon.mcp_auth import (
    extract_loopback_oauth_callback_url,
    extract_oauth_callback_url,
)


def test_configured_callback_requires_exact_redirect_endpoint() -> None:
    redirect_uri = "http://127.0.0.1:6359/callback"
    callback = f"{redirect_uri}?code=secret&state=opaque"
    assert extract_oauth_callback_url(callback) is None
    assert extract_oauth_callback_url(f"<{callback}>", redirect_uri=redirect_uri) == callback
    for altered in (
        callback.replace("6359", "6360"),
        callback.replace("127.0.0.1", "localhost"),
        callback.replace("/callback", "/wrong"),
    ):
        assert extract_loopback_oauth_callback_url(altered) == altered
        assert extract_oauth_callback_url(altered, redirect_uri=redirect_uri) is None


@pytest.mark.parametrize(
    "callback",
    [
        "http://attacker.example:6359/callback?code=secret&state=opaque",
        "http://localhost.attacker.example/callback?code=secret&state=opaque",
        "http://user@127.0.0.1:6359/callback?code=secret&state=opaque",
        "http://[::1]:65536/callback?code=secret&state=opaque",
        "http://127.0.0.1:0/callback?code=secret&state=opaque",
        "http://127.0.0.1:6359/callback?code=secret&state=opaque#fragment",
        "http://127.0.0.1:6359/callback?code=secret&state=opaque\nextra",
        "http://127.0.0.1:6359/callback?code=secret",
    ],
)
def test_callback_shape_rejects_unsafe_or_incomplete_urls(callback: str) -> None:
    assert extract_loopback_oauth_callback_url(callback) is None


def test_callback_shape_recognizes_ipv6_loopback_denial() -> None:
    callback = "http://[::1]:6359/callback?error=access_denied&state=opaque"
    assert extract_loopback_oauth_callback_url(callback) == callback
