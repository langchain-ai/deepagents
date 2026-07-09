"""Tests for MCP OAuth helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import httpx
import pytest
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthToken

from deepagents_code.mcp_auth import (
    FileTokenStorage,
    MCPReauthRequiredError,
    find_oauth_challenge,
    find_reauth_required,
    format_login_failure,
    resolve_headers,
)

_RESOURCE_METADATA_URL = "https://mcp.example.com/.well-known/oauth-protected-resource"
_BEARER_CHALLENGE = f'Bearer resource_metadata="{_RESOURCE_METADATA_URL}"'
"""A minimal RFC 9728 Bearer challenge pointing at the resource metadata."""


def _http_status_error(
    status_code: int,
    *,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
) -> httpx.HTTPStatusError:
    """Build an `httpx.HTTPStatusError` with a canned response."""
    request = httpx.Request("GET", "https://mcp.example.com/")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` and `DEFAULT_STATE_DIR` into a temp directory.

    `Path.home` is patched for code that resolves it at call time;
    `DEFAULT_STATE_DIR` is patched for code (like `mcp_auth._tokens_dir`)
    that pulls from the import-time-frozen constant in `model_config`.
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


class TestResolveHeaders:
    """Tests for static MCP header interpolation."""

    def test_resolves_single_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single `${VAR}` placeholder resolves to its env value."""
        monkeypatch.setenv("FOO", "bar")
        assert resolve_headers({"Authorization": "Bearer ${FOO}"}) == {
            "Authorization": "Bearer bar"
        }

    def test_resolves_multiple_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple placeholders resolve left-to-right."""
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")
        assert resolve_headers({"X-Combo": "${A}-${B}"}) == {"X-Combo": "alpha-beta"}

    def test_non_string_value_raises(self) -> None:
        """Header values must be strings."""
        with pytest.raises(TypeError, match="must be a string"):
            resolve_headers({"X-Bad": 123}, server_name="srv")  # ty: ignore

    def test_unset_env_var_raises(self) -> None:
        """Unset placeholders fail with a helpful message."""
        with pytest.raises(RuntimeError, match="unset env var"):
            resolve_headers({"Authorization": "Bearer ${MISSING}"})

    def test_plain_text_value_is_unchanged(self) -> None:
        """Strings without placeholders pass through unchanged."""
        assert resolve_headers({"X-Plain": "hello"}) == {"X-Plain": "hello"}


def _make_tokens(access_token: str = "at"):
    return OAuthToken(
        access_token=access_token,
        token_type="Bearer",
        refresh_token="rt",
        expires_in=3600,
    )


def _make_client_info():
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def _make_oauth_metadata(token_endpoint: str = "https://auth.example/token"):
    from mcp.shared.auth import AnyHttpUrl, OAuthMetadata

    return OAuthMetadata(
        issuer=AnyHttpUrl("https://auth.example"),
        authorization_endpoint=AnyHttpUrl("https://auth.example/authorize"),
        token_endpoint=AnyHttpUrl(token_endpoint),
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
    )


def _make_client_info_with_secret(
    auth_method: Literal["client_secret_basic", "client_secret_post", "none"],
):
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    # Public clients (`none`) carry no secret; confidential clients do.
    client_secret = None if auth_method == "none" else "client-secret"
    return OAuthClientInformationFull(
        client_id="client-id",
        client_secret=client_secret,
        token_endpoint_auth_method=auth_method,
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def _make_client_info_with_loopback(port: int):
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=[AnyUrl(f"http://localhost:{port}/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorage:
    """Tests for the file-backed OAuth token store."""

    async def test_missing_file_returns_none(self) -> None:
        """Missing token files return `None` for both tokens and client info."""
        storage = FileTokenStorage("notion")
        assert await storage.get_tokens() is None
        assert await storage.get_client_info() is None

    async def test_round_trip_tokens_and_client_info(self) -> None:
        """Tokens and client info round-trip through disk storage."""
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        got_ci = await storage.get_client_info()
        got_tok = await storage.get_tokens()

        assert got_ci is not None
        assert got_tok is not None
        assert got_ci.client_id == "client-id"
        assert got_tok.access_token == "at"

    async def test_sets_file_permissions_on_posix(self, fake_home: Path) -> None:
        """Token files are created with private user-only permissions."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())

        token_path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        assert token_path.exists()
        if hasattr(token_path, "stat"):
            assert token_path.stat().st_mode & 0o777 == 0o600

    async def test_corrupt_file_raises(self, fake_home: Path) -> None:
        """Corrupt files fail with a remediation hint."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        storage = FileTokenStorage("notion")

        with pytest.raises(RuntimeError, match="Delete the file"):
            await storage.get_tokens()

    async def test_server_names_are_isolated(self) -> None:
        """Different servers use different token files."""
        alpha = FileTokenStorage("alpha")
        beta = FileTokenStorage("beta")
        await alpha.set_tokens(_make_tokens())
        await beta.set_tokens(_make_tokens())

        got_alpha = await alpha.get_tokens()
        got_beta = await beta.get_tokens()

        assert got_alpha is not None
        assert got_beta is not None

    async def test_same_server_name_with_different_urls_isolated(self) -> None:
        """Same-named servers on different endpoints use separate files."""
        alpha = FileTokenStorage("github", server_url="https://alpha.example/mcp")
        beta = FileTokenStorage("github", server_url="https://beta.example/mcp")
        await alpha.set_tokens(_make_tokens("alpha-token"))
        await beta.set_tokens(_make_tokens("beta-token"))

        got_alpha = await alpha.get_tokens()
        got_beta = await beta.get_tokens()

        assert alpha.path != beta.path
        assert got_alpha is not None
        assert got_alpha.access_token == "alpha-token"
        assert got_beta is not None
        assert got_beta.access_token == "beta-token"

    @pytest.mark.parametrize(
        "name",
        [
            "../escape",
            "../../etc/cron.d/evil",
            "name/with/slashes",
            "name\\with\\backslashes",
            "name with spaces",
            "name\x00null",
            "..",
            ".",
            "",
        ],
    )
    def test_unsafe_server_name_rejected(self, name: str) -> None:
        """Names that could traverse out of the tokens dir are rejected.

        Guards against path traversal via attacker-controlled `mcpServers`
        keys (Corridor finding d5d5b0c1).
        """
        with pytest.raises(ValueError, match="Invalid MCP server name"):
            FileTokenStorage(name)

    async def test_set_tokens_records_absolute_expiry(self) -> None:
        """`set_tokens` writes an `expires_at` sidecar derived from `expires_in`."""
        storage = FileTokenStorage("notion")
        before = time.time()
        await storage.set_tokens(_make_tokens())
        after = time.time()

        got = await storage.get_expires_at()
        assert got is not None
        # 3600 from `_make_tokens`; widen the wall-clock window to absorb
        # GC pauses on busy CI runners.
        assert before + 3600 <= got <= after + 3600 + 1.0

    async def test_set_tokens_and_client_info_records_expiry(self) -> None:
        """The combined writer also persists `expires_at`."""
        storage = FileTokenStorage("notion")
        before = time.time()
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())
        after = time.time()

        got = await storage.get_expires_at()
        assert got is not None
        assert before + 3600 <= got <= after + 3600 + 1.0

    async def test_get_expires_at_returns_none_for_legacy_file(
        self, fake_home: Path
    ) -> None:
        """Token files written before this field existed return `None`."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "tokens": {"access_token": "x"}}))
        storage = FileTokenStorage("notion")

        assert await storage.get_expires_at() is None

    async def test_get_expires_at_rejects_non_numeric(self, fake_home: Path) -> None:
        """A garbage sidecar value falls back to `None` rather than raising."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "tokens": {"access_token": "x"},
                    "expires_at": "soon",
                }
            )
        )
        storage = FileTokenStorage("notion")

        assert await storage.get_expires_at() is None

    async def test_set_tokens_clears_stale_expiry_when_expires_in_absent(self) -> None:
        """Writing a token without `expires_in` removes any prior `expires_at`."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())
        assert await storage.get_expires_at() is not None

        # Some providers omit `expires_in` on refresh; the sidecar must not
        # linger from the prior write or the next cold start will use a
        # bogus expiry.
        await storage.set_tokens(
            OAuthToken(access_token="x2", token_type="Bearer", refresh_token="rt2")
        )
        assert await storage.get_expires_at() is None

    async def test_round_trip_oauth_metadata(self) -> None:
        """Public OAuth metadata round-trips beside token state."""
        storage = FileTokenStorage("notion")
        metadata = _make_oauth_metadata()

        assert await storage.get_oauth_metadata() is None
        await storage.set_oauth_metadata(metadata)

        stored = await storage.get_oauth_metadata()
        assert stored is not None
        assert str(stored.token_endpoint) == "https://auth.example/token"


@pytest.mark.usefixtures("fake_home")
class TestExpiryAwareOAuthClientProvider:
    """Tests for cold-start expiry restoration on the OAuth client provider."""

    async def test_initialize_restores_expiry_minus_safety_margin(self) -> None:
        """A live `expires_at` is loaded into `context.token_expiry_time`."""
        from deepagents_code.mcp_auth import (
            _REFRESH_SAFETY_MARGIN_SECONDS,
            build_oauth_provider,
        )

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())
        expected_expires_at = await storage.get_expires_at()
        assert expected_expires_at is not None

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        await provider._initialize()

        assert provider.context.token_expiry_time == (
            expected_expires_at - _REFRESH_SAFETY_MARGIN_SECONDS
        )
        assert provider.context.is_token_valid() is True
        assert provider.context.can_refresh_token() is True

    async def test_initialize_treats_expired_token_as_invalid(self) -> None:
        """A past `expires_at` makes the loaded token report as invalid."""
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())
        path = storage.path
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 60  # already expired
        path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        await provider._initialize()

        assert provider.context.is_token_valid() is False
        assert provider.context.can_refresh_token() is True

    async def test_initialize_legacy_file_forces_refresh_when_refresh_token_present(
        self,
    ) -> None:
        """No sidecar + refresh token => assume expired so refresh path fires."""
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        # Write a legacy-format file (no `expires_at`).
        path = storage.path
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "client_info": json.loads(
                        _make_client_info().model_dump_json(exclude_none=True)
                    ),
                    "tokens": json.loads(
                        _make_tokens().model_dump_json(exclude_none=True)
                    ),
                }
            )
        )

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        await provider._initialize()

        # The exact sentinel float is documented in the source; assert the
        # observable behavior (token invalid, refresh path reachable) rather
        # than pinning the magic value.
        assert provider.context.is_token_valid() is False
        assert provider.context.can_refresh_token() is True

    async def test_initialize_legacy_file_without_refresh_token_leaves_expiry_unset(
        self,
    ) -> None:
        """Legacy file without `refresh_token` cannot pre-empt expiry."""
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        path = storage.path
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "client_info": json.loads(
                        _make_client_info().model_dump_json(exclude_none=True)
                    ),
                    "tokens": {"access_token": "stale", "token_type": "Bearer"},
                }
            )
        )

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        await provider._initialize()

        # No refresh_token means there's nothing to refresh with, so the
        # provider must leave token_expiry_time at its default (None). The
        # stale Bearer will go out, hit 401, and fall into the SDK's full
        # re-auth path — there's no shortcut available.
        assert provider.context.token_expiry_time is None
        assert provider.context.can_refresh_token() is False

    async def test_initialize_with_storage_lacking_get_expires_at(self) -> None:
        """Custom `TokenStorage` impls without `get_expires_at` still initialize."""
        from deepagents_code.mcp_auth import build_oauth_provider

        class _MinimalStorage(TokenStorage):
            """`TokenStorage` that omits the optional `get_expires_at` method."""

            def __init__(self) -> None:
                self._tokens: OAuthToken | None = _make_tokens()
                self._client_info = _make_client_info()

            async def get_tokens(self) -> OAuthToken | None:
                return self._tokens

            async def set_tokens(self, tokens: OAuthToken) -> None:
                self._tokens = tokens

            async def get_client_info(self):
                return self._client_info

            async def set_client_info(self, client_info) -> None:
                self._client_info = client_info

        provider = build_oauth_provider(
            server_name="custom",
            server_url="https://mcp.example.com/mcp",
            storage=_MinimalStorage(),
        )
        await provider._initialize()

        # Without the optional sidecar accessor, the provider falls back to
        # the upstream SDK's behavior: no expiry known, token treated as
        # valid until a 401 forces re-auth.
        assert provider.context.token_expiry_time is None
        assert provider.context.current_tokens is not None

    async def test_delegated_flow_forwards_responses_into_sdk(
        self,
        fake_home: Path,
    ) -> None:
        """Responses sent into the outer flow reach the delegated SDK flow.

        Regression test: the override used to delegate with `async for`, which
        advances the inner SDK generator via `__anext__()` (`asend(None)`) and
        discards the HTTP responses httpx feeds back through `asend(response)`.
        The SDK's `response = yield request` then saw `None` and raised
        `AttributeError: 'NoneType' object has no attribute 'status_code'`,
        surfacing as the `ExceptionGroup` users hit on MCP OAuth login. With a
        valid stored token the pre-emptive discovery branch is skipped, so the
        first response forwarded is the one whose `status_code` the SDK reads.
        """
        del fake_home
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )

        # The valid token is attached and the request is yielded unchanged.
        first_request = await anext(flow)
        assert first_request.headers["Authorization"] == "Bearer at"

        # Feeding a 401 back must reach the SDK's `response.status_code` check
        # and trigger metadata discovery — not raise AttributeError.
        discovery_request = await flow.asend(httpx.Response(401, request=first_request))
        assert "/.well-known/oauth-protected-resource" in str(discovery_request.url)
        await flow.aclose()

    @pytest.mark.parametrize(
        ("interactive", "expected"),
        [(False, True), (True, False)],
    )
    async def test_delegated_flow_toggles_reauth_log_suppression(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        interactive: bool,
        expected: bool,
    ) -> None:
        """The contextvar is set during delegation only for non-interactive runs.

        Guards the wiring between `build_oauth_provider(interactive=...)` and the
        filter: the SDK flow logs synchronously inside the delegated generator,
        so the suppression flag must be visible there. A fake SDK flow records
        what the contextvar reads at that point.
        """
        del fake_home
        import httpx
        from mcp.client.auth import OAuthClientProvider

        from deepagents_code.mcp_auth import (
            _SUPPRESS_EXPECTED_REAUTH_LOGS,
            build_oauth_provider,
        )

        observed: dict[str, bool] = {}

        async def fake_flow(
            self: OAuthClientProvider,
            request: httpx.Request,
        ):
            del self
            observed["suppressed"] = _SUPPRESS_EXPECTED_REAUTH_LOGS.get()
            _ = yield request

        monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_flow)

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=interactive,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )
        await anext(flow)
        await flow.aclose()

        assert observed["suppressed"] is expected
        # The flag never leaks past the flow.
        assert _SUPPRESS_EXPECTED_REAUTH_LOGS.get() is False

    async def test_delegated_flow_forwards_responses_on_every_iteration(
        self,
        fake_home: Path,
    ) -> None:
        """The pump loop forwards responses on every round-trip, not just one.

        Guards against a regression that primes the inner generator correctly
        but then reverts to discarding subsequent sends (e.g. back toward
        `async for`): the SDK's protected-resource-metadata discovery walks
        several URLs, sending a response into the delegated generator each
        time. Each forwarded response must advance discovery to the next URL.
        """
        del fake_home
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )

        first_request = await anext(flow)
        # First forwarded response (401) advances to the path-scoped PRM URL.
        prm_path_request = await flow.asend(httpx.Response(401, request=first_request))
        assert str(prm_path_request.url).endswith(
            "/.well-known/oauth-protected-resource/mcp"
        )
        # Second forwarded response (404) must also reach the SDK and advance
        # discovery to the root PRM URL — proving the loop didn't stop after
        # the first send.
        prm_root_request = await flow.asend(
            httpx.Response(404, request=prm_path_request)
        )
        assert str(prm_root_request.url).endswith(
            "/.well-known/oauth-protected-resource"
        )
        await flow.aclose()


@pytest.mark.usefixtures("fake_home")
class TestRefreshTokenSerialization:
    """Cross-process-safe refresh serialization to avoid refresh-token reuse.

    The LangSmith OAuth server rotates refresh tokens and revokes the entire
    identity+client token family when an already-rotated token is replayed, so
    the provider must reload the on-disk token under a lock before refreshing.
    """

    async def test_refresh_lock_path_is_sibling_of_token_file(self) -> None:
        """The lock lives beside the token file and never replaces it."""
        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        assert storage.refresh_lock_path == storage.path.with_name(
            f"{storage.path.name}.lock"
        )
        assert storage.refresh_lock_path.parent == storage.path.parent
        assert storage.refresh_lock_path != storage.path

    async def test_skips_refresh_when_peer_already_rotated_on_disk(self) -> None:
        """A peer's fresh token is reloaded and used instead of refreshing.

        Guards the reuse fix: if this provider still has a stale token in
        memory but disk already holds a peer's rotated token, it must attach
        the reloaded token rather than replay its own (now-revoked) refresh
        token.
        """
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_oauth_metadata(_make_oauth_metadata())
        await storage.set_tokens(_make_tokens(access_token="stale"))
        # Backdate the sidecar so the loaded token reports as expired.
        path = storage.path
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 60
        path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )
        # Initialize with the stale token held in memory.
        await provider._initialize()
        assert provider.context.is_token_valid() is False

        # A peer rotates the token on disk: fresh access token, future expiry.
        await storage.set_tokens(
            OAuthToken(
                access_token="peer-rotated",
                token_type="Bearer",
                refresh_token="rt-new",
                expires_in=3600,
            )
        )

        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )
        first_request = await anext(flow)
        # No refresh round-trip: the reloaded token is attached directly, and
        # the first yielded request is the actual server call.
        assert first_request.headers["Authorization"] == "Bearer peer-rotated"
        assert str(first_request.url) == "https://mcp.notion.com/mcp"
        await flow.aclose()

    async def test_performs_locked_refresh_and_persists_rotation(self) -> None:
        """A still-stale token triggers exactly one refresh, then persists it."""
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_oauth_metadata(
            _make_oauth_metadata("https://auth.example/token")
        )
        await storage.set_tokens(_make_tokens(access_token="stale"))
        path = storage.path
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 60
        path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )

        refresh_request = await anext(flow)
        assert refresh_request.method == "POST"
        assert str(refresh_request.url) == "https://auth.example/token"
        body = refresh_request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "refresh_token=rt" in body

        token_response = httpx.Response(
            200,
            json={
                "access_token": "at-rotated",
                "token_type": "Bearer",
                "refresh_token": "rt-rotated",
                "expires_in": 3600,
            },
            request=refresh_request,
        )
        actual_request = await flow.asend(token_response)
        assert actual_request.headers["Authorization"] == "Bearer at-rotated"
        assert str(actual_request.url) == "https://mcp.notion.com/mcp"
        await flow.aclose()

        # The rotated pair is persisted so the next process reads it too.
        persisted = await storage.get_tokens()
        assert persisted is not None
        assert persisted.access_token == "at-rotated"
        assert persisted.refresh_token == "rt-rotated"

    async def test_locked_refresh_rejection_delegates_to_sdk_reauth_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rejected locked refresh does not bypass the SDK re-auth fallback."""
        from mcp.client.auth import OAuthClientProvider

        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_oauth_metadata(
            _make_oauth_metadata("https://auth.example/token")
        )
        await storage.set_tokens(_make_tokens(access_token="stale"))
        path = storage.path
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 60
        path.write_text(json.dumps(data))

        delegated: dict[str, bool] = {}

        async def fake_sdk_flow(
            self: OAuthClientProvider,
            request: httpx.Request,
        ):
            delegated["entered"] = True
            assert self.context.current_tokens is None
            _ = yield request

        monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_sdk_flow)

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )

        async def raise_refresh_failure(response: httpx.Response) -> bool:
            msg = "refresh token rejected"
            raise httpx.HTTPStatusError(
                msg,
                request=response.request,
                response=response,
            )

        monkeypatch.setattr(
            provider,
            "_handle_refresh_response",
            raise_refresh_failure,
        )

        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )

        refresh_request = await anext(flow)
        assert str(refresh_request.url) == "https://auth.example/token"

        actual_request = await flow.asend(httpx.Response(401, request=refresh_request))

        assert delegated == {"entered": True}
        assert str(actual_request.url) == "https://mcp.notion.com/mcp"
        await flow.aclose()

    async def _build_stale_refreshable_provider(
        self,
    ) -> tuple[Any, FileTokenStorage]:
        """Build a provider whose on-disk token is expired but refreshable.

        Shared setup for the lock-behavior tests: client info and OAuth
        metadata are persisted and the sidecar expiry is backdated so the
        refresh branch fires against `https://auth.example/token`.
        """
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_oauth_metadata(
            _make_oauth_metadata("https://auth.example/token")
        )
        await storage.set_tokens(_make_tokens(access_token="stale"))
        path = storage.path
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 60
        path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
            interactive=False,
        )
        return provider, storage

    async def test_lock_timeout_skips_refresh_token_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A peer-held lock timeout avoids replaying the refresh token.

        The wait is shrunk so the contended acquire times out promptly; the
        provider must not fire either its locked refresh or the delegated SDK
        refresh while a peer may still be using the same rotating refresh token.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        provider, storage = await self._build_stale_refreshable_provider()
        monkeypatch.setattr(mcp_auth, "_REFRESH_LOCK_TIMEOUT_SECONDS", 0.1)

        # A peer holds the refresh lock for the duration of this flow, forcing
        # the provider's acquire to time out.
        peer_lock = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer_lock.acquire()
        try:
            caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
            flow = provider.async_auth_flow(
                httpx.Request("POST", "https://mcp.notion.com/mcp")
            )
            actual_request = await anext(flow)
            assert str(actual_request.url) == "https://mcp.notion.com/mcp"
            assert "Authorization" not in actual_request.headers
            await flow.aclose()
        finally:
            peer_lock.release()

        assert any(
            "skipping refresh to avoid refresh-token reuse" in record.getMessage()
            for record in caplog.records
        )

    async def test_refresh_waits_for_peer_holding_lock_then_proceeds(self) -> None:
        """The flow blocks on the refresh lock until the peer releases it.

        Proves the serialization the PR exists for: while a peer holds the file
        lock the provider cannot reach the refresh, and it proceeds only once
        the lock is free. Guards against a regression that drops the file lock
        (e.g. reverting to the per-provider `context.lock`).
        """
        from filelock import FileLock

        provider, storage = await self._build_stale_refreshable_provider()

        async def drive_to_refresh() -> httpx.Request:
            # Keep the whole generator lifecycle on one task: its inner
            # `anyio` lock is task-affine, so driving `anext` here and
            # `aclose` on another task would raise on release.
            flow = provider.async_auth_flow(
                httpx.Request("POST", "https://mcp.notion.com/mcp")
            )
            try:
                return await anext(flow)
            finally:
                await flow.aclose()

        peer_lock = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer_lock.acquire()
        task = asyncio.ensure_future(drive_to_refresh())
        try:
            # While the peer holds the lock, the flow can't reach the refresh.
            await asyncio.sleep(0.2)
            assert not task.done()

            peer_lock.release()

            # Once the lock frees, the provider acquires it and yields the refresh.
            refresh_request = await asyncio.wait_for(task, timeout=5)
            assert str(refresh_request.url) == "https://auth.example/token"
        finally:
            if peer_lock.is_locked:
                peer_lock.release()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_refresh_lock_released_after_successful_refresh(self) -> None:
        """A completed locked refresh frees the lock for the next caller.

        Defends against a self-deadlock regression: if the guard failed to
        release, the next process to refresh this server would hang.
        """
        from filelock import FileLock, Timeout

        provider, storage = await self._build_stale_refreshable_provider()

        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.notion.com/mcp")
        )
        refresh_request = await anext(flow)
        token_response = httpx.Response(
            200,
            json={
                "access_token": "at-rotated",
                "token_type": "Bearer",
                "refresh_token": "rt-rotated",
                "expires_in": 3600,
            },
            request=refresh_request,
        )
        await flow.asend(token_response)
        await flow.aclose()

        # The guard must have released; a fresh holder takes the lock at once.
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        try:
            await asyncio.to_thread(probe.acquire, timeout=0)
        except Timeout:
            pytest.fail("refresh lock was not released after a successful refresh")
        else:
            probe.release()


@pytest.mark.usefixtures("fake_home")
class TestBasicAuthClientIdStripping:
    """Tests for dropping the duplicate body `client_id` under HTTP Basic auth."""

    def _build_provider(
        self,
        auth_method: Literal["client_secret_basic", "client_secret_post", "none"],
    ):
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("pylon")
        provider = build_oauth_provider(
            server_name="pylon",
            server_url="https://mcp.usepylon.com/mcp",
            storage=storage,
            interactive=False,
        )
        provider.context.client_info = _make_client_info_with_secret(auth_method)
        return provider

    def test_basic_auth_drops_body_client_id(self) -> None:
        """`client_secret_basic` carries credentials in the header, not the body.

        This wrapper strips the redundant body `client_id`. The SDK itself
        already strips `client_secret` for Basic auth, which the final
        assertion pins (see `test_sdk_still_injects_client_id_under_basic_auth`
        for the contract the wrapper depends on).
        """
        provider = self._build_provider("client_secret_basic")

        data, headers = provider.context.prepare_token_auth(
            {
                "grant_type": "authorization_code",
                "client_id": "client-id",
                "client_secret": "client-secret",
            },
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert headers["Authorization"].startswith("Basic ")
        assert "client_id" not in data
        assert "client_secret" not in data  # stripped by the SDK, not this wrapper

    def test_post_auth_retains_body_client_id(self) -> None:
        """`client_secret_post` keeps both fields in the body and adds no header."""
        provider = self._build_provider("client_secret_post")

        data, headers = provider.context.prepare_token_auth(
            {
                "grant_type": "authorization_code",
                "client_id": "client-id",
            },
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert "Authorization" not in headers
        assert data["client_id"] == "client-id"
        assert data["client_secret"] == "client-secret"

    def test_none_auth_retains_body_client_id(self) -> None:
        """`none` (public client) sends no header, so the body keeps `client_id`."""
        provider = self._build_provider("none")

        data, headers = provider.context.prepare_token_auth(
            {
                "grant_type": "authorization_code",
                "client_id": "client-id",
            },
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert "Authorization" not in headers
        assert data["client_id"] == "client-id"

    def test_sdk_still_injects_client_id_under_basic_auth(self) -> None:
        """Pin the SDK contract this wrapper depends on.

        Unwrapped, the SDK leaves `client_id` in the token-request body under
        Basic auth (it strips only `client_secret`). If upstream ever strips
        `client_id` too, this wrapper becomes a silent no-op; this test fails
        loudly instead, flagging the workaround as obsolete.
        """
        from mcp.client.auth.oauth2 import OAuthContext

        provider = self._build_provider("client_secret_basic")

        # Call the SDK's method via the class to bypass the instance-level wrap
        # installed in `__init__` and observe the SDK's own behavior.
        data, headers = OAuthContext.prepare_token_auth(
            provider.context,
            {
                "grant_type": "authorization_code",
                "client_id": "client-id",
                "client_secret": "client-secret",
            },
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert headers["Authorization"].startswith("Basic ")
        assert data["client_id"] == "client-id"
        assert "client_secret" not in data


class TestExpectedReauthLogFilter:
    """Tests for suppressing noisy SDK OAuth logs during non-interactive reauth."""

    def test_suppresses_expected_sdk_oauth_logs(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Expected non-interactive reauth logs are replaced by our login hint."""
        from deepagents_code.mcp_auth import _SUPPRESS_EXPECTED_REAUTH_LOGS

        sdk_logger = logging.getLogger("mcp.client.auth.oauth2")
        caplog.set_level(logging.WARNING, logger="mcp.client.auth.oauth2")
        server = "notion"
        reauth = MCPReauthRequiredError(server)
        msg = "boom"
        unexpected = RuntimeError(msg)
        token = _SUPPRESS_EXPECTED_REAUTH_LOGS.set(True)
        try:
            sdk_logger.warning("Token refresh failed: 400")
            sdk_logger.error(
                "OAuth flow error",
                exc_info=(type(reauth), reauth, reauth.__traceback__),
            )
            sdk_logger.error(
                "OAuth flow error",
                exc_info=(type(unexpected), unexpected, unexpected.__traceback__),
            )
        finally:
            _SUPPRESS_EXPECTED_REAUTH_LOGS.reset(token)

        messages = [record.getMessage() for record in caplog.records]
        assert messages == ["OAuth flow error"]
        exc_info = caplog.records[0].exc_info
        assert exc_info is not None
        assert isinstance(exc_info[1], RuntimeError)

    def test_transient_refresh_failure_is_not_suppressed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Transient refresh statuses (5xx/429) stay visible, not relabeled reauth.

        The SDK logs `Token refresh failed: <status>` for any non-200. A `503`
        means the provider is down and the refresh token is still valid, so the
        operator must see it rather than be steered toward a pointless re-login.
        """
        from deepagents_code.mcp_auth import _SUPPRESS_EXPECTED_REAUTH_LOGS

        sdk_logger = logging.getLogger("mcp.client.auth.oauth2")
        caplog.set_level(logging.WARNING, logger="mcp.client.auth.oauth2")
        token = _SUPPRESS_EXPECTED_REAUTH_LOGS.set(True)
        try:
            sdk_logger.warning("Token refresh failed: 503")
            sdk_logger.warning("Token refresh failed: 429")
            sdk_logger.warning("Token refresh failed: 400")
        finally:
            _SUPPRESS_EXPECTED_REAUTH_LOGS.reset(token)

        messages = [record.getMessage() for record in caplog.records]
        assert messages == [
            "Token refresh failed: 503",
            "Token refresh failed: 429",
        ]

    def test_passes_through_when_not_suppressing(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With the contextvar unset, the process-wide filter is inert.

        The filter is installed on the SDK logger for every consumer of that
        logger, so its default-off behavior guards against globally swallowing
        real OAuth errors outside a non-interactive reauth window.
        """
        sdk_logger = logging.getLogger("mcp.client.auth.oauth2")
        caplog.set_level(logging.WARNING, logger="mcp.client.auth.oauth2")
        reauth = MCPReauthRequiredError("notion")

        # No `_SUPPRESS_EXPECTED_REAUTH_LOGS.set(...)`: contextvar at default.
        sdk_logger.warning("Token refresh failed: 400")
        sdk_logger.error(
            "OAuth flow error",
            exc_info=(type(reauth), reauth, reauth.__traceback__),
        )

        messages = [record.getMessage() for record in caplog.records]
        assert messages == ["Token refresh failed: 400", "OAuth flow error"]


class TestFindReauthRequired:
    """Tests for unwrapping nested re-auth errors."""

    def test_returns_direct_error(self) -> None:
        """Direct `MCPReauthRequiredError` instances are returned unchanged."""
        exc = MCPReauthRequiredError("srv")
        assert find_reauth_required(exc) is exc

    def test_finds_error_inside_exception_group(self) -> None:
        """Nested exception groups are searched recursively."""
        exc = ExceptionGroup(
            "outer", [RuntimeError("x"), MCPReauthRequiredError("srv")]
        )
        found = find_reauth_required(exc)
        assert isinstance(found, MCPReauthRequiredError)
        assert found.server_name == "srv"

    def test_finds_error_via_cause_chain(self) -> None:
        """`raise X from MCPReauthRequiredError(...)` is unwrapped."""
        reauth = MCPReauthRequiredError("srv")
        outer_msg = "outer"
        try:
            try:
                raise reauth
            except MCPReauthRequiredError as inner:
                raise RuntimeError(outer_msg) from inner
        except RuntimeError as exc:
            found = find_reauth_required(exc)
        assert found is reauth

    def test_finds_error_via_context(self) -> None:
        """Implicit `__context__` chains are searched."""
        reauth = MCPReauthRequiredError("srv")
        outer_msg = "outer"
        try:
            try:
                raise reauth
            except MCPReauthRequiredError:
                raise RuntimeError(outer_msg)  # noqa: B904
        except RuntimeError as exc:
            found = find_reauth_required(exc)
        assert found is reauth

    def test_returns_none_when_absent(self) -> None:
        """Pure exception trees without reauth errors yield `None`."""
        exc = ExceptionGroup("outer", [RuntimeError("x"), ValueError("y")])
        assert find_reauth_required(exc) is None

    def test_handles_cyclic_chain(self) -> None:
        """Self-referencing `__context__` cycles terminate without recursion."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        assert find_reauth_required(a) is None


class TestFindOauthChallenge:
    """Tests for detecting a 401 OAuth challenge in an exception tree."""

    def test_direct_401_with_challenge(self) -> None:
        """A 401 carrying an RFC 9728 Bearer challenge yields its URL."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_header_match_is_case_insensitive(self) -> None:
        """The scheme and parameter matching ignore casing."""
        exc = _http_status_error(
            401,
            headers={
                "www-authenticate": (
                    f'bearer resource_METADATA="{_RESOURCE_METADATA_URL}"'
                )
            },
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_multiparam_bearer_challenge(self) -> None:
        """`resource_metadata` is found after other Bearer auth-params."""
        exc = _http_status_error(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer error="invalid_token", '
                    'error_description="The access token expired", '
                    f'resource_metadata="{_RESOURCE_METADATA_URL}"'
                )
            },
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_bearer_not_first_in_multischeme_line(self) -> None:
        """A Bearer challenge behind another scheme on one line is detected."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": f'Basic realm="mcp", {_BEARER_CHALLENGE}'},
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_bearer_across_repeated_headers(self) -> None:
        """A Bearer challenge on a second `WWW-Authenticate` line is detected."""
        exc = _http_status_error(
            401,
            headers=[
                ("WWW-Authenticate", 'Basic realm="mcp"'),
                (
                    "WWW-Authenticate",
                    _BEARER_CHALLENGE,
                ),
            ],
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_without_challenge_header_ignored(self) -> None:
        """A 401 lacking `WWW-Authenticate` is not an OAuth challenge."""
        exc = _http_status_error(401)
        assert find_oauth_challenge(exc) is None

    def test_401_basic_challenge_ignored(self) -> None:
        """A non-OAuth auth challenge is not treated as an MCP login prompt."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Basic realm="mcp"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_401_bearer_without_resource_metadata_ignored(self) -> None:
        """A Bearer challenge with params but no `resource_metadata` is ignored."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_401_resource_metadata_substring_not_matched(self) -> None:
        """`resource_metadata` embedded in another token is not a match."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Bearer error="x_resource_metadata_y"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_non_401_status_ignored(self) -> None:
        """Other status codes never count as a challenge."""
        exc = _http_status_error(
            403,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        assert find_oauth_challenge(exc) is None

    def test_found_inside_exception_group(self) -> None:
        """Nested exception groups are searched recursively."""
        exc = ExceptionGroup(
            "outer",
            [
                RuntimeError("x"),
                _http_status_error(
                    401,
                    headers={"WWW-Authenticate": (_BEARER_CHALLENGE)},
                ),
            ],
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_found_via_cause_chain(self) -> None:
        """`raise X from HTTPStatusError(...)` is unwrapped."""
        challenge = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        wrapped = RuntimeError("wrapped")
        wrapped.__cause__ = challenge
        assert find_oauth_challenge(wrapped) == _RESOURCE_METADATA_URL

    def test_found_via_context_chain(self) -> None:
        """Implicit chaining (`__context__`) is unwrapped, not only `__cause__`."""
        challenge = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        wrapped = RuntimeError("wrapped")
        wrapped.__context__ = challenge
        assert find_oauth_challenge(wrapped) == _RESOURCE_METADATA_URL

    def test_returns_none_when_absent(self) -> None:
        """Trees without a 401 challenge yield `None`."""
        exc = ExceptionGroup("outer", [RuntimeError("x"), ValueError("y")])
        assert find_oauth_challenge(exc) is None

    def test_handles_cyclic_chain(self) -> None:
        """Self-referencing `__context__` cycles terminate without recursion."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        assert find_oauth_challenge(a) is None


class TestFormatLoginFailure:
    """Tests for the token-safe summary helper used in app + CLI logs."""

    def test_returns_reauth_message_for_nested_reauth_error(self) -> None:
        """ExceptionGroup wrapping `MCPReauthRequiredError` surfaces its message."""
        exc = ExceptionGroup(
            "anyio task group",
            [RuntimeError("upstream"), MCPReauthRequiredError("notion")],
        )
        summary = format_login_failure(exc)
        assert "notion" in summary
        assert "Run `/mcp login notion`" in summary

    def test_omits_message_for_unknown_exception_types(self) -> None:
        """Unrecognized exceptions degrade to a class-name chain — no `str()`.

        Tokens can hide in `args`/`repr` of unfamiliar MCP-SDK error types;
        the helper must never include those payloads.
        """

        class FakeMcpError(RuntimeError):
            pass

        sentinel = "TOKEN_PAYLOAD_DO_NOT_LEAK"
        exc = FakeMcpError(sentinel)
        summary = format_login_failure(exc)
        assert sentinel not in summary
        assert "FakeMcpError" in summary

    def test_includes_message_for_known_loopback_errors(self) -> None:
        """Loopback-internal exceptions are token-free and may include their message."""
        from deepagents_code.mcp_auth import _LoopbackCallbackTimeoutError

        exc = _LoopbackCallbackTimeoutError("Callback timed out")
        summary = format_login_failure(exc)
        assert "Callback timed out" in summary
        assert "_LoopbackCallbackTimeoutError" in summary

    def test_walks_cause_chain_into_class_names(self) -> None:
        """A chained unknown exception still surfaces every link's class name."""

        class OuterError(RuntimeError):
            pass

        class InnerError(RuntimeError):
            pass

        inner_msg = "inner-payload"
        outer_msg = "outer-payload"
        try:
            try:
                raise InnerError(inner_msg)  # noqa: TRY301
            except InnerError as inner:
                raise OuterError(outer_msg) from inner
        except OuterError as exc:
            summary = format_login_failure(exc)
        assert "OuterError" in summary
        assert "InnerError" in summary
        assert inner_msg not in summary
        assert outer_msg not in summary


class TestAppendQueryParams:
    """Tests for `_append_query_params` URL manipulation."""

    def test_adds_params_to_url_without_query(self) -> None:
        """Params are appended when the URL has no query string."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "T123"})
        assert "team=T123" in result

    def test_overwrites_existing_same_key(self) -> None:
        """Existing same-key query params are replaced, not merged."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x?team=OLD", {"team": "NEW"})
        assert "team=NEW" in result
        assert "team=OLD" not in result

    def test_url_encodes_special_characters(self) -> None:
        """Special characters in values are properly URL-encoded."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "a b&c"})
        assert "team=a+b%26c" in result


class TestPasteBackHandlers:
    """Tests for the interactive OAuth paste-back callback handler."""

    async def test_callback_parses_code_and_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Callback URL with `code` and `state` yields both values."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr(
            "builtins.input", lambda _: "https://localhost/?code=abc&state=xyz"
        )
        code, state = await callback()
        assert code == "abc"
        assert state == "xyz"

    async def test_callback_missing_code_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """URL without `code` raises a clear error."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr("builtins.input", lambda _: "https://localhost/?other=1")
        with pytest.raises(RuntimeError, match="missing the 'code' parameter"):
            await callback()

    async def test_callback_surfaces_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`error=` in the callback URL surfaces provider-side denials."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr(
            "builtins.input",
            lambda _: (
                "https://localhost/?error=access_denied"
                "&error_description=User%20declined"
            ),
        )
        with pytest.raises(RuntimeError, match="access_denied"):
            await callback()


class TestBuildOAuthProvider:
    """Tests for `build_oauth_provider` branching."""

    def test_slack_url_is_detected(self) -> None:
        """The Slack URL detector treats slack.com subdomains as Slack."""
        from deepagents_code.mcp_providers.slack import _is_slack_mcp_url

        assert _is_slack_mcp_url("https://slack.com/mcp")
        assert _is_slack_mcp_url("https://deep.slack.com/mcp")
        assert not _is_slack_mcp_url("https://mcp.notion.com/mcp")

    def test_slack_provider_uses_fixed_loopback_port(self) -> None:
        """SlackProvider uses a fixed port matching the Slack app registration."""
        from deepagents_code.mcp_providers.slack import (
            _SLACK_LOOPBACK_PORT,
            SlackProvider,
        )

        provider = SlackProvider()
        assert provider.supports_loopback_callback() is True
        assert provider.loopback_port() == _SLACK_LOOPBACK_PORT

    def test_slack_branch_sets_public_client_metadata(self) -> None:
        """Slack branch configures a public OAuth client using the loopback URI."""
        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _SLACK_REDIRECT_URI

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://slack.com/mcp",
            storage=FileTokenStorage("slack"),
        )
        metadata = provider.context.client_metadata
        assert metadata.token_endpoint_auth_method == "none"
        assert metadata.redirect_uris is not None
        assert [str(uri) for uri in metadata.redirect_uris] == [_SLACK_REDIRECT_URI]

    def test_interactive_mode_maps_to_reauth_log_suppression(
        self,
        fake_home: Path,
    ) -> None:
        """Only non-interactive providers suppress expected reauth SDK logs.

        Interactive sessions keep the SDK's OAuth diagnostics; non-interactive
        runs replace the expected reauth noise with our login hint.
        """
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        non_interactive = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
            interactive=False,
        )
        interactive = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
            interactive=True,
        )

        assert cast("Any", non_interactive)._suppress_expected_reauth_logs is True
        assert cast("Any", interactive)._suppress_expected_reauth_logs is False

    async def test_refresh_uses_cached_oauth_metadata_endpoint(
        self,
        fake_home: Path,
    ) -> None:
        """Expired tokens refresh against cached metadata, not guessed `/token`."""
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _preseed_slack_client_info

        token_endpoint = "https://slack.com/api/oauth.v2.user.access"
        storage = FileTokenStorage("slack", server_url="https://mcp.slack.com/mcp")
        await _preseed_slack_client_info(storage)
        await storage.set_oauth_metadata(_make_oauth_metadata(token_endpoint))
        await storage.set_tokens(_make_tokens())
        data = json.loads(storage.path.read_text())
        data["expires_at"] = time.time() - 60
        storage.path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://mcp.slack.com/mcp",
            storage=storage,
            interactive=False,
        )
        await provider._initialize()
        refresh_request = await provider._refresh_token()

        assert provider.context.oauth_metadata is not None
        assert str(refresh_request.url) == token_endpoint

    async def test_refresh_discovers_and_caches_oauth_metadata_endpoint(
        self,
        fake_home: Path,
    ) -> None:
        """Legacy token files discover metadata before refreshing."""
        del fake_home
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _preseed_slack_client_info

        token_endpoint = "https://slack.com/api/oauth.v2.user.access"
        storage = FileTokenStorage("slack", server_url="https://mcp.slack.com/mcp")
        await _preseed_slack_client_info(storage)
        await storage.set_tokens(_make_tokens())
        data = json.loads(storage.path.read_text())
        data["expires_at"] = time.time() - 60
        storage.path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://mcp.slack.com/mcp",
            storage=storage,
            interactive=False,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.slack.com/mcp")
        )

        prm_path_request = await anext(flow)
        assert str(prm_path_request.url).endswith(
            "/.well-known/oauth-protected-resource/mcp"
        )
        prm_root_request = await flow.asend(
            httpx.Response(404, request=prm_path_request)
        )
        assert str(prm_root_request.url).endswith(
            "/.well-known/oauth-protected-resource"
        )
        auth_metadata_request = await flow.asend(
            httpx.Response(
                200,
                request=prm_root_request,
                json={
                    "resource": "https://mcp.slack.com",
                    "authorization_servers": ["https://mcp.slack.com"],
                },
            )
        )
        assert str(auth_metadata_request.url).endswith(
            "/.well-known/oauth-authorization-server"
        )
        refresh_request = await flow.asend(
            httpx.Response(
                200,
                request=auth_metadata_request,
                json={
                    "issuer": "https://slack.com",
                    "authorization_endpoint": "https://slack.com/oauth/v2_user/authorize",
                    "token_endpoint": token_endpoint,
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                },
            )
        )

        assert str(refresh_request.url) == token_endpoint
        stored = await storage.get_oauth_metadata()
        assert stored is not None
        assert str(stored.token_endpoint) == token_endpoint
        await flow.aclose()

    async def test_refresh_falls_back_when_preemptive_metadata_discovery_raises(
        self,
        fake_home: Path,
    ) -> None:
        """Transient metadata discovery errors still defer to SDK refresh."""
        del fake_home
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _preseed_slack_client_info

        storage = FileTokenStorage("slack", server_url="https://mcp.slack.com/mcp")
        await _preseed_slack_client_info(storage)
        await storage.set_tokens(_make_tokens())
        data = json.loads(storage.path.read_text())
        data["expires_at"] = time.time() - 60
        storage.path.write_text(json.dumps(data))

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://mcp.slack.com/mcp",
            storage=storage,
            interactive=False,
        )
        flow = provider.async_auth_flow(
            httpx.Request("POST", "https://mcp.slack.com/mcp")
        )

        metadata_request = await anext(flow)
        refresh_request = await flow.athrow(
            httpx.TransportError("metadata unavailable", request=metadata_request)
        )

        assert str(refresh_request.url).endswith("/token")
        await flow.aclose()

    async def test_full_login_persists_discovered_oauth_metadata(
        self,
        fake_home: Path,
    ) -> None:
        """Metadata discovered during full login is cached for later refreshes."""
        del fake_home
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _preseed_slack_client_info

        storage = FileTokenStorage("slack", server_url="https://mcp.slack.com/mcp")
        await _preseed_slack_client_info(storage)
        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://mcp.slack.com/mcp",
            storage=storage,
            interactive=False,
        )
        await provider._initialize()
        # Simulate the SDK's 401-path discovery populating the context during a
        # full browser login, just before the token exchange completes.
        provider.context.oauth_metadata = _make_oauth_metadata()

        assert await storage.get_oauth_metadata() is None
        token_json = json.loads(_make_tokens().model_dump_json(exclude_none=True))
        await provider._handle_token_response(httpx.Response(200, json=token_json))

        stored = await storage.get_oauth_metadata()
        assert stored is not None
        assert str(stored.token_endpoint) == "https://auth.example/token"

    def test_generic_branch_uses_loopback_callback(self) -> None:
        """Non-Slack URLs (including Notion) use a local callback server redirect."""
        from deepagents_code.mcp_auth import build_oauth_provider

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        assert re.fullmatch(r"http://localhost:\d+/callback", redirect_uri)
        # Generic (non-Slack) providers default to client-secret auth, so the
        # Slack-only `token_endpoint_auth_method="none"` override must not
        # leak into this branch.
        assert metadata.token_endpoint_auth_method != "none"

    def test_generic_branch_reuses_stored_loopback_port(self, fake_home: Path) -> None:
        """A persisted DCR redirect URI pins the callback port across launches."""
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(51208)))
        first = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        second = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        first_metadata = first.context.client_metadata
        second_metadata = second.context.client_metadata
        assert first_metadata.redirect_uris is not None
        assert second_metadata.redirect_uris is not None
        assert str(first_metadata.redirect_uris[0]) == "http://localhost:51208/callback"
        assert (
            str(second_metadata.redirect_uris[0]) == "http://localhost:51208/callback"
        )

    def test_fixed_loopback_port_wins_over_stored_port(self, fake_home: Path) -> None:
        """Provider-fixed callback ports take precedence over stored DCR ports."""
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _SLACK_REDIRECT_URI

        storage = FileTokenStorage("slack")
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(51208)))
        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://slack.com/mcp",
            storage=storage,
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == _SLACK_REDIRECT_URI

    def test_generic_branch_random_port_when_stored_uri_non_loopback(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-loopback stored URI falls back to a fresh random port.

        A token is seeded so the stale-registration self-heal is skipped
        (`discard_client_info_if_loopback_unusable` only fires when no token is
        persisted); that keeps this test focused on the random-port fallback,
        distinct from `test_build_oauth_provider_clears_stale_portless_registration`.
        """
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        monkeypatch.setattr(
            "deepagents_code.mcp_auth._choose_loopback_port", lambda: 60001
        )
        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info()))  # localhost, no port
        asyncio.run(storage.set_tokens(_make_tokens()))  # blocks self-heal discard
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == "http://localhost:60001/callback"
        assert "http://localhost/callback" in caplog.text
        assert "not a reusable loopback callback URI" in caplog.text

    def test_stored_loopback_port(self, fake_home: Path) -> None:
        """The storage helper extracts ports only from valid loopback URIs."""
        del fake_home

        storage = FileTokenStorage("notion")
        # No token file on disk yet.
        assert storage.stored_loopback_port() is None
        # Loopback URI with explicit port — reused.
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(54321)))
        assert storage.stored_loopback_port() == 54321

    @pytest.mark.parametrize(
        "uri",
        [
            "https://localhost:5000/callback",
            "http://127.0.0.1:5000/callback",
            "http://localhost:5000/cb",
            "http://localhost:notaport/callback",
        ],
    )
    def test_stored_loopback_port_rejects_non_reusable_uris(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture, uri: str
    ) -> None:
        """Stored ports are reused only for the exact loopback callback shape."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "client_info": {
                        "client_id": "client-id",
                        "redirect_uris": [uri],
                        "grant_types": ["authorization_code", "refresh_token"],
                        "response_types": ["code"],
                    },
                }
            ),
            encoding="utf-8",
        )

        assert storage.stored_loopback_port() is None
        assert uri in caplog.text
        assert "not a reusable loopback callback URI" in caplog.text

    def test_stored_loopback_port_warns_when_token_file_unreadable(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unreadable token files fall back with a warning breadcrumb."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_bytes(b"{not json")

        assert storage.stored_loopback_port() is None
        assert "unreadable during loopback port lookup" in caplog.text
        assert "Failed to read MCP token file" in caplog.text

    async def test_non_interactive_reauth_handlers_raise(self) -> None:
        """In non-interactive mode, both OAuth handlers raise re-auth errors."""
        from deepagents_code.mcp_auth import _make_reauth_required_handlers

        redirect, callback = _make_reauth_required_handlers("srv")
        with pytest.raises(MCPReauthRequiredError):
            await redirect("https://auth.example/")
        with pytest.raises(MCPReauthRequiredError):
            await callback()


class TestLoopbackHandlers:
    """Tests for the local OAuth callback server."""

    async def test_loopback_callback_returns_code_and_state(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A browser callback to the loopback URI completes the handler."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{redirect_uri}?code=abc&state=xyz")

        assert response.status_code == 200
        code, state = await callback_handler()
        assert code == "abc"
        assert state == "xyz"

    async def test_loopback_callback_surfaces_provider_error(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """Provider-side callback errors propagate with a useful message."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{redirect_uri}?error=access_denied&error_description=User%20declined"
            )

        assert response.status_code == 400
        with pytest.raises(RuntimeError, match="access_denied"):
            await callback_handler()

    async def test_loopback_callback_missing_code_raises(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A callback URL missing the `code` parameter sends 400 and raises."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{redirect_uri}?state=xyz")

        assert response.status_code == 400
        with pytest.raises(RuntimeError, match="missing the 'code' parameter"):
            await callback_handler()

    async def test_loopback_falls_back_when_browser_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the browser cannot open, callback() falls back to paste-back at once."""
        from deepagents_code.mcp_auth import build_oauth_provider

        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        code, state = await callback_handler()
        assert code == "fallback"
        assert state == "s"

    async def test_loopback_falls_back_on_bind_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bind failure in redirect() causes callback() to fall back to paste-back."""
        from deepagents_code.mcp_auth import (
            _LoopbackOAuthCallbackServer,
            build_oauth_provider,
        )

        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        monkeypatch.setattr(
            _LoopbackOAuthCallbackServer,
            "start",
            lambda _self: (_ for _ in ()).throw(OSError("Address already in use")),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        code, state = await callback_handler()
        assert code == "fallback"
        assert state == "s"

    async def test_loopback_falls_back_when_webbrowser_get_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No-browser environments (headless, SSH) skip the 300s loopback wait.

        `webbrowser.open` can return `True` in some headless setups even
        when nothing actually launches. `webbrowser.get` raising
        `webbrowser.Error` is the reliable signal that no browser is
        available — the redirect handler must trip the paste-back path
        without binding a socket or burning the timeout.
        """
        import webbrowser

        from deepagents_code.mcp_auth import build_oauth_provider

        def _raise_no_browser(*_a: object, **_kw: object) -> object:
            no_browser_msg = "no browser"
            raise webbrowser.Error(no_browser_msg)

        monkeypatch.setattr("webbrowser.get", _raise_no_browser)
        # Sanity: webbrowser.open intentionally returns True to prove we
        # never call it when get() fails first.
        monkeypatch.setattr(
            "webbrowser.open",
            lambda _url: pytest.fail("webbrowser.open should not be called"),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        code, state = await callback_handler()
        assert code == "fallback"
        assert state == "s"

    async def test_loopback_falls_back_on_callback_timeout(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A loopback callback that never arrives falls through to paste-back.

        Regression guard for `_LoopbackCallbackTimeoutError`. Without this
        path, a user whose browser opens but never redirects would hang
        for the full `_LOOPBACK_CALLBACK_TIMEOUT` (300s).
        """
        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("deepagents_code.mcp_auth._LOOPBACK_CALLBACK_TIMEOUT", 0.05)
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=after_timeout",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        code, _state = await callback_handler()
        assert code == "after_timeout"

    async def test_loopback_repeat_request_after_error_shows_error_page(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A duplicate request after a failed callback must not show success HTML.

        Regression guard: previously `_handle_get` early-returned success
        whenever the future was done, even if the future resolved with an
        exception. A second browser hit (prefetch, favicon) would render
        "You're signed in" while the worker was actually surfacing the
        underlying error.
        """
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        assert redirect_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            first = await client.get(f"{redirect_uri}?error=access_denied")
            second = await client.get(f"{redirect_uri}?code=late")

        assert first.status_code == 400
        # Second request must surface the prior error state, not success.
        assert second.status_code == 400
        assert "You're signed in" not in second.text
        assert "did not complete" in second.text


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorageExtras:
    """Extended storage tests (migration, atomic writes)."""

    async def test_version_mismatch_raises(self, fake_home: Path) -> None:
        """Token files with an unknown version fail with a remediation hint."""
        storage = FileTokenStorage("notion")
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 999, "tokens": {}}))

        with pytest.raises(RuntimeError, match="unsupported version"):
            await storage.get_tokens()

    async def test_set_tokens_and_client_info_atomic(self, fake_home: Path) -> None:
        """Atomic setter writes both fields in a single on-disk payload."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())

        token_path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        raw = token_path.read_text()
        data = json.loads(raw)
        assert "tokens" in data
        assert "client_info" in data
        assert data["tokens"]["access_token"] == "at"
        assert data["client_info"]["client_id"] == "client-id"

    async def test_discard_removes_portless_registration_without_tokens(
        self, fake_home: Path
    ) -> None:
        """A portless loopback registration with no tokens is removed."""
        del fake_home
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())  # localhost, no port

        assert storage.discard_client_info_if_loopback_unusable() is True
        assert await storage.get_client_info() is None

    async def test_discard_keeps_ported_loopback_registration(
        self, fake_home: Path
    ) -> None:
        """A reusable ported loopback registration is left intact."""
        del fake_home
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info_with_loopback(51208))

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert await storage.get_client_info() is not None

    async def test_discard_keeps_registration_when_tokens_present(
        self, fake_home: Path
    ) -> None:
        """A still-usable token blocks discard so refresh isn't downgraded."""
        del fake_home
        storage = FileTokenStorage("notion")
        # Portless registration, but a persisted token can still authenticate.
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert await storage.get_client_info() is not None

    async def test_discard_noop_without_client_info(self, fake_home: Path) -> None:
        """No persisted registration means nothing to discard."""
        del fake_home
        storage = FileTokenStorage("notion")

        assert storage.discard_client_info_if_loopback_unusable() is False

    def test_discard_returns_false_and_warns_on_unreadable_file(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt token file is surfaced (not silently swallowed)."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_bytes(b"{not json")

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert "unreadable while checking for a stale client registration" in (
            caplog.text
        )

    async def test_discard_returns_false_and_keeps_file_when_write_fails(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed atomic write leaves the registration intact and warns."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())  # portless, no tokens
        # Occupy the temp path with a directory so the real atomic write fails
        # with an OSError instead of replacing the token file — no mocks needed.
        tmp = storage.path.with_suffix(storage.path.suffix + ".tmp")
        tmp.mkdir()

        assert storage.discard_client_info_if_loopback_unusable() is False
        # The original registration must still be on disk.
        assert await storage.get_client_info() is not None
        assert "Could not remove stale MCP client registration" in caplog.text

    def test_build_oauth_provider_clears_stale_portless_registration(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Interactive loopback login drops a stale portless registration.

        Regression: a portless `http://localhost/callback` registration (left
        by an earlier non-loopback login) was reused with a fresh random port,
        so the authorize request sent the stale `client_id` with a
        redirect_uri it was never registered for and the server rejected it
        with "invalid or missing redirect_uri". The build must instead discard
        the registration so the handshake re-runs DCR with a matching URI.
        """
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        monkeypatch.setattr(
            "deepagents_code.mcp_auth._choose_loopback_port", lambda: 60001
        )
        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info()))  # localhost, no port

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )

        # Stale registration gone, so the SDK will re-register via DCR.
        assert asyncio.run(storage.get_client_info()) is None
        # The authorize request will carry the freshly bound loopback URI.
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == "http://localhost:60001/callback"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `asyncio.sleep` with a yield so device-flow tests stay fast."""
    real_sleep = asyncio.sleep

    async def _fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


@pytest.mark.usefixtures("no_sleep")
class TestDeviceFlow:
    """Tests for the OAuth 2.0 Device Authorization Grant helper."""

    async def test_happy_path_returns_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful poll returns the issued `OAuthToken`."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(200, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_slow_down_increases_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`slow_down` errors bump the poll interval and continue polling."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(200, json={"error": "slow_down"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_pending_on_http_400_still_polls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Providers returning HTTP 400 for `authorization_pending` still poll."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_error_surfaces_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-recoverable errors surface the provider's description."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            return httpx.Response(
                200,
                json={"error": "access_denied", "error_description": "nope"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="access_denied"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )

    async def test_device_code_request_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 4xx on the initial device-code request raises `RuntimeError`."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={})

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="Device code request failed"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )


@pytest.mark.usefixtures("fake_home")
class TestLogin:
    """Tests for the interactive OAuth login entrypoint."""

    async def test_login_persists_tokens(self) -> None:
        """Successful login persists tokens to the server-specific file."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(
                OAuthToken(access_token="new", token_type="Bearer")
            )
            await storage.set_client_info(_make_client_info())

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                },
                ui=CliOAuthInteraction(),
            )

        storage = FileTokenStorage(
            "notion",
            server_url="https://mcp.notion.com/mcp",
        )
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_allows_http_server_without_explicit_oauth(self) -> None:
        """Auto-detected servers (no `auth: oauth`) can still run OAuth login."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(
                OAuthToken(access_token="new", token_type="Bearer")
            )
            await storage.set_client_info(_make_client_info())

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                },
                ui=CliOAuthInteraction(),
            )

        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_rejects_stdio_server(self) -> None:
        """OAuth login is limited to HTTP/SSE transports."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with pytest.raises(ValueError, match="only valid for http/sse"):
            await login(
                server_name="srv",
                server_config={"command": "echo", "auth": "oauth"},
                ui=CliOAuthInteraction(),
            )

    async def test_login_propagates_static_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured static headers flow into the OAuth handshake connection."""
        from deepagents_code.mcp_auth import login

        monkeypatch.setenv("MCP_GATEWAY_TOKEN", "gw-token")
        captured: dict[str, Any] = {}

        async def _fake_handshake(connections: dict) -> None:
            await asyncio.sleep(0)
            captured.update(next(iter(connections.values())))

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                    "headers": {
                        "X-Tenant": "acme",
                        "Authorization": "Bearer ${MCP_GATEWAY_TOKEN}",
                    },
                },
                ui=CliOAuthInteraction(),
            )

        assert captured["headers"] == {
            "X-Tenant": "acme",
            "Authorization": "Bearer gw-token",
        }

    async def test_login_unset_env_var_in_headers_raises(self) -> None:
        """Unset env vars in static headers fail before the handshake."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with pytest.raises(RuntimeError, match="unset env var"):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                    "headers": {"Authorization": "Bearer ${MISSING_VAR}"},
                },
                ui=CliOAuthInteraction(),
            )

    async def test_github_login_runs_device_flow_and_seeds_client(self) -> None:
        """GitHub URLs short-circuit to device flow and persist client info."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_providers.github import _GITHUB_MCP_CLIENT_ID

        async def _fake_device_flow(
            *,
            device_code_url: str,
            token_url: str,
            client_id: str,
            scope: str | None = None,
            ui: object | None = None,
        ) -> OAuthToken:
            del device_code_url, token_url, client_id, scope, ui
            return OAuthToken(access_token="gh-tok", token_type="Bearer")

        handshake_called = False

        async def _handshake_should_not_run(connections: dict) -> None:
            del connections
            nonlocal handshake_called
            handshake_called = True

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with (
            patch(
                "deepagents_code.mcp_providers.github._run_device_flow",
                _fake_device_flow,
            ),
            patch(
                "deepagents_code.mcp_auth._drive_handshake",
                _handshake_should_not_run,
            ),
        ):
            await login(
                server_name="github",
                server_config={
                    "type": "http",
                    "url": "https://api.githubcopilot.com/mcp/",
                    "auth": "oauth",
                },
                ui=CliOAuthInteraction(),
            )

        assert handshake_called is False, (
            "GitHub login must use device flow, not the authorization-code handshake."
        )
        storage = FileTokenStorage(
            "github",
            server_url="https://api.githubcopilot.com/mcp/",
        )
        tokens = await storage.get_tokens()
        client_info = await storage.get_client_info()
        assert tokens is not None
        assert tokens.access_token == "gh-tok"
        assert client_info is not None
        assert client_info.client_id == _GITHUB_MCP_CLIENT_ID

    async def test_slack_login_routes_team_into_redirect_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slack login threads the entered team id into the interactive URL."""
        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import OAuthInteraction

        class _CapturingUI:
            def __init__(self) -> None:
                self.authorize_urls: list[tuple[str, bool]] = []

            async def show_authorize_url(
                self, url: str, *, opened_in_browser: bool
            ) -> None:
                self.authorize_urls.append((url, opened_in_browser))

            async def request_callback_url(self) -> str:
                msg = "not expected in this test"
                raise AssertionError(msg)

            async def show_device_code(
                self, *, verification_uri: str, user_code: str, expires_in: int
            ) -> None: ...

            async def prompt_slack_team_id(self) -> str | None:
                return "T01234567"

            async def show_success(self, message: str) -> None: ...

            async def show_notice(self, message: str) -> None: ...

            async def show_error(self, message: str) -> None: ...

        # Structural check: all required protocol methods are present.
        protocol_methods = [
            "show_authorize_url",
            "request_callback_url",
            "show_device_code",
            "prompt_slack_team_id",
            "show_success",
            "show_notice",
            "show_error",
        ]
        ui_instance = _CapturingUI()
        assert all(callable(getattr(ui_instance, m, None)) for m in protocol_methods)

        ui = _CapturingUI()

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            provider = connection["auth"]
            redirect = provider.context.redirect_handler
            await redirect("https://slack.com/oauth/v2/authorize?client_id=x")
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="slack",
                server_config={
                    "type": "http",
                    "url": "https://slack.com/mcp",
                    "auth": "oauth",
                },
                ui=ui,
            )

        assert ui.authorize_urls, "authorize URL must be shown"
        shown_url, _opened = ui.authorize_urls[0]
        assert "team=T01234567" in shown_url

    async def test_slack_preseed_is_idempotent(self) -> None:
        """Preseeding Slack client info a second time reads rather than writes."""
        from deepagents_code.mcp_providers.slack import (
            _SLACK_MCP_CLIENT_ID,
            _preseed_slack_client_info,
        )

        storage = FileTokenStorage(
            "slack",
            server_url="https://slack.com/mcp",
        )
        await _preseed_slack_client_info(storage)
        first = await storage.get_client_info()
        assert first is not None
        first_mtime = storage.path.stat().st_mtime_ns

        # Calling a second time must not rewrite the token file.
        await _preseed_slack_client_info(storage)
        second = await storage.get_client_info()
        assert second is not None
        assert second.client_id == _SLACK_MCP_CLIENT_ID
        assert storage.path.stat().st_mtime_ns == first_mtime


@pytest.mark.usefixtures("fake_home", "no_sleep")
class TestDeviceFlowTimeout:
    """Timeout-path coverage for `_run_device_flow`."""

    async def test_device_flow_times_out_when_pending_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The device-code deadline expires when polling never resolves."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        # expires_in=0 means the deadline fires on the
                        # first loop iteration after sleep returns.
                        "expires_in": 0,
                        "interval": 0,
                    },
                )
            return httpx.Response(200, json={"error": "authorization_pending"})

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="Device flow timed out"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )

    async def test_device_code_response_missing_required_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider response missing `verification_uri` surfaces as RuntimeError."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"device_code": "d", "user_code": "U", "expires_in": 30},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="missing required fields"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorageWriteFailures:
    """Partial-write failure cleanup for `FileTokenStorage._write`."""

    async def test_replace_failure_removes_tmp_and_leaves_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup when `tmp.replace` fails mid-write.

        The `.tmp` file must be unlinked and any existing primary token
        file must remain untouched so a failed write never clobbers
        existing credentials.
        """
        storage = FileTokenStorage("acme")
        await storage.set_client_info(_make_client_info())
        original_bytes = storage.path.read_bytes()

        real_replace = Path.replace

        def _failing_replace(self: Path, target: Path | str) -> None:
            if self.suffix == ".tmp":
                msg = "simulated"
                raise OSError(msg)
            real_replace(self, target)

        monkeypatch.setattr(Path, "replace", _failing_replace)

        with pytest.raises(OSError, match="simulated"):
            await storage.set_tokens(_make_tokens("new"))

        tmp = storage.path.with_suffix(storage.path.suffix + ".tmp")
        assert not tmp.exists(), ".tmp must be cleaned up after replace failure"
        assert storage.path.read_bytes() == original_bytes, (
            "primary token file must not be clobbered when write fails"
        )
