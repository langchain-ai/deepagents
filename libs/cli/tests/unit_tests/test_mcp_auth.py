"""Tests for MCP OAuth helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from deepagents_cli.mcp_auth import (
    FileTokenStorage,
    MCPReauthRequiredError,
    find_reauth_required,
    resolve_headers,
)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` to a temp directory."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
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
            resolve_headers({"X-Bad": 123}, server_name="srv")  # type: ignore[dict-item]

    def test_unset_env_var_raises(self) -> None:
        """Unset placeholders fail with a helpful message."""
        with pytest.raises(RuntimeError, match="unset env var"):
            resolve_headers({"Authorization": "Bearer ${MISSING}"})

    def test_plain_text_value_is_unchanged(self) -> None:
        """Strings without placeholders pass through unchanged."""
        assert resolve_headers({"X-Plain": "hello"}) == {"X-Plain": "hello"}


def _make_tokens(access_token: str = "at"):
    from mcp.shared.auth import OAuthToken

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

        token_path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"
        assert token_path.exists()
        if hasattr(token_path, "stat"):
            assert token_path.stat().st_mode & 0o777 == 0o600

    async def test_corrupt_file_raises(self, fake_home: Path) -> None:
        """Corrupt files fail with a remediation hint."""
        path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"
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


class TestAppendQueryParams:
    """Tests for `_append_query_params` URL manipulation."""

    def test_adds_params_to_url_without_query(self) -> None:
        """Params are appended when the URL has no query string."""
        from deepagents_cli.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "T123"})
        assert "team=T123" in result

    def test_overwrites_existing_same_key(self) -> None:
        """Existing same-key query params are replaced, not merged."""
        from deepagents_cli.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x?team=OLD", {"team": "NEW"})
        assert "team=NEW" in result
        assert "team=OLD" not in result

    def test_url_encodes_special_characters(self) -> None:
        """Special characters in values are properly URL-encoded."""
        from deepagents_cli.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "a b&c"})
        assert "team=a+b%26c" in result


class TestPasteBackHandlers:
    """Tests for the interactive OAuth paste-back callback handler."""

    async def test_callback_parses_code_and_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Callback URL with `code` and `state` yields both values."""
        from deepagents_cli.mcp_auth import _make_paste_back_handlers

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
        from deepagents_cli.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr("builtins.input", lambda _: "https://localhost/?other=1")
        with pytest.raises(RuntimeError, match="missing the 'code' parameter"):
            await callback()

    async def test_callback_surfaces_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`error=` in the callback URL surfaces provider-side denials."""
        from deepagents_cli.mcp_auth import _make_paste_back_handlers

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
        from deepagents_cli.mcp_providers.slack import _is_slack_mcp_url

        assert _is_slack_mcp_url("https://slack.com/mcp")
        assert _is_slack_mcp_url("https://deep.slack.com/mcp")
        assert not _is_slack_mcp_url("https://mcp.notion.com/mcp")

    def test_slack_branch_sets_public_client_metadata(self) -> None:
        """Slack branch configures a public OAuth client (no token secret)."""
        from deepagents_cli.mcp_auth import build_oauth_provider

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://slack.com/mcp",
            storage=FileTokenStorage("slack"),
        )
        metadata = provider.context.client_metadata
        assert metadata.token_endpoint_auth_method == "none"
        assert metadata.redirect_uris is not None
        assert [str(uri) for uri in metadata.redirect_uris] == ["https://localhost/"]

    def test_generic_branch_uses_localhost_callback(self) -> None:
        """Non-Slack URLs use the generic `http://localhost/callback` redirect."""
        from deepagents_cli.mcp_auth import build_oauth_provider

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert [str(uri) for uri in metadata.redirect_uris] == [
            "http://localhost/callback"
        ]
        # Generic (non-Slack) providers default to client-secret auth, so the
        # Slack-only `token_endpoint_auth_method="none"` override must not
        # leak into this branch.
        assert metadata.token_endpoint_auth_method != "none"

    async def test_non_interactive_reauth_handlers_raise(self) -> None:
        """In non-interactive mode, both OAuth handlers raise re-auth errors."""
        from deepagents_cli.mcp_auth import _make_reauth_required_handlers

        redirect, callback = _make_reauth_required_handlers("srv")
        with pytest.raises(MCPReauthRequiredError):
            await redirect("https://auth.example/")
        with pytest.raises(MCPReauthRequiredError):
            await callback()


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorageExtras:
    """Extended storage tests (migration, atomic writes)."""

    async def test_version_mismatch_raises(self, fake_home: Path) -> None:
        """Token files with an unknown version fail with a remediation hint."""
        storage = FileTokenStorage("notion")
        path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 999, "tokens": {}}))

        with pytest.raises(RuntimeError, match="unsupported version"):
            await storage.get_tokens()

    async def test_set_tokens_and_client_info_atomic(self, fake_home: Path) -> None:
        """Atomic setter writes both fields in a single on-disk payload."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())

        raw = (fake_home / ".deepagents" / "mcp-tokens" / "notion.json").read_text()
        data = json.loads(raw)
        assert "tokens" in data
        assert "client_info" in data
        assert data["tokens"]["access_token"] == "at"
        assert data["client_info"]["client_id"] == "client-id"


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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import login

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(
                OAuthToken(access_token="new", token_type="Bearer")
            )
            await storage.set_client_info(_make_client_info())

        with patch("deepagents_cli.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                },
            )

        storage = FileTokenStorage(
            "notion",
            server_url="https://mcp.notion.com/mcp",
        )
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_rejects_non_oauth_server(self) -> None:
        """Only `auth: oauth` servers support the login command."""
        from deepagents_cli.mcp_auth import login

        with pytest.raises(ValueError, match="does not use OAuth"):
            await login(
                server_name="srv",
                server_config={"transport": "http", "url": "https://example.com"},
            )

    async def test_login_rejects_stdio_server(self) -> None:
        """OAuth login is limited to HTTP/SSE transports."""
        from deepagents_cli.mcp_auth import login

        with pytest.raises(ValueError, match="only valid for http/sse"):
            await login(
                server_name="srv",
                server_config={"command": "echo", "auth": "oauth"},
            )

    async def test_login_propagates_static_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured static headers flow into the OAuth handshake connection."""
        from deepagents_cli.mcp_auth import login

        monkeypatch.setenv("MCP_GATEWAY_TOKEN", "gw-token")
        captured: dict[str, Any] = {}

        async def _fake_handshake(connections: dict) -> None:
            await asyncio.sleep(0)
            captured.update(next(iter(connections.values())))

        with patch("deepagents_cli.mcp_auth._drive_handshake", _fake_handshake):
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
            )

        assert captured["headers"] == {
            "X-Tenant": "acme",
            "Authorization": "Bearer gw-token",
        }

    async def test_login_unset_env_var_in_headers_raises(self) -> None:
        """Unset env vars in static headers fail before the handshake."""
        from deepagents_cli.mcp_auth import login

        with pytest.raises(RuntimeError, match="unset env var"):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                    "headers": {"Authorization": "Bearer ${MISSING_VAR}"},
                },
            )

    async def test_github_login_runs_device_flow_and_seeds_client(self) -> None:
        """GitHub URLs short-circuit to device flow and persist client info."""
        from mcp.shared.auth import OAuthToken

        from deepagents_cli.mcp_auth import login
        from deepagents_cli.mcp_providers.github import _GITHUB_MCP_CLIENT_ID

        async def _fake_device_flow(
            *,
            device_code_url: str,
            token_url: str,
            client_id: str,
            scope: str | None = None,
        ) -> OAuthToken:
            del device_code_url, token_url, client_id, scope
            return OAuthToken(access_token="gh-tok", token_type="Bearer")

        handshake_called = False

        async def _handshake_should_not_run(connections: dict) -> None:
            del connections
            nonlocal handshake_called
            handshake_called = True

        with (
            patch(
                "deepagents_cli.mcp_providers.github._run_device_flow",
                _fake_device_flow,
            ),
            patch(
                "deepagents_cli.mcp_auth._drive_handshake",
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

    async def test_slack_login_routes_team_into_redirect_url(self) -> None:
        """Slack login threads the entered team id into the interactive URL."""
        from deepagents_cli.mcp_auth import login

        captured_urls: list[str] = []

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            provider = connection["auth"]
            redirect = provider.context.redirect_handler
            await redirect("https://slack.com/oauth/v2/authorize?client_id=x")
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            from mcp.shared.auth import OAuthToken

            await storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))

        async def _fake_prompt_team() -> str | None:
            return "T01234567"

        with (
            patch(
                "deepagents_cli.mcp_providers.slack._prompt_slack_team",
                _fake_prompt_team,
            ),
            patch(
                "deepagents_cli.mcp_auth._drive_handshake",
                _fake_handshake,
            ),
            patch(
                "builtins.print",
                lambda *args, **_kwargs: captured_urls.extend(
                    s for s in args if isinstance(s, str)
                ),
            ),
        ):
            await login(
                server_name="slack",
                server_config={
                    "type": "http",
                    "url": "https://slack.com/mcp",
                    "auth": "oauth",
                },
            )

        joined = "\n".join(captured_urls)
        assert "team=T01234567" in joined

    async def test_slack_preseed_is_idempotent(self) -> None:
        """Preseeding Slack client info a second time reads rather than writes."""
        from deepagents_cli.mcp_providers.slack import (
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

        from deepagents_cli.mcp_auth import _run_device_flow

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

        from deepagents_cli.mcp_auth import _run_device_flow

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
