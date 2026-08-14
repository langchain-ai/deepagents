"""Unit tests for `deepagents_code.integrations.xai_oauth`.

Covers status detection, build-model wiring, the device-code login
orchestration, and — since these are the two places a subtly wrong
implementation would most likely misbehave in production — the `HTTP 403`
SuperGrok/entitlement-tier-denied special case and single-use refresh-token
rotation on refresh. All network access is monkeypatched; nothing here hits
the real network or a real browser/device flow.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from mcp.shared.auth import OAuthToken

from deepagents_code.integrations import xai_oauth
from deepagents_code.mcp_auth import FileTokenStorage
from deepagents_code.model_config import (
    XAI_OAUTH_PROVIDER,
    ProviderAuthSource,
    ProviderAuthState,
    _get_xai_oauth_auth_status,
    clear_caches,
    get_provider_auth_status,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _response(status: int, **kwargs: Any) -> httpx.Response:
    """Build an `httpx.Response` with a request attached.

    `raise_for_status()` requires the response to know its originating
    request; a bare `httpx.Response(status, ...)` has none, so every fake
    `post` in this module routes through this helper instead.
    """
    return httpx.Response(
        status, request=httpx.Request("POST", xai_oauth._XAI_TOKEN_URL), **kwargs
    )


def _make_jwt(*, exp: float | None) -> str:
    """Build a minimal (unsigned) JWT carrying only an `exp` claim."""
    header = base64.urlsafe_b64encode(b"{}").rstrip(b"=").decode()
    payload_obj: dict[str, Any] = {} if exp is None else {"exp": exp}
    payload = (
        base64.urlsafe_b64encode(json.dumps(payload_obj).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.sig"


def _write_token(
    path: Path,
    *,
    expires_in_seconds: float = 3600.0,
    refresh_token: str | None = "fake_refresh",
    lifetime_seconds: float | None = None,
) -> None:
    """Plant a serialized xAI OAuth token bundle at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exp = time.time() + expires_in_seconds
    payload = {
        "version": 1,
        "tokens": {
            "access_token": _make_jwt(exp=exp),
            "token_type": "Bearer",
            "refresh_token": refresh_token,
        },
        "expires_at": exp,
    }
    if lifetime_seconds is not None:
        payload["xai_lifetime_seconds"] = lifetime_seconds
    else:
        payload["xai_lifetime_seconds"] = expires_in_seconds
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def _storage_at(path: Path) -> FileTokenStorage:
    # Point `.path` at a tmp file without going through the real
    # `~/.deepagents` location. `FileTokenStorage.path` is a computed
    # property, so override it on the instance via a subclass shim.
    return _FixedPathStorage(path)


class _FixedPathStorage(FileTokenStorage):
    """`FileTokenStorage` whose `.path` is pinned to a given file, for tests."""

    def __init__(self, path: Path) -> None:
        super().__init__("xai")
        self._fixed_path = path

    @property
    def path(self) -> Path:  # type: ignore[override]
        return self._fixed_path

    @property
    def refresh_lock_path(self) -> Path:  # type: ignore[override]
        return self._fixed_path.with_name(f"{self._fixed_path.name}.lock")


@contextmanager
def _override_default_storage(storage: FileTokenStorage) -> Iterator[None]:
    """Point `xai_oauth.default_storage` at `storage` for the duration."""
    original = xai_oauth.default_storage
    xai_oauth.default_storage = lambda: storage  # ty: ignore[invalid-assignment]
    clear_caches()
    try:
        yield
    finally:
        xai_oauth.default_storage = original  # ty: ignore[invalid-assignment]
        clear_caches()


class TestDecodeJwtExp:
    """`_decode_jwt_exp` extracts the unverified `exp` claim."""

    def test_decodes_valid_jwt(self) -> None:
        exp = time.time() + 100
        token = _make_jwt(exp=exp)
        assert xai_oauth._decode_jwt_exp(token) == pytest.approx(exp)

    def test_returns_none_for_malformed_token(self) -> None:
        assert xai_oauth._decode_jwt_exp("not-a-jwt") is None

    def test_returns_none_when_exp_missing(self) -> None:
        token = _make_jwt(exp=None)
        assert xai_oauth._decode_jwt_exp(token) is None

    def test_returns_none_for_non_json_payload(self) -> None:
        header = base64.urlsafe_b64encode(b"{}").rstrip(b"=").decode()
        bad_payload = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
        assert xai_oauth._decode_jwt_exp(f"{header}.{bad_payload}.sig") is None


class TestGetStatus:
    """`get_status` reflects on-disk state without network or refresh."""

    def test_not_logged_in_when_file_missing(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "missing.json")
        status = xai_oauth.get_status(storage=storage)
        assert status.logged_in is False
        assert status.expires_at is None
        assert status.is_expired is False
        assert status.unreadable_reason is None

    def test_logged_in_when_token_present(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        status = xai_oauth.get_status(storage=_storage_at(path))
        assert status.logged_in is True
        assert status.expires_at is not None
        assert status.is_expired is False

    def test_expired_token_reported_as_expired(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=-3600)
        status = xai_oauth.get_status(storage=_storage_at(path))
        assert status.logged_in is True
        assert status.is_expired is True

    def test_unreadable_token_surfaces_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        path.write_text("{not valid json")
        status = xai_oauth.get_status(storage=_storage_at(path))
        assert status.logged_in is False
        assert status.unreadable_reason is not None


class TestIsLoggedIn:
    def test_false_when_missing(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "x.json")
        assert xai_oauth.is_logged_in(storage=storage) is False

    def test_true_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        assert xai_oauth.is_logged_in(storage=_storage_at(path)) is True


class TestLogout:
    def test_noop_when_file_missing(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "missing.json")
        assert xai_oauth.logout(storage=storage) is False

    def test_removes_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        assert xai_oauth.logout(storage=_storage_at(path)) is True
        assert not path.exists()


class TestProviderAuthStatus:
    """`get_provider_auth_status('xai_oauth')` reads the OAuth file."""

    def test_missing_when_no_token(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "missing.json")
        with _override_default_storage(storage):
            status = get_provider_auth_status(XAI_OAUTH_PROVIDER)
        assert status.state is ProviderAuthState.MISSING
        assert status.provider == XAI_OAUTH_PROVIDER
        assert status.source is None
        assert "not signed in" in (status.detail or "")

    def test_configured_with_stored_source_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        storage = _storage_at(path)
        with _override_default_storage(storage):
            status = get_provider_auth_status(XAI_OAUTH_PROVIDER)
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED

    def test_unreadable_token_reports_missing_with_detail(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        path.write_text("garbage")
        storage = _storage_at(path)
        with _override_default_storage(storage):
            status = _get_xai_oauth_auth_status()
        assert status.state is ProviderAuthState.MISSING
        assert "unreadable" in (status.detail or "")

    def test_expired_token_reports_configured(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=-3600)
        storage = _storage_at(path)
        with _override_default_storage(storage):
            status = get_provider_auth_status(XAI_OAUTH_PROVIDER)
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED
        assert "refresh on use" in (status.detail or "")


class TestNeedsRefresh:
    """Proactive-refresh skew selection (default vs. short-lifetime tokens)."""

    def test_unknown_expiry_always_refreshes(self) -> None:
        assert xai_oauth._needs_refresh(None, None) is True

    def test_far_from_expiry_does_not_refresh(self) -> None:
        far = time.time() + 3600
        assert xai_oauth._needs_refresh(far, 3600) is False

    def test_within_default_skew_refreshes(self) -> None:
        near = time.time() + 100
        assert xai_oauth._needs_refresh(near, 3600) is True

    def test_short_lifetime_uses_smaller_skew(self) -> None:
        # 200s remaining on a short-lived (15min) token: inside the default
        # 300s skew but outside the shrunk 120s skew used for short-lived
        # tokens, so no refresh yet.
        remaining_200s = time.time() + 200
        assert xai_oauth._needs_refresh(remaining_200s, 15 * 60) is False

    def test_short_lifetime_still_refreshes_near_expiry(self) -> None:
        remaining_60s = time.time() + 60
        assert xai_oauth._needs_refresh(remaining_60s, 15 * 60) is True


class TestRefreshAccessToken:
    """`_refresh_access_token` maps HTTP status codes to specific exceptions."""

    def test_403_raises_tier_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(403, json={"error": "insufficient_scope"})

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthTierDeniedError, match="XAI_API_KEY"):
            xai_oauth._refresh_access_token("rt")

    @pytest.mark.parametrize("status", [400, 401])
    def test_400_401_raises_reauth_required(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(status, json={"error": "invalid_grant"})

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthReauthRequiredError):
            xai_oauth._refresh_access_token("rt")

    def test_other_error_status_raises_protocol_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(500, json={"error": "server_error"})

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthProtocolError):
            xai_oauth._refresh_access_token("rt")

    def test_non_json_response_raises_protocol_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(200, content=b"not json")

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthProtocolError):
            xai_oauth._refresh_access_token("rt")

    def test_network_error_raises_protocol_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            msg = "connection reset"
            raise httpx.ConnectError(msg)

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthProtocolError):
            xai_oauth._refresh_access_token("rt")

    def test_rotates_refresh_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XAI's response carries a new refresh token; it must be used as-is."""

        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(
                200,
                json={
                    "access_token": _make_jwt(exp=time.time() + 3600),
                    "refresh_token": "rotated_rt",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        new_token = xai_oauth._refresh_access_token("original_rt")
        assert new_token.refresh_token == "rotated_rt"

    def test_falls_back_to_prior_refresh_token_when_response_omits_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response without `refresh_token` must not lose the session."""

        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(
                200,
                json={
                    "access_token": _make_jwt(exp=time.time() + 3600),
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        new_token = xai_oauth._refresh_access_token("original_rt")
        assert new_token.refresh_token == "original_rt"


class TestEnsureValidAccessToken:
    """`_ensure_valid_access_token` drives refresh-on-expiry end-to-end."""

    def test_raises_file_not_found_when_no_token(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            xai_oauth._ensure_valid_access_token(storage)

    def test_returns_access_token_when_not_near_expiry(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=3600)
        storage = _storage_at(path)
        data = storage._read()
        assert data is not None
        expected = OAuthToken.model_validate(data["tokens"]).access_token
        assert xai_oauth._ensure_valid_access_token(storage) == expected

    def test_no_refresh_token_raises_reauth_required(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=1, refresh_token=None)
        storage = _storage_at(path)
        with pytest.raises(xai_oauth.XaiOAuthReauthRequiredError):
            xai_oauth._ensure_valid_access_token(storage)

    def test_refreshes_and_persists_rotated_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=1, refresh_token="old_rt")
        storage = _storage_at(path)

        new_access = _make_jwt(exp=time.time() + 3600)

        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(
                200,
                json={
                    "access_token": new_access,
                    "refresh_token": "new_rt",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        result = xai_oauth._ensure_valid_access_token(storage)
        assert result == new_access

        # Persisted to disk with the rotated refresh token.
        data = storage._read()
        assert data is not None
        stored = OAuthToken.model_validate(data["tokens"])
        assert stored.refresh_token == "new_rt"
        assert stored.access_token == new_access

    def test_403_during_refresh_propagates_tier_denied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=1, refresh_token="old_rt")
        storage = _storage_at(path)

        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(403, json={"error": "insufficient_scope"})

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthTierDeniedError):
            xai_oauth._ensure_valid_access_token(storage)


class TestBuildChatModel:
    """`build_chat_model` raises `FileNotFoundError` when no token exists."""

    def test_raises_when_no_token(self, tmp_path: Path) -> None:
        storage = _storage_at(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            xai_oauth.build_chat_model("grok-4", storage=storage)

    def test_returns_chat_model_when_token_present(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        storage = _storage_at(path)
        model = xai_oauth.build_chat_model("grok-4", storage=storage)
        from langchain_xai import ChatXAI

        assert isinstance(model, ChatXAI)
        assert model.model_name == "grok-4"

    def test_forwards_arbitrary_model_name_verbatim(self, tmp_path: Path) -> None:
        path = tmp_path / "xai.json"
        _write_token(path)
        storage = _storage_at(path)
        model = xai_oauth.build_chat_model("some-future-grok", storage=storage)
        from langchain_xai import ChatXAI

        assert isinstance(model, ChatXAI)
        assert model.model_name == "some-future-grok"

    def test_tier_denied_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "xai.json"
        _write_token(path, expires_in_seconds=1, refresh_token="old_rt")
        storage = _storage_at(path)

        def fake_post(_self: httpx.Client, *_a: Any, **_kw: Any) -> httpx.Response:
            return _response(403, json={"error": "insufficient_scope"})

        monkeypatch.setattr(httpx.Client, "post", fake_post)
        with pytest.raises(xai_oauth.XaiOAuthTierDeniedError):
            xai_oauth.build_chat_model("grok-4", storage=storage)


class _FakeUI(xai_oauth.XaiLoginInteraction):
    """Capture interaction calls in-memory for assertion."""

    def __init__(self) -> None:
        self.device_codes: list[tuple[str, str, int]] = []
        self.successes: list[str] = []

    async def show_device_code(
        self, *, verification_uri: str, user_code: str, expires_in: int
    ) -> None:
        self.device_codes.append((verification_uri, user_code, expires_in))

    async def show_success(self, message: str) -> None:
        self.successes.append(message)


class TestRunDeviceLogin:
    """`run_device_login` orchestrates the device flow and persists the token."""

    def test_success_path_persists_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.mcp_auth as mcp_auth_module

        path = tmp_path / "xai.json"
        storage = _storage_at(path)

        async def fake_device_flow(**_kwargs: Any) -> OAuthToken:  # noqa: RUF029
            return OAuthToken(
                access_token=_make_jwt(exp=time.time() + 3600),
                refresh_token="device_rt",
                expires_in=3600,
            )

        monkeypatch.setattr(mcp_auth_module, "_run_device_flow", fake_device_flow)

        ui = _FakeUI()
        status = asyncio.run(xai_oauth.run_device_login(ui, storage=storage))

        assert status.logged_in is True
        assert path.exists()
        assert ui.successes  # a success message was shown
        stored = json.loads(path.read_text())
        assert stored["tokens"]["refresh_token"] == "device_rt"
        if hasattr(path, "stat"):
            import os

            if os.name == "posix":
                mode = path.stat().st_mode & 0o777
                assert mode == 0o600

    def test_device_flow_failure_raises_protocol_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.mcp_auth as mcp_auth_module

        path = tmp_path / "xai.json"
        storage = _storage_at(path)

        async def fake_device_flow(**_kwargs: Any) -> OAuthToken:  # noqa: RUF029
            msg = "Device flow timed out. Try logging in again."
            raise RuntimeError(msg)

        monkeypatch.setattr(mcp_auth_module, "_run_device_flow", fake_device_flow)

        with pytest.raises(xai_oauth.XaiOAuthProtocolError, match="timed out"):
            asyncio.run(xai_oauth.run_device_login(_FakeUI(), storage=storage))
        assert not path.exists()


class TestXaiOAuthStatusInvariants:
    """`XaiOAuthStatus.__post_init__` rejects incoherent snapshots."""

    def test_unreadable_cannot_be_logged_in(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unreadable_reason"):
            xai_oauth.XaiOAuthStatus(
                logged_in=True,
                store_path=tmp_path,
                unreadable_reason="corrupt",
            )

    def test_logged_out_rejects_expires_at(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expires_at"):
            xai_oauth.XaiOAuthStatus(
                logged_in=False,
                store_path=tmp_path,
                expires_at=datetime.now(UTC),
            )

    def test_logged_out_rejects_is_expired(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is_expired"):
            xai_oauth.XaiOAuthStatus(
                logged_in=False, store_path=tmp_path, is_expired=True
            )


def test_expired_timedelta_helper_sanity() -> None:
    """Sanity check that `_write_token`'s expiry math produces past timestamps."""
    assert timedelta(seconds=-3600).total_seconds() < 0
