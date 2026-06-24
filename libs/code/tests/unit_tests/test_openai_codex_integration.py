"""Unit tests for `deepagents_code.integrations.openai_codex`.

Covers status detection, build-model wiring, and the orchestration of the
browser-loopback OAuth flow without ever hitting the network or opening a
real browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import warnings
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from deepagents_code.integrations import openai_codex
from deepagents_code.model_config import (
    CODEX_PROVIDER,
    ProviderAuthSource,
    ProviderAuthState,
    _get_codex_auth_status,
    clear_caches,
    get_provider_auth_status,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _write_token(
    path: Path,
    *,
    expires_in: timedelta = timedelta(hours=1),
    plan_type: str | None = "plus",
    account_id: str | None = "acct_abc",
) -> None:
    """Plant a serialized token bundle at `path` with private perms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "access_token": "fake_access",
        "refresh_token": "fake_refresh",
        "expires_at": (datetime.now(UTC) + expires_in).isoformat(),
        "account_id": account_id,
        "plan_type": plan_type,
        "user_id": "user_xyz",
        "id_token": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


@contextmanager
def _override_default_store(path: Path) -> Iterator[None]:
    """Point `openai_codex.default_store_path` at `path` for the duration."""
    original = openai_codex.default_store_path
    setattr(openai_codex, "default_store_path", lambda: path)  # noqa: B010  # restored in finally
    clear_caches()
    try:
        yield
    finally:
        setattr(openai_codex, "default_store_path", original)  # noqa: B010  # restore original
        clear_caches()


@contextmanager
def _ignore_codex_experimental_warning() -> Iterator[None]:
    """Suppress the dependency's expected experimental class warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`_ChatOpenAICodex` is experimental",
            category=UserWarning,
        )
        yield


class TestGetStatus:
    """`get_status` reflects on-disk state without network or refresh."""

    def test_not_logged_in_when_file_missing(self, tmp_path: Path) -> None:
        status = openai_codex.get_status(store_path=tmp_path / "missing.json")
        assert status.logged_in is False
        assert status.account_id is None
        assert status.plan_type is None
        assert status.expires_at is None
        assert status.is_expired is False
        assert status.unreadable_reason is None

    def test_logged_in_when_token_present(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path)
        status = openai_codex.get_status(store_path=path)
        assert status.logged_in is True
        assert status.account_id == "acct_abc"
        assert status.plan_type == "plus"
        assert status.expires_at is not None
        assert status.is_expired is False

    def test_expired_token_reported_as_expired(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path, expires_in=timedelta(seconds=-3600))
        status = openai_codex.get_status(store_path=path)
        assert status.logged_in is True
        assert status.is_expired is True

    def test_unreadable_token_surfaces_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text("{not valid json")
        status = openai_codex.get_status(store_path=path)
        assert status.logged_in is False
        assert status.unreadable_reason is not None
        assert "is not valid JSON" in status.unreadable_reason


class TestIsLoggedIn:
    """Convenience predicate over `get_status`."""

    def test_false_when_missing(self, tmp_path: Path) -> None:
        assert openai_codex.is_logged_in(store_path=tmp_path / "x.json") is False

    def test_true_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path)
        assert openai_codex.is_logged_in(store_path=path) is True


class TestLogout:
    """Logout returns `True` only when a file was actually removed."""

    def test_noop_when_file_missing(self, tmp_path: Path) -> None:
        assert openai_codex.logout(store_path=tmp_path / "missing.json") is False

    def test_removes_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path)
        assert openai_codex.logout(store_path=path) is True
        assert not path.exists()


class TestProviderAuthStatus:
    """`get_provider_auth_status('openai_codex')` reads the OAuth file."""

    def test_missing_when_no_token(self, tmp_path: Path) -> None:
        with _override_default_store(tmp_path / "missing.json"):
            status = get_provider_auth_status(CODEX_PROVIDER)
        assert status.state is ProviderAuthState.MISSING
        assert status.provider == CODEX_PROVIDER
        assert status.source is None
        assert "not signed in" in (status.detail or "")

    def test_configured_with_stored_source_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path)
        with _override_default_store(path):
            status = get_provider_auth_status(CODEX_PROVIDER)
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED
        assert "plus" in (status.detail or "")

    def test_unreadable_token_reports_missing_with_detail(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text("garbage")
        with _override_default_store(path):
            status = _get_codex_auth_status()
        assert status.state is ProviderAuthState.MISSING
        assert "unreadable" in (status.detail or "")

    def test_expired_token_reports_configured(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path, expires_in=timedelta(seconds=-3600))
        with _override_default_store(path):
            status = get_provider_auth_status(CODEX_PROVIDER)
        assert status.state is ProviderAuthState.CONFIGURED
        assert status.source is ProviderAuthSource.STORED
        assert "refresh on use" in (status.detail or "")


class TestBuildChatModel:
    """`build_chat_model` raises `FileNotFoundError` when no token exists."""

    def test_raises_when_no_token(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            openai_codex.build_chat_model(
                "gpt-5.2-codex", store_path=tmp_path / "missing.json"
            )

    def test_returns_chat_model_when_token_present(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        _write_token(path)
        with _ignore_codex_experimental_warning():
            model = openai_codex.build_chat_model(
                "gpt-5.2-codex",
                store_path=path,
                http_socket_options=[],
            )
        from langchain_openai.chat_models.codex import (
            _ChatOpenAICodex,
        )

        assert isinstance(model, _ChatOpenAICodex)
        assert model.model_name == "gpt-5.2-codex"

    def test_forwards_arbitrary_model_name_verbatim(self, tmp_path: Path) -> None:
        """Non-allowlisted model names still reach the API unchanged.

        `CODEX_MODELS` only curates the discoverable list (`get_available_models`
        and `get_model_profiles`); an explicit `openai_codex:<name>` spec must
        construct a model that forwards `<name>` to the backend verbatim.
        """
        path = tmp_path / "auth.json"
        _write_token(path)
        with _ignore_codex_experimental_warning():
            model = openai_codex.build_chat_model(
                "dsjhfbshjdf",
                store_path=path,
                http_socket_options=[],
            )
        from langchain_openai.chat_models.codex import (
            _ChatOpenAICodex,
        )

        assert isinstance(model, _ChatOpenAICodex)
        assert model.model_name == "dsjhfbshjdf"


class _FakeUI(openai_codex.CodexLoginInteraction):
    """Capture interaction calls in-memory for assertion."""

    def __init__(self) -> None:
        self.urls: list[tuple[str, bool]] = []

    async def show_authorize_url(self, url: str, *, opened_in_browser: bool) -> None:
        self.urls.append((url, opened_in_browser))


class TestRunBrowserLogin:
    """`run_browser_login` orchestrates PKCE + callback wait + token exchange."""

    def test_success_path_persists_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Happy path: URL surfaced, callback yields code, token saved."""
        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"

        captured_state: dict[str, str] = {}
        # Patch `_build_authorize_url` to capture the state token so the
        # fake callback can echo it back unchanged.
        original_build = o._build_authorize_url

        def _capture_build(**kwargs: Any) -> str:
            captured_state["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture_build)
        monkeypatch.setattr(o, "webbrowser", _DummyBrowser())
        # Don't actually launch a browser.
        import webbrowser as _wb

        monkeypatch.setattr(_wb, "open", lambda *_a, **_kw: False)

        # Fake the loopback callback to immediately return a code + matching
        # state without binding a real socket.
        def fake_wait(**_kwargs: Any) -> dict[str, str]:
            return {"code": "fake_code", "state": captured_state["state"]}

        monkeypatch.setattr(openai_codex, "_wait_for_oauth_callback", fake_wait)

        # Fake the token exchange.
        token_payload = {
            "access_token": "tk_access",
            "refresh_token": "tk_refresh",
            "expires_in": 3600,
            "id_token": None,
        }

        def fake_post_form(
            _url: str, _data: dict[str, str], **_kw: Any
        ) -> dict[str, Any]:
            return token_payload

        monkeypatch.setattr(o, "_post_form", fake_post_form)

        ui = _FakeUI()
        result = asyncio.run(
            openai_codex.run_browser_login(ui, store_path=path, open_browser=False)
        )

        assert result.logged_in is True
        assert path.exists()
        # Stored file is well-formed and matches the fake response.
        stored = json.loads(path.read_text())
        assert stored["access_token"] == "tk_access"
        assert stored["refresh_token"] == "tk_refresh"
        # File perms: 0600 where supported.
        if os.name == "posix":
            mode = path.stat().st_mode & 0o777
            assert mode == 0o600
        # URL surfaced exactly once with `opened_in_browser=False`.
        assert len(ui.urls) == 1
        assert ui.urls[0][1] is False
        assert "client_id=" in ui.urls[0][0]

    def test_state_mismatch_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CSRF mismatch must fail closed before any code exchange."""
        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"

        def fake_wait(**_kwargs: Any) -> dict[str, str]:
            return {
                "code": "fake_code",
                "state": secrets.token_urlsafe(8),
            }

        monkeypatch.setattr(openai_codex, "_wait_for_oauth_callback", fake_wait)

        def _explode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            msg = "should not reach token exchange on state mismatch"
            raise AssertionError(msg)

        monkeypatch.setattr(o, "_post_form", _explode)

        with pytest.raises(RuntimeError, match="state mismatch"):
            asyncio.run(
                openai_codex.run_browser_login(
                    _FakeUI(), store_path=path, open_browser=False
                )
            )
        assert not path.exists()

    def test_callback_error_surfaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OAuth `error=` response is wrapped in a `RuntimeError`."""
        import langchain_openai.chatgpt_oauth as o

        captured: dict[str, str] = {}
        original_build = o._build_authorize_url

        def _capture(**kwargs: Any) -> str:
            captured["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture)
        monkeypatch.setattr(
            openai_codex,
            "_wait_for_oauth_callback",
            lambda **_kw: {
                "state": captured["state"],
                "error": "access_denied",
                "error_description": "user said no",
            },
        )
        with pytest.raises(RuntimeError, match="access_denied"):
            asyncio.run(
                openai_codex.run_browser_login(
                    _FakeUI(),
                    store_path=tmp_path / "auth.json",
                    open_browser=False,
                )
            )

    def test_cancel_event_raises_cancelled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When `cancel_event` is set during the callback wait, we raise."""
        import langchain_openai.chatgpt_oauth as o

        cancel = threading.Event()
        captured: dict[str, str] = {}
        original_build = o._build_authorize_url

        def _capture(**kwargs: Any) -> str:
            captured["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture)

        def fake_wait(**_kw: Any) -> dict[str, str]:
            cancel.set()
            return {"code": "fake_code", "state": captured["state"]}

        monkeypatch.setattr(openai_codex, "_wait_for_oauth_callback", fake_wait)
        with pytest.raises(openai_codex.CodexLoginCancelledError):
            asyncio.run(
                openai_codex.run_browser_login(
                    _FakeUI(),
                    store_path=tmp_path / "auth.json",
                    open_browser=False,
                    cancel_event=cancel,
                )
            )

    def test_missing_code_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A callback with the right state but no `code` must fail closed."""
        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"
        captured: dict[str, str] = {}
        original_build = o._build_authorize_url

        def _capture(**kwargs: Any) -> str:
            captured["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture)
        monkeypatch.setattr(
            openai_codex,
            "_wait_for_oauth_callback",
            lambda **_kw: {"state": captured["state"]},
        )

        def _explode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            msg = "should not exchange a token when no code was returned"
            raise AssertionError(msg)

        monkeypatch.setattr(o, "_post_form", _explode)

        with pytest.raises(RuntimeError, match="authorization code"):
            asyncio.run(
                openai_codex.run_browser_login(
                    _FakeUI(), store_path=path, open_browser=False
                )
            )
        assert not path.exists()

    def test_browser_launch_oserror_falls_back_to_manual_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A browser launcher raising `OSError` must not abort sign-in."""
        import webbrowser as _wb

        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"
        captured: dict[str, str] = {}
        original_build = o._build_authorize_url

        def _capture(**kwargs: Any) -> str:
            captured["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture)

        def _raise_oserror(*_a: Any, **_kw: Any) -> bool:
            msg = "no such browser binary"
            raise OSError(msg)

        # `OSError` is not a `webbrowser.Error`; the flow must still continue.
        monkeypatch.setattr(_wb, "open", _raise_oserror)
        monkeypatch.setattr(
            openai_codex,
            "_wait_for_oauth_callback",
            lambda **_kw: {"code": "fake_code", "state": captured["state"]},
        )
        monkeypatch.setattr(
            o,
            "_post_form",
            lambda *_a, **_kw: {
                "access_token": "tk",
                "refresh_token": "rk",
                "expires_in": 3600,
                "id_token": None,
            },
        )

        ui = _FakeUI()
        result = asyncio.run(
            openai_codex.run_browser_login(ui, store_path=path, open_browser=True)
        )
        assert result.logged_in is True
        # URL surfaced with `opened_in_browser=False` so the user can copy it.
        assert len(ui.urls) == 1
        assert ui.urls[0][1] is False


class TestWaitForOAuthCallback:
    """The cancel-aware callback wait releases the port promptly on cancel."""

    def test_closes_server_on_cancel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set `cancel_event` raises and closes the callback server."""
        import http.server

        class FakeHTTPServer:
            def __init__(self, _address: tuple[str, int], _handler: type[Any]) -> None:
                self.timeout: float | None = None
                self.closed = False

            def handle_request(self) -> None:
                msg = "cancelled wait should not handle requests"
                raise AssertionError(msg)

            def server_close(self) -> None:
                self.closed = True

        servers: list[FakeHTTPServer] = []

        def fake_http_server(
            address: tuple[str, int], handler: type[Any]
        ) -> FakeHTTPServer:
            server = FakeHTTPServer(address, handler)
            servers.append(server)
            return server

        monkeypatch.setattr(http.server, "HTTPServer", fake_http_server)

        cancel = threading.Event()
        cancel.set()
        with pytest.raises(openai_codex.CodexLoginCancelledError):
            openai_codex._wait_for_oauth_callback(
                host="localhost",
                port=1455,
                callback_path="/auth/callback",
                timeout=5.0,
                cancel_event=cancel,
                poll_interval=0.05,
            )

        assert len(servers) == 1
        assert servers[0].closed is True

    def test_bind_failure_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bind failure surfaces as a `RuntimeError`, not `OSError`."""
        import http.server

        def fake_http_server(_address: tuple[str, int], _handler: type[Any]) -> None:
            msg = "address already in use"
            raise OSError(msg)

        monkeypatch.setattr(http.server, "HTTPServer", fake_http_server)

        with pytest.raises(RuntimeError, match="Could not bind"):
            openai_codex._wait_for_oauth_callback(
                host="localhost",
                port=1455,
                callback_path="/auth/callback",
                timeout=1.0,
                poll_interval=0.05,
            )


class TestCodexAuthStatusInvariants:
    """`CodexAuthStatus.__post_init__` rejects incoherent snapshots.

    These guard the cross-field rules the attribute docs promise; without a
    test the guards are dead code that silently rots if a guard is removed.
    """

    def test_unreadable_cannot_be_logged_in(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unreadable_reason"):
            openai_codex.CodexAuthStatus(
                logged_in=True,
                store_path=tmp_path,
                unreadable_reason="corrupt",
            )

    def test_logged_in_requires_expires_at(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expires_at"):
            openai_codex.CodexAuthStatus(logged_in=True, store_path=tmp_path)

    def test_logged_out_rejects_expires_at(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expires_at"):
            openai_codex.CodexAuthStatus(
                logged_in=False,
                store_path=tmp_path,
                expires_at=datetime.now(UTC),
            )

    def test_logged_out_rejects_is_expired(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is_expired"):
            openai_codex.CodexAuthStatus(
                logged_in=False, store_path=tmp_path, is_expired=True
            )

    def test_logged_out_rejects_plan_metadata(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"account_id|plan_type"):
            openai_codex.CodexAuthStatus(
                logged_in=False, store_path=tmp_path, plan_type="pro"
            )


class TestRunBrowserLoginCancelDuringExchange:
    """A cancel landing *during* the token exchange must not save a token."""

    def test_cancel_during_exchange_raises_and_saves_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"
        cancel = threading.Event()
        captured: dict[str, str] = {}
        original_build = o._build_authorize_url

        def _capture(**kwargs: Any) -> str:
            captured["state"] = kwargs["state"]
            return original_build(**kwargs)

        monkeypatch.setattr(o, "_build_authorize_url", _capture)
        # Callback arrives cleanly (cancel not yet set), so the post-callback
        # guard passes; the cancel is then set *inside* the token exchange.
        monkeypatch.setattr(
            openai_codex,
            "_wait_for_oauth_callback",
            lambda **_kw: {"code": "fake_code", "state": captured["state"]},
        )

        def _cancel_mid_exchange(
            _url: str, _data: dict[str, str], **_kw: Any
        ) -> dict[str, Any]:
            cancel.set()
            return {
                "access_token": "tk",
                "refresh_token": "rk",
                "expires_in": 3600,
                "id_token": None,
            }

        monkeypatch.setattr(o, "_post_form", _cancel_mid_exchange)
        with pytest.raises(openai_codex.CodexLoginCancelledError):
            asyncio.run(
                openai_codex.run_browser_login(
                    _FakeUI(),
                    store_path=path,
                    open_browser=False,
                    cancel_event=cancel,
                )
            )
        assert not path.exists()


class TestBuildChatModelRefresh:
    """A revoked refresh token surfaces as `CodexAuthExpiredError`.

    Mapping the raw upstream error lets `create_model` route it to the
    `/auth` flow rather than a generic failure.
    """

    def test_refresh_failure_raises_codex_auth_expired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import langchain_openai.chatgpt_oauth as o

        path = tmp_path / "auth.json"
        _write_token(path)

        def _raise_refresh(_self: object) -> object:
            msg = "refresh token revoked"
            raise o._ChatGPTOAuthRefreshError(msg)

        monkeypatch.setattr(
            o._FileChatGPTOAuthTokenProvider, "get_token", _raise_refresh
        )
        with pytest.raises(openai_codex.CodexAuthExpiredError):
            openai_codex.build_chat_model("gpt-5.2-codex", store_path=path)


class _DummyBrowser:
    """Minimal `webbrowser` stand-in that always reports "no browser"."""

    Error = Exception

    @staticmethod
    def open(_url: str, *_args: Any, **_kwargs: Any) -> bool:
        return False
