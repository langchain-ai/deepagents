import time
from unittest.mock import MagicMock, patch

import pytest

from deepagents_codex.auth import (
    _generate_pkce_pair,
    get_auth_status,
    logout,
    refresh_access_token,
)
from deepagents_codex.errors import CodexAuthError
from deepagents_codex.status import CodexAuthStatus
from deepagents_codex.store import CodexAuthStore, CodexCredentials


class TestPKCE:
    def test_generate_pair(self) -> None:
        verifier, challenge = _generate_pkce_pair()
        assert len(verifier) > 40
        assert len(challenge) > 20
        assert verifier != challenge

    def test_pairs_are_unique(self) -> None:
        pair1 = _generate_pkce_pair()
        pair2 = _generate_pkce_pair()
        assert pair1[0] != pair2[0]


class TestGetAuthStatus:
    def test_not_authenticated(self, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "missing.json")
        info = get_auth_status(store=store)
        assert info.status == CodexAuthStatus.NOT_AUTHENTICATED

    def test_authenticated(self, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "creds.json")
        creds = CodexCredentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() + 3600,
            user_email="test@example.com",
        )
        store.save(creds)
        info = get_auth_status(store=store)
        assert info.status == CodexAuthStatus.AUTHENTICATED
        assert info.user_email == "test@example.com"

    def test_expired(self, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "creds.json")
        creds = CodexCredentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() - 100,
        )
        store.save(creds)
        info = get_auth_status(store=store)
        assert info.status == CodexAuthStatus.EXPIRED

    def test_corrupt(self, tmp_path) -> None:
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("not valid json{{{", encoding="utf-8")
        store = CodexAuthStore(path=creds_path)
        info = get_auth_status(store=store)
        assert info.status == CodexAuthStatus.CORRUPT


class TestLogout:
    def test_logout_existing(self, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "creds.json")
        creds = CodexCredentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() + 3600,
        )
        store.save(creds)
        assert logout(store=store) is True
        assert store.load() is None

    def test_logout_missing(self, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "missing.json")
        assert logout(store=store) is False


class TestRefreshToken:
    @patch("deepagents_codex.auth.httpx.Client")
    def test_refresh_success(self, mock_client_cls, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "creds.json")
        old_creds = CodexCredentials(
            access_token="old_tok",
            refresh_token="ref_tok",
            expires_at=time.time() - 100,
            user_email="test@example.com",
        )
        store.save(old_creds)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "new_tok",
            "refresh_token": "new_ref",
            "expires_in": 3600,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        new_creds = refresh_access_token(old_creds, store=store)
        assert new_creds.access_token == "new_tok"
        assert new_creds.user_email == "test@example.com"

    @patch("deepagents_codex.auth.httpx.Client")
    def test_refresh_failure(self, mock_client_cls, tmp_path) -> None:
        store = CodexAuthStore(path=tmp_path / "creds.json")
        old_creds = CodexCredentials(
            access_token="old_tok",
            refresh_token="ref_tok",
            expires_at=time.time() - 100,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(CodexAuthError, match="Token refresh failed"):
            refresh_access_token(old_creds, store=store)
