"""Tests for deepagents_cli.mcp_auth."""

from pathlib import Path
from unittest.mock import patch

import pytest
from mcp.shared.auth import AnyUrl, OAuthClientInformationFull, OAuthToken

from deepagents_cli.mcp_auth import resolve_headers


class TestResolveHeaders:
    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "secret-value")
        assert resolve_headers({"Authorization": "Bearer ${FOO}"}) == {
            "Authorization": "Bearer secret-value"
        }

    def test_multiple_vars_in_one_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")
        assert resolve_headers({"X-Combo": "${A}-${B}"}) == {"X-Combo": "alpha-beta"}

    def test_missing_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(RuntimeError) as exc_info:
            resolve_headers(
                {"Authorization": "Bearer ${MISSING_VAR}"},
                server_name="linear",
            )
        msg = str(exc_info.value)
        assert "MISSING_VAR" in msg
        assert "Authorization" in msg
        assert "linear" in msg

    def test_escape_double_dollar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOT_LOOKED_UP", raising=False)
        assert resolve_headers({"X-Literal": "$${NOT_LOOKED_UP}"}) == {
            "X-Literal": "${NOT_LOOKED_UP}"
        }

    def test_dollar_not_followed_by_brace_untouched(self) -> None:
        assert resolve_headers({"X-Price": "price=$5"}) == {"X-Price": "price=$5"}

    def test_non_string_value_rejected(self) -> None:
        with pytest.raises(TypeError):
            resolve_headers({"X-Bad": 123}, server_name="srv")  # type: ignore[dict-item]

    def test_empty_headers_returns_empty_dict(self) -> None:
        assert resolve_headers({}) == {}

    def test_no_substitution_when_no_placeholders(self) -> None:
        assert resolve_headers({"X-Plain": "hello"}) == {"X-Plain": "hello"}


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect `~/.deepagents/` into a per-test tmp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _make_tokens() -> OAuthToken:
    return OAuthToken(
        access_token="at", token_type="Bearer", refresh_token="rt", expires_in=3600
    )


def _make_client_info() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="cid",
        redirect_uris=[AnyUrl("http://localhost/callback")],
    )


class TestFileTokenStorage:
    @pytest.mark.asyncio
    async def test_get_tokens_when_file_missing(self, fake_home: Path) -> None:
        from deepagents_cli.mcp_auth import FileTokenStorage

        storage = FileTokenStorage("notion")
        assert await storage.get_tokens() is None
        assert await storage.get_client_info() is None

    @pytest.mark.asyncio
    async def test_set_and_get_round_trip(self, fake_home: Path) -> None:
        from deepagents_cli.mcp_auth import FileTokenStorage

        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        got_ci = await storage.get_client_info()
        got_tok = await storage.get_tokens()
        assert got_ci is not None
        assert got_tok is not None
        assert got_ci.client_id == "cid"
        assert got_tok.access_token == "at"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(__import__("os"), "geteuid"),
        reason="POSIX file-mode semantics",
    )
    async def test_file_and_dir_permissions(self, fake_home: Path) -> None:
        import os

        from deepagents_cli.mcp_auth import FileTokenStorage

        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())
        token_path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"
        dir_path = token_path.parent
        assert token_path.exists()
        assert (os.stat(token_path).st_mode & 0o777) == 0o600
        assert (os.stat(dir_path).st_mode & 0o777) == 0o700

    @pytest.mark.asyncio
    async def test_atomic_write_no_partial_on_failure(
        self, fake_home: Path
    ) -> None:
        from deepagents_cli.mcp_auth import FileTokenStorage

        storage = FileTokenStorage("notion")
        token_path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"

        with patch("os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                await storage.set_tokens(_make_tokens())
        assert not token_path.exists()

    @pytest.mark.asyncio
    async def test_version_mismatch_raises(self, fake_home: Path) -> None:
        from deepagents_cli.mcp_auth import FileTokenStorage

        storage = FileTokenStorage("notion")
        # Force-write a bad file bypassing the model.
        token_path = fake_home / ".deepagents" / "mcp-tokens" / "notion.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text('{"version": 999, "tokens": {}}')

        with pytest.raises(RuntimeError, match="version"):
            await storage.get_tokens()

    @pytest.mark.asyncio
    async def test_two_servers_do_not_collide(self, fake_home: Path) -> None:
        from deepagents_cli.mcp_auth import FileTokenStorage

        a = FileTokenStorage("notion")
        b = FileTokenStorage("linear")
        await a.set_tokens(
            OAuthToken(access_token="a-tok", token_type="Bearer")
        )
        await b.set_tokens(
            OAuthToken(access_token="b-tok", token_type="Bearer")
        )
        got_a = await a.get_tokens()
        got_b = await b.get_tokens()
        assert got_a is not None and got_a.access_token == "a-tok"
        assert got_b is not None and got_b.access_token == "b-tok"


import io


class TestBuildOAuthProvider:
    def test_returns_oauth_client_provider(self, fake_home: Path) -> None:
        from mcp.client.auth import OAuthClientProvider

        from deepagents_cli.mcp_auth import FileTokenStorage, build_oauth_provider

        storage = FileTokenStorage("notion")
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        assert isinstance(provider, OAuthClientProvider)

    @pytest.mark.asyncio
    async def test_redirect_handler_prints_url(
        self, fake_home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from deepagents_cli.mcp_auth import _make_paste_back_handlers

        redirect, _ = _make_paste_back_handlers()
        await redirect("https://issuer.example.com/authorize?x=1")
        captured = capsys.readouterr()
        assert "https://issuer.example.com/authorize?x=1" in captured.out

    @pytest.mark.asyncio
    async def test_callback_handler_parses_code_and_state(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_cli import mcp_auth

        _, callback = mcp_auth._make_paste_back_handlers()
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(
                "http://localhost/callback?code=ABC&state=XYZ\n"
            ),
        )
        code, state = await callback()
        assert code == "ABC"
        assert state == "XYZ"

    @pytest.mark.asyncio
    async def test_callback_handler_rejects_missing_code(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_cli import mcp_auth

        _, callback = mcp_auth._make_paste_back_handlers()
        monkeypatch.setattr(
            "sys.stdin", io.StringIO("http://localhost/callback\n")
        )
        with pytest.raises(RuntimeError, match="code"):
            await callback()
