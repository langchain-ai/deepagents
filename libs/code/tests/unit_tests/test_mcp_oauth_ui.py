"""Tests proving MCP OAuth login can be driven without stdin/stdout.

These tests use a programmable `RecordingOAuthInteraction` in place of
the CLI prompt surface. They demonstrate that any TUI or test surface
can satisfy the `OAuthInteraction` Protocol and drive `mcp_auth.login`
end-to-end without touching `builtins.input`, `print`, or stdin.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code.mcp_auth import FileTokenStorage

if TYPE_CHECKING:
    from deepagents_code.mcp_oauth_ui import OAuthInteraction


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect token state into a temp directory."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


class RecordingOAuthInteraction:
    """`OAuthInteraction` whose responses are pre-programmed by the test.

    Captures the messages the login flow would have shown to a user and
    serves canned answers from queues for each prompt. Implements the
    Protocol structurally — no inheritance, no I/O.
    """

    def __init__(
        self,
        *,
        callback_urls: list[str] | None = None,
        slack_team_ids: list[str | None] | None = None,
    ) -> None:
        """Seed the interaction with canned prompt answers.

        Args:
            callback_urls: URLs to return from `request_callback_url`, in order.
            slack_team_ids: Team IDs (or `None`) to return from
                `prompt_slack_team_id`, in order.
        """
        self._callback_urls = list(callback_urls or [])
        self._slack_team_ids = list(slack_team_ids or [])
        self.authorize_urls: list[tuple[str, bool]] = []
        self.device_codes: list[tuple[str, str, int]] = []
        self.successes: list[str] = []
        self.notices: list[str] = []
        self.errors: list[str] = []

    async def show_authorize_url(self, url: str, *, opened_in_browser: bool) -> None:
        """Record the URL display call."""
        self.authorize_urls.append((url, opened_in_browser))

    async def request_callback_url(self) -> str:
        """Return the next queued callback URL or raise if none remain."""
        if not self._callback_urls:
            msg = "No queued callback URL"
            raise RuntimeError(msg)
        return self._callback_urls.pop(0)

    async def show_device_code(
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Record the device-code display call."""
        self.device_codes.append((verification_uri, user_code, expires_in))

    async def prompt_slack_team_id(self) -> str | None:
        """Return the next queued Slack team ID."""
        if not self._slack_team_ids:
            return None
        return self._slack_team_ids.pop(0)

    async def show_success(self, message: str) -> None:
        """Record a success message."""
        self.successes.append(message)

    async def show_notice(self, message: str) -> None:
        """Record a progress notice."""
        self.notices.append(message)

    async def show_error(self, message: str) -> None:
        """Record an error message."""
        self.errors.append(message)


def test_recording_ui_satisfies_protocol() -> None:
    """`RecordingOAuthInteraction` structurally satisfies `OAuthInteraction`."""
    protocol_methods = [
        "show_authorize_url",
        "request_callback_url",
        "show_device_code",
        "show_success",
        "show_notice",
        "show_error",
    ]
    ui = RecordingOAuthInteraction()
    for method in protocol_methods:
        assert callable(getattr(ui, method, None)), (
            f"RecordingOAuthInteraction missing protocol method: {method}"
        )


class TestLoginWithoutStdio:
    """`login()` runs end-to-end with a programmable UI implementation."""

    @pytest.mark.usefixtures("fake_home")
    async def test_paste_back_login_with_recording_ui(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slack-style paste-back login completes without using stdin or stdout."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login

        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        captured: list[str] = []

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            provider = connection["auth"]
            await provider.context.redirect_handler(
                "https://slack.com/oauth/v2/authorize?client_id=x"
            )
            code, _state = await provider.context.callback_handler()
            captured.append(code)
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))

        ui = RecordingOAuthInteraction(
            callback_urls=["https://localhost/?code=abc&state=xyz"],
            slack_team_ids=["T01234567"],
        )

        # Guard: if the flow secretly fell back to stdin, the test must fail
        # loudly instead of hanging.
        def _input_should_not_run(_prompt: str) -> str:
            msg = "login() reached builtins.input even though a UI was provided"
            raise AssertionError(msg)

        monkeypatch.setattr("builtins.input", _input_should_not_run)

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

        assert captured == ["abc"]
        assert ui.authorize_urls, "authorize URL should have been displayed"
        shown_url, opened = ui.authorize_urls[0]
        assert "team=T01234567" in shown_url
        assert opened is False
        assert ui.successes, "login() must report success via the UI"
        assert "Logged in to MCP server 'slack'" in ui.successes[-1]

    @pytest.mark.usefixtures("fake_home")
    async def test_github_device_flow_with_recording_ui(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GitHub device-flow login renders codes through the UI surface."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login

        async def _fake_device_flow(
            *,
            device_code_url: str,
            token_url: str,
            client_id: str,
            scope: str | None = None,
            ui: OAuthInteraction | None = None,
        ) -> OAuthToken:
            del device_code_url, token_url, client_id, scope
            assert ui is not None, "device flow must receive the UI surface"
            await ui.show_device_code(
                verification_uri="https://github.com/login/device",
                user_code="ABCD-1234",
                expires_in=900,
            )
            return OAuthToken(access_token="gh-tok", token_type="Bearer")

        async def _handshake_should_not_run(_connections: dict) -> None:  # noqa: RUF029
            msg = "GitHub login must short-circuit to device flow"
            raise AssertionError(msg)

        ui = RecordingOAuthInteraction()

        monkeypatch.setattr(
            "builtins.input",
            lambda _: (_ for _ in ()).throw(
                AssertionError("device flow must not touch stdin")
            ),
        )

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
                ui=ui,
            )

        assert ui.device_codes == [
            ("https://github.com/login/device", "ABCD-1234", 900)
        ]
        assert ui.successes, "device-flow login must report success via the UI"

    @pytest.mark.usefixtures("fake_home")
    async def test_ui_success_message_never_contains_token_material(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tokens must not leak into UI messages or notices."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login

        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        secret = "super-secret-access-token"

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            provider = connection["auth"]
            await provider.context.redirect_handler(
                "https://slack.com/oauth/v2/authorize?client_id=x"
            )
            await provider.context.callback_handler()
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(
                OAuthToken(access_token=secret, token_type="Bearer")
            )

        ui = RecordingOAuthInteraction(
            callback_urls=["https://localhost/?code=abc"],
            slack_team_ids=[None],
        )

        monkeypatch.setattr(
            "builtins.input",
            lambda _: (_ for _ in ()).throw(
                AssertionError("login() should not reach stdin")
            ),
        )

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

        for url, _opened in ui.authorize_urls:
            assert secret not in url
        for message in ui.successes:
            assert secret not in message
        for message in ui.notices:
            assert secret not in message


class TestCliOAuthInteraction:
    """The default CLI implementation still uses stdin/stdout."""

    async def test_cli_request_callback_url_reads_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`request_callback_url` returns the trimmed `input()` result."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        monkeypatch.setattr(
            "builtins.input", lambda _: "  https://localhost/?code=abc  "
        )
        url = await CliOAuthInteraction().request_callback_url()
        assert url == "https://localhost/?code=abc"

    async def test_cli_request_callback_url_eof_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`EOFError` on stdin becomes a `RuntimeError` with a remediation hint."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        with pytest.raises(RuntimeError, match="No callback URL received"):
            await CliOAuthInteraction().request_callback_url()

    async def test_cli_show_authorize_url_prints_browser_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Browser-opened branch mentions the auto-launch."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        await CliOAuthInteraction().show_authorize_url(
            "https://example/auth", opened_in_browser=True
        )
        out = capsys.readouterr().out
        assert "Opened your browser" in out
        assert "https://example/auth" in out

    async def test_cli_show_authorize_url_prints_paste_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Paste-back branch instructs the user to copy the callback URL."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        await CliOAuthInteraction().show_authorize_url(
            "https://example/auth", opened_in_browser=False
        )
        out = capsys.readouterr().out
        assert "paste the full" in out.lower()
        assert "https://example/auth" in out

    async def test_cli_prompt_slack_team_id_returns_none_when_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty input means `None` so callers fall back to Slack's chooser."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        monkeypatch.setattr("builtins.input", lambda _: "   ")
        assert await CliOAuthInteraction().prompt_slack_team_id() is None

    async def test_cli_prompt_slack_team_id_returns_stripped_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`prompt_slack_team_id` returns the stripped team ID when non-blank."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        monkeypatch.setattr("builtins.input", lambda _: "  T01234567  ")
        assert await CliOAuthInteraction().prompt_slack_team_id() == "T01234567"

    async def test_cli_prompt_slack_team_id_returns_none_on_eof(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`EOFError` from closed stdin returns `None` instead of propagating."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        assert await CliOAuthInteraction().prompt_slack_team_id() is None

    async def test_cli_show_device_code_prints_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`show_device_code` writes the verification URI and user code to stdout."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        await CliOAuthInteraction().show_device_code(
            verification_uri="https://github.com/login/device",
            user_code="ABCD-1234",
            expires_in=900,
        )
        out = capsys.readouterr().out
        assert "https://github.com/login/device" in out
        assert "ABCD-1234" in out
        assert "900" in out
        assert capsys.readouterr().err == ""

    async def test_cli_show_success_prints_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`show_success` writes to stdout, not stderr."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        await CliOAuthInteraction().show_success("Logged in to MCP server 'test'.")
        captured = capsys.readouterr()
        assert "Logged in to MCP server 'test'" in captured.out
        assert captured.err == ""

    async def test_cli_show_notice_prints_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`show_notice` writes to stdout, not stderr."""
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        await CliOAuthInteraction().show_notice("Falling back to paste-back flow.")
        captured = capsys.readouterr()
        assert "Falling back to paste-back flow." in captured.out
        assert captured.err == ""
