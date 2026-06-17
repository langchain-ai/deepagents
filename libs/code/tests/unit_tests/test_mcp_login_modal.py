"""Tests for the in-TUI MCP OAuth login modal."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from deepagents_code.mcp_oauth_ui import OAuthInteraction
from deepagents_code.widgets.mcp_login import MCPLoginCancelledError, MCPLoginScreen


class _LoginTestApp(App[None]):
    """Minimal app wrapper for testing `MCPLoginScreen`."""

    def compose(self) -> ComposeResult:
        yield Static("base")


async def _wait_for_prompt(screen: MCPLoginScreen) -> None:
    """Poll until the screen's input row is displayed.

    The login screen only shows its input widget once an interaction
    method like `request_callback_url` has run far enough to flip
    `display = True`. Tests need to wait for that to inject values; a
    short polling loop with a 2s deadline is sufficient because the
    production code that flips the flag runs synchronously inside
    `_await_input`.
    """
    async with asyncio.timeout(2.0):
        while True:
            try:
                visible = screen.query_one("#ml-input", Input).display
            except Exception:  # noqa: BLE001  # screen may still be mounting
                visible = False
            if visible:
                return
            await asyncio.sleep(0.01)  # production polls UI state


def test_mcp_login_screen_implements_oauth_interaction_protocol() -> None:
    """`MCPLoginScreen` satisfies the `OAuthInteraction` Protocol."""
    screen = MCPLoginScreen("notion")
    protocol_methods = [
        "show_authorize_url",
        "request_callback_url",
        "show_device_code",
        "show_success",
        "show_notice",
        "show_error",
    ]
    for method in protocol_methods:
        assert callable(getattr(screen, method, None)), (
            f"MCPLoginScreen missing protocol method: {method}"
        )


def test_mcp_login_screen_has_no_slack_team_prompt() -> None:
    """`MCPLoginScreen` does not implement `prompt_slack_team_id`; Slack's browser page picks the workspace."""  # noqa: E501
    screen = MCPLoginScreen("notion")
    assert not hasattr(screen, "prompt_slack_team_id")


class TestMCPLoginScreen:
    """Behavior tests that drive the modal through `pilot`."""

    async def test_callback_url_round_trip(self) -> None:
        """Typing a callback URL into the input resolves `request_callback_url`."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://auth.example/", opened_in_browser=True
            )

            async def submit_after_delay() -> None:
                await _wait_for_prompt(screen)
                input_widget = screen.query_one("#ml-input", Input)
                value = "https://localhost/?code=abc&state=xyz"
                input_widget.value = value
                input_widget.post_message(Input.Submitted(input_widget, value))

            submit_task = asyncio.create_task(submit_after_delay())
            result = await screen.request_callback_url()
            await submit_task
            assert result == "https://localhost/?code=abc&state=xyz"

    async def test_escape_cancels_pending_prompt(self) -> None:
        """Pressing Escape mid-prompt raises `RuntimeError` for the worker."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            async def press_escape_after_delay() -> None:
                await _wait_for_prompt(screen)
                await pilot.press("escape")

            press_task = asyncio.create_task(press_escape_after_delay())
            with pytest.raises(MCPLoginCancelledError, match="cancelled"):
                await screen.request_callback_url()
            await press_task

    async def test_show_authorize_url_does_not_block(self) -> None:
        """`show_authorize_url` is non-blocking and updates the visible link."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=False
            )
            await pilot.pause()
            # The link widget should be visible and contain the URL.
            assert screen._link_widget is not None
            assert screen._link_widget.display
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "https://example.test/auth" in link_text

    async def test_browser_opened_authorize_url_is_collapsed(self) -> None:
        """The happy path hides the raw authorize URL behind manual fallback."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("slack")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=True
            )
            await pilot.pause()

            assert screen._link_widget is not None
            assert screen._link_widget.display
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "Show manual authorization URL" in link_text
            assert "https://example.test/auth" not in link_text

    async def test_enter_toggles_browser_opened_authorize_url(self) -> None:
        """Enter expands and collapses the manual authorization URL fallback."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("slack")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=True
            )
            await pilot.press("enter")
            await pilot.pause()

            assert screen._link_widget is not None
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "Hide manual authorization URL" in link_text
            assert "https://example.test/auth" in link_text

            await pilot.press("enter")
            await pilot.pause()
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "Show manual authorization URL" in link_text
            assert "https://example.test/auth" not in link_text

    async def test_browser_wait_spinner_animates_status_not_title(self) -> None:
        """The browser-wait spinner advances in the status line, not the title."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("slack")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=True
            )
            assert screen._status_widget is not None
            assert screen._title_widget is not None
            status_before = str(screen._status_widget._Static__content)  # ty: ignore
            title_before = str(screen._title_widget._Static__content)  # ty: ignore

            screen._tick_spinner()
            status_after = str(screen._status_widget._Static__content)  # ty: ignore
            title_after = str(screen._title_widget._Static__content)  # ty: ignore

            assert status_before != status_after
            assert "Status:" in status_after
            assert title_after == title_before

    async def test_device_code_renders_user_code(self) -> None:
        """`show_device_code` displays the user code and verification URL."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("github")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_device_code(
                verification_uri="https://github.com/login/device",
                user_code="ABCD-1234",
                expires_in=900,
            )
            await pilot.pause()
            assert screen._link_widget is not None
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "https://github.com/login/device" in link_text
            assert "ABCD-1234" in link_text

    async def test_show_success_does_not_leak_tokens(self) -> None:
        """Tokens passed by mistake stay only in the status line."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            # Caller is responsible for not leaking; we assert the modal
            # does not also paste the message into the authorize-URL slot.
            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=True
            )
            await screen.show_success("Logged in to 'notion'.")
            await pilot.pause()
            assert screen._link_widget is not None
            assert not screen._link_widget.display
            link_text = str(screen._link_widget._Static__content)  # ty: ignore
            assert "Logged in" not in link_text
            assert screen._history_widget is not None
            assert not screen._history_widget.display

    async def test_finish_success_hides_history_and_auth_fallback(self) -> None:
        """Clean success leaves only the final status message visible."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("langsmith")
            app.push_screen(screen)
            await pilot.pause()

            await screen.show_authorize_url(
                "https://example.test/auth", opened_in_browser=True
            )
            await screen.show_success("Logged in to MCP server 'langsmith'.")
            screen.finish(
                success=True,
                message=(
                    "Logged in to 'langsmith'. Reconnect required to load new tools."
                ),
            )
            await pilot.pause()

            assert screen._link_widget is not None
            assert not screen._link_widget.display
            assert screen._history_widget is not None
            assert not screen._history_widget.display
            assert screen._status_widget is not None
            status_text = str(screen._status_widget._Static__content)  # ty: ignore
            assert "Reconnect required" in status_text


class TestMCPLoginScreenWithLoginCoroutine:
    """End-to-end: drive `mcp_auth.login` through the modal."""

    async def test_paste_back_login_through_modal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`mcp_auth.login` completes when the user pastes a callback URL.

        Stubs `_drive_handshake` so no network calls happen; the modal
        only prompts for the callback URL (no Slack team ID — the TUI
        defers workspace selection to Slack's browser page).
        """
        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import FileTokenStorage, login

        fake_home = tmp_path
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR",
            fake_home / ".deepagents" / ".state",
        )

        captured_urls: list[str] = []

        async def _fake_handshake(connections: dict) -> None:
            server_name, connection = next(iter(connections.items()))
            provider = connection["auth"]
            await provider.context.redirect_handler(
                "https://slack.com/oauth/v2/authorize?client_id=x"
            )
            code, _state = await provider.context.callback_handler()
            captured_urls.append(code)
            storage = FileTokenStorage(server_name, server_url=connection["url"])
            await storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))

        monkeypatch.setattr(
            "deepagents_code.mcp_auth._drive_handshake", _fake_handshake
        )

        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("slack")
            app.push_screen(screen)
            await pilot.pause()

            async def drive_prompts() -> None:
                # One prompt: callback URL only. No Slack team ID prompt in TUI.
                await _wait_for_prompt(screen)
                input_widget = screen.query_one("#ml-input", Input)
                pending = screen._pending_input
                value = "https://localhost/?code=abc"
                input_widget.value = value
                input_widget.post_message(Input.Submitted(input_widget, value))
                if pending is not None:
                    async with asyncio.timeout(2.0):
                        await pending

            driver = asyncio.create_task(drive_prompts())
            await login(
                server_name="slack",
                server_config={
                    "type": "http",
                    "url": "https://slack.com/mcp",
                    "auth": "oauth",
                },
                ui=screen,
            )
            await driver

        assert captured_urls == ["abc"]


class TestMCPLoginScreenEdgeCases:
    """Guards and idempotency tests for `MCPLoginScreen`."""

    async def test_await_input_raises_when_already_cancelled(self) -> None:
        """Calling a prompt method on an already-cancelled screen raises immediately."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            screen._cancelled = True
            with pytest.raises(MCPLoginCancelledError, match="cancelled before"):
                await screen.request_callback_url()

    async def test_finish_double_call_is_idempotent(self) -> None:
        """A second call to `finish` is a no-op and does not raise."""
        app = _LoginTestApp()
        async with app.run_test() as pilot:
            screen = MCPLoginScreen("notion")
            app.push_screen(screen)
            await pilot.pause()

            screen.finish(success=True, message="Done.")
            # Second call must not raise and must not change the outcome.
            screen.finish(success=False, message="Should be ignored.")
            assert screen._outcome == "success"


class TestMCPLoginPointer:
    """Pointer affordance over links and the manual-URL disclosure row."""

    def test_hover_over_link_uses_pointer(self) -> None:
        """Hovering an inline authorize link sets a pointer cursor."""
        screen = MCPLoginScreen("notion")
        event = SimpleNamespace(
            style=SimpleNamespace(link="https://example.test/auth"), widget=None
        )
        screen.on_mouse_move(event)  # ty: ignore[invalid-argument-type]
        assert screen.styles.pointer == "pointer"

    def test_hover_over_disclosure_row_uses_pointer(self) -> None:
        """Hovering the manual-URL disclosure row sets a pointer cursor."""
        screen = MCPLoginScreen("notion")
        row = Static("Show manual authorization URL")
        screen._link_widget = row
        event = SimpleNamespace(style=SimpleNamespace(link=None), widget=row)
        screen.on_mouse_move(event)  # ty: ignore[invalid-argument-type]
        assert screen.styles.pointer == "pointer"

    def test_hover_elsewhere_uses_default(self) -> None:
        """Hovering neither a link nor the disclosure row keeps the default cursor."""
        screen = MCPLoginScreen("notion")
        screen._link_widget = Static("x")
        event = SimpleNamespace(style=SimpleNamespace(link=None), widget=None)
        screen.on_mouse_move(event)  # ty: ignore[invalid-argument-type]
        assert screen.styles.pointer == "default"

    def test_leave_resets_pointer(self) -> None:
        """Leaving the modal resets the cursor to default."""
        screen = MCPLoginScreen("notion")
        link_event = SimpleNamespace(
            style=SimpleNamespace(link="https://example.test/auth"), widget=None
        )
        screen.on_mouse_move(link_event)  # ty: ignore[invalid-argument-type]
        screen.on_leave()
        assert screen.styles.pointer == "default"
