"""Tests for offering a server restart when a Tavily key is saved via `/auth`.

`web_search` is bound only when Tavily is configured at server spawn time
(see `server_graph._build_tools`), so a key added to an already-running server
takes effect only after a respawn. Saving a Tavily key in the `/auth` manager
should flag — and, once the manager closes, offer — that restart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from deepagents_code.app import DeepAgentsApp

if TYPE_CHECKING:
    import pytest


def _fake_tavily_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `apply_stored_service_credentials` export a Tavily key to the env."""

    def _apply() -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    monkeypatch.setattr(
        "deepagents_code.model_config.apply_stored_service_credentials",
        _apply,
    )


class TestNoteWebSearchRestart:
    """`_note_web_search_restart_if_needed` gating."""

    def test_flags_restart_when_running_server_lacks_tavily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A saved Tavily key on a Tavily-less running server arms the offer."""
        from deepagents_code.config import settings

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is True

    def test_ignores_non_tavily_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model-provider keys don't gate a spawn-time tool, so no offer."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("openai")

        assert app._pending_web_search_restart is False

    def test_skips_when_server_already_has_tavily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server spawned with Tavily already bound `web_search`."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "tvly-existing")

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False

    def test_skips_when_no_owned_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no owned subprocess there is nothing to respawn."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        app._server_proc = None
        app._server_kwargs = None

        app._note_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False


class TestOfferRestartForWebSearch:
    """`_offer_restart_for_web_search` messaging and restart dispatch."""

    async def test_no_owned_server_recommends_relaunch(self) -> None:
        """A remote/not-owned server can't be `/restart`ed — recommend relaunch."""
        app = DeepAgentsApp()
        app._server_proc = None
        app._server_kwargs = None
        app._mount_message = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        contents = " ".join(
            str(c.args[0]._content)
            for c in app._mount_message.await_args_list  # ty: ignore
        )
        assert "Relaunch dcode" in contents
        assert "web search" in contents
        assert "/restart" not in contents

    async def test_busy_recommends_restart(self) -> None:
        """An owned-but-busy server points at `/restart`, never a relaunch."""
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = True
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        contents = " ".join(
            str(c.args[0]._content)
            for c in app._mount_message.await_args_list  # ty: ignore
        )
        assert "/restart" in contents
        assert "relaunch" not in contents.lower()

    async def test_restart_choice_dispatches_restart(self) -> None:
        """Choosing restart respawns the owned server via `_restart_after_install`."""
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock(return_value="restart")  # ty: ignore
        app._restart_after_install = AsyncMock(return_value=True)  # ty: ignore

        await app._offer_restart_for_web_search()

        app._restart_after_install.assert_awaited_once_with(  # ty: ignore
            "Tavily API key"
        )

    async def test_restart_state_flip_surfaces_fallback(self) -> None:
        """A chosen restart that can't run surfaces a web-search fallback hint."""
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock(return_value="restart")  # ty: ignore
        app._restart_after_install = AsyncMock(return_value=False)  # ty: ignore

        await app._offer_restart_for_web_search()

        contents = " ".join(
            str(c.args[0]._content)
            for c in app._mount_message.await_args_list  # ty: ignore
        )
        assert "Couldn't restart the server automatically to enable web" in contents


class TestCredentialSavedHandler:
    """The `/auth` credential-saved handler threads the service through."""

    async def test_saved_tavily_key_arms_restart_offer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving Tavily flags the offer without waiting for the manager to close."""
        from deepagents_code.config import settings
        from deepagents_code.tui.widgets.auth import AuthManagerScreen

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._server_proc = MagicMock()
            app._server_kwargs = {"model_name": "anthropic:fake"}
            app._resume_server_after_auth_change = AsyncMock()  # ty: ignore

            app.on_auth_manager_screen_credential_saved(
                AuthManagerScreen.CredentialSaved("tavily")
            )
            await pilot.pause()

            assert app._pending_web_search_restart is True
