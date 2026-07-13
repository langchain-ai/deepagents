"""Tests for offering a server restart when a Tavily key is saved via `/auth`.

`web_search` is bound only when Tavily is configured at server spawn time
(see `server_graph._build_tools`), so a key added to an already-running server
takes effect only after a respawn. Saving a Tavily key in the `/auth` manager
should flag — and, once the manager closes, offer — that restart.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code.app import DeepAgentsApp


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

    def test_skips_when_export_produces_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the eager export lands no `TAVILY_API_KEY`, stay disarmed.

        A stored entry can be empty/malformed, in which case
        `apply_stored_service_credentials` exports nothing; without an env key
        a respawn couldn't bind `web_search`, so there is nothing to offer.
        """
        from deepagents_code.config import settings

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(settings, "tavily_api_key", None)
        monkeypatch.setattr(
            "deepagents_code.model_config.apply_stored_service_credentials",
            lambda: None,
        )

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False

    def test_deleted_tavily_key_clears_pending_restart_and_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting Tavily disarms the deferred restart and removes its env bridge."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-deleted")

        app = DeepAgentsApp()
        app._pending_web_search_restart = True
        app._auth_exported_tavily = True
        app._auth_exported_tavily_original = None

        app._clear_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False
        assert "TAVILY_API_KEY" not in os.environ

    def test_deleted_tavily_key_clears_env_after_offer_was_consumed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting Tavily still removes `/auth` export after prompt scheduling."""
        from deepagents_code.config import settings

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("tavily")
        app._maybe_offer_deferred_web_search_restart()
        app._clear_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False
        assert "TAVILY_API_KEY" not in os.environ

    def test_deleted_tavily_key_restores_original_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting Tavily restores a shell key that `/auth` temporarily replaced."""
        from deepagents_code.config import settings

        monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-shell")
        monkeypatch.setattr(settings, "tavily_api_key", None)
        _fake_tavily_export(monkeypatch)

        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}

        app._note_web_search_restart_if_needed("tavily")
        app._maybe_offer_deferred_web_search_restart()
        app._clear_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False
        assert os.environ["TAVILY_API_KEY"] == "tvly-from-shell"

    def test_deleted_non_tavily_key_leaves_pending_restart_and_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model-provider deletes do not affect Tavily restart state."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-still-present")

        app = DeepAgentsApp()
        app._pending_web_search_restart = True

        app._clear_web_search_restart_if_needed("openai")

        assert app._pending_web_search_restart is True
        assert os.environ["TAVILY_API_KEY"] == "tvly-still-present"

    def test_deleted_tavily_key_preserves_env_when_restart_was_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting a stored key must not clear an independent shell env key."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-shell")

        app = DeepAgentsApp()
        app._pending_web_search_restart = False

        app._clear_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False
        assert os.environ["TAVILY_API_KEY"] == "tvly-from-shell"

    def test_deleted_tavily_key_preserves_env_when_only_restart_flag_was_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one-shot prompt flag alone does not prove `/auth` owns the env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-shell")

        app = DeepAgentsApp()
        app._pending_web_search_restart = True

        app._clear_web_search_restart_if_needed("tavily")

        assert app._pending_web_search_restart is False
        assert os.environ["TAVILY_API_KEY"] == "tvly-from-shell"


class TestOfferRestartForWebSearch:
    """`_offer_restart_for_web_search` messaging and restart dispatch.

    Each test pins `settings.tavily_api_key` to `None` so the idempotent
    "already configured" guard never short-circuits the offer under test; the
    guard itself is covered by `test_skips_offer_when_tavily_already_configured`.
    """

    async def test_no_owned_server_recommends_relaunch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remote/not-owned server can't be `/restart`ed — recommend relaunch."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
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

    async def test_busy_recommends_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An owned-but-busy server points at `/restart`, never a relaunch."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
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

    async def test_restart_choice_dispatches_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Choosing restart respawns the owned server via `_restart_after_install`."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
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

    async def test_restart_state_flip_surfaces_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chosen restart that can't run surfaces a web-search fallback hint."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
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

    async def test_skips_offer_when_tavily_already_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A respawn that already bound `web_search` suppresses a redundant offer.

        Guards the install-on-select-in-the-same-session case: the install
        auto-restarted the server (reloading config and rebinding `web_search`),
        so by the time the reopened manager closes there is nothing left to do.
        """
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "tvly-now-configured")
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock()  # ty: ignore
        app._restart_after_install = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        app._push_screen_wait.assert_not_awaited()  # ty: ignore
        app._restart_after_install.assert_not_awaited()  # ty: ignore
        app._mount_message.assert_not_awaited()  # ty: ignore

    async def test_later_choice_shows_no_followup_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deferring at the prompt is an informed choice — no extra hint follows.

        The prompt's own button was the call to action, so the shared helper
        stays silent (neither restarts nor mounts a `/restart` hint) when the
        user picks "later".
        """
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock(return_value="later")  # ty: ignore
        app._restart_after_install = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        app._restart_after_install.assert_not_awaited()  # ty: ignore
        app._mount_message.assert_not_awaited()  # ty: ignore

    async def test_watchdog_timeout_falls_back_to_manual_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt that never resolves is bounded by the watchdog → manual hint."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock(side_effect=TimeoutError())  # ty: ignore
        app._restart_after_install = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        app._restart_after_install.assert_not_awaited()  # ty: ignore
        contents = " ".join(
            str(c.args[0]._content)
            for c in app._mount_message.await_args_list  # ty: ignore
        )
        assert "/restart" in contents
        assert "web search" in contents

    async def test_mount_failure_falls_back_to_manual_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the modal can't be mounted at all, degrade to the manual hint."""
        from deepagents_code.config import settings

        monkeypatch.setattr(settings, "tavily_api_key", None)
        app = DeepAgentsApp()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "anthropic:fake"}
        app._agent_running = False
        app._connecting = False
        app._mount_message = AsyncMock()  # ty: ignore
        app._push_screen_wait = AsyncMock(  # ty: ignore
            side_effect=RuntimeError("stack hijacked")
        )
        app._restart_after_install = AsyncMock()  # ty: ignore

        await app._offer_restart_for_web_search()

        app._restart_after_install.assert_not_awaited()  # ty: ignore
        contents = " ".join(
            str(c.args[0]._content)
            for c in app._mount_message.await_args_list  # ty: ignore
        )
        assert "/restart" in contents
        assert "web search" in contents


class TestCredentialSavedHandler:
    """The `/auth` credential-saved handler threads the provider through."""

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

    async def test_deleted_tavily_key_disarms_restart_offer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting Tavily clears the offer before the manager-close callback runs."""
        from deepagents_code.tui.widgets.auth import AuthManagerScreen

        monkeypatch.setenv("TAVILY_API_KEY", "tvly-deleted")

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._pending_web_search_restart = True
            app._auth_exported_tavily = True
            app._auth_exported_tavily_original = None

            app.on_auth_manager_screen_credential_deleted(
                AuthManagerScreen.CredentialDeleted("tavily")
            )
            await pilot.pause()

            assert app._pending_web_search_restart is False
            assert "TAVILY_API_KEY" not in os.environ

    async def test_resume_is_scheduled_before_web_search_bookkeeping(self) -> None:
        """The credentials-blocked resume is scheduled even if bookkeeping raises.

        The handler kicks off `_resume_server_after_auth_change` (the response
        the user is waiting on) before the secondary web-search bookkeeping, so
        a failure in the latter can never preempt the resume.
        """
        from deepagents_code.tui.widgets.auth import AuthManagerScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._resume_server_after_auth_change = AsyncMock()  # ty: ignore
            app._note_web_search_restart_if_needed = MagicMock(  # ty: ignore
                side_effect=RuntimeError("bookkeeping blew up")
            )

            with pytest.raises(RuntimeError):
                app.on_auth_manager_screen_credential_saved(
                    AuthManagerScreen.CredentialSaved("tavily")
                )
            await pilot.pause()

            app._resume_server_after_auth_change.assert_awaited_once()  # ty: ignore


class TestDeferredOfferAfterManagerCloses:
    """The armed offer is consumed only when the `/auth` manager closes.

    Spies on `call_after_refresh` rather than asserting the offer coroutine
    actually ran: scheduling is synchronous inside the dismiss callback, so it
    is deterministic, whereas the post-refresh invocation is timing-dependent.
    """

    @staticmethod
    def _spy_call_after_refresh(app: DeepAgentsApp) -> list[object]:
        """Record callbacks scheduled via `call_after_refresh`, still delegating."""
        scheduled: list[object] = []
        real = app.call_after_refresh

        def _spy(callback: object, *args: object, **kwargs: object) -> object:
            scheduled.append(callback)
            return real(callback, *args, **kwargs)  # ty: ignore

        app.call_after_refresh = _spy  # ty: ignore
        return scheduled

    async def test_offer_deferred_until_manager_closes(self) -> None:
        """An armed offer is scheduled on manager close — never while it is open."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._resume_server_after_auth_change = AsyncMock()  # ty: ignore
            app._launch_web_search_restart_prompt = MagicMock()  # ty: ignore
            scheduled = self._spy_call_after_refresh(app)
            app._pending_web_search_restart = True

            await app._show_auth_manager()
            await pilot.pause()

            # Manager is open: nothing scheduled and the flag is still armed.
            assert app._launch_web_search_restart_prompt not in scheduled
            assert app._pending_web_search_restart is True

            # Closing the manager (no pending install) consumes the flag once
            # and schedules the offer.
            app.screen.dismiss(None)
            await pilot.pause()

            assert app._pending_web_search_restart is False
            assert app._launch_web_search_restart_prompt in scheduled

    async def test_flag_not_reoffered_on_a_later_unrelated_close(self) -> None:
        """Once consumed, a later manager close does not re-schedule the offer."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._resume_server_after_auth_change = AsyncMock()  # ty: ignore
            app._launch_web_search_restart_prompt = MagicMock()  # ty: ignore
            scheduled = self._spy_call_after_refresh(app)
            app._pending_web_search_restart = True

            await app._show_auth_manager()
            await pilot.pause()
            app.screen.dismiss(None)
            await pilot.pause()
            assert scheduled.count(app._launch_web_search_restart_prompt) == 1

            # A second, unrelated open/close must not re-schedule the offer.
            await app._show_auth_manager()
            await pilot.pause()
            app.screen.dismiss(None)
            await pilot.pause()

            assert scheduled.count(app._launch_web_search_restart_prompt) == 1

    async def test_flag_rides_along_when_close_kicks_off_install(self) -> None:
        """A close that starts a provider install must not consume the offer.

        When the manager closes with a `pending_install_extra`, the app hands
        off to `_install_provider_then_reopen_auth` (which reopens the manager
        or surfaces the offer on its non-reopen exits). Consuming the flag here
        would strand the restart the user armed, so it must ride along untouched.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._resume_server_after_auth_change = AsyncMock()  # ty: ignore
            app._launch_web_search_restart_prompt = MagicMock()  # ty: ignore
            app._install_provider_then_reopen_auth = AsyncMock()  # ty: ignore
            scheduled = self._spy_call_after_refresh(app)
            app._pending_web_search_restart = True

            await app._show_auth_manager()
            await pilot.pause()
            # Simulate the user having confirmed installing an uninstalled
            # provider: the manager records the extra to install on close.
            app.screen.pending_install_extra = "baseten"  # ty: ignore
            app.screen.pending_install_provider = "baseten"  # ty: ignore

            app.screen.dismiss(None)
            await pilot.pause()

            # The flag rides along: neither consumed nor scheduled here.
            assert app._pending_web_search_restart is True
            assert app._launch_web_search_restart_prompt not in scheduled
            app._install_provider_then_reopen_auth.assert_awaited_once()  # ty: ignore
