"""Tests for the `/auth` prompt and manager screens."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, OptionList, Static

from deepagents_code import auth_store
from deepagents_code.widgets.auth import AuthManagerScreen, AuthPromptScreen, AuthResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the credential store into a temp directory."""
    state_dir = tmp_path / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)
    return state_dir


class _AuthHostApp(App[None]):
    """Minimal host app for pushing the auth screens."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_result: AuthResult | None = None
        self.prompt_dismissed = False

    def compose(self) -> ComposeResult:
        """Render a placeholder root."""
        yield Container(id="main")

    def show_prompt(
        self, provider: str, env_var: str | None, *, reason: str | None = None
    ) -> None:
        """Push the prompt and capture the dismissal result."""

        def handle(result: AuthResult | None) -> None:
            self.prompt_result = result
            self.prompt_dismissed = True

        self.push_screen(AuthPromptScreen(provider, env_var, reason=reason), handle)

    def show_manager(self) -> None:
        """Push the manager screen."""
        self.push_screen(AuthManagerScreen())


@pytest.mark.usefixtures("fake_state_dir")
class TestAuthPromptScreen:
    """Behavioral tests for the API-key prompt."""

    async def test_input_is_password_masked(self) -> None:
        """The key input is masked so the secret never echoes."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            assert app.screen.query_one("#auth-prompt-input", Input).password is True

    async def test_paste_and_submit_persists(self) -> None:
        """Submitting a non-empty value writes to the store and dismisses True."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            inp = app.screen.query_one("#auth-prompt-input", Input)
            inp.value = "sk-ant-test-12345"
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_dismissed is True
        assert app.prompt_result is AuthResult.SAVED
        assert auth_store.get_stored_key("anthropic") == "sk-ant-test-12345"

    async def test_empty_submit_shows_error_and_does_not_dismiss(self) -> None:
        """Empty input renders an inline error instead of dismissing."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            err = app.screen.query_one("#auth-prompt-error", Static)
            assert "cannot be empty" in str(err.content)
        assert app.prompt_dismissed is False
        assert auth_store.get_stored_key("anthropic") is None

    async def test_escape_cancels(self) -> None:
        """Escape dismisses with `CANCELLED` and writes nothing."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            inp = app.screen.query_one("#auth-prompt-input", Input)
            inp.value = "should-not-be-saved"
            await pilot.press("escape")
            await pilot.pause()
        assert app.prompt_dismissed is True
        assert app.prompt_result is AuthResult.CANCELLED
        assert auth_store.get_stored_key("openai") is None

    async def test_ctrl_d_opens_confirm_then_deletes(self) -> None:
        """Ctrl+D opens the confirmation modal; Enter completes the delete."""
        from deepagents_code.widgets.auth import DeleteCredentialConfirmScreen

        auth_store.set_stored_key("openai", "to-be-removed")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert isinstance(app.screen, DeleteCredentialConfirmScreen)
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_dismissed is True
        assert app.prompt_result is AuthResult.DELETED
        assert auth_store.get_stored_key("openai") is None

    async def test_ctrl_d_then_escape_keeps_credential(self) -> None:
        """Esc on the confirm modal returns to the prompt without deleting."""
        from deepagents_code.widgets.auth import DeleteCredentialConfirmScreen

        auth_store.set_stored_key("openai", "still-here")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert isinstance(app.screen, DeleteCredentialConfirmScreen)
            await pilot.press("escape")
            await pilot.pause()
        assert app.prompt_dismissed is False
        assert auth_store.get_stored_key("openai") == "still-here"

    async def test_ctrl_d_noop_without_existing_credential(self) -> None:
        """Ctrl+D does nothing when there's no stored key to delete."""
        from deepagents_code.widgets.auth import (
            AuthPromptScreen,
            DeleteCredentialConfirmScreen,
        )

        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            # Stay on the prompt — no confirm modal pushed.
            assert not isinstance(app.screen, DeleteCredentialConfirmScreen)
            assert isinstance(app.screen, AuthPromptScreen)

    async def test_title_shows_stored_when_existing(self) -> None:
        """Title surfaces a `(stored)` marker when a key already exists."""
        auth_store.set_stored_key("anthropic", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            title = app.screen.query_one(".auth-prompt-title", Static)
            assert "stored" in str(title.content)

    async def test_title_omits_stored_when_no_credential(self) -> None:
        """Title doesn't claim a stored key when one doesn't exist."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            title = app.screen.query_one(".auth-prompt-title", Static)
            assert "stored" not in str(title.content)

    async def test_init_does_not_crash_on_corrupt_store(
        self, fake_state_dir: Path
    ) -> None:
        """A corrupt auth.json must not crash the prompt at construction."""
        path = fake_state_dir / "auth.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            # Pushing must not raise; the screen should mount and show
            # an inline warning instead.
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            assert isinstance(app.screen, AuthPromptScreen)
            error_widgets = app.screen.query(".auth-prompt-error")
            warning_text = " ".join(str(w.render()) for w in error_widgets)
            assert "unreadable" in warning_text

    async def test_helper_text_mentions_env_var(self) -> None:
        """Helper text shows the canonical env-var name as a hint."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            meta = app.screen.query_one(".auth-prompt-meta", Static)
            assert "OPENAI_API_KEY" in str(meta.content)

    async def test_no_logging_of_secret(self, caplog: pytest.LogCaptureFixture) -> None:
        """Submitting a key never lands its value in widget logs."""
        secret = "sk-do-not-log-zzz"
        app = _AuthHostApp()
        with caplog.at_level("DEBUG"):
            async with app.run_test() as pilot:
                app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
                await pilot.pause()
                inp = app.screen.query_one("#auth-prompt-input", Input)
                inp.value = secret
                await pilot.press("enter")
                await pilot.pause()
        for record in caplog.records:
            assert secret not in record.getMessage()


@pytest.mark.usefixtures("fake_state_dir")
class TestAuthManagerScreen:
    """Behavioral tests for the manager listing."""

    async def test_lists_known_providers(self) -> None:
        """Every well-known provider appears in the option list."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
        assert "anthropic" in ids
        assert "openai" in ids

    async def test_stored_provider_shows_stored_badge(self) -> None:
        """Stored providers render a `[stored]` badge in their option label."""
        auth_store.set_stored_key("openai", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label: Any = None
            for i in range(options.option_count):
                opt = options.get_option_at_index(i)
                if opt.id == "openai":
                    label = opt.prompt
                    break
        assert label is not None
        assert "stored" in str(label)

    async def test_env_badge_shows_canonical_when_only_canonical_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canonical env var only → label shows the canonical name."""
        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "openai"
            )
        assert "[env: OPENAI_API_KEY]" in label

    async def test_env_badge_shows_prefixed_when_prefixed_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixed env var present → label shows the prefixed name."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "from-prefix")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "openai"
            )
        assert "[env: DEEPAGENTS_CODE_OPENAI_API_KEY]" in label

    async def test_env_badge_prefers_prefixed_when_both_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both set → label shows the prefixed variant (matches resolve order)."""
        monkeypatch.setenv("OPENAI_API_KEY", "canonical")
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "openai"
            )
        assert "[env: DEEPAGENTS_CODE_OPENAI_API_KEY]" in label

    async def test_only_installed_well_known_providers_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hardcoded providers without an installed package are hidden.

        When `openai` is "installed", `openai_codex` rides along — it shares
        the same `langchain-openai` package, so the manager surfaces the
        OAuth-backed twin alongside the API-key entry.
        """
        # Pretend only `openai` and `anthropic` are installed.
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
        assert ids == {"openai", "openai_codex", "anthropic"}

    async def test_stored_provider_shown_even_when_uninstalled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored credential remains visible after its package is uninstalled.

        Lets the user clean up stale credentials without reinstalling the
        provider's LangChain package first.
        """
        auth_store.set_stored_key("groq", "k")
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
        assert "groq" in ids
        assert "openai" in ids

    async def test_description_includes_docs_link(self) -> None:
        """The manager description carries a clickable link to providers docs."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            copy = app.screen.query_one(".auth-manager-copy", Static)
            content = str(copy.content)
        assert "Lists installed providers" in content
        assert "Docs" in content
        # URL is embedded as a Textual link style — assert the link target
        # surfaces in the rendered span representation.
        assert "providers" in repr(copy.content) or "providers" in content

    async def test_footer_lists_full_action_set(self) -> None:
        """Footer mentions add/replace/delete (delete happens via the prompt)."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            help_text = app.screen.query_one(".auth-manager-help", Static)
        assert "add/replace/delete" in str(help_text.content)

    async def test_corrupt_store_surfaces_warning_banner(
        self, fake_state_dir: Path
    ) -> None:
        """A corrupt auth.json shows a visible banner in the manager."""
        path = fake_state_dir / "auth.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            warnings = app.screen.query(".auth-manager-warning")
            assert warnings, "expected a corruption warning banner to render"
            text = " ".join(str(w.render()) for w in warnings)
        assert "unreadable" in text


class TestCodexAuthInManager:
    """`/auth` -> `openai_codex` routes to the OAuth screen, not the API key prompt.

    These tests cover the dispatch in `AuthManagerScreen` itself; the
    behavior of the OAuth flow (PKCE, callback, token exchange) is covered
    by `test_openai_codex_integration.py` so we don't repeat the network /
    fake-`webbrowser` plumbing here.
    """

    async def test_codex_option_visible_when_openai_installed(self) -> None:
        """`langchain-openai` is a hard dep, so `openai_codex` is always shown."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
        assert "openai_codex" in ids

    async def test_codex_badge_reflects_signed_out_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing token store renders the "[sign in to chatgpt]" badge."""
        import langchain_openai.chatgpt_oauth as o

        from deepagents_code.model_config import clear_caches

        monkeypatch.setattr(o, "DEFAULT_STORE_PATH", tmp_path / "missing.json")
        clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            target = None
            for i in range(options.option_count):
                opt = options.get_option_at_index(i)
                if opt.id == "openai_codex":
                    target = opt
                    break
        assert target is not None
        assert "sign in to chatgpt" in str(target.prompt).lower()

    async def test_codex_selection_pushes_oauth_screen_when_signed_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Choosing `openai_codex` while signed out opens the OAuth modal."""
        import langchain_openai.chatgpt_oauth as o

        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.codex_auth import CodexAuthScreen

        # Stub the OAuth flow so the modal does not try to bind a real
        # loopback port or run a token exchange when it mounts.
        async def _fake_run(  # noqa: RUF029  # async signature dictated by protocol
            *_args: object, **_kwargs: object
        ) -> codex_integration.CodexAuthStatus:
            return codex_integration.CodexAuthStatus(
                logged_in=False, store_path=tmp_path / "missing.json"
            )

        monkeypatch.setattr(codex_integration, "run_browser_login", _fake_run)

        monkeypatch.setattr(o, "DEFAULT_STORE_PATH", tmp_path / "missing.json")
        clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            target_index: int | None = None
            for i in range(options.option_count):
                if options.get_option_at_index(i).id == "openai_codex":
                    target_index = i
                    break
            assert target_index is not None
            options.highlighted = target_index
            # We just need to observe that the screen is pushed *before* the
            # fake worker finishes; capture the screen class via the
            # `screen_stack` instead of asserting on `app.screen` (which the
            # fast fake worker may have already popped).
            pushed: list[type] = []
            original = app.push_screen

            def _capture(screen, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                pushed.append(type(screen))
                return original(screen, *args, **kwargs)

            monkeypatch.setattr(app, "push_screen", _capture)
            await pilot.press("enter")
            await pilot.pause()
        assert CodexAuthScreen in pushed

    async def test_codex_selection_when_signed_in_shows_signout_overlay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A logged-in user sees the sign-out / re-auth overlay instead."""
        import json
        from datetime import datetime, timedelta

        import langchain_openai.chatgpt_oauth as o

        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.auth import CodexSignedInScreen

        path = tmp_path / "auth.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "access_token": "fake",
                    "refresh_token": "fake",
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    "account_id": "acct",
                    "plan_type": "plus",
                    "user_id": "u",
                    "id_token": None,
                }
            )
        )
        path.chmod(0o600)
        monkeypatch.setattr(o, "DEFAULT_STORE_PATH", path)
        clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            target_index: int | None = None
            for i in range(options.option_count):
                if options.get_option_at_index(i).id == "openai_codex":
                    target_index = i
                    break
            assert target_index is not None
            options.highlighted = target_index
            pushed: list[type] = []
            original = app.push_screen

            def _capture(screen, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                pushed.append(type(screen))
                return original(screen, *args, **kwargs)

            monkeypatch.setattr(app, "push_screen", _capture)
            await pilot.press("enter")
            await pilot.pause()
        assert CodexSignedInScreen in pushed
