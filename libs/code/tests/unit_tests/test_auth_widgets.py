"""Tests for the `/auth` prompt and manager screens."""

from __future__ import annotations

import asyncio
from datetime import UTC
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, OptionList, Static

from deepagents_code import auth_store, model_config
from deepagents_code.config import get_glyphs
from deepagents_code.widgets.auth import (
    PROVIDER_API_KEY_URLS,
    PROVIDER_DISPLAY_NAMES,
    AuthManagerScreen,
    AuthPromptScreen,
    AuthResult,
    _is_safe_acquisition_url,
    _provider_display_name,
)
from deepagents_code.widgets.codex_auth import CodexAuthScreen

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from textual.content import Content


@pytest.fixture(autouse=True)
def _restore_model_caches() -> Iterator[None]:
    """Reset model-config caches after tests that repoint `DEFAULT_CONFIG_PATH`.

    A few tests patch the config path to isolate base-URL resolution; clearing
    on teardown stops their throwaway config from leaking into later tests via
    the cached singleton.
    """
    yield
    model_config.clear_caches()


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
        self.credential_saved_count = 0

    def compose(self) -> ComposeResult:
        """Render a placeholder root."""
        yield Container(id="main")

    def show_prompt(
        self,
        provider: str,
        env_var: str | None,
        *,
        reason: str | None = None,
        allow_empty_submit: bool = False,
        input_placeholder: str | None = None,
        submit_label: str | None = None,
    ) -> None:
        """Push the prompt and capture the dismissal result."""

        def handle(result: AuthResult | None) -> None:
            self.prompt_result = result
            self.prompt_dismissed = True

        self.push_screen(
            AuthPromptScreen(
                provider,
                env_var,
                reason=reason,
                allow_empty_submit=allow_empty_submit,
                input_placeholder=input_placeholder,
                submit_label=submit_label,
            ),
            handle,
        )

    def show_manager(self, *, initial_provider: str | None = None) -> None:
        """Push the manager screen."""
        self.push_screen(AuthManagerScreen(initial_provider=initial_provider))

    def on_auth_manager_screen_credential_saved(
        self, _event: AuthManagerScreen.CredentialSaved
    ) -> None:
        """Record credential-save notifications from the manager."""
        self.credential_saved_count += 1


class TestCodexAuthScreen:
    """Behavioral tests for ChatGPT sign-in modal helpers."""

    def test_link_hover_uses_pointer(self) -> None:
        """Hovering the inline authorize link sets a pointer cursor."""
        screen = CodexAuthScreen()
        link_event = SimpleNamespace(style=SimpleNamespace(link="https://example.com"))
        screen.on_mouse_move(link_event)  # ty: ignore[invalid-argument-type]
        assert screen.styles.pointer == "pointer"
        screen.on_leave()
        assert screen.styles.pointer == "default"


@pytest.mark.usefixtures("fake_state_dir")
class TestAuthPromptScreen:
    """Behavioral tests for the API-key prompt."""

    def test_provider_key_urls_point_at_login_or_key_pages(self) -> None:
        """Built-in key links route users directly to current provider key pages."""
        assert (
            PROVIDER_API_KEY_URLS["anthropic"]
            == "https://platform.claude.com/login?returnTo=%2Fsettings%2Fkeys"
        )
        assert (
            PROVIDER_API_KEY_URLS["cohere"]
            == "https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys"
        )
        assert (
            PROVIDER_API_KEY_URLS["baseten"]
            == "https://docs.baseten.co/organization/api-keys"
        )
        assert (
            PROVIDER_API_KEY_URLS["fireworks"]
            == "https://app.fireworks.ai/settings/users/api-keys"
        )
        assert PROVIDER_API_KEY_URLS["google_genai"] == (
            "https://aistudio.google.com/api-keys"
        )
        assert PROVIDER_API_KEY_URLS["huggingface"] == (
            "https://huggingface.co/login?next=%2Fsettings%2Ftokens"
        )
        assert PROVIDER_API_KEY_URLS["ibm"] == "https://cloud.ibm.com/iam/apikeys"
        assert (
            PROVIDER_API_KEY_URLS["litellm"]
            == "https://docs.litellm.ai/docs/proxy/virtual_keys"
        )
        assert (
            PROVIDER_API_KEY_URLS["openrouter"]
            == "https://openrouter.ai/workspaces/default/keys"
        )

    def test_provider_metadata_maps_reference_known_providers(self) -> None:
        """Map keys stay in sync with real providers so none silently never resolve."""
        # Services (e.g. LangSmith tracing) may carry a display name and key URL
        # but live in SERVICE_API_KEY_ENV rather than PROVIDER_API_KEY_ENV.
        known = (
            set(model_config.PROVIDER_API_KEY_ENV)
            | set(model_config.SERVICE_API_KEY_ENV)
            | {model_config.CODEX_PROVIDER}
        )
        assert set(PROVIDER_DISPLAY_NAMES) <= known
        # Codex uses ChatGPT login, not an API-key page, so it has no key URL.
        assert set(PROVIDER_API_KEY_URLS) <= (
            set(model_config.PROVIDER_API_KEY_ENV)
            | set(model_config.SERVICE_API_KEY_ENV)
        )

    def test_langsmith_service_has_display_name_and_key_url(self) -> None:
        """LangSmith tracing surfaces a branded label and a key-acquisition URL."""
        assert PROVIDER_DISPLAY_NAMES[model_config.LANGSMITH_SERVICE] == (
            "LangSmith (tracing)"
        )
        assert PROVIDER_API_KEY_URLS[model_config.LANGSMITH_SERVICE] == (
            "https://smith.langchain.com/settings"
        )

    def test_every_known_provider_has_a_display_name(self) -> None:
        """A new provider can't ship without a branded `/auth` label.

        The title-cased fallback would otherwise mask the omission silently, so
        this pins the reverse direction: every canonical provider must appear in
        `PROVIDER_DISPLAY_NAMES`.
        """
        assert set(model_config.PROVIDER_API_KEY_ENV) <= set(PROVIDER_DISPLAY_NAMES)

    def test_providers_without_key_url_are_intentionally_omitted(self) -> None:
        """Only providers with no self-serve key page may skip `PROVIDER_API_KEY_URLS`.

        `azure_openai` keys live on a per-resource page (special-cased in the
        instructions) and `google_vertexai` uses application-default
        credentials rather than an API-key page. Any other omission is an
        oversight that should fail here rather than ship a generic docs link.
        """
        no_self_serve_key_page = {"azure_openai", "google_vertexai"}
        missing = set(model_config.PROVIDER_API_KEY_ENV) - set(PROVIDER_API_KEY_URLS)
        assert missing == no_self_serve_key_page

    def test_display_name_falls_back_to_title_cased_key(self) -> None:
        """Unmapped provider keys degrade to a readable title-cased label."""
        assert _provider_display_name("acme_gateway") == "Acme Gateway"

    def test_config_display_name_overrides_builtin_map(self) -> None:
        """A configured `display_name` wins over the built-in label.

        Pins the resolution order (config > built-in map) so reordering the
        `or` chain in `_provider_display_name` would fail a test instead of
        silently shadowing user config.
        """
        config = model_config.ModelConfig(
            providers={"openai": {"display_name": "Custom OpenAI"}}
        )
        assert _provider_display_name("openai", config) == "Custom OpenAI"
        # Without the override, resolution falls through to the built-in map.
        empty = model_config.ModelConfig()
        builtin_label = PROVIDER_DISPLAY_NAMES["openai"]
        assert _provider_display_name("openai", empty) == builtin_label

    def test_is_safe_acquisition_url_rejects_non_http_schemes(self) -> None:
        """Only http/https URLs are eligible to render as clickable links."""
        assert _is_safe_acquisition_url("https://example.com/keys") is True
        assert _is_safe_acquisition_url("http://example.com/keys") is True
        assert _is_safe_acquisition_url("javascript:alert(1)") is False
        assert _is_safe_acquisition_url("file:///etc/passwd") is False
        assert _is_safe_acquisition_url("not a url") is False

    async def test_input_is_password_masked(self) -> None:
        """The key input is masked so the secret never echoes."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            assert app.screen.query_one("#auth-prompt-input", Input).password is True

    async def test_provider_instructions_precede_advanced_details(self) -> None:
        """The default prompt starts with acquisition guidance, not env precedence."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            storage_note = app.screen.query_one("#auth-prompt-storage-note", Static)
            key_meta = app.screen.query_one("#auth-prompt-key-meta", Static)
            base_url_label = app.screen.query_one("#auth-prompt-base-url-label", Static)
            base_url = app.screen.query_one("#auth-prompt-base-url", Input)
            toggle = app.screen.query_one("#auth-prompt-advanced-toggle", Static)
            assert "Sign in to Anthropic" in str(instructions.content)
            assert "create or copy an API key" in str(instructions.content)
            assert "Anthropic key page" in str(instructions.content)
            assert "Deep Agents Code stores the above key locally" in str(
                storage_note.content
            )
            assert "Advanced (F2)" in str(toggle.content)
            assert base_url_label.display is False
            assert "Base URL override" in str(base_url_label.content)
            assert base_url.placeholder == "Base URL"
            assert key_meta.display is False
            assert base_url.display is False

    async def test_env_resolved_key_is_called_out_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opening a provider with env auth shows the current credential source."""
        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            title = app.screen.query_one(".auth-prompt-title", Static)
            status = app.screen.query_one("#auth-prompt-env-status", Static)
            assert get_glyphs().checkmark in str(title.content)
            assert "Replace key for OpenAI" in str(title.content)
            assert "set from environment variable OPENAI_API_KEY" in str(status.content)
            assert "use a different key for Deep Agents Code" in str(status.content)
            content = cast("Content", status.content)
            assert any("$success" in str(span.style) for span in content.spans)

    async def test_prefixed_env_key_status_notes_it_is_already_scoped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefixed env auth doesn't invite users to scope the key again."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "from-prefixed")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            title = app.screen.query_one(".auth-prompt-title", Static)
            status = app.screen.query_one("#auth-prompt-env-status", Static)
            text = str(status.content)
            assert get_glyphs().warning in str(title.content)
            assert "DEEPAGENTS_CODE_OPENAI_API_KEY" in text
            assert "scoped env var takes priority" in text
            assert "saved key will be used only when" in text
            assert "DEEPAGENTS_CODE_OPENAI_API_KEY is unset" in text
            assert "use a different key" not in text
            content = cast("Content", status.content)
            assert any("$warning" in str(span.style) for span in content.spans)

    async def test_prefixed_env_key_status_shows_when_stored_key_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scoped env override is shown even when stored auth exists."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "from-prefixed")
        auth_store.set_stored_key("openai", "from-store")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            title = app.screen.query_one(".auth-prompt-title", Static)
            status = app.screen.query_one("#auth-prompt-env-status", Static)
            text = str(status.content)
            assert get_glyphs().warning in str(title.content)
            assert "Replace key for OpenAI (stored)" in str(title.content)
            assert "DEEPAGENTS_CODE_OPENAI_API_KEY" in text
            assert "scoped env var takes priority" in text
            assert "saved key will be used only when" in text

    async def test_azure_instructions_name_resource_keys_page(self) -> None:
        """Azure keys live on a resource-specific Keys and Endpoint page."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("azure_openai", "AZURE_OPENAI_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "Find your key in your Azure OpenAI resource" in text
            assert "Keys and Endpoint page" in text
            assert "Sign in to Azure OpenAI" not in text

    async def test_openai_instructions_name_minimum_key_permissions(self) -> None:
        """OpenAI keys note the minimum Restricted scopes Deep Agents Code needs."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "Sign in to OpenAI" in text
            assert "create or copy an API key" in text
            assert "Minimum permissions needed" in text
            assert "under Model capabilities" in text
            assert "Write access to Responses (/v1/responses)" in text
            assert "For older models" in text
            assert "Request access to Chat completions (/v1/chat/completions)" in text

    async def test_anthropic_instructions_warn_against_subscription_plans(
        self,
    ) -> None:
        """Anthropic keys note subscription plans don't work for API calls."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "Sign in to Anthropic" in text
            assert "create or copy an API key" in text
            assert "Subscription plans (Claude Pro/Max, Claude Code) cannot be" in text
            assert "pay-as-you-go billing" in text

    async def test_provider_instructions_use_config_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configured arbitrary providers can customize auth instructions."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
display_name = "My Gateway"
api_key_url = "https://gateway.example/keys"
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("my_gateway", "MY_GATEWAY_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            assert "Sign in to My Gateway" in str(instructions.content)
            assert "My Gateway key page" in str(instructions.content)

    async def test_unmapped_provider_links_to_setup_docs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Providers with no key page point at setup docs, not a key link."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("my_gateway", "MY_GATEWAY_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "Sign in to My Gateway" in text
            assert "My Gateway setup docs" in text
            assert "key page" not in text

    async def test_unsafe_configured_key_url_is_not_rendered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-http `api_key_url` is dropped rather than rendered as a link."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
display_name = "My Gateway"
api_key_url = "javascript:alert(1)"
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("my_gateway", "MY_GATEWAY_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            content = cast("Content", instructions.content)
            assert "javascript:" not in str(content)
            # The malformed URL is dropped, so no clickable link survives.
            assert not any("javascript" in str(span.style) for span in content.spans)
            # With the configured URL rejected and no built-in entry, it falls
            # back to the setup-docs link.
            assert "My Gateway setup docs" in str(content)

    async def test_unsafe_configured_key_url_surfaces_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected `api_key_url` is surfaced in the UI, not just the log."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
display_name = "My Gateway"
api_key_url = "javascript:alert(1)"
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("my_gateway", "MY_GATEWAY_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "configured api_key_url was ignored" in text
            assert "unsupported URL scheme" in text

    async def test_anthropic_notice_and_rejected_url_notice_coexist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider-specific and rejected-URL notices both render when both fire."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic]
api_key_url = "javascript:alert(1)"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            instructions = app.screen.query_one("#auth-prompt-key-instructions", Static)
            text = str(instructions.content)
            assert "Subscription plans (Claude Pro/Max, Claude Code) cannot be" in text
            assert "configured api_key_url was ignored" in text

    async def test_f2_toggles_advanced_details(self) -> None:
        """Advanced endpoint and env-var details are hidden behind F2."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            await pilot.press("f2")
            await pilot.pause()
            assert app.screen.query_one("#auth-prompt-key-meta", Static).display is True
            assert (
                app.screen.query_one("#auth-prompt-base-url-label", Static).display
                is True
            )
            assert app.screen.query_one("#auth-prompt-base-url", Input).display is True
            await pilot.press("f2")
            await pilot.pause()
            key_meta = app.screen.query_one("#auth-prompt-key-meta", Static)
            base_url_label = app.screen.query_one("#auth-prompt-base-url-label", Static)
            base_url = app.screen.query_one("#auth-prompt-base-url", Input)
            assert key_meta.display is False
            assert base_url_label.display is False
            assert base_url.display is False

    async def test_mouse_move_without_widget_does_not_crash(self) -> None:
        """Mouse moves over empty modal space still reset the pointer safely."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            event = SimpleNamespace(style=SimpleNamespace(link=None), widget=None)
            screen = app.screen
            if not isinstance(screen, AuthPromptScreen):
                pytest.fail(f"expected AuthPromptScreen, got {type(screen).__name__}")
            screen.on_mouse_move(event)  # ty: ignore[invalid-argument-type]
            assert screen.styles.pointer == "default"

    async def test_click_toggles_advanced_details(self) -> None:
        """Clicking the disclosure row opens Advanced settings."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            await pilot.click("#auth-prompt-advanced-toggle")
            await pilot.pause()
            assert app.screen.query_one("#auth-prompt-key-meta", Static).display is True
            assert (
                app.screen.query_one("#auth-prompt-base-url-label", Static).display
                is True
            )
            assert app.screen.query_one("#auth-prompt-base-url", Input).display is True

    async def test_collapsing_advanced_restores_key_input_focus(self) -> None:
        """Collapsing Advanced returns focus to the key input for fast entry."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            await pilot.press("f2")  # expand
            await pilot.pause()
            await pilot.press("f2")  # collapse
            await pilot.pause()
            key_input = app.screen.query_one("#auth-prompt-input", Input)
            assert app.screen.focused is key_input

    async def test_link_and_toggle_hover_use_pointer(self) -> None:
        """A pointer shows over links and the Advanced row; leaving resets it."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("anthropic", "ANTHROPIC_API_KEY")
            await pilot.pause()
            screen = app.screen
            if not isinstance(screen, AuthPromptScreen):
                pytest.fail(f"expected AuthPromptScreen, got {type(screen).__name__}")
            toggle = screen.query_one("#auth-prompt-advanced-toggle", Static)
            link_event = SimpleNamespace(
                style=SimpleNamespace(link="https://example.com"), widget=None
            )
            screen.on_mouse_move(link_event)  # ty: ignore[invalid-argument-type]
            assert screen.styles.pointer == "pointer"
            toggle_event = SimpleNamespace(
                style=SimpleNamespace(link=None), widget=toggle
            )
            screen.on_mouse_move(toggle_event)  # ty: ignore[invalid-argument-type]
            assert screen.styles.pointer == "pointer"
            screen.on_leave()
            assert screen.styles.pointer == "default"

    async def test_existing_base_url_expands_advanced_by_default(self) -> None:
        """Stored endpoint values remain visible when replacing a key."""
        auth_store.set_stored_key(
            "openai", "sk-existing", base_url="https://proxy.example/v1"
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            base_url_label = app.screen.query_one("#auth-prompt-base-url-label", Static)
            base_url = app.screen.query_one("#auth-prompt-base-url", Input)
            assert base_url_label.display is True
            assert base_url.display is True
            assert base_url.value == "https://proxy.example/v1"

    async def test_existing_project_expands_advanced_by_default(self) -> None:
        """A stored LangSmith project keeps the advanced section visible."""
        auth_store.set_stored_key("langsmith", "lsv2_existing", project="my-app")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("langsmith", "LANGSMITH_API_KEY")
            await pilot.pause()
            project_label = app.screen.query_one("#auth-prompt-project-label", Static)
            project_input = app.screen.query_one("#auth-prompt-project", Input)
            assert project_label.display is True
            assert project_input.display is True
            assert project_input.value == "my-app"

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

    async def test_langsmith_submit_applies_tracing_env_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving LangSmith auth activates tracing in the running process."""
        import os

        for var in (
            "LANGSMITH_API_KEY",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_PROJECT",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
            "DEEPAGENTS_CODE_LANGSMITH_TRACING",
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
        ):
            monkeypatch.delenv(var, raising=False)

        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("langsmith", "LANGSMITH_API_KEY")
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "lsv2_live"
            await pilot.press("enter")
            await pilot.pause()

        assert app.prompt_result is AuthResult.SAVED
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_live"
        assert os.environ["LANGSMITH_TRACING"] == "true"

    async def test_base_url_round_trips_on_submit(self) -> None:
        """A base URL typed alongside the key is persisted as the pair."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "sk-key"
            app.screen.query_one(
                "#auth-prompt-base-url", Input
            ).value = "  https://proxy.example/v1  "
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_result is AuthResult.SAVED
        assert auth_store.get_stored_key("openai") == "sk-key"
        # Whitespace is stripped before storage.
        assert auth_store.get_stored_base_url("openai") == "https://proxy.example/v1"

    async def test_submit_from_base_url_field_saves_pair(self) -> None:
        """Enter in the base-URL field saves the pair, not just the key field.

        `on_input_submitted` reads both inputs regardless of which one fired, so
        submitting from either field must persist the same key + endpoint.
        """
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            await pilot.press("f2")
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "sk-key"
            base_url_field = app.screen.query_one("#auth-prompt-base-url", Input)
            base_url_field.value = "https://proxy.example/v1"
            base_url_field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_result is AuthResult.SAVED
        assert auth_store.get_stored_key("openai") == "sk-key"
        assert auth_store.get_stored_base_url("openai") == "https://proxy.example/v1"

    async def test_blank_base_url_field_stores_no_endpoint(self) -> None:
        """A whitespace-only base URL stores nothing (uses the provider default)."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "sk-key"
            app.screen.query_one("#auth-prompt-base-url", Input).value = "   "
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_result is AuthResult.SAVED
        assert auth_store.get_stored_base_url("openai") is None

    async def test_existing_base_url_prefills_field(self) -> None:
        """Reopening the prompt pre-fills the stored endpoint for editing."""
        auth_store.set_stored_key("openai", "k", base_url="https://stored.example/v1")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            base_url_field = app.screen.query_one("#auth-prompt-base-url", Input)
            assert base_url_field.value == "https://stored.example/v1"

    async def test_langsmith_prompt_saves_key_and_project(self) -> None:
        """The LangSmith prompt persists a key plus its custom project name."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("langsmith", "LANGSMITH_API_KEY")
            await pilot.pause()
            await pilot.press("f2")
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "lsv2_test"
            project_field = app.screen.query_one("#auth-prompt-project", Input)
            project_field.value = "my-app"
            project_field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        assert app.prompt_result is AuthResult.SAVED
        assert auth_store.get_stored_key("langsmith") == "lsv2_test"
        assert auth_store.get_stored_project("langsmith") == "my-app"

    async def test_langsmith_prompt_has_no_base_url_field(self) -> None:
        """The LangSmith prompt swaps the base-URL field for the project field."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("langsmith", "LANGSMITH_API_KEY")
            await pilot.pause()
            assert not app.screen.query("#auth-prompt-base-url")
            assert app.screen.query("#auth-prompt-project")

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

    async def test_optional_empty_submit_cancels_without_error(self) -> None:
        """Onboarding can use the auth prompt as an optional setup step."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt(
                "tavily",
                "TAVILY_API_KEY",
                allow_empty_submit=True,
                input_placeholder="Tavily API key (optional)",
                submit_label="Enter save/skip",
            )
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert app.prompt_dismissed is True
        assert app.prompt_result is AuthResult.CANCELLED
        assert auth_store.get_stored_key("tavily") is None

    async def test_optional_prompt_customizes_new_user_copy(self) -> None:
        """Onboarding can explain Tavily without a separate key-entry modal."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt(
                "tavily",
                "TAVILY_API_KEY",
                reason="Web search is optional. Press Enter to skip.",
                allow_empty_submit=True,
                input_placeholder="Tavily API key (optional)",
                submit_label="Enter save/skip",
            )
            await pilot.pause()

            key_input = app.screen.query_one("#auth-prompt-input", Input)
            help_text = app.screen.query_one(".auth-prompt-help", Static)
            copy = "\n".join(str(widget.content) for widget in app.screen.query(Static))
            has_storage_note = bool(app.screen.query("#auth-prompt-storage-note"))

        assert key_input.placeholder == "Tavily API key (optional)"
        assert key_input.password is True
        assert "Web search is optional" in copy
        assert "Enter save/skip" in str(help_text.content)
        # Services (Tavily) omit the storage note entirely — the title and
        # reason already explain the key, so it would only be redundant copy.
        assert has_storage_note is False
        assert "stores the above key locally" not in copy

    async def test_optional_prompt_surfaces_save_failure_and_stays_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed credential write shows an inline error and keeps the modal."""

        def _raise(*_args: object, **_kwargs: object) -> auth_store.WriteOutcome:
            msg = "credential store is not writable"
            raise RuntimeError(msg)

        monkeypatch.setattr(auth_store, "set_stored_key", _raise)

        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("tavily", "TAVILY_API_KEY", allow_empty_submit=True)
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "tvly-key"
            await pilot.press("enter")
            await pilot.pause()
            err = app.screen.query_one("#auth-prompt-error", Static)
            error_text = str(err.content)

        # Surfaced in-modal, not silently swallowed, and the modal stays open so
        # onboarding cannot proceed as if the key were saved.
        assert "Could not save credential" in error_text
        assert "credential store is not writable" in error_text
        assert app.prompt_dismissed is False

    async def test_optional_prompt_notifies_store_warnings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chmod-style warnings from the store reach the user via `notify`."""
        notices: list[tuple[str, str | None]] = []

        def _capture_notify(
            message: str, *_args: object, severity: str | None = None, **_kwargs: object
        ) -> None:
            notices.append((str(message), severity))

        def _warn(*_args: object, **_kwargs: object) -> auth_store.WriteOutcome:
            return auth_store.WriteOutcome(warnings=("credential file is not private",))

        monkeypatch.setattr(auth_store, "set_stored_key", _warn)

        app = _AuthHostApp()
        async with app.run_test() as pilot:
            monkeypatch.setattr(app, "notify", _capture_notify)
            app.show_prompt("tavily", "TAVILY_API_KEY", allow_empty_submit=True)
            await pilot.pause()
            app.screen.query_one("#auth-prompt-input", Input).value = "tvly-key"
            await pilot.press("enter")
            await pilot.pause()

        assert ("credential file is not private", "warning") in notices
        assert app.prompt_result is AuthResult.SAVED

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

    async def test_ctrl_d_quits_without_existing_credential(self) -> None:
        """Ctrl+D falls through to quit when there's no stored key to delete.

        The `priority` binding would otherwise swallow the app-level
        Ctrl+D=quit, leaving the key dead in the modal.
        """
        from deepagents_code.widgets.auth import DeleteCredentialConfirmScreen

        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            # No confirm modal — there's nothing to delete.
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert not isinstance(app.screen, DeleteCredentialConfirmScreen)
            # The key fell through to quit instead of being swallowed.
            assert app._exit is True

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
            # The MISSING fallback renders no env-source block and an unprefixed
            # title (no warning/checkmark glyph), confirming the cosmetic-only
            # degradation the construction guard promises.
            assert not app.screen.query("#auth-prompt-env-status")
            title_text = str(app.screen.query_one(".auth-prompt-title", Static).content)
            glyphs = get_glyphs()
            assert glyphs.warning not in title_text
            assert glyphs.checkmark not in title_text

    async def test_helper_text_describes_precedence(self) -> None:
        """Helper text names both env vars and their order vs the stored key.

        A stored key sits between the plain var (which it beats) and the
        `DEEPAGENTS_CODE_`-prefixed var (which beats it). The meta line must
        convey that ordering, not imply the three are interchangeable.
        """
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            meta = app.screen.query_one("#auth-prompt-key-meta", Static)
            text = str(meta.content)
            assert "Deep Agents Code stores" not in text
            assert (
                "Alternatively, environment variables can be used in place "
                "of the key stored above." in text
            )
            assert "DEEPAGENTS_CODE_OPENAI_API_KEY" in text
            assert "Deep Agents Code-only key" in text
            assert "highest priority" in text
            assert "OPENAI_API_KEY" in text
            assert "share a key with other provider SDK tools" in text
            assert "used only when no scoped or stored key exists" in text
            assert "Configuration docs" in text

    async def test_base_url_hint_names_endpoint_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a known endpoint var but no survivor set, name it as a hint."""
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml")
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            hint = app.screen.query_one("#auth-prompt-base-url-hint", Static)
            text = str(hint.content)
            assert "Override the provider endpoint for this stored key" in text
            assert "Leave blank to use the provider default" in text
            assert "Env override order" in text
            assert "DEEPAGENTS_CODE_OPENAI_BASE_URL, then OPENAI_BASE_URL" in text
            # It must not claim blank *uses* the plain var (it gets cleared).
            assert "use OPENAI_BASE_URL" not in text

    async def test_base_url_hint_generic_without_endpoint_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider with no base-URL env var falls back to the generic line."""
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml")
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            # `google_vertexai` has an API-key env var but no base-URL mapping.
            app.show_prompt("google_vertexai", "GOOGLE_CLOUD_PROJECT")
            await pilot.pause()
            hint = app.screen.query_one("#auth-prompt-base-url-hint", Static)
            text = str(hint.content)
            assert "Override the provider endpoint for this stored key" in text
            assert "Leave blank to use the provider default" in text
            assert "Env override order" not in text

    async def test_base_url_hint_names_surviving_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The surviving env var is named (not its value) so blank is unambiguous."""
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml")
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_OPENAI_BASE_URL", "https://scoped.example/v1"
        )
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("openai", "OPENAI_API_KEY")
            await pilot.pause()
            hint = app.screen.query_one("#auth-prompt-base-url-hint", Static)
            text = str(hint.content)
            assert "Override the provider endpoint for this stored key" in text
            assert "Leave blank to use DEEPAGENTS_CODE_OPENAI_BASE_URL" in text
            assert "Env override order" in text
            # The URL value itself is not leaked into the hint.
            assert "scoped.example" not in text

    async def test_base_url_hint_names_surviving_alternate_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The blank-endpoint hint includes alternate env vars that still resolve."""
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "none.toml")
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_BASETEN_API_BASE", "https://legacy.example/v1"
        )
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_prompt("baseten", "BASETEN_API_KEY")
            await pilot.pause()
            hint = app.screen.query_one("#auth-prompt-base-url-hint", Static)
            text = str(hint.content)
            assert "Leave blank to use DEEPAGENTS_CODE_BASETEN_API_BASE" in text
            assert "DEEPAGENTS_CODE_BASETEN_BASE_URL, then BASETEN_BASE_URL" in text
            assert "DEEPAGENTS_CODE_BASETEN_API_BASE, then BASETEN_API_BASE" in text
            assert "legacy.example" not in text

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

    async def test_prompt_save_posts_credential_saved_event(self) -> None:
        """Saving a key notifies the app before the manager itself closes."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)

            screen._on_prompt_closed(AuthResult.SAVED)
            await pilot.pause()

        assert app.credential_saved_count == 1

    async def test_prompt_cancel_does_not_post_credential_saved_event(self) -> None:
        """Cancelling or deleting credentials should not trigger startup recovery."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)

            screen._on_prompt_closed(AuthResult.CANCELLED)
            screen._on_prompt_closed(AuthResult.DELETED)
            await pilot.pause()

        assert app.credential_saved_count == 0

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
        if "openai_codex" in ids:
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "openai_codex"
            )
            assert "OpenAI Codex (ChatGPT login)" in label

    async def test_configured_provider_uses_display_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configured display names appear in the auth manager list."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.my_gateway]
display_name = "My Gateway"
models = ["my-model"]
api_key_env = "MY_GATEWAY_API_KEY"
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "my_gateway"
            )
        assert "My Gateway" in label
        assert "my_gateway" not in label

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

    async def test_configured_providers_sort_to_top(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Providers with a credential float above unconfigured ones.

        `openai` sorts after `anthropic` alphabetically, but storing a key for
        it should lift it to the top of the installed-provider group while the
        rest stay in alphabetical order.
        """
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        auth_store.set_stored_key("openai", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
        assert ids.index("openai") < ids.index("anthropic")

    async def test_configured_services_sort_with_configured_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configured services float above missing provider rows."""
        for var in (
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "DEEPAGENTS_CODE_TAVILY_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        monkeypatch.setenv("TAVILY_API_KEY", "from-env")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
        assert ids.index("tavily") < ids.index("anthropic")

    async def test_uninstalled_providers_stay_below_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Greyed-out install-on-select entries sort after every installed row.

        An uninstalled provider routes to an install prompt rather than a key
        entry, so it stays at the bottom even when an installed-but-unconfigured
        provider (`anthropic`) sorts below a configured one (`openai`). The
        unconfigured installed row makes the float load-bearing: a list that
        ignored the credential float would order `anthropic` before `openai`.
        `groq` stays last because uninstalled extras are appended after the
        whole manageable block, regardless of the float.
        """
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "DEEPAGENTS_CODE_GROQ_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider != "groq",
        )
        auth_store.set_stored_key("openai", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
        assert "groq" in ids
        # Configured openai floats above unconfigured anthropic, and the
        # uninstalled groq extra stays below both.
        assert ids.index("openai") < ids.index("anthropic") < ids.index("groq")

    async def test_refresh_options_resorts_after_saving_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving a key re-floats its provider on the next options refresh.

        The float is only useful if the live save/clear path re-sorts; this
        exercises `_refresh_options` rather than the initial render. `openai`
        starts below `anthropic` alphabetically and must overtake it once a key
        is stored.
        """
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)

            def current_ids() -> list[str | None]:
                return [
                    options.get_option_at_index(i).id
                    for i in range(options.option_count)
                ]

            before = current_ids()
            assert before.index("anthropic") < before.index("openai")

            auth_store.set_stored_key("openai", "k")
            screen = cast("AuthManagerScreen", app.screen)
            screen._refresh_options()  # exercise the live re-sort path
            await pilot.pause()
            after = current_ids()
        assert after.index("openai") < after.index("anthropic")

    async def test_configured_group_preserves_alphabetical_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Within each group, entries keep alphabetical order.

        The sort key is `(0|1, name)`, so the name tiebreaker must keep both the
        configured block and the unconfigured block alphabetical. A regression
        that dropped the name element (e.g. returned a bare flag) would scramble
        order within a group while still floating configured entries.
        """
        for var in (
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "COHERE_API_KEY",
            "DEEPAGENTS_CODE_COHERE_API_KEY",
            "MISTRAL_API_KEY",
            "DEEPAGENTS_CODE_MISTRAL_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {
                "openai": ["gpt-5.4"],
                "anthropic": ["claude-opus-4-7"],
                "cohere": ["command"],
                "mistralai": ["mistral-large"],
            },
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        auth_store.set_stored_key("anthropic", "k")
        auth_store.set_stored_key("openai", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
        # Configured group stays alphabetical, unconfigured group stays
        # alphabetical, and the whole configured block sits above the rest.
        assert ids.index("anthropic") < ids.index("openai")
        assert ids.index("openai") < ids.index("cohere")
        assert ids.index("cohere") < ids.index("mistralai")

    async def test_env_configured_provider_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider configured only via an env var floats like a stored key.

        The sort keys off resolved auth status, not the stored-key set, so an
        env-only credential must float too. `openai` is configured via
        `OPENAI_API_KEY` alone and overtakes unconfigured `anthropic`.
        """
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
            openai_label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "openai"
            )
        assert ids.index("openai") < ids.index("anthropic")
        # The env-resolved status object must be threaded through to the badge,
        # not just the sort key: an env-only credential renders `[env set: …]`.
        assert "env set" in openai_label

    async def test_stored_service_is_not_duplicated(self) -> None:
        """Stored non-model services appear once in the manager list."""
        auth_store.set_stored_key("tavily", "k")
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = [
                options.get_option_at_index(i).id for i in range(options.option_count)
            ]
        assert ids.count("tavily") == 1

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
        assert "[env set: OPENAI_API_KEY]" in label

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
        assert "[env set: DEEPAGENTS_CODE_OPENAI_API_KEY]" in label

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
        assert "[env set: DEEPAGENTS_CODE_OPENAI_API_KEY]" in label

    async def test_only_installed_well_known_providers_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installed providers show as live rows; uninstalled ones grey out.

        When `openai` is "installed", `openai_codex` rides along — it shares
        the same `langchain-openai` package, so the manager surfaces the
        OAuth-backed twin alongside the API-key entry. With every known
        package reported installed, no greyed install-on-select rows appear,
        so the listing equals the installed set.
        """
        # Pretend only `openai` and `anthropic` are installed.
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        # Report every known package installed so no greyed-out
        # install-on-select rows are appended to the listing.
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: True,
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
        # Non-model services (Tavily search, LangSmith tracing) are always
        # listed for key entry.
        assert ids == {
            "openai",
            "openai_codex",
            "anthropic",
            "tavily",
            "langsmith",
        }

    async def test_selecting_service_opens_prompt_for_its_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting a service routes to the key prompt bound to its env var.

        Services must not fall through to the model-provider branch, which
        would look up a credential env var the service doesn't have.
        """
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            event = SimpleNamespace(option=SimpleNamespace(id="tavily"))
            screen.on_option_list_option_selected(cast("Any", event))
            await pilot.pause()
            prompt = app.screen
            assert isinstance(prompt, AuthPromptScreen)
            assert prompt._provider == "tavily"
            assert prompt._env_var == "TAVILY_API_KEY"

    async def test_service_row_shows_stored_badge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored service key renders the `[stored]` badge on its row.

        Confirms the service-aware status branch in `_format_label` is wired
        up — a regression to `get_provider_auth_status` would `KeyError`.
        """
        auth_store.set_stored_key("tavily", "k")
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            options = app.screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "tavily"
            )
        assert "stored" in label

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

    async def test_uninstalled_known_provider_shown_greyed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A known provider whose package is missing appears, greyed out.

        Only `openai`/`anthropic` are installed and `groq` reports no package,
        so `groq` is surfaced as a `[not installed]` install-on-select entry
        for discoverability rather than being hidden.
        """
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider in {"openai", "anthropic"},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            options = screen.query_one("#auth-manager-options", OptionList)
            label = next(
                str(options.get_option_at_index(i).prompt)
                for i in range(options.option_count)
                if options.get_option_at_index(i).id == "groq"
            )
            install_extras = dict(screen._install_extras)
        assert "not installed" in label
        assert install_extras.get("groq") == "groq"

    async def test_selecting_uninstalled_provider_prompts_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting a greyed-out provider opens the install confirmation."""
        from deepagents_code.widgets.install_confirm import (
            InstallProviderConfirmScreen,
        )

        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider == "openai",
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            event = SimpleNamespace(option=SimpleNamespace(id="groq"))
            screen.on_option_list_option_selected(cast("Any", event))
            await pilot.pause()
            assert isinstance(app.screen, InstallProviderConfirmScreen)

    async def test_confirming_install_records_extra_and_dismisses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirming the install records the extra and dismisses the manager."""
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider == "openai",
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            screen._prompt_install_provider("groq", "groq")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        assert screen.pending_install_extra == "groq"
        assert screen.pending_install_provider == "groq"

    async def test_reopening_manager_highlights_initial_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reopening after an install highlights the just-installed provider.

        Simulates the post-install reopen: `groq` is now installed and listed,
        and passing it as `initial_provider` lands the cursor on its row rather
        than resetting to index 0.
        """
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "groq": ["llama-3"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider in {"openai", "groq"},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager(initial_provider="groq")
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            options = screen.query_one("#auth-manager-options", OptionList)
            assert options.highlighted is not None
            assert options.get_option_at_index(options.highlighted).id == "groq"

    async def test_reopening_manager_ignores_unknown_initial_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An initial provider absent from the list leaves the cursor at the top."""
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider == "openai",
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager(initial_provider="not-a-real-provider")
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            options = screen.query_one("#auth-manager-options", OptionList)
            assert options.highlighted == 0

    async def test_cancelling_install_leaves_manager_without_pending_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declining the install records nothing and keeps the user on the manager."""
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider == "openai",
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            screen._prompt_install_provider("groq", "groq")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert screen.pending_install_extra is None
            assert screen.pending_install_provider is None
            assert isinstance(app.screen, AuthManagerScreen)

    async def test_disabled_known_provider_not_offered_for_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider disabled in config is neither shown nor install-on-select.

        `groq` reports no package and would normally surface greyed-out, but an
        explicit `enabled = false` in config keeps it out of the listing
        entirely — the same gate the model switcher honors.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.groq]
enabled = false
""")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"], "anthropic": ["claude-opus-4-7"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider in {"openai", "anthropic"},
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            options = screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
            install_extras = dict(screen._install_extras)
        assert "groq" not in install_extras
        assert "groq" not in ids

    async def test_known_provider_without_extra_not_offered_for_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A known provider with no curated extra is skipped, not greyed-out.

        Without a `provider_install_extra` mapping there is nothing to install,
        so the provider must not appear as an install-on-select row pointing at
        a `None` extra.
        """
        monkeypatch.setattr(
            "deepagents_code.widgets.auth.get_available_models",
            lambda: {"openai": ["gpt-5.4"]},
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda provider: provider != "groq",
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.provider_install_extra",
            lambda provider: None if provider == "groq" else provider,
        )
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            screen = cast("AuthManagerScreen", app.screen)
            options = screen.query_one("#auth-manager-options", OptionList)
            ids = {
                options.get_option_at_index(i).id for i in range(options.option_count)
            }
            install_extras = dict(screen._install_extras)
        assert "groq" not in install_extras
        assert "groq" not in ids

    async def test_description_includes_docs_link(self) -> None:
        """The manager description carries a clickable link to providers docs."""
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            copy = app.screen.query_one(".auth-manager-copy", Static)
            content = str(copy.content)
        assert "Lists installed model providers" in content
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
        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.model_config import clear_caches

        monkeypatch.setattr(
            codex_integration, "default_store_path", lambda: tmp_path / "missing.json"
        )
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

        monkeypatch.setattr(
            codex_integration, "default_store_path", lambda: tmp_path / "missing.json"
        )
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

    async def test_codex_oauth_cancel_dismisses_modal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Esc after the OAuth worker starts dismisses the modal as cancelled."""
        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.widgets.codex_auth import CodexAuthScreen

        async def _fake_run(
            *_args: object, **_kwargs: object
        ) -> codex_integration.CodexAuthStatus:
            await asyncio.Event().wait()
            return codex_integration.CodexAuthStatus(
                logged_in=True, store_path=tmp_path / "auth.json"
            )

        monkeypatch.setattr(codex_integration, "run_browser_login", _fake_run)

        results: list[bool | None] = []
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.push_screen(CodexAuthScreen(), results.append)
            await pilot.pause()
            await pilot.press("escape")
            for _ in range(5):
                await pilot.pause()
                if results:
                    break
        assert results == [False]

    async def test_codex_selection_when_signed_in_shows_signout_overlay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A logged-in user sees the sign-out / re-auth overlay instead."""
        import json
        from datetime import datetime, timedelta

        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.codex_auth import CodexSignedInScreen

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
        monkeypatch.setattr(codex_integration, "default_store_path", lambda: path)
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

    @staticmethod
    def _write_token(path: Path) -> None:
        """Plant a valid (unexpired) token bundle at `path` with 0600 perms."""
        import json
        from datetime import datetime, timedelta

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
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    async def test_signout_dispatch_deletes_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`SIGN_OUT` from the overlay deletes the stored token on disk."""
        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.codex_auth import CodexSignedInAction

        path = tmp_path / "auth.json"
        self._write_token(path)
        monkeypatch.setattr(codex_integration, "default_store_path", lambda: path)
        clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            manager = cast("AuthManagerScreen", app.screen)
            manager._on_codex_signed_in_closed(CodexSignedInAction.SIGN_OUT)
            await pilot.pause()
        assert not path.exists()

    async def test_reauth_dispatch_pushes_oauth_screen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`REAUTH` from the overlay opens a fresh sign-in flow."""
        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.codex_auth import (
            CodexAuthScreen,
            CodexSignedInAction,
        )

        store = tmp_path / "missing.json"

        async def _fake_run(  # noqa: RUF029  # async signature dictated by protocol
            *_args: object, **_kwargs: object
        ) -> codex_integration.CodexAuthStatus:
            return codex_integration.CodexAuthStatus(logged_in=False, store_path=store)

        monkeypatch.setattr(codex_integration, "run_browser_login", _fake_run)
        monkeypatch.setattr(codex_integration, "default_store_path", lambda: store)
        clear_caches()
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            app.show_manager()
            await pilot.pause()
            manager = cast("AuthManagerScreen", app.screen)
            pushed: list[type] = []
            original = app.push_screen

            def _capture(screen, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                pushed.append(type(screen))
                return original(screen, *args, **kwargs)

            monkeypatch.setattr(app, "push_screen", _capture)
            manager._on_codex_signed_in_closed(CodexSignedInAction.REAUTH)
            await pilot.pause()
        assert CodexAuthScreen in pushed

    async def test_codex_oauth_success_dismisses_true_with_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker SUCCESS path dismisses `True` and toasts the plan."""
        from datetime import datetime, timedelta

        from deepagents_code.integrations import openai_codex as codex_integration
        from deepagents_code.widgets.codex_auth import CodexAuthScreen

        async def _fake_run(  # noqa: RUF029  # async signature dictated by protocol
            *_args: object, **_kwargs: object
        ) -> codex_integration.CodexAuthStatus:
            return codex_integration.CodexAuthStatus(
                logged_in=True,
                store_path=tmp_path / "auth.json",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                plan_type="plus",
                account_id="acct",
            )

        monkeypatch.setattr(codex_integration, "run_browser_login", _fake_run)

        results: list[bool | None] = []
        notices: list[str] = []
        app = _AuthHostApp()
        async with app.run_test() as pilot:
            original_notify = app.notify

            def _capture_notify(message, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                notices.append(str(message))
                return original_notify(message, *args, **kwargs)

            monkeypatch.setattr(app, "notify", _capture_notify)
            app.push_screen(CodexAuthScreen(), results.append)
            for _ in range(10):
                await pilot.pause()
                if results:
                    break
        assert results == [True]
        assert any("plus" in note for note in notices)
