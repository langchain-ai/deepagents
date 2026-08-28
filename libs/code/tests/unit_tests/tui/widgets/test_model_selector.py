"""Tests for ModelSelectorScreen."""

import asyncio
import tomllib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_code import clipboard as clipboard_module
from deepagents_code._paths import PATHS
from deepagents_code.config import get_glyphs
from deepagents_code.model_config import (
    ModelProfileEntry,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
)
from deepagents_code.tui.widgets.model_selector import (
    MAIN_MODEL_DEFAULT_SCOPE,
    ModelSelectorScreen,
)


@pytest.fixture(autouse=True)
def _seed_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    """Seed credentials so dismissal tests aren't blocked by missing keys.

    The selector now opens an auth prompt when the highlighted provider
    has no key. Most tests in this file just want to assert dismissal
    behavior, so we seed env vars for the providers their fixtures use
    and redirect the credential store into a clean temp dir.

    Also redirects `DEFAULT_CONFIG_PATH` to a (nonexistent) temp file and
    clears the process-wide config cache. The selector now resolves provider
    labels via `ModelConfig.load()`, which defaults to the developer's real
    `~/.deepagents/config.toml`; a local `display_name`/`short_name` override
    would otherwise flip label assertions. This keeps the suite hermetic.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Strip dotenv-loaded prefixed variants so monkeypatched canonical vars
    # win in `resolve_env_var`'s lookup order.
    for var in ("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "DEEPAGENTS_CODE_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
    )
    from deepagents_code import model_config

    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    # The cache is a module global that monkeypatch can't undo; clear it before
    # and after so neither a prior test's real config nor this fixture's empty
    # config leaks across tests.
    model_config.clear_caches()
    yield
    model_config.clear_caches()


_FILTER_TEST_MODELS: list[tuple[str, str]] = [
    ("anthropic:claude-sonnet-4-5", "anthropic"),
    ("anthropic:claude-opus-4-8", "anthropic"),
    ("anthropic:claude-haiku-4-5", "anthropic"),
    ("openai:gpt-4", "openai"),
    ("openai:gpt-5.5", "openai"),
    ("openrouter:anthropic/claude-sonnet-4.7", "openrouter"),
]


def _model_selector_for_filtering() -> ModelSelectorScreen:
    """Create a selector with deterministic model data for filter unit tests."""
    screen = ModelSelectorScreen(
        current_model="claude-sonnet-4-5",
        current_provider="anthropic",
        default_scope=MAIN_MODEL_DEFAULT_SCOPE,
    )
    screen._recommended_only = False
    screen._unfiltered_models = list(_FILTER_TEST_MODELS)
    screen._all_models = list(_FILTER_TEST_MODELS)
    screen._filtered_models = list(_FILTER_TEST_MODELS)
    screen._recent_specs = []
    screen._install_extras = {}
    screen._selected_index = screen._find_current_model_index()
    return screen


class ModelSelectorTestApp(App):
    """Test app for ModelSelectorScreen."""

    def __init__(self) -> None:
        super().__init__()
        self.result: tuple[str, str] | None = None
        self.callback_results: list[tuple[str, str] | None] = []
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the model selector screen.

        Starts in the full-list (`_recommended_only=False`) state so that
        legacy assertions about the full catalog continue to hold. Tests for
        the recommended-only toggle construct their own screen directly.
        """

        def handle_result(result: tuple[str, str] | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ModelSelectorScreen(
            current_model="claude-sonnet-4-5",
            current_provider="anthropic",
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        screen._recommended_only = False
        self.push_screen(screen, handle_result)

    def show_selector_with_result_callback(self) -> None:
        """Show the model selector using its direct result callback."""
        screen = ModelSelectorScreen(
            current_model="claude-sonnet-4-5",
            current_provider="anthropic",
            result_callback=self.callback_results.append,
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        screen._recommended_only = False
        self.push_screen(screen)


class AppWithEscapeBinding(App):
    """Test app that has a conflicting escape binding like DeepAgentsApp.

    This reproduces the real-world scenario where the app binds escape
    to action_interrupt, which would intercept escape before the modal.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.result: tuple[str, str] | None = None
        self.dismissed = False
        self.interrupt_called = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def action_interrupt(self) -> None:
        """Handle escape - dismiss modal if present, otherwise mark as called."""
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.interrupt_called = True

    def show_selector(self) -> None:
        """Show the model selector screen."""

        def handle_result(result: tuple[str, str] | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ModelSelectorScreen(
            current_model="claude-sonnet-4-5",
            current_provider="anthropic",
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        self.push_screen(screen, handle_result)


class TestModelSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    async def test_escape_dismisses_modal(self) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Press ESC - this should dismiss the modal
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None


class TestModelSelectorChrome:
    """Tests for model selector title and description chrome."""

    async def test_curated_selector_help_hides_esc_hint(self) -> None:
        """Onboarding model selection keeps Escape bound but hides its hint."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)

            # Deliberately not the shared `modal_navigation_hint` copy: Tab
            # autocompletes here, so advertising "Tab/Shift+Tab navigate"
            # would misdescribe it. Shift+Tab still works via
            # `_SupportsReverseNav`; it is simply unadvertised.
            assert "navigate" in str(help_text.content)
            assert "Tab/Shift+Tab navigate" not in str(help_text.content)
            assert "Tab autocomplete" in str(help_text.content)
            assert "Esc skip setup" not in str(help_text.content)
            assert "Esc close" not in str(help_text.content)
            assert "Esc cancel" not in str(help_text.content)

    @pytest.mark.parametrize("curated", [False, True])
    async def test_selector_uses_compact_sizing(self, *, curated: bool) -> None:
        """Model selection should size like the integration summary."""
        app = ModelSelectorTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = ModelSelectorScreen(
                curated=curated, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()

            container = screen.query_one(Vertical)
            body = screen.query_one(".model-list", VerticalScroll)
            help_text = screen.query_one(".model-selector-help", Static)

        assert container.region.y >= 0
        assert container.region.y + container.region.height <= app.size.height
        assert help_text.region.y + help_text.region.height <= app.size.height
        max_height = body.styles.max_height
        assert max_height is not None
        assert max_height.cells is not None
        assert max_height.cells <= 16

    async def test_standard_selector_help_shows_close_hint(self) -> None:
        """The regular `/model` selector should advertise Escape dismissal."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)

            assert "Tab autocomplete" in str(help_text.content)
            # Standard mode still advertises the default-setting shortcut that
            # curated/onboarding mode hides.
            assert "Ctrl+S set default" in str(help_text.content)
            assert "Esc close" in str(help_text.content)

    async def test_standard_selector_help_wraps_to_two_rows(self) -> None:
        """The standard footer is wider than the modal, so it must wrap.

        With a clamped one-row `height` the trailing `Ctrl+R recommended`
        hint was clipped off the end; `height: auto` lets it wrap instead.
        """
        app = ModelSelectorTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)

            assert "Ctrl+R recommended" in str(help_text.content)
            # `content` holds the full string even when a one-row clamp clips it
            # off-screen, so the rendered `region.height` is the load-bearing
            # assertion that actually catches the regression.
            assert help_text.region.height >= 2
            assert help_text.region.y + help_text.region.height <= app.size.height


class TestRecommendedToggle:
    """Tests for the Ctrl+R recommended-only toggle in `/model`."""

    async def test_info_line_reflects_active_search(self) -> None:
        """Typing a filter should avoid stale recommended-only copy."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            info = screen.query_one("#model-selector-info", Static)
            assert "Showing recommended models" in str(info.content)

            for char in "gpt":
                await pilot.press(char)
            await pilot.pause()

            assert "Searching all models" in str(info.content)
            assert "Showing recommended models" not in str(info.content)


class TestDefaultModelScope:
    """Tests for which stored preference Ctrl+S writes.

    The selector is shared by `/model` and the `/auto model` classifier picker,
    so a scope mix-up would let the classifier picker retarget the model the
    agent itself runs on.
    """

    @staticmethod
    def _stub_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
        """Reduce the catalog to a single deterministic row."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

    async def test_write_failure_raises_a_toast_naming_the_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The toast is the load-bearing half of the failure UX.

        The footer notice is one short line; the toast carries the remedy and
        stays up for 10s. Nothing else asserts it fires, so a refactor could
        drop it and leave only the footer with no test noticing.

        It must not claim a single cause: the writers return `False` for an
        unwritable file, unparseable TOML, and a `[models]` section of the wrong
        shape alike, so "check permissions" alone sends users to inspect
        permissions that are already correct.
        """
        from deepagents_code.tui.widgets import model_selector

        self._stub_catalog(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                default_scope=model_selector.AUTO_CLASSIFIER_DEFAULT_SCOPE._replace(
                    save=lambda _spec: False
                )
            )
            app.push_screen(screen)
            await pilot.pause()

            notified: list[tuple[str, object]] = []
            monkeypatch.setattr(
                screen,
                "notify",
                lambda msg, **kw: notified.append((msg, kw.get("severity"))),
            )

            await pilot.press("ctrl+s")
            await pilot.pause()

        assert notified
        message, severity = notified[0]
        assert severity == "error"
        assert PATHS.display(PATHS.profile.config_file) in message
        assert "unwritable" in message
        assert "malformed" in message

    async def test_install_refusal_remedy_uses_recovery_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The install-required toast names an environment-correct command.

        Plain `pip install ...` would target the caller's current Python
        environment; for `uv tool` installs (the supported path) that leaves
        the provider unavailable to `dcode`. The remedy must flow through
        `safe_install_extra_recovery_command` like `/install` failures do.
        """
        from deepagents_code.tui.widgets import model_selector
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        monkeypatch.setattr(
            "deepagents_code.update_check.install_extra_recovery_command",
            lambda _extra: "uv tool install deepagents-code --with langchain-anthropic",
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._install_extras = {"anthropic": "anthropic"}

            notified: list[str] = []
            monkeypatch.setattr(screen, "notify", lambda msg, **_: notified.append(msg))

            await pilot.press("ctrl+s")
            await pilot.pause()

        assert notified
        assert (
            "uv tool install deepagents-code --with langchain-anthropic"
            in (notified[0])
        )
        assert "pip install" not in notified[0]

    async def test_ctrl_s_clears_stored_model_with_missing_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Ctrl+S clears a stored spec even when its provider was uninstalled.

        The install-required refusal exists so a spec that can never build is
        not *stored*; applying it to an already-stored row would block the
        only in-app path for removing the now-unusable persisted value.
        """
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        (tmp_path / "config.toml").write_text(
            '[models]\nauto_classifier = "anthropic:claude-sonnet-5"\n',
            encoding="utf-8",
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._install_extras = {"anthropic": "anthropic"}

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Default classifier model cleared" in str(help_widget.content)
            assert screen._default_spec is None

        with (tmp_path / "config.toml").open("rb") as handle:
            data = tomllib.load(handle)
        assert "auto_classifier" not in data["models"]


class TestNamesToggle:
    """Tests for the Ctrl+N names/raw-spec toggle in `/model`."""

    async def test_names_toggle_clobbers_ctrl_s_success_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pending Ctrl+S restore timer must not resurrect a stale hint.

        Ctrl+N and Ctrl+S both write the footer. Toggling while a success
        message is up replaces it immediately, and the message's uncancelled
        3s timer must then restore the hint for the *new* mode — which only
        holds because `_restore_help_text` recomputes rather than replaying a
        string captured when the message was set.
        """
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()
            screen._default_spec = None

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "Default set to" in str(help_widget.content)

            await pilot.press("ctrl+n")
            await pilot.pause()
            assert "Ctrl+N names" in str(help_widget.content)

            # Stand in for the 3s timer the Ctrl+S message left pending.
            screen._restore_help_text()
            assert "Ctrl+N names" in str(help_widget.content)

    async def test_names_toggle_preserves_ctrl_s_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+N must not wipe the only notice that saving the default failed."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            # Inject a failing writer through the screen's scope, since that is
            # the only path `action_set_default` saves through.
            screen = ModelSelectorScreen(
                default_scope=model_selector.MAIN_MODEL_DEFAULT_SCOPE._replace(
                    save=lambda _spec: False
                )
            )
            app.push_screen(screen)
            await pilot.pause()
            screen._default_spec = None

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "Failed to save default" in str(help_widget.content)

            await pilot.press("ctrl+n")
            await pilot.pause()

            # Rows still flip; only the footer refresh is held back.
            assert screen._show_specs
            assert "anthropic:claude-sonnet-5" in str(screen._option_widgets[0].content)
            assert "Failed to save default" in str(help_widget.content)

            screen._restore_help_text()
            assert "Ctrl+N names" in str(help_widget.content)


class TestRecentModelsSection:
    """Tests for the "Recent" pseudo-provider section pinned at the top."""

    async def test_recent_row_uses_short_brand_over_verbose_display_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Recent tag uses the compact brand, not the verbose auth label."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"openai_codex": ["gpt-5.5"]},
        )
        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["openai_codex:gpt-5.5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            recent = screen._option_widgets[0]
            assert recent.model_spec == "openai_codex:gpt-5.5"
            text = str(recent.content)
            assert "(OpenAI Codex)" in text
            # The verbose auth label must not leak into the compact tag.
            assert "Subscription login" not in text


class TestModelSelectorAvailabilityHint:
    """Tests for the API-keys hint shown above the standard model list."""


class TestModelSelectorKeyboardNavigation:
    """Tests for keyboard navigation in the modal."""


class TestModelSelectorAuthRouting:
    """Selecting a credential-less model routes to the right auth modal."""

    @staticmethod
    def _patch_missing_auth(monkeypatch: pytest.MonkeyPatch) -> None:
        """Force every provider to report missing (start-blocking) creds."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_provider_auth_status",
            lambda provider: ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider=provider,
                detail="missing",
            ),
        )

    @staticmethod
    def _capture_pushes(
        monkeypatch: pytest.MonkeyPatch, app: App
    ) -> list[tuple[object, Callable[[bool | None], None] | None]]:
        """Replace `app.push_screen` with a recorder of (screen, callback)."""
        pushed: list[tuple[object, Callable[[bool | None], None] | None]] = []

        def _capture(
            target: object,
            callback: Callable[[bool | None], None] | None = None,
            *_a: object,
            **_k: object,
        ) -> None:
            pushed.append((target, callback))

        monkeypatch.setattr(app, "push_screen", _capture)
        return pushed

    async def test_missing_codex_creds_opens_confirm_not_api_key_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enter on a codex model with no creds opens the sign-in confirm gate.

        `openai_codex` authenticates via ChatGPT OAuth and has no API key, so
        the generic key/base-url `AuthPromptScreen` must not appear. The OAuth
        flow itself is gated behind a confirmation modal, not launched yet.
        """
        from deepagents_code.tui.widgets.auth import AuthConfirmScreen, AuthPromptScreen
        from deepagents_code.tui.widgets.codex_auth import CodexAuthScreen

        self._patch_missing_auth(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            pushed: list[object] = []
            monkeypatch.setattr(
                screen.app,
                "push_screen",
                lambda s, *_a, **_k: pushed.append(s),
            )

            screen._select_with_auth_check("openai_codex:gpt-5.5", "openai_codex")

            assert len(pushed) == 1
            assert isinstance(pushed[0], AuthConfirmScreen)
            assert not isinstance(pushed[0], AuthPromptScreen)
            assert not isinstance(pushed[0], CodexAuthScreen)
            assert app.dismissed is False

    async def test_codex_confirm_proceeds_to_oauth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirming the gate launches the OAuth flow without dismissing."""
        from deepagents_code.tui.widgets.auth import AuthConfirmScreen
        from deepagents_code.tui.widgets.codex_auth import CodexAuthScreen

        self._patch_missing_auth(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            pushed = self._capture_pushes(monkeypatch, screen.app)

            screen._prompt_codex_sign_in("openai_codex:gpt-5.5", "openai_codex")

            assert isinstance(pushed[0][0], AuthConfirmScreen)
            # Simulate the user confirming on the gate.
            on_confirm = pushed[0][1]
            assert on_confirm is not None
            on_confirm(True)
            await pilot.pause()

            assert any(isinstance(s, CodexAuthScreen) for s, _ in pushed)
            assert app.dismissed is False

    async def test_codex_confirm_declined_stays_on_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declining the gate returns to the selector without OAuth."""
        from deepagents_code.tui.widgets.auth import AuthConfirmScreen
        from deepagents_code.tui.widgets.codex_auth import CodexAuthScreen

        self._patch_missing_auth(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            pushed = self._capture_pushes(monkeypatch, screen.app)

            screen._prompt_codex_sign_in("openai_codex:gpt-5.5", "openai_codex")
            assert isinstance(pushed[0][0], AuthConfirmScreen)
            # Simulate the user declining on the gate.
            on_confirm = pushed[0][1]
            assert on_confirm is not None
            on_confirm(False)
            await pilot.pause()

            assert not any(isinstance(s, CodexAuthScreen) for s, _ in pushed)
            assert app.dismissed is False

    async def test_missing_api_key_provider_opens_api_key_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-codex providers still open the API-key/base-url prompt."""
        from deepagents_code.tui.widgets.auth import AuthPromptScreen

        self._patch_missing_auth(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            pushed: list[object] = []
            monkeypatch.setattr(
                screen.app,
                "push_screen",
                lambda s, *_a, **_k: pushed.append(s),
            )

            screen._select_with_auth_check("openai:gpt-5.1", "openai")

            assert len(pushed) == 1
            assert isinstance(pushed[0], AuthPromptScreen)


class TestModelSelectorFiltering:
    """Tests for search filtering."""

    async def test_empty_allowlist_explains_policy(self) -> None:
        """An empty policy shows its cause instead of offering custom models."""
        from deepagents_code import model_config

        model_config.DEFAULT_CONFIG_PATH.write_text(
            "[models]\nallowed = []\n",
            encoding="utf-8",
        )
        model_config.clear_caches()

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            options = screen.query_one("#model-options", Container)
            content = " ".join(str(widget.content) for widget in options.query(Static))
            # An empty filter must not be blamed on a typo, and the message
            # names the policy rather than offering a custom spec.
            assert "models.allowed permits no models" in content
            assert "press Enter" not in content

    async def test_nonspec_filter_is_not_blamed_on_policy(self) -> None:
        """A typo in the filter box is not the administrator's fault.

        `is_model_allowed` rejects any non-spec string, so testing the filter
        text directly would attribute every mistyped filter -- and every empty
        one -- to `models.allowed`.
        """
        from deepagents_code import model_config

        model_config.DEFAULT_CONFIG_PATH.write_text(
            '[models]\nallowed = ["anthropic:claude-sonnet-5"]\n',
            encoding="utf-8",
        )
        model_config.clear_caches()

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            await pilot.press("z", "z", "z", "q")
            await pilot.pause()

            options = screen.query_one("#model-options", Container)
            content = " ".join(str(widget.content) for widget in options.query(Static))
            assert "models.allowed" not in content

    async def test_blocked_spec_filter_names_the_policy_and_allowed_models(
        self,
    ) -> None:
        """Typing a real but blocked spec does name the policy, and what is allowed."""
        from deepagents_code import model_config

        model_config.DEFAULT_CONFIG_PATH.write_text(
            '[models]\nallowed = ["anthropic:claude-sonnet-5"]\n',
            encoding="utf-8",
        )
        model_config.clear_caches()

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            screen._filter_text = "openai:blocked"
            screen._filtered_models = []
            await screen._update_display()
            await pilot.pause()

            options = screen.query_one("#model-options", Container)
            content = " ".join(str(widget.content) for widget in options.query(Static))
            assert "not allowed by the configured models.allowed" in content
            # Naming the permitted specs is the only way out when they are not
            # discoverable, so the empty state must carry them.
            assert "anthropic:claude-sonnet-5" in content

    def test_enter_selects_highlighted_model_not_filter_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enter selects highlighted model, not raw filter text."""
        screen = _model_selector_for_filtering()
        selected: tuple[str, str] | None = None

        def record(model_spec: str, provider: str) -> None:
            nonlocal selected
            selected = (model_spec, provider)

        screen._filter_text = "anthropic:claude"
        screen._update_filtered_list()
        monkeypatch.setattr(screen, "_select_with_auth_check", record)

        assert len(screen._filtered_models) > 0

        screen.action_select()

        assert selected is not None
        model_spec, provider = selected
        assert model_spec != "anthropic:claude"
        assert provider == "anthropic"


class TestModelSelectorCurrentModelPreselection:
    """Tests for pre-selecting the current model when opening the selector."""

    async def test_non_discovered_current_model_is_visible_and_preselected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remote-only current model survives the recommended-only subset."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"openai": ["gpt-5.6-sol"]},
        )
        monkeypatch.setattr(model_selector, "get_model_profiles", lambda **_kwargs: {})
        monkeypatch.setattr(model_selector, "load_recent_models", list)
        monkeypatch.setattr(
            model_selector,
            "get_provider_auth_status",
            lambda provider: ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider=provider,
                source=ProviderAuthSource.ENV,
            ),
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                current_model="remote-model",
                current_provider="server_provider",
                recommended_models={"openai:gpt-5.6-sol": "GPT-5.6 Sol"},
                default_scope=None,
                check_provider_requirements=False,
            )
            app.push_screen(screen)
            await pilot.pause()

            current = ("server_provider:remote-model", "server_provider")
            assert current in screen._filtered_models
            assert screen._filtered_models[screen._selected_index] == current


class TestModelSelectorFuzzyMatching:
    """Tests for fuzzy search filtering."""

    def test_enter_selects_fuzzy_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pressing Enter after fuzzy filtering should select the top result."""
        screen = _model_selector_for_filtering()
        selected: tuple[str, str] | None = None

        def record(model_spec: str, provider: str) -> None:
            nonlocal selected
            selected = (model_spec, provider)

        screen._filter_text = "claude"
        screen._update_filtered_list()
        monkeypatch.setattr(screen, "_select_with_auth_check", record)

        assert len(screen._filtered_models) > 0

        screen.action_select()

        assert selected is not None
        model_spec, _ = selected
        assert "claude" in model_spec.lower()

    def test_fuzzy_dotted_version_needs_friendly_name(self) -> None:
        """Negative control: without the friendly name, "4.8" can't match the spec.

        Pins that the previous test passes because of the folded-in name, not
        some incidental spec match — guarding the friendly-name search feature.
        """
        screen = _model_selector_for_filtering()

        # Neutralize the friendly name (return the hyphenated model portion, as
        # the raw spec already carries) so "4.8" has no dotted form to match.
        screen._get_model_display_name = (  # ty: ignore[invalid-assignment]
            lambda spec: spec.split(":", 1)[-1]
        )
        screen._filter_text = "opus 4.8"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert "anthropic:claude-opus-4-8" not in specs

    def test_fuzzy_matches_provider_friendly_label(self) -> None:
        """The provider display label — not just the key — is searchable.

        Searches "subscription", which appears in neither the spec
        (`openai_codex:gpt-5.2`), the friendly model name ("GPT-5.2"), nor the
        provider key (`openai_codex`) — only in the resolved display label
        "OpenAI (Subscription login)". So a match proves the provider-label
        branch of the haystack is doing the work; there is no other source for
        it. Guards against the label term being silently dropped.
        """
        screen = _model_selector_for_filtering()
        # Inject a codex row locally rather than mutate the shared fixture that
        # other filter tests count on.
        codex = ("openai_codex:gpt-5.2", "openai_codex")
        for models in (
            screen._unfiltered_models,
            screen._all_models,
            screen._filtered_models,
        ):
            models.append(codex)
        screen._filter_text = "subscription"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert specs == ["openai_codex:gpt-5.2"], (
            f"'subscription' should match only via the provider label. Got: {specs}"
        )


class TestFilteredModelsWidgetSync:
    """Tests that _filtered_models indices match _option_widgets after display."""


class TestAvailabilityOrdering:
    """The default view floats usable providers above unavailable ones."""

    @staticmethod
    def _status(state: ProviderAuthState, provider: str) -> ProviderAuthStatus:
        if state is ProviderAuthState.CONFIGURED:
            return ProviderAuthStatus(
                state=state, provider=provider, source=ProviderAuthSource.STORED
            )
        if state is ProviderAuthState.MISSING:
            return ProviderAuthStatus(
                state=state, provider=provider, env_var=f"{provider.upper()}_API_KEY"
            )
        return ProviderAuthStatus(state=state, provider=provider)

    async def test_available_provider_floats_to_top_in_default_view(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured provider listed last renders first when unfiltered."""
        from deepagents_code.tui.widgets import model_selector

        def fake_auth(provider: str) -> ProviderAuthStatus:
            state = (
                ProviderAuthState.CONFIGURED
                if provider == "openai_codex"
                else ProviderAuthState.MISSING
            )
            return self._status(state, provider)

        monkeypatch.setattr(model_selector, "get_provider_auth_status", fake_auth)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._curated = False
            screen._recommended_only = False
            screen._filter_text = ""
            screen._recent_specs = []
            screen._install_extras = {}
            # Codex (the only configured provider) is declared last.
            models = [
                ("anthropic:claude-opus-4-8", "anthropic"),
                ("openai:gpt-5.5", "openai"),
                ("openai_codex:gpt-5.5", "openai_codex"),
            ]
            screen._unfiltered_models = list(models)
            screen._all_models = list(models)
            screen._filtered_models = list(models)
            screen._selected_index = 0

            await screen._update_display()

            providers = [provider for _, provider in screen._filtered_models]
            assert providers[0] == "openai_codex"
            assert providers.index("openai_codex") < providers.index("anthropic")
            assert providers.index("openai_codex") < providers.index("openai")
            # The reorder must carry the highlight with its model: anthropic
            # was selected at index 0 and now sits at index 1, so the remapped
            # selected index must still resolve to the anthropic entry.
            assert screen._filtered_models[screen._selected_index] == (
                "anthropic:claude-opus-4-8",
                "anthropic",
            )

    async def test_search_view_keeps_score_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A filtered search ignores availability and keeps fuzzy-score order."""
        from deepagents_code.tui.widgets import model_selector

        def fake_auth(provider: str) -> ProviderAuthStatus:
            state = (
                ProviderAuthState.CONFIGURED
                if provider == "openai_codex"
                else ProviderAuthState.MISSING
            )
            return self._status(state, provider)

        monkeypatch.setattr(model_selector, "get_provider_auth_status", fake_auth)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._curated = False
            screen._recommended_only = False
            screen._recent_specs = []
            screen._install_extras = {}
            screen._all_models = [
                ("openai:gpt-5.5", "openai"),
                ("openai_codex:gpt-5.5", "openai_codex"),
            ]
            # Simulate a score-sorted filtered list with the missing-credential
            # provider ranked first; availability must not reorder it.
            screen._filter_text = "gpt"
            screen._filtered_models = [
                ("openai:gpt-5.5", "openai"),
                ("openai_codex:gpt-5.5", "openai_codex"),
            ]
            screen._selected_index = 0

            await screen._update_display()

            providers = [provider for _, provider in screen._filtered_models]
            assert providers[0] == "openai"

    async def test_equal_rank_providers_keep_declared_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-rank providers keep their declared order (stable sort)."""
        from deepagents_code.tui.widgets import model_selector

        def fake_auth(provider: str) -> ProviderAuthStatus:
            state = (
                ProviderAuthState.CONFIGURED
                if provider == "openai_codex"
                else ProviderAuthState.MISSING
            )
            return self._status(state, provider)

        monkeypatch.setattr(model_selector, "get_provider_auth_status", fake_auth)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._curated = False
            screen._recommended_only = False
            screen._filter_text = ""
            screen._recent_specs = []
            screen._install_extras = {}
            # Two missing-credential providers declared non-alphabetically, plus
            # a configured provider declared last. The configured one must float
            # up (proving the sort actually ran), while the two missing ones keep
            # their declared order rather than being alphabetized.
            models = [
                ("openai:gpt-5.5", "openai"),
                ("anthropic:claude-opus-4-8", "anthropic"),
                ("openai_codex:gpt-5.5", "openai_codex"),
            ]
            screen._unfiltered_models = list(models)
            screen._all_models = list(models)
            screen._filtered_models = list(models)
            screen._selected_index = 0

            await screen._update_display()

            providers = [provider for _, provider in screen._filtered_models]
            assert providers[0] == "openai_codex"
            assert providers.index("openai") < providers.index("anthropic")

    async def test_recent_stays_pinned_above_availability_sort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recent entry pins to the top even when its provider is unusable."""
        from deepagents_code.tui.widgets import model_selector

        def fake_auth(provider: str) -> ProviderAuthStatus:
            state = (
                ProviderAuthState.CONFIGURED
                if provider == "openai_codex"
                else ProviderAuthState.MISSING
            )
            return self._status(state, provider)

        monkeypatch.setattr(model_selector, "get_provider_auth_status", fake_auth)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._curated = False
            screen._recommended_only = False
            screen._filter_text = ""
            screen._install_extras = {}
            # Anthropic is the user's recent pick but has no credential; the
            # configured codex provider is usable. The recent section must still
            # lead, and the availability sort must order the grouped section
            # below it (codex above the missing-credential anthropic).
            models = [
                ("anthropic:claude-opus-4-8", "anthropic"),
                ("openai_codex:gpt-5.5", "openai_codex"),
            ]
            screen._recent_specs = ["anthropic:claude-opus-4-8"]
            screen._unfiltered_models = list(models)
            screen._all_models = list(models)
            screen._filtered_models = list(models)
            screen._selected_index = 0

            await screen._update_display()

            providers = [provider for _, provider in screen._filtered_models]
            # Recent entry pinned at the very top, ahead of the grouped section.
            assert screen._filtered_models[0] == (
                "anthropic:claude-opus-4-8",
                "anthropic",
            )
            # The grouped section (everything after the pinned recent) is
            # availability-sorted: the usable provider leads it.
            assert providers[1] == "openai_codex"


class TestCuratedModelSelection:
    """Tests for onboarding curated model selection."""

    def test_curated_initial_selection_starts_at_top(self) -> None:
        """Onboarding should highlight the first model, not the current one."""
        screen = ModelSelectorScreen(
            current_model="claude-opus-4-7",
            current_provider="anthropic",
            curated=True,
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        screen._filtered_models = [
            ("openai:gpt-5.5", "openai"),
            ("anthropic:claude-opus-4-7", "anthropic"),
        ]

        assert screen._find_current_model_index() == 1
        assert screen._initial_selected_index() == 0


class TestFormatOptionLabel:
    """Tests for _format_option_label."""

    def test_all_suffixes_coexist(self) -> None:
        """Current + default + deprecated all render together."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:old-model",
            selected=False,
            current=True,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
            is_default=True,
            status="deprecated",
        )
        assert "(current)" in label.plain
        assert "(default)" in label.plain
        assert "(deprecated)" in label.plain

    def test_missing_credentials_warning_styles_model(self) -> None:
        """Missing credentials should warn on the model row."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:claude-sonnet-4-5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider="anthropic",
                env_var="ANTHROPIC_API_KEY",
            ),
        )
        from deepagents_code.theme import DARK_COLORS

        assert DARK_COLORS.warning in label.markup

    def test_install_required_dims_spec_when_not_selected(self) -> None:
        """Uninstalled providers render dimmed, overriding the missing-creds warning."""
        from deepagents_code.theme import DARK_COLORS

        label = ModelSelectorScreen._format_option_label(
            "baseten:some-model",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider="baseten",
                env_var="BASETEN_API_KEY",
            ),
            install_required=True,
        )
        assert "dim" in label.markup
        # The dim branch takes precedence over the blocks_start warning color.
        assert DARK_COLORS.warning not in label.markup


class TestFormatAuthIndicator:
    """Tests for provider auth indicator labels."""

    def test_configured_auth_renders_no_indicator(self) -> None:
        """Configured credentials hide the indicator to keep headers clean."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="openai",
                env_var="OPENAI_API_KEY",
                source=ProviderAuthSource.ENV,
            ),
            get_glyphs(),
        )

        assert indicator == ""

    def test_ollama_local_auth_has_no_checkmark(self) -> None:
        """Local Ollama uses its own detail, not the CONFIGURED empty indicator."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.NOT_REQUIRED,
                provider="ollama",
                detail="local provider",
            ),
            get_glyphs(),
        )

        assert indicator == "local provider"

    def test_missing_auth_uses_generic_message(self) -> None:
        """Missing credentials show a generic label, not the env var name."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider="anthropic",
                env_var="ANTHROPIC_API_KEY",
            ),
            get_glyphs(),
        )

        assert "missing credentials" in indicator
        assert "ANTHROPIC_API_KEY" not in indicator


class TestGetModelStatus:
    """Tests for _get_model_status profile lookup."""


def _bare_selector() -> ModelSelectorScreen:
    """Build an uninitialized selector with just the display-name dependencies.

    `_get_model_display_name` is a pure lookup over `_profiles` and
    `_recommended_models`, so these tests skip `__init__` (which needs a running
    app). Both attributes are set explicitly: the screen must not carry a
    fallback to the *main-model* recommendation set, since the classifier
    selector passes its own and would otherwise be labelled from the wrong one.
    """
    from deepagents_code.tui.widgets import model_selector

    screen = ModelSelectorScreen.__new__(ModelSelectorScreen)
    screen._recommended_models = model_selector._RECOMMENDED_MODELS
    return screen


class TestModelDetailFooter:
    """Tests for the model detail footer in the selector."""

    def test_format_footer_non_numeric_tokens(self) -> None:
        """Non-numeric token values render gracefully instead of crashing."""
        from deepagents_code.config import UNICODE_GLYPHS

        entry = ModelProfileEntry(
            profile={"max_input_tokens": "unlimited", "max_output_tokens": 64000},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        text = str(result)
        assert "unlimited" in text
        assert "64.0K" in text


class TestModelSelectorAuthGate:
    """Selecting a provider with missing creds opens the auth prompt."""

    async def test_blocked_provider_opens_auth_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enter on a model whose provider has no key opens the prompt."""
        from deepagents_code.tui.widgets.auth import AuthPromptScreen

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, AuthPromptScreen)
        # Selector did not dismiss; the prompt is in the foreground instead.
        assert app.dismissed is False

    async def test_save_key_in_prompt_dismisses_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saving a key in the prompt dismisses the selector with the model."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            # Type a key into the auth prompt input and submit
            from textual.widgets import Input as _Input

            inp = app.screen.query_one("#auth-prompt-input", _Input)
            inp.value = "stored-from-prompt"
            await pilot.press("enter")
            await pilot.pause()
        assert app.dismissed is True
        assert app.result is not None
        assert app.result[1] == "anthropic"


class TestModelSelectorInstallRouting:
    """Selecting a model whose provider is not installed prompts to install."""

    async def test_load_model_data_surfaces_installed_unprofiled_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recommended models missing from an installed provider's profiles surface."""
        from deepagents_code import config_manifest
        from deepagents_code.tui.widgets import model_selector

        spec = "fireworks:accounts/fireworks/models/kimi-k3"
        assert spec in model_selector._RECOMMENDED_MODELS

        # Provider is installed/discoverable but its profiles omit the curated
        # model, mirroring an upstream profile list that lags the hardcoded set.
        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"fireworks": ["accounts/fireworks/models/some-other-model"]},
        )
        monkeypatch.setattr(
            config_manifest,
            "is_provider_package_installed",
            lambda provider: provider == "fireworks",
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = {model_spec for model_spec, _ in all_models}
        assert spec in specs
        # Surfaced as a normal selectable row, not an install-required one.
        assert "fireworks" not in install_extras

    async def test_load_model_data_marks_config_listed_missing_provider_uninstalled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config-listed models do not make a missing provider look installed."""
        from deepagents_code import config_manifest
        from deepagents_code.tui.widgets import model_selector

        spec = "baseten:moonshotai/Kimi-K3"
        assert spec in model_selector._RECOMMENDED_MODELS

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"baseten": ["moonshotai/config-listed-model"]},
        )
        monkeypatch.setattr(
            config_manifest,
            "is_provider_package_installed",
            lambda provider: provider != "baseten",
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        assert spec in {model_spec for model_spec, _ in all_models}
        assert install_extras.get("baseten") == "baseten"

    async def test_load_model_data_does_not_duplicate_profiled_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recommended model already in profiles surfaces exactly once."""
        from deepagents_code import config_manifest
        from deepagents_code.tui.widgets import model_selector

        spec = "fireworks:accounts/fireworks/models/kimi-k3"
        model = spec.split(":", 1)[1]
        assert spec in model_selector._RECOMMENDED_MODELS

        # The provider is installed and its profiles already surface the curated
        # model, so the recommended-merge must not re-append it.
        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"fireworks": [model]},
        )
        monkeypatch.setattr(
            config_manifest,
            "is_provider_package_installed",
            lambda provider: provider == "fireworks",
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = [model_spec for model_spec, _ in all_models]
        assert specs.count(spec) == 1
        assert "fireworks" not in install_extras

    async def test_load_model_data_surfaces_multiple_unprofiled_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every unprofiled recommended spec for one installed provider surfaces."""
        from deepagents_code import config_manifest
        from deepagents_code.tui.widgets import model_selector

        expected = {
            spec
            for spec in model_selector._RECOMMENDED_MODELS
            if spec.startswith("fireworks:")
        }
        # Guard against the curated set shrinking below the multi-spec case the
        # test is meant to exercise.
        assert len(expected) > 1

        # Provider installed/discoverable, but its profiles list none of the
        # curated specs, so each must be added as a normal selectable row.
        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"fireworks": ["accounts/fireworks/models/some-other-model"]},
        )
        monkeypatch.setattr(
            config_manifest,
            "is_provider_package_installed",
            lambda provider: provider == "fireworks",
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = {model_spec for model_spec, _ in all_models}
        assert expected <= specs
        assert "fireworks" not in install_extras

    async def test_curated_uninstalled_provider_defers_to_launch_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Onboarding selections install from the launch flow before auth."""
        from deepagents_code.tui.widgets import model_selector

        results: list[tuple[str, str] | None] = []
        screen = ModelSelectorScreen(
            current_model="openai:gpt-5.5",
            current_provider="openai",
            curated=True,
            result_callback=results.append,
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore
        monkeypatch.setattr(
            "deepagents_code.config_manifest.provider_install_extra",
            lambda _provider: "baseten",
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.is_provider_package_installed",
            lambda _provider: False,
        )
        monkeypatch.setattr(
            model_selector,
            "get_provider_auth_status",
            lambda _provider: pytest.fail("auth should wait until after install"),
        )

        screen._select_with_auth_check("baseten:zai-org/GLM-5.2", "baseten")

        assert results == [("baseten:zai-org/GLM-5.2", "baseten")]
        dismiss.assert_called_once_with(("baseten:zai-org/GLM-5.2", "baseten"))

    async def test_select_uninstalled_provider_prompts_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting an uninstalled provider opens the install-confirm modal."""
        import importlib.util

        if importlib.util.find_spec("langchain_baseten") is not None:
            pytest.skip("langchain_baseten is installed in this environment")

        from deepagents_code.tui.widgets.install_confirm import (
            InstallProviderConfirmScreen,
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            pushed: list[tuple[object, Callable[[bool | None], None] | None]] = []
            monkeypatch.setattr(
                screen.app,
                "push_screen",
                lambda s, cb=None, *_a, **_k: pushed.append((s, cb)),
            )

            screen._select_with_auth_check(
                "baseten:moonshotai/Kimi-K2.7-Code", "baseten"
            )

            assert len(pushed) == 1
            assert isinstance(pushed[0][0], InstallProviderConfirmScreen)
            assert app.dismissed is False

    async def test_navigation_preserves_install_required_dim(self) -> None:
        """Cursoring onto then off an install-required row keeps it dimmed.

        Regression: `_move_selection` re-rendered the deselected row without
        the `install_required` flag, so uninstalled rows turned bright after
        the cursor passed over them and never reverted.
        """
        install_spec = "baseten:moonshotai/Kimi-K3"
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            screen._curated = False
            screen._recommended_only = False
            screen._install_extras = {"baseten": "baseten"}
            screen._unfiltered_models = [
                ("openai:gpt-5.5", "openai"),
                (install_spec, "baseten"),
            ]
            screen._all_models = list(screen._unfiltered_models)
            screen._filtered_models = list(screen._unfiltered_models)
            screen._filter_text = ""
            screen._selected_index = 0
            await screen._update_display()
            await pilot.pause()

            install_widget = next(
                w for w in screen._option_widgets if w.model_spec == install_spec
            )
            assert "dim" in install_widget.content.markup

            # Move the cursor onto the install-required row, then back off it.
            screen._move_selection(1)
            await pilot.pause()
            assert screen._selected_index == 1
            # While highlighted the row is intentionally bright: CSS owns the
            # selected row and `_format_option_label` only dims when
            # `not selected`. This guards the selected-row relabel so it keeps
            # threading the correct state.
            assert "dim" not in install_widget.content.markup
            screen._move_selection(-1)
            await pilot.pause()

            assert "dim" in install_widget.content.markup

    async def test_navigation_preserves_install_required_dim_in_recent(self) -> None:
        """The Recent-section copy of an install-required row stays dimmed too.

        `_move_selection` is section-agnostic, but the Recent section builds
        its rows through a separate call site than the provider groups. An
        install-required model also surfaces at the top as a recent pick, so
        cursoring onto then off that Recent row must re-dim it just the same.
        """
        install_spec = "baseten:moonshotai/Kimi-K3"
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            screen._curated = False
            screen._recommended_only = False
            screen._install_extras = {"baseten": "baseten"}
            screen._unfiltered_models = [
                ("openai:gpt-5.5", "openai"),
                (install_spec, "baseten"),
            ]
            screen._all_models = list(screen._unfiltered_models)
            screen._filtered_models = list(screen._unfiltered_models)
            # The install-required model is also a recent pick, so it renders
            # both at the top (Recent) and in its provider group.
            screen._recent_specs = [install_spec]
            screen._filter_text = ""
            screen._selected_index = 0
            await screen._update_display()
            await pilot.pause()

            # Recents render first; index 0 is the Recent-section install row.
            recent_install = screen._option_widgets[0]
            assert recent_install.model_spec == install_spec
            # Check the model NAME's dim specifically: the Recent row always
            # carries a dim `(Baseten)` provider tag, so a bare "dim" substring
            # search can't tell install-required dimming from the tag. The name
            # is wrapped in `[dim]...` only when install-required and unselected.
            name_dim = "[dim]Kimi K3"
            # The provider tag disambiguates the cross-provider Recent row and
            # must survive `_move_selection`'s incremental relabel — which
            # re-derives the label from the widget's persisted `show_provider`.
            provider_tag = "(Baseten)"
            assert name_dim in recent_install.content.markup
            assert provider_tag in recent_install.content.markup
            # `_update_display` keeps the openai row highlighted (rendered
            # order: recent install, openai, provider-group install).
            assert screen._selected_index == 1

            # Move the cursor onto the Recent install row, then back off it.
            screen._move_selection(-1)
            await pilot.pause()
            assert screen._selected_index == 0
            assert name_dim not in recent_install.content.markup
            # The tag persists regardless of selection (it's the relabel path).
            assert provider_tag in recent_install.content.markup
            screen._move_selection(1)
            await pilot.pause()
            assert screen._selected_index == 1

            assert name_dim in recent_install.content.markup
            assert provider_tag in recent_install.content.markup

    async def test_remote_selection_skips_local_provider_requirements(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code.tui.widgets import model_selector

        results: list[tuple[str, str] | None] = []
        screen = ModelSelectorScreen(
            check_provider_requirements=False,
            default_scope=None,
            result_callback=results.append,
        )
        dismiss = MagicMock()
        screen.dismiss = dismiss  # ty: ignore
        monkeypatch.setattr(
            "deepagents_code.config_manifest.provider_install_extra",
            lambda _provider: pytest.fail("local install should not be checked"),
        )
        monkeypatch.setattr(
            model_selector,
            "get_provider_auth_status",
            lambda _provider: pytest.fail("local credentials should not be checked"),
        )

        screen._select_with_auth_check(
            "server_provider:remote-model", "server_provider"
        )

        result = ("server_provider:remote-model", "server_provider")
        assert results == [result]
        dismiss.assert_called_once_with(result)
