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
from textual.screen import ModalScreen
from textual.widgets import Input, Static

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

    async def test_escape_works_when_input_focused(self) -> None:
        """ESC should work even when the filter input is focused."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Type something to ensure input is focused
            await pilot.press("c", "l", "a", "u", "d", "e")
            await pilot.pause()

            # Press ESC - should still dismiss
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None

    async def test_escape_calls_direct_result_callback(self) -> None:
        """The direct result callback should receive dismiss results."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector_with_result_callback()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.callback_results == [None]

    async def test_escape_with_conflicting_app_binding(self) -> None:
        """ESC should dismiss modal even when app has its own escape binding.

        This test reproduces the bug where DeepAgentsApp's escape binding
        for action_interrupt would intercept escape before the modal could
        handle it, causing the modal to not close.
        """
        app = AppWithEscapeBinding()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Press ESC - this should dismiss the modal, not call action_interrupt
            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None
            # The interrupt action should NOT have been called because modal was open
            assert app.interrupt_called is False


class TestModelSelectorChrome:
    """Tests for model selector title and description chrome."""

    async def test_optional_title_and_description_render(self) -> None:
        """A custom title and description should render above the filter."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                title="Choose a Recommended Model",
                description="Curated models backed by evals.",
                default_scope=MAIN_MODEL_DEFAULT_SCOPE,
            )
            app.push_screen(screen)
            await pilot.pause()

            title = screen.query_one(".model-selector-title", Static)
            description = screen.query_one(".model-selector-description", Static)

            assert "Choose a Recommended Model" in str(title.content)
            assert "Curated models backed by evals." in str(description.content)

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

            assert "Tab autocomplete" in str(help_text.content)
            assert "Esc skip setup" not in str(help_text.content)
            assert "Esc cancel" not in str(help_text.content)

    async def test_curated_selector_help_hides_default_hint(self) -> None:
        """Onboarding model selection should not advertise default changes."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)

            assert "Ctrl+S" not in str(help_text.content)
            assert "set default" not in str(help_text.content)

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

    async def test_standard_selector_help_hides_cancel_hint(self) -> None:
        """The regular /model selector should not leave a trailing separator."""
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
            assert "Esc cancel" not in str(help_text.content)

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

    async def test_curated_selector_help_stays_one_row(self) -> None:
        """The shorter curated footer must not over-wrap once the clamp is gone.

        `height: auto` lets the standard footer wrap, but the curated line drops
        the Ctrl+S/Ctrl+R hints and fits one row — pin it so a future width or
        hint change that pushes it to two rows fails loudly.
        """
        app = ModelSelectorTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)

            assert help_text.region.height == 1


class TestRecommendedToggle:
    """Tests for the Ctrl+R recommended-only toggle in `/model`."""

    def test_custom_recommendations_replace_standard_list(self) -> None:
        """Specialized pickers should be able to supply their own shortlist."""
        recommendations = {
            "anthropic:claude-haiku-4-5": "Claude Haiku 4.5",
        }
        screen = ModelSelectorScreen(
            recommended_models=recommendations,
            include_recent_models=False,
            default_scope=MAIN_MODEL_DEFAULT_SCOPE,
        )
        screen._recent_specs = ["openai:gpt-5.5"]
        models = [
            ("anthropic:claude-haiku-4-5", "anthropic"),
            ("openai:gpt-5.5", "openai"),
        ]

        assert screen._apply_subset(models) == models[:1]
        assert (
            screen._get_model_display_name("anthropic:claude-haiku-4-5")
            == "Claude Haiku 4.5"
        )

    async def test_default_view_is_recommended(self) -> None:
        """Opening `/model` should land on the curated recommended subset."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._recommended_only is True
            info = screen.query_one("#model-selector-info", Static)
            assert "Showing recommended models" in str(info.content)

    def test_search_uses_full_list_from_recommended_view(self) -> None:
        """Typing a filter should search beyond the recommended subset."""
        screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
        screen._unfiltered_models = [
            ("openai:gpt-5.6-luna", "openai"),
            ("openai:gpt-4o", "openai"),
        ]
        screen._all_models = screen._apply_subset(screen._unfiltered_models)

        assert screen._recommended_only is True
        assert screen._all_models == [("openai:gpt-5.6-luna", "openai")]

        screen._filter_text = "gpt-4o"
        screen._update_filtered_list()

        assert screen._filtered_models == [("openai:gpt-4o", "openai")]

    def test_recent_codex_keeps_recommended_provider_order(self) -> None:
        """A recent Codex model should stay between OpenAI and OpenRouter."""
        screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
        screen._recent_specs = ["openai_codex:gpt-5.6-luna"]
        all_models = [
            ("anthropic:claude-sonnet-5", "anthropic"),
            ("openai:gpt-5.6-luna", "openai"),
            ("openai_codex:gpt-5.6-luna", "openai_codex"),
            ("openrouter:anthropic/claude-sonnet-5", "openrouter"),
        ]

        providers = [provider for _, provider in screen._apply_subset(all_models)]

        assert providers.index("openai") < providers.index("openai_codex")
        assert providers.index("openai_codex") < providers.index("openrouter")
        assert providers[0] != "openai_codex"

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

    async def test_toggle_expands_to_full_list(self) -> None:
        """Ctrl+R from the default recommended view should expand to all."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            recommended_count = len(screen._filtered_models)

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert screen._recommended_only is False
            assert len(screen._filtered_models) >= recommended_count

    async def test_toggle_round_trip_restores_recommended(self) -> None:
        """Pressing Ctrl+R twice should return to the recommended subset."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            original = list(screen._filtered_models)

            await pilot.press("ctrl+r")
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()

            assert screen._recommended_only is True
            assert screen._filtered_models == original

    async def test_toggle_updates_info_line(self) -> None:
        """Info line should advertise the inverse state after toggling."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            info = screen.query_one("#model-selector-info", Static)
            assert "Ctrl+R for all" in str(info.content)

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert "Ctrl+R for recommended" in str(info.content)

    async def test_toggle_disabled_in_curated_onboarding_mode(self) -> None:
        """Curated/onboarding mode should ignore Ctrl+R."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            before = list(screen._filtered_models)

            await pilot.press("ctrl+r")
            await pilot.pause()

            assert screen._recommended_only is False
            assert screen._filtered_models == before

    async def test_help_text_advertises_toggle_in_standard_mode(self) -> None:
        """Standard `/model` help footer should mention Ctrl+R."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+R" in str(help_text.content)

    async def test_help_text_omits_toggle_in_curated_mode(self) -> None:
        """Onboarding's curated help footer should not mention Ctrl+R."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+R" not in str(help_text.content)


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

    async def test_ctrl_s_writes_default_for_main_scope(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`/model`'s Ctrl+S still stores `[models].default`."""
        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+S set default" in str(help_widget.content)

            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Default set to anthropic:claude-sonnet-5" in str(
                help_widget.content
            )

        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        assert data["models"]["default"] == "anthropic:claude-sonnet-5"
        assert "auto_classifier" not in data["models"]

    async def test_ctrl_s_writes_auto_classifier_for_classifier_scope(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The `/auto model` picker stores `[models].auto_classifier`."""
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+S set classifier default" in str(help_widget.content)

            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Default classifier model set to anthropic:claude-sonnet-5" in str(
                help_widget.content
            )

        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        assert data["models"]["auto_classifier"] == "anthropic:claude-sonnet-5"
        assert "default" not in data["models"]

    async def test_second_ctrl_s_clears_classifier_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pressing Ctrl+S on the stored classifier removes it."""
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Default classifier model cleared" in str(help_widget.content)

        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        assert "auto_classifier" not in data["models"]

    async def test_classifier_scope_marks_stored_model_as_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The `(default)` marker reflects the stored classifier, not `default`."""
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        (tmp_path / "config.toml").write_text(
            "[models]\n"
            'default = "openai:gpt-5.6-luna"\n'
            'auto_classifier = "anthropic:claude-sonnet-5"\n',
            encoding="utf-8",
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._default_spec == "anthropic:claude-sonnet-5"
            assert "(default)" in str(screen._option_widgets[0].content)

    async def test_no_scope_disables_ctrl_s_and_omits_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A scopeless picker must not persist the *agent's* model.

        The grader pickers (`/goal model`, `/rubric model`) have no config key
        of their own; inheriting the default scope would make Ctrl+S there
        rewrite `[models].default`.
        """
        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=None)
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+S" not in str(help_widget.content)
            # Ctrl+R/Ctrl+N still advertised — only the Ctrl+S hint drops out.
            assert "Ctrl+R recommended" in str(help_widget.content)

            await pilot.press("ctrl+s")
            await pilot.pause()

            assert screen._default_spec is None

        assert not config_path.exists()

    async def test_no_scope_omits_marker_for_the_agents_stored_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A scopeless picker must not mark the *agent's* stored default.

        Sibling coverage writes no config, so `_default_spec is None` holds
        there no matter what the scope branch reads. Storing `[models].default`
        for the row on screen makes the assertion discriminating: a picker whose
        Ctrl+S is deliberately inert would otherwise render `(default)` beside a
        preference it cannot touch.
        """
        self._stub_catalog(monkeypatch)
        (tmp_path / "config.toml").write_text(
            '[models]\ndefault = "anthropic:claude-sonnet-5"\n', encoding="utf-8"
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=None)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._default_spec is None
            assert "(default)" not in str(screen._option_widgets[0].content)

    async def test_stored_spec_is_stripped_so_it_can_be_toggled_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Whitespace around a hand-edited spec must not orphan the value.

        The launch resolvers strip, so an unstripped read here would match no
        row: no `(default)` marker, and Ctrl+S would take the *save* branch
        instead of clearing — leaving no in-app way to remove the stored spec.
        """
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[models]\nauto_classifier = "  anthropic:claude-sonnet-5  "\n',
            encoding="utf-8",
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._default_spec == "anthropic:claude-sonnet-5"
            assert "(default)" in str(screen._option_widgets[0].content)

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Default classifier model cleared" in str(help_widget.content)

        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        assert "auto_classifier" not in data["models"]

    async def test_blank_stored_spec_reads_as_nothing_stored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A blank stored value degrades to "nothing stored", as at launch."""
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        (tmp_path / "config.toml").write_text(
            '[models]\nauto_classifier = "   "\n', encoding="utf-8"
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._default_spec is None

    async def test_clear_failure_shows_persistent_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A failed clear reports the noun and does not erase itself."""
        from deepagents_code.tui.widgets import model_selector

        self._stub_catalog(monkeypatch)
        (tmp_path / "config.toml").write_text(
            '[models]\nauto_classifier = "anthropic:claude-sonnet-5"\n',
            encoding="utf-8",
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                default_scope=model_selector.AUTO_CLASSIFIER_DEFAULT_SCOPE._replace(
                    clear=lambda: False
                )
            )
            app.push_screen(screen)
            await pilot.pause()

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Failed to clear default classifier model" in str(
                help_widget.content
            )
            assert screen._help_error_shown is True
            # The stored spec is untouched, so the row keeps its marker.
            assert screen._default_spec == "anthropic:claude-sonnet-5"

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
        assert "~/.deepagents/config.toml" in message
        assert "unwritable" in message
        assert "malformed" in message

    async def test_success_warns_when_an_env_var_overrides_the_stored_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored classifier the environment outranks changes nothing at launch.

        `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL` beats `[models].auto_classifier`,
        so without this warning the success message and the `(default)` marker
        both imply the keypress changed which model reviews actions.
        """
        from deepagents_code import _env_vars
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "openai:gpt-5.6-luna")

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
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

            # The write still happened; only the claim about effect is qualified.
            assert screen._default_spec == "anthropic:claude-sonnet-5"

        assert notified
        message, severity = notified[0]
        assert severity == "warning"
        assert "Default classifier model saved" in message
        assert _env_vars.AUTO_CLASSIFIER_MODEL in message
        assert "is currently set; if it remains set" in message
        assert "next launch" in message

    async def test_success_warns_when_env_override_is_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported-but-empty override also outranks the stored key.

        The resolver rejects a blank `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL`
        and falls back to the main model, so the stored classifier still will
        not run — the warning must fire on presence, not on a non-empty value.
        """
        from deepagents_code import _env_vars
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "")

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
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

            assert screen._default_spec == "anthropic:claude-sonnet-5"

        assert notified
        message, severity = notified[0]
        assert severity == "warning"
        assert "Default classifier model saved" in message
        assert _env_vars.AUTO_CLASSIFIER_MODEL in message
        assert "is currently set; if it remains set" in message
        assert "next launch" in message

    async def test_main_scope_success_raises_no_override_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`[models].default` has no env var, so `/model` stays quiet."""
        self._stub_catalog(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            notified: list[str] = []
            monkeypatch.setattr(screen, "notify", lambda msg, **_: notified.append(msg))

            await pilot.press("ctrl+s")
            await pilot.pause()

        assert notified == []

    async def test_ctrl_s_refuses_install_required_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A spec whose provider is missing can never build, so don't store it."""
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)
        config_path = tmp_path / "config.toml"

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._install_extras = {"anthropic": "anthropic"}

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "not installed" in str(help_widget.content)
            assert screen._default_spec is None

        assert not config_path.exists()

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

    async def test_failure_cancels_pending_success_restore_timer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Ctrl+S failure must not be erased by a prior success's timer.

        Successful saves schedule a 3-second footer restore; a failed save
        schedules nothing. If the failure lands before the prior timer fires,
        the orphaned timer would wipe the persistent error notice unless
        `_fail` cancels it.
        """
        from deepagents_code.tui.widgets import model_selector

        self._stub_catalog(monkeypatch)

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
            timer = screen._help_restore_timer
            assert timer is not None

            # The row just saved is now the stored spec, so the next Ctrl+S
            # takes the clear branch. Swap in a failing clear before the
            # success timer fires.
            screen._default_scope = model_selector.MAIN_MODEL_DEFAULT_SCOPE._replace(
                clear=lambda: False
            )
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "Failed to clear default" in str(help_widget.content)
            assert screen._help_restore_timer is None
            # `Timer.stop()` cancels the underlying task, so the orphaned
            # timer can never fire and erase the error notice.
            assert timer._task is None

    async def test_success_restarts_pending_restore_timer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second successful Ctrl+S must not orphan the first timer.

        Two quick successful toggles schedule two restores; overwriting the
        handle without stopping the first leaves it active but untracked. Its
        callback would clear the handle while the second timer is pending, so
        a later `_fail` could not cancel the real timer — and that timer would
        erase the persistent error notice. The first timer must be stopped
        before the second is scheduled.
        """
        self._stub_catalog(monkeypatch)

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
            first_timer = screen._help_restore_timer
            assert first_timer is not None

            # The row just saved is now the stored spec, so the next Ctrl+S
            # takes the clear branch — a second success.
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "Default cleared" in str(help_widget.content)

            second_timer = screen._help_restore_timer
            assert second_timer is not None
            assert second_timer is not first_timer
            # `Timer.stop()` cancels the underlying task, so the superseded
            # timer can never fire and clobber the newer handle or message.
            assert first_timer._task is None

    async def test_names_toggle_does_not_orphan_the_success_restore_timer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+N between a success and a failure must not orphan the timer.

        `_restore_help_text` drops the timer handle without stopping it, which is
        correct when the timer is its own caller. `action_toggle_names` calls it
        synchronously, so without an explicit stop the still-live timer becomes
        untracked, `_fail` has nothing to cancel, and the orphan fires ~3s later
        and erases the failure notice — the exact wipe the handle exists to
        prevent, reached through a different door.
        """
        from deepagents_code.tui.widgets import model_selector

        self._stub_catalog(monkeypatch)

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
            success_timer = screen._help_restore_timer
            assert success_timer is not None

            # Ctrl+N clobbers the success message and restores the hint. The
            # timer it superseded must be stopped, not merely forgotten.
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._help_restore_timer is None
            assert success_timer._task is None

            screen._default_scope = model_selector.MAIN_MODEL_DEFAULT_SCOPE._replace(
                clear=lambda: False
            )
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "Failed to clear default" in str(help_widget.content)

            # Give the orphaned timer's original deadline time to pass.
            await asyncio.sleep(3.2)
            await pilot.pause()

            assert "Failed to clear default" in str(help_widget.content)
            assert screen._help_error_shown is True

    async def test_install_refusal_restores_the_footer_on_its_timer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The install refusal is correctable, so it must not pin the footer.

        A write failure persists because the user has to go fix a file. The
        refusal is fixed by moving the cursor to an installed row, and its 10s
        toast already carries the remedy — so pinning `_help_error_shown` for the
        life of the modal would also stop Ctrl+N from ever refreshing the hint.
        """
        from deepagents_code.tui.widgets.model_selector import (
            AUTO_CLASSIFIER_DEFAULT_SCOPE,
        )

        self._stub_catalog(monkeypatch)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=AUTO_CLASSIFIER_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            screen._install_extras = {"anthropic": "anthropic"}

            help_widget = screen.query_one(".model-selector-help", Static)
            await pilot.press("ctrl+s")
            await pilot.pause()

            assert "not installed" in str(help_widget.content)
            assert screen._help_restore_timer is not None

            await asyncio.sleep(3.2)
            await pilot.pause()

            assert "not installed" not in str(help_widget.content)
            assert "Ctrl+S set classifier default" in str(help_widget.content)
            assert screen._help_error_shown is False


class TestNamesToggle:
    """Tests for the Ctrl+N names/raw-spec toggle in `/model`."""

    async def test_ctrl_n_toggles_rows_between_names_and_specs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+N flips rows between the friendly name and the raw spec."""
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

            row = screen._option_widgets[0]
            assert row.model_spec == "anthropic:claude-sonnet-5"
            assert not screen._show_specs
            text = str(row.content)
            assert "Claude Sonnet 5" in text
            assert "anthropic:claude-sonnet-5" not in text

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert screen._show_specs
            text = str(screen._option_widgets[0].content)
            assert "anthropic:claude-sonnet-5" in text
            assert "Claude Sonnet 5" not in text

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert not screen._show_specs
            text = str(screen._option_widgets[0].content)
            assert "Claude Sonnet 5" in text
            assert "anthropic:claude-sonnet-5" not in text

    async def test_help_text_advertises_names_toggle_in_standard_mode(self) -> None:
        """Standard `/model` help footer should mention Ctrl+N."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+N" in str(help_text.content)

    async def test_help_text_names_toggle_hint_follows_display_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Ctrl+N footer hint advertises what the next press switches to."""
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

            help_widget = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+N IDs" in str(help_widget.content)

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert "Ctrl+N names" in str(help_widget.content)
            assert "Ctrl+N IDs" not in str(help_widget.content)

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert "Ctrl+N IDs" in str(help_widget.content)
            assert "Ctrl+N names" not in str(help_widget.content)

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

    async def test_help_text_omits_names_toggle_in_curated_mode(self) -> None:
        """Onboarding's curated help footer should not mention Ctrl+N."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            help_text = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+N" not in str(help_text.content)

    async def test_names_toggle_available_in_curated_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Curated/onboarding mode still honors Ctrl+N (presentation only)."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert screen._show_specs
            assert "anthropic:claude-sonnet-5" in str(screen._option_widgets[0].content)
            # The toggle refreshes the footer, but curated mode advertises no
            # Ctrl+N hint to begin with — the press must not leak one in.
            help_widget = screen.query_one(".model-selector-help", Static)
            assert "Ctrl+N" not in str(help_widget.content)

    async def test_show_specs_persists_across_filter_rebuild(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spec mode survives the full rebuild that filtering triggers.

        `_relabel_options` (the in-place Ctrl+N path) is not the only renderer:
        typing into the filter runs `_update_display`, which rebuilds every row
        from scratch. That rebuild path must honor `_show_specs` too, so toggling
        specs on and then filtering keeps the surviving rows in spec mode.
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

            await pilot.press("ctrl+n")
            await pilot.pause()
            assert screen._show_specs

            # Typing filters the list, rebuilding rows via `_update_display`.
            await pilot.press("c", "l", "a", "u", "d", "e")
            await pilot.pause()

            assert screen._show_specs
            specs = [str(w.content) for w in screen._option_widgets]
            assert any("anthropic:claude-sonnet-5" in s for s in specs)
            assert all("Claude Sonnet 5" not in s for s in specs)

    async def test_selection_preserved_after_names_toggle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Toggling names keeps the highlighted row highlighted.

        Uses a multi-row list with the cursor moved off row 0 so a regression
        that dropped the selection (e.g. hardcoding `selected=False` in
        `_relabel_options`) would surface as a missing cursor glyph.
        """
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"], "openai": ["gpt-5.5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

        cursor = get_glyphs().cursor

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            screen._recommended_only = False
            app.push_screen(screen)
            await pilot.pause()

            assert len(screen._option_widgets) >= 2

            await pilot.press("down")
            await pilot.pause()
            selected = screen._selected_index
            assert selected != 0

            await pilot.press("ctrl+n")
            await pilot.pause()

            assert screen._selected_index == selected
            for index, widget in enumerate(screen._option_widgets):
                text = str(widget.content)
                if index == selected:
                    assert text.startswith(f"{cursor} ")
                else:
                    assert not text.startswith(f"{cursor} ")

    async def test_names_toggle_drops_provider_tag_in_recent_section(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In spec mode a Recent row shows the bare spec with no `(provider)` tag.

        Recent rows (`show_provider=True`) append a dim ` (provider)` tag after
        the friendly name to disambiguate the same model across providers. The
        raw spec already embeds the provider, so spec mode must drop that tag
        rather than print the provider twice.
        """
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"anthropic": ["claude-sonnet-5"]},
        )
        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["anthropic:claude-sonnet-5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            recent_rows = [w for w in screen._option_widgets if w.show_provider]
            assert recent_rows, "expected a Recent-section row"
            before = str(recent_rows[0].content)
            assert "Claude Sonnet 5" in before
            assert "(Anthropic)" in before

            await pilot.press("ctrl+n")
            await pilot.pause()

            recent_rows = [w for w in screen._option_widgets if w.show_provider]
            after = str(recent_rows[0].content)
            assert "anthropic:claude-sonnet-5" in after
            assert "Claude Sonnet 5" not in after
            assert "(Anthropic)" not in after


class TestRecentModelsSection:
    """Tests for the "Recent" pseudo-provider section pinned at the top."""

    async def test_recent_header_renders_when_recents_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A populated recents file should produce a `Recent` header."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["anthropic:claude-opus-4-7"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            headers = [
                str(h.content)
                for h in screen.query(".model-provider-header").results(Static)
            ]
            assert headers, "expected at least one provider header"
            assert "Recent" in headers[0]

    async def test_no_recent_header_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No recents file means the Recent header is not rendered."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(model_selector, "load_recent_models", list)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            headers = [
                str(h.content)
                for h in screen.query(".model-provider-header").results(Static)
            ]
            assert not any("Recent" in h for h in headers)

    async def test_provider_header_uses_friendly_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider headers render the friendly label, not the raw config key."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"openai_codex": ["gpt-5.5"]},
        )
        monkeypatch.setattr(model_selector, "load_recent_models", list)

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            headers = [
                str(h.content)
                for h in screen.query(".model-provider-header").results(Static)
            ]
            assert any("OpenAI Codex (ChatGPT login)" in h for h in headers)
            assert not any("openai_codex" in h for h in headers)

    async def test_recent_row_shows_name_and_provider_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recent rows show the friendly name plus a brand `(provider)` tag."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"openai": ["gpt-5.5"]},
        )
        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["openai:gpt-5.5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            recent = screen._option_widgets[0]
            assert recent.model_spec == "openai:gpt-5.5"
            assert recent.show_provider
            text = str(recent.content)
            assert "GPT-5.5" in text
            # No brand override for openai: falls back to the display name.
            assert "(OpenAI)" in text
            assert "openai:gpt-5.5" not in text

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
            assert "ChatGPT login" not in text

    async def test_recent_entries_appear_in_provider_section_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recent spec is also kept in its real provider section below."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["anthropic:claude-opus-4-7"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            specs = [w.model_spec for w in screen._option_widgets]
            assert specs.count("anthropic:claude-opus-4-7") == 2

    async def test_recent_entries_do_not_duplicate_on_refresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refreshing an open selector should not compound recent entries."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {
                "openai": ["gpt-5.5"],
                "openai_codex": ["gpt-5.5"],
            },
        )
        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["openai_codex:gpt-5.5", "openai:gpt-5.5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            await screen._update_display()

            specs = [w.model_spec for w in screen._option_widgets]
            assert specs.count("openai:gpt-5.5") == 2
            assert specs.count("openai_codex:gpt-5.5") == 2

    async def test_refresh_preserves_provider_section_recent_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refreshing should not move selection from provider row to Recent."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {
                "openai": ["gpt-5.5"],
                "openai_codex": ["gpt-5.5"],
            },
        )
        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["openai_codex:gpt-5.5", "openai:gpt-5.5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            provider_index = [
                i
                for i, entry in enumerate(screen._filtered_models)
                if entry == ("openai_codex:gpt-5.5", "openai_codex")
            ][1]
            screen._selected_index = provider_index

            await screen._update_display()

            assert screen._selected_index == provider_index
            assert screen._filtered_models[screen._selected_index] == (
                "openai_codex:gpt-5.5",
                "openai_codex",
            )

    async def test_recent_entries_survive_recommended_toggle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recents not on the curated list should still show in recommended mode."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["anthropic:claude-sonnet-4-5"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            assert screen._recommended_only is True
            specs = [spec for spec, _ in screen._filtered_models]
            assert "anthropic:claude-sonnet-4-5" in specs

    async def test_recent_section_hidden_during_onboarding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Curated onboarding never shows Recent, even if the MRU is populated.

        Guards the `include_recent` gating in `_load_model_data` (see its
        docstring for why the startup auto-detected fallback must not surface
        as a "Recent" entry the user never chose).
        """
        from deepagents_code.tui.widgets import model_selector

        recent_called = False

        def _tracked_load_recent_models() -> list[str]:
            nonlocal recent_called
            recent_called = True
            # Deliberately a recommended model so it survives curated filtering:
            # on a revert it would reach the rendered "Recent" header, keeping
            # the header assertion below an effective regression guard.
            return ["anthropic:claude-opus-4-7"]

        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            _tracked_load_recent_models,
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            assert recent_called is False
            assert screen._recent_specs == []
            headers = [
                str(h.content)
                for h in screen.query(".model-provider-header").results(Static)
            ]
            assert not any("Recent" in h for h in headers)

    async def test_recent_section_hidden_during_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the user is searching, recents fold into the match results."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector,
            "load_recent_models",
            lambda: ["anthropic:claude-opus-4-7"],
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            headers = [
                str(h.content)
                for h in screen.query(".model-provider-header").results(Static)
            ]
            assert not any("Recent" in h for h in headers)


class TestModelSelectorAvailabilityHint:
    """Tests for the API-keys hint shown above the standard model list."""

    async def test_hint_renders_in_non_curated_mode(self) -> None:
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(default_scope=MAIN_MODEL_DEFAULT_SCOPE)
            app.push_screen(screen)
            await pilot.pause()

            info = screen.query_one("#model-selector-info", Static)

            assert info.display is True

    async def test_hint_absent_in_curated_mode(self) -> None:
        """Onboarding's curated picker shares no copy with the standard selector."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            screen = ModelSelectorScreen(
                curated=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
            app.push_screen(screen)
            await pilot.pause()

            assert not screen.query("#model-selector-info")


class TestModelSelectorKeyboardNavigation:
    """Tests for keyboard navigation in the modal."""

    async def test_down_arrow_moves_selection(self) -> None:
        """Down arrow should move selection down."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            initial_index = screen._selected_index

            await pilot.press("down")
            await pilot.pause()

            assert screen._selected_index == initial_index + 1

    async def test_up_arrow_moves_selection(self) -> None:
        """Up arrow should move selection up (wrapping to end if at 0)."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            initial_index = screen._selected_index
            count = len(screen._filtered_models)

            await pilot.press("up")
            await pilot.pause()

            # Should move up by one, wrapping if at 0
            expected = (initial_index - 1) % count
            assert screen._selected_index == expected

    async def test_enter_selects_model(self) -> None:
        """Enter should select the current model and dismiss."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None
            assert isinstance(app.result, tuple)
            assert len(app.result) == 2


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

    async def test_typing_filters_models(self) -> None:
        """Typing in the filter input should filter models."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a filter
            await pilot.press("c", "l", "a", "u", "d", "e")
            await pilot.pause()

            assert screen._filter_text == "claude"

    def test_custom_model_spec_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """User can enter a custom provider:model spec."""
        screen = _model_selector_for_filtering()
        result: tuple[str, str] | None = None

        class FakeInput:
            value = "custom:my-model"

        screen._filtered_models = []
        monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: FakeInput())

        def record(value: tuple[str, str] | None) -> None:
            nonlocal result
            result = value

        monkeypatch.setattr(screen, "_dismiss_with_result", record)

        screen.action_select()

        assert result == ("custom:my-model", "custom")

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

    async def test_current_model_is_preselected(self) -> None:
        """Opening the selector should pre-select the current model, not first."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # The test app sets current model to "anthropic:claude-sonnet-4-5"
            # Find its index in the filtered models
            current_spec = "anthropic:claude-sonnet-4-5"
            expected_index = None
            for i, (model_spec, _) in enumerate(screen._filtered_models):
                if model_spec == current_spec:
                    expected_index = i
                    break

            assert expected_index is not None, f"{current_spec} not found in models"
            assert screen._selected_index == expected_index, (
                f"Expected current model at index {expected_index} to be selected, "
                f"but index {screen._selected_index} was selected instead"
            )

    async def test_clearing_filter_reselects_current_model(self) -> None:
        """Clearing the filter should re-select the current model."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Find the current model's index
            current_spec = "anthropic:claude-sonnet-4-5"
            current_index = None
            for i, (model_spec, _) in enumerate(screen._filtered_models):
                if model_spec == current_spec:
                    current_index = i
                    break
            assert current_index is not None

            # Type something that filters to no/few results
            await pilot.press("x", "y", "z")
            await pilot.pause()

            # Now clear the filter by backspacing
            await pilot.press("backspace", "backspace", "backspace")
            await pilot.pause()

            # Selection should be back to the current model
            assert screen._selected_index == current_index, (
                f"After clearing filter, expected index {current_index} "
                f"but got {screen._selected_index}"
            )


class TestModelSelectorFuzzyMatching:
    """Tests for fuzzy search filtering."""

    def test_fuzzy_exact_substring_still_works(self) -> None:
        """Exact substring matches should still work with fuzzy matching."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "claude"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert any("claude" in s for s in specs), (
            f"'claude' substring should match. Got: {specs}"
        )

    def test_fuzzy_subsequence_match(self) -> None:
        """Subsequence queries like 'cs45' should match 'claude-sonnet-4-5'."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "cs45"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert any("claude-sonnet-4-5" in s for s in specs), (
            f"'cs45' should fuzzy-match claude-sonnet-4-5. Got: {specs}"
        )

    def test_fuzzy_across_hyphen(self) -> None:
        """Queries should match across hyphens (e.g., 'gpt4' matches 'gpt-5.5')."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "gpt4"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert any("gpt-4" in s for s in specs), (
            f"'gpt4' should fuzzy-match gpt-4 models. Got: {specs}"
        )

    def test_fuzzy_case_insensitive(self) -> None:
        """Fuzzy matching should be case-insensitive."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "CLAUDE"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert any("claude" in s for s in specs), (
            f"'CLAUDE' should case-insensitively match claude models. Got: {specs}"
        )

    def test_fuzzy_no_match(self) -> None:
        """A query that matches nothing should produce an empty filtered list."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "xyz999qqq"
        screen._update_filtered_list()

        assert len(screen._filtered_models) == 0

    def test_fuzzy_ranking_better_match_first(self) -> None:
        """Better fuzzy matches should rank higher than weaker matches."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "claude"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert len(specs) > 0
        assert "claude" in specs[0].lower()

    def test_empty_filter_shows_all(self) -> None:
        """Empty filter should show all models in original order."""
        screen = _model_selector_for_filtering()
        screen._filter_text = ""
        screen._update_filtered_list()

        assert len(screen._filtered_models) == len(screen._all_models)

    def test_whitespace_filter_shows_all(self) -> None:
        """Whitespace-only filter should be treated as empty."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "   "
        screen._update_filtered_list()

        assert len(screen._filtered_models) == len(screen._all_models)

    def test_selection_clamped_on_filter(self) -> None:
        """Selected index should stay valid when filter results shrink."""
        screen = _model_selector_for_filtering()
        screen._selected_index = 5
        screen._filter_text = "claude"
        screen._update_filtered_list()

        assert screen._filtered_models, "Filter should match claude models"
        assert screen._selected_index == 0, (
            "Fuzzy filter should reset selection to best match (index 0)"
        )

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

    def test_fuzzy_space_separated_tokens(self) -> None:
        """Space-separated tokens should each fuzzy-match independently."""
        screen = _model_selector_for_filtering()
        screen._filter_text = "claude sonnet"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert any("claude" in s and "sonnet" in s for s in specs), (
            f"'claude sonnet' should match claude-sonnet models. Got: {specs}"
        )

    def test_fuzzy_matches_friendly_name_dotted_version(self) -> None:
        """A dotted version in the friendly name matches where the spec can't.

        `anthropic:claude-opus-4-8` hyphenates its version, so the "4.8" token
        cannot subsequence-match the spec. The friendly name "Claude Opus 4.8"
        (from the recommended list) supplies the dotted form.
        """
        screen = _model_selector_for_filtering()
        screen._filter_text = "opus 4.8"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert "anthropic:claude-opus-4-8" in specs, (
            f"friendly name should let 'opus 4.8' match. Got: {specs}"
        )

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

        Searches "chatgpt", which appears in neither the spec
        (`openai_codex:gpt-5.2`), the friendly model name ("GPT-5.2"), nor the
        provider key (`openai_codex`) — only in the resolved display label
        "OpenAI Codex (ChatGPT login)". So a match proves the provider-label
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
        screen._filter_text = "chatgpt"
        screen._update_filtered_list()

        specs = [spec for spec, _ in screen._filtered_models]
        assert specs == ["openai_codex:gpt-5.2"], (
            f"'chatgpt' should match only via the provider label. Got: {specs}"
        )

    async def test_tab_noop_when_no_matches(self) -> None:
        """Tab should do nothing when filter matches no models."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type gibberish that matches nothing
            for char in "xyz999qqq":
                await pilot.press(char)
            await pilot.pause()

            assert len(screen._filtered_models) == 0

            # Press tab - should not crash or change input
            await pilot.press("tab")
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.value == "xyz999qqq"

    async def test_tab_autocompletes_after_navigation(self) -> None:
        """Tab should autocomplete the model navigated to, not just index 0."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a partial filter
            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            assert len(screen._filtered_models) > 1, (
                "Need multiple claude matches to test navigation"
            )

            # Navigate down to select a different model
            await pilot.press("down")
            await pilot.pause()

            assert screen._selected_index == 1
            expected_spec, _ = screen._filtered_models[1]

            # Press tab - should autocomplete the navigated-to model
            await pilot.press("tab")
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.value == expected_spec

    async def test_tab_autocompletes_selected_model(self) -> None:
        """Tab should replace search text with the selected model spec."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a partial filter
            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            assert len(screen._filtered_models) > 0
            expected_spec, _ = screen._filtered_models[screen._selected_index]

            # Press tab - should replace filter text with selected model spec
            await pilot.press("tab")
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.value == expected_spec

    async def test_navigation_after_fuzzy_filter(self) -> None:
        """Arrow keys should work correctly on fuzzy-filtered results."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            count = len(screen._filtered_models)
            assert count > 1, "Need multiple claude matches to test navigation"
            initial = screen._selected_index
            await pilot.press("down")
            await pilot.pause()
            assert screen._selected_index == (initial + 1) % count


class TestFilteredModelsWidgetSync:
    """Tests that _filtered_models indices match _option_widgets after display."""

    def test_display_reorders_filtered_models_to_match_widgets(self) -> None:
        """After _update_display, _filtered_models order matches _option_widgets.

        Fuzzy search sorts by score, which can interleave providers. The
        display groups models by provider. _filtered_models must be reordered
        to match so that _update_footer looks up the correct model for the
        highlighted widget index.
        """
        screen = _bare_selector()
        # Simulate score-sorted filtered list that interleaves providers
        screen._filtered_models = [
            ("openai:gpt-5", "openai"),
            ("anthropic:claude-opus", "anthropic"),
            ("openai:gpt-4", "openai"),
            ("anthropic:claude-sonnet", "anthropic"),
        ]
        screen._selected_index = 0

        # Group by provider (same logic as _update_display)
        by_provider: dict[str, list[tuple[str, str]]] = {}
        for spec, prov in screen._filtered_models:
            by_provider.setdefault(prov, []).append((spec, prov))

        grouped: list[tuple[str, str]] = []
        for entries in by_provider.values():
            grouped.extend(entries)

        # Verify that grouping reorders: openai models cluster, then anthropic
        assert grouped == [
            ("openai:gpt-5", "openai"),
            ("openai:gpt-4", "openai"),
            ("anthropic:claude-opus", "anthropic"),
            ("anthropic:claude-sonnet", "anthropic"),
        ]
        # The original _filtered_models had anthropic:claude-opus at index 1
        # but after grouping it moves to index 2. Without the fix,
        # navigating to widget index 1 (openai:gpt-4) would look up
        # _filtered_models[1] = anthropic:claude-opus — wrong model.
        assert screen._filtered_models[1] != grouped[1]


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

    def test_provider_availability_rank_orders_states(self) -> None:
        """Usable < unknown < missing < not-installed, regardless of auth."""
        screen = _bare_selector()
        screen._install_extras = {"baseten": "baseten"}
        rank = screen._provider_availability_rank

        configured = rank(
            "openai_codex", self._status(ProviderAuthState.CONFIGURED, "openai_codex")
        )
        not_required = rank(
            "ollama", self._status(ProviderAuthState.NOT_REQUIRED, "ollama")
        )
        # Ambient/managed auth is just as usable as an explicit credential, so
        # both must collapse into the available tier rather than falling
        # through to the missing-credential rank.
        implicit = rank("bedrock", self._status(ProviderAuthState.IMPLICIT, "bedrock"))
        managed = rank(
            "custom_cls", self._status(ProviderAuthState.MANAGED, "custom_cls")
        )
        unknown = rank("custom", self._status(ProviderAuthState.UNKNOWN, "custom"))
        missing = rank("openai", self._status(ProviderAuthState.MISSING, "openai"))
        # A configured but not-installed provider still sinks to the bottom.
        uninstalled = rank(
            "baseten", self._status(ProviderAuthState.CONFIGURED, "baseten")
        )

        assert configured == not_required == implicit == managed
        assert configured < unknown < missing < uninstalled

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

    def test_opus_5_is_recommended(self) -> None:
        """Opus 5 should be discoverable in the frontier picker subset."""
        from deepagents_code.tui.widgets import model_selector

        all_models = [
            ("anthropic:claude-opus-5", "anthropic"),
            ("anthropic:claude-opus-4-5", "anthropic"),
        ]

        curated = ModelSelectorScreen._curate_models(all_models)

        assert model_selector._RECOMMENDED_MODELS["anthropic:claude-opus-5"] == (
            "Claude Opus 5"
        )
        assert curated == all_models[:1]

    def test_sonnet_5_is_recommended(self) -> None:
        """Sonnet 5 should be part of the frontier picker subset."""
        from deepagents_code.tui.widgets import model_selector

        all_models = [
            ("anthropic:claude-sonnet-5", "anthropic"),
            ("openrouter:anthropic/claude-sonnet-5", "openrouter"),
            ("openai:gpt-4o", "openai"),
        ]

        curated = ModelSelectorScreen._curate_models(all_models)

        assert "anthropic:claude-sonnet-5" in model_selector._RECOMMENDED_MODELS
        assert (
            "openrouter:anthropic/claude-sonnet-5" in model_selector._RECOMMENDED_MODELS
        )
        assert "anthropic:claude-sonnet-4-6" not in model_selector._RECOMMENDED_MODELS
        assert (
            "openrouter:anthropic/claude-sonnet-4.6"
            not in model_selector._RECOMMENDED_MODELS
        )
        assert curated == all_models[:2]

    def test_curated_models_filter_frontier_in_default_order(self) -> None:
        """Onboarding curation should preserve the model switcher's order."""
        all_models = [
            ("openai:gpt-5.6-luna", "openai"),
            ("anthropic:claude-sonnet-4-5", "anthropic"),
            ("openai:gpt-5.6-sol", "openai"),
            ("anthropic:claude-opus-5", "anthropic"),
            ("google_genai:gemini-3.6-flash", "google_genai"),
            ("anthropic:claude-opus-4-8", "anthropic"),
        ]

        curated = ModelSelectorScreen._curate_models(all_models)

        assert curated == [
            ("openai:gpt-5.6-luna", "openai"),
            ("openai:gpt-5.6-sol", "openai"),
            ("anthropic:claude-opus-5", "anthropic"),
            ("google_genai:gemini-3.6-flash", "google_genai"),
            ("anthropic:claude-opus-4-8", "anthropic"),
        ]

    def test_curated_models_limit_to_frontier_subset(self) -> None:
        """Current/default models outside the frontier subset should stay hidden."""
        all_models = [
            ("openai:gpt-5.3-codex", "openai"),
            ("anthropic:claude-opus-4-8", "anthropic"),
            ("anthropic:claude-sonnet-4-5", "anthropic"),
        ]

        curated = ModelSelectorScreen._curate_models(all_models)

        assert curated == [
            ("anthropic:claude-opus-4-8", "anthropic"),
        ]

    def test_curated_models_fall_back_when_frontier_unavailable(self) -> None:
        """Onboarding should show normal switcher entries if frontier is absent."""
        all_models = [
            ("anthropic:claude-sonnet-4-5", "anthropic"),
            ("openai:gpt-5.3-codex", "openai"),
        ]

        curated = ModelSelectorScreen._curate_models(all_models)

        assert curated == [
            ("anthropic:claude-sonnet-4-5", "anthropic"),
            ("openai:gpt-5.3-codex", "openai"),
        ]

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

    def test_deprecated_model_shows_tag(self) -> None:
        """Deprecated models should show a red (deprecated) tag."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:old-model",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
            status="deprecated",
        )
        from deepagents_code.theme import DARK_COLORS

        assert "(deprecated)" in label.plain
        assert DARK_COLORS.error in label.markup

    def test_non_deprecated_model_no_tag(self) -> None:
        """Models without deprecated status should not show the tag."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:claude-sonnet-4-5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
            status=None,
        )
        assert "(deprecated)" not in label.plain

    def test_other_status_renders_yellow(self) -> None:
        """Non-deprecated statuses (e.g., beta) render yellow, not red."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:new-model",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
            status="beta",
        )
        assert "(deprecated)" not in label.plain
        from deepagents_code.theme import DARK_COLORS

        assert "(beta)" in label.plain
        assert DARK_COLORS.warning in label.markup

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

    def test_no_auth_required_does_not_warning_style_model(self) -> None:
        """No-auth local providers should not look like missing credentials."""
        label = ModelSelectorScreen._format_option_label(
            "ollama:llama3",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.NOT_REQUIRED,
                provider="ollama",
                detail="local provider",
            ),
        )
        from deepagents_code.theme import DARK_COLORS

        assert DARK_COLORS.warning not in label.markup

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

    def test_selected_row_carries_no_inline_color(self) -> None:
        """A highlighted row leaves color to CSS so the accent stays readable."""
        from deepagents_code.theme import DARK_COLORS

        label = ModelSelectorScreen._format_option_label(
            "baseten:some-model",
            selected=True,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider="baseten",
                env_var="BASETEN_API_KEY",
            ),
            install_required=True,
            status="deprecated",
        )
        # Neither the install-required dim nor a semantic hue may survive the
        # ($primary bg, $background fg) flip that CSS applies to the row.
        assert "dim" not in label.markup
        assert DARK_COLORS.warning not in label.markup
        assert DARK_COLORS.error not in label.markup
        assert "(deprecated)" in label.plain

    def test_uses_display_name_when_provided(self) -> None:
        """Provider-grouped rows render the profile name, not the full spec."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:claude-sonnet-4-5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
            display_name="Claude Sonnet 4.5",
        )
        assert "Claude Sonnet 4.5" in label.plain
        assert "anthropic:claude-sonnet-4-5" not in label.plain

    def test_shows_full_spec_when_no_display_name(self) -> None:
        """With no display_name the full spec is shown (static-method default)."""
        label = ModelSelectorScreen._format_option_label(
            "anthropic:claude-sonnet-4-5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="anthropic",
                source=ProviderAuthSource.ENV,
            ),
        )
        assert "anthropic:claude-sonnet-4-5" in label.plain

    def test_appends_provider_tag_for_recent(self) -> None:
        """Recent rows show the name plus a `(provider)` tag, not the raw spec."""
        label = ModelSelectorScreen._format_option_label(
            "openai:gpt-5.5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="openai",
                source=ProviderAuthSource.ENV,
            ),
            display_name="GPT-5.5",
            provider_label="openai",
        )
        assert "GPT-5.5" in label.plain
        assert "(openai)" in label.plain
        assert "openai:gpt-5.5" not in label.plain

    def test_no_provider_tag_without_label(self) -> None:
        """Provider-grouped rows omit the `(provider)` tag."""
        label = ModelSelectorScreen._format_option_label(
            "openai:gpt-5.5",
            selected=False,
            current=False,
            auth_status=ProviderAuthStatus(
                state=ProviderAuthState.CONFIGURED,
                provider="openai",
                source=ProviderAuthSource.ENV,
            ),
            display_name="GPT-5.5",
        )
        assert "GPT-5.5" in label.plain
        assert "(openai)" not in label.plain


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

    def test_missing_auth_without_env_var_uses_generic_message(self) -> None:
        """MISSING without env_var falls back to a generic missing-creds label."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider="custom",
            ),
            get_glyphs(),
        )

        assert "missing credentials" in indicator

    def test_implicit_auth_uses_detail(self) -> None:
        """IMPLICIT state surfaces its detail string."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.IMPLICIT,
                provider="google_vertexai",
                detail="implicit auth",
            ),
            get_glyphs(),
        )

        assert indicator == "implicit auth"

    def test_managed_auth_uses_detail(self) -> None:
        """MANAGED state surfaces its detail string."""
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.MANAGED,
                provider="custom",
                detail="custom auth",
            ),
            get_glyphs(),
        )

        assert indicator == "custom auth"

    def test_unknown_auth_uses_question_glyph(self) -> None:
        """UNKNOWN state prefixes the detail with the question glyph."""
        glyphs = get_glyphs()
        indicator = ModelSelectorScreen._format_auth_indicator(
            ProviderAuthStatus(
                state=ProviderAuthState.UNKNOWN,
                provider="ollama",
                detail="remote endpoint; set OLLAMA_API_KEY if auth is required",
            ),
            glyphs,
        )

        assert indicator.startswith(glyphs.question)
        assert "OLLAMA_API_KEY" in indicator


class TestGetModelStatus:
    """Tests for _get_model_status profile lookup."""

    def test_returns_status_when_present(self) -> None:
        """Status is returned when profile entry has the key."""
        screen = _bare_selector()
        screen._profiles = {
            "anthropic:old-model": ModelProfileEntry(
                profile={"status": "deprecated"},
                overridden_keys=frozenset(),
            ),
        }
        assert screen._get_model_status("anthropic:old-model") == "deprecated"

    def test_returns_none_when_no_profile_entry(self) -> None:
        """None is returned when model spec is not in profiles."""
        screen = _bare_selector()
        screen._profiles = {}
        assert screen._get_model_status("anthropic:missing") is None

    def test_returns_none_when_no_status_key(self) -> None:
        """None is returned when profile exists but has no status key."""
        screen = _bare_selector()
        screen._profiles = {
            "anthropic:model": ModelProfileEntry(
                profile={"max_input_tokens": 200000},
                overridden_keys=frozenset(),
            ),
        }
        assert screen._get_model_status("anthropic:model") is None

    def test_returns_none_when_profile_empty(self) -> None:
        """None is returned when profile dict is empty."""
        screen = _bare_selector()
        screen._profiles = {
            "anthropic:model": ModelProfileEntry(
                profile={},
                overridden_keys=frozenset(),
            ),
        }
        assert screen._get_model_status("anthropic:model") is None


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


class TestGetModelDisplayName:
    """Tests for _get_model_display_name resolution."""

    def test_uses_the_injected_recommendation_set(self) -> None:
        """A caller-supplied set is used, not the main-model default.

        `/auto model` passes `_AUTO_CLASSIFIER_RECOMMENDED_MODELS`, so falling
        back to the main-model set would label classifier rows from the wrong
        catalog.
        """
        screen = ModelSelectorScreen.__new__(ModelSelectorScreen)
        screen._profiles = {}
        screen._recommended_models = {"openai:tiny": "Tiny Reviewer"}

        assert screen._get_model_display_name("openai:tiny") == "Tiny Reviewer"
        # A spec only in the main-model catalog must not resolve here.
        assert screen._get_model_display_name("anthropic:claude-opus-5") == (
            "claude-opus-5"
        )

    def test_returns_profile_name_when_present(self) -> None:
        """The human-readable profile name wins over the raw model id."""
        screen = _bare_selector()
        screen._profiles = {
            "anthropic:claude-sonnet-4-5": ModelProfileEntry(
                profile={"name": "Claude Sonnet 4.5"},
                overridden_keys=frozenset(),
            ),
        }
        assert (
            screen._get_model_display_name("anthropic:claude-sonnet-4-5")
            == "Claude Sonnet 4.5"
        )

    def test_falls_back_to_model_id_when_no_name(self) -> None:
        """A profile without a `name` key falls back to the model portion."""
        screen = _bare_selector()
        screen._profiles = {
            "anthropic:claude-sonnet-4-5": ModelProfileEntry(
                profile={"max_input_tokens": 200000},
                overridden_keys=frozenset(),
            ),
        }
        assert (
            screen._get_model_display_name("anthropic:claude-sonnet-4-5")
            == "claude-sonnet-4-5"
        )

    @pytest.mark.parametrize(
        ("spec", "name"),
        [
            ("fireworks:accounts/fireworks/models/kimi-k3", "Kimi K3"),
            ("meta:muse-spark-1.1", "Muse Spark 1.1"),
            ("meta:muse-spark-1.2", "Muse Spark 1.2"),
            ("openai:gpt-5.6-luna", "GPT-5.6 Luna"),
            ("openai:gpt-5.6-sol", "GPT-5.6 Sol"),
            ("openai:gpt-5.6-terra", "GPT-5.6 Terra"),
            ("openai_codex:gpt-5.6-luna", "GPT-5.6 Luna"),
            ("openai_codex:gpt-5.6-sol", "GPT-5.6 Sol"),
            ("openai_codex:gpt-5.6-terra", "GPT-5.6 Terra"),
            ("xai:grok-4.5", "Grok 4.5"),
        ],
    )
    def test_uses_recommended_name_when_no_profile(self, spec: str, name: str) -> None:
        """Uninstalled recommendations use the hardcoded name, not the raw id."""
        from deepagents_code.tui.widgets import model_selector

        screen = _bare_selector()
        screen._profiles = {}
        assert spec in model_selector._RECOMMENDED_MODELS
        assert screen._get_model_display_name(spec) == name

    def test_profile_name_wins_over_recommended_name(self) -> None:
        """A loaded profile's `name` takes precedence over the hardcoded one."""
        from deepagents_code.tui.widgets import model_selector

        spec = "openai:gpt-5.6-luna"
        assert spec in model_selector._RECOMMENDED_MODELS
        screen = _bare_selector()
        screen._profiles = {
            spec: ModelProfileEntry(
                profile={"name": "GPT-5.6 Luna (from profile)"},
                overridden_keys=frozenset(),
            ),
        }
        assert screen._get_model_display_name(spec) == "GPT-5.6 Luna (from profile)"

    def test_falls_back_to_model_id_when_not_recommended(self) -> None:
        """A non-recommended spec absent from profiles falls back to the model."""
        screen = _bare_selector()
        screen._profiles = {}
        assert (
            screen._get_model_display_name("openai:some-unlisted-model")
            == "some-unlisted-model"
        )

    def test_ignores_empty_name(self) -> None:
        """An empty `name` string falls back rather than rendering blank."""
        screen = _bare_selector()
        screen._profiles = {
            "openai:some-unlisted-model": ModelProfileEntry(
                profile={"name": ""},
                overridden_keys=frozenset(),
            ),
        }
        assert (
            screen._get_model_display_name("openai:some-unlisted-model")
            == "some-unlisted-model"
        )

    def test_preserves_colon_in_model_id_fallback(self) -> None:
        """Only the leading `provider:` is stripped in the fallback path."""
        screen = _bare_selector()
        screen._profiles = {}
        assert (
            screen._get_model_display_name("ollama:some-model:cloud")
            == "some-model:cloud"
        )

    def test_returns_bare_spec_unchanged(self) -> None:
        """A spec without a `provider:` prefix is returned as-is."""
        screen = _bare_selector()
        screen._profiles = {}
        assert screen._get_model_display_name("gpt-5.5") == "gpt-5.5"


class TestModelDetailFooter:
    """Tests for the model detail footer in the selector."""

    def test_format_footer_full_profile(self) -> None:
        """Full profile renders token counts, modalities, and capabilities."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
                "text_inputs": True,
                "image_inputs": True,
                "pdf_inputs": False,
                "reasoning_output": True,
                "tool_calling": True,
                "structured_output": False,
            },
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        text = str(result)
        assert "200.0K" in text
        assert "64.0K" in text
        assert "text" in text
        assert "image" in text
        assert "tool calling" in text
        assert "reasoning" in text
        # No override marker
        assert "* =" not in text

    def test_format_footer_no_profile(self) -> None:
        """None profile shows 'Model profile not available'."""
        from deepagents_code.config import UNICODE_GLYPHS

        result = ModelSelectorScreen._format_footer(None, UNICODE_GLYPHS)
        assert "Model profile not available :(" in str(result)

    def test_format_footer_overridden_fields(self) -> None:
        """Overridden fields show yellow * marker and override legend."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={
                "max_input_tokens": 100000,
                "max_output_tokens": 64000,
                "tool_calling": True,
            },
            overridden_keys=frozenset({"max_input_tokens"}),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        text = str(result)
        assert "*" in text
        assert "= override" in text
        from deepagents_code.theme import DARK_COLORS

        assert DARK_COLORS.warning in result.markup

    def test_format_footer_partial_profile(self) -> None:
        """Profile with only token counts still renders without crash."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": 4096},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        text = str(result)
        assert "4096" in text or "4.1K" in text or "4.0K" in text
        # Should not crash and should have content
        assert "No profile data" not in text

    def test_format_footer_empty_profile(self) -> None:
        """Empty profile dict shows 'Model profile not available'."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "Model profile not available :(" in str(result)

    def test_format_footer_override_on_non_displayed_key(self) -> None:
        """Override on a non-displayed key should not show legend."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": 4096, "supports_thinking": True},
            overridden_keys=frozenset({"supports_thinking"}),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "= override" not in str(result)

    def test_format_footer_non_numeric_tokens(self) -> None:
        """Non-numeric token values render gracefully instead of crashing."""
        from deepagents_code.config import UNICODE_GLYPHS
        from deepagents_code.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": "unlimited", "max_output_tokens": 64000},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        text = str(result)
        assert "unlimited" in text
        assert "64.0K" in text

    async def test_footer_updates_on_navigation(self) -> None:
        """Footer content changes when navigating to a different model."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            footer = screen.query_one("#model-detail-footer", Static)
            initial_content = str(footer.content)
            assert "Context:" in initial_content or "No profile" in initial_content

            await pilot.press("down")
            await pilot.pause()

            updated_content = str(footer.content)
            assert "Context:" in updated_content or "No profile" in updated_content

    async def test_footer_shows_on_mount(self) -> None:
        """Footer is populated with structural content on initial mount."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            footer = screen.query_one("#model-detail-footer", Static)
            content = str(footer.content)
            assert "Context:" in content or "No profile" in content

    async def test_footer_no_model_when_filter_empty(self) -> None:
        """Footer shows 'No model selected' when filter matches nothing."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            for char in "xyz999qqq":
                await pilot.press(char)
            # Pump several frames so all deferred call_after_refresh
            # callbacks complete after the last keystroke
            for _ in range(5):
                await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)
            assert len(screen._filtered_models) == 0
            footer = screen.query_one("#model-detail-footer", Static)
            assert "No model selected" in str(footer.content)


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

    async def test_curated_screen_loads_uninstalled_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Onboarding includes install-required recommended models."""
        from deepagents_code.tui.widgets import model_selector

        captured: dict[str, bool] = {}

        def load_model_data(
            _cli_override: dict[str, Any] | None,
            *,
            include_uninstalled: bool = True,
            include_recent: bool = True,
            recommended_models: Mapping[str, str] | None = None,
            # Mirrors production: required and nullable, so the stub cannot
            # outlive the contract it fakes.
            default_scope: model_selector.DefaultModelScope | None,
        ) -> model_selector._ModelData:
            del recommended_models, default_scope
            captured["include_uninstalled"] = include_uninstalled
            captured["include_recent"] = include_recent
            return model_selector._ModelData(
                [("baseten:zai-org/GLM-5.2", "baseten")],
                None,
                {},
                [],
                {"baseten": "baseten"},
            )

        monkeypatch.setattr(
            ModelSelectorScreen,
            "_load_model_data",
            staticmethod(load_model_data),
        )

        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                ModelSelectorScreen(
                    current_model="openai:gpt-5.5",
                    current_provider="openai",
                    curated=True,
                    default_scope=MAIN_MODEL_DEFAULT_SCOPE,
                )
            )
            await pilot.pause()

        assert captured["include_uninstalled"] is True
        # Curated onboarding must skip the recent-models MRU at the call site,
        # independent of the rendering-level guard in
        # test_recent_section_hidden_during_onboarding.
        assert captured["include_recent"] is False

    async def test_load_model_data_surfaces_uninstalled_recommended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recommended models from uninstalled providers are surfaced."""
        from deepagents_code import config_manifest
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector, "get_available_models", lambda: {"openai": ["gpt-5.5"]}
        )
        monkeypatch.setattr(
            config_manifest,
            "is_provider_package_installed",
            lambda provider: provider not in {"baseten", "ollama"},
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = {spec for spec, _ in all_models}
        assert any(spec.startswith("baseten:") for spec in specs)
        assert any(spec.startswith("ollama:") for spec in specs)
        assert install_extras.get("baseten") == "baseten"
        assert install_extras.get("ollama") == "ollama"

    async def test_load_model_data_orders_installed_recommended_before_uninstalled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installed-provider recommendations sort before install-required rows."""
        from deepagents_code.tui.widgets import model_selector

        installed_spec = "ollama:glm-5.2:cloud"
        uninstalled_spec = "fireworks:accounts/fireworks/models/deepseek-v4-pro"
        monkeypatch.setattr(
            model_selector,
            "_RECOMMENDED_MODELS",
            {installed_spec: "GLM 5.2", uninstalled_spec: "DeepSeek V4 Pro"},
        )
        monkeypatch.setattr(
            model_selector,
            "get_available_models",
            lambda: {"ollama": ["local-model"]},
        )
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda package: object() if package == "langchain_ollama" else None,
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=True, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = [model_spec for model_spec, _ in all_models]
        assert specs.index(installed_spec) < specs.index(uninstalled_spec)
        assert install_extras.get("fireworks") == "fireworks"
        assert "ollama" not in install_extras

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

    async def test_load_model_data_skips_uninstalled_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicitly disabling uninstalled recommendations hides providers."""
        from deepagents_code.tui.widgets import model_selector

        monkeypatch.setattr(
            model_selector, "get_available_models", lambda: {"openai": ["gpt-5.5"]}
        )

        all_models, _default, _profiles, _recent, install_extras = (
            ModelSelectorScreen._load_model_data(
                None, include_uninstalled=False, default_scope=MAIN_MODEL_DEFAULT_SCOPE
            )
        )

        specs = {spec for spec, _ in all_models}
        assert not any(spec.startswith("baseten:") for spec in specs)
        assert install_extras == {}

    async def test_load_model_data_respects_disabled_uninstalled_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
        tmp_path: Path,
    ) -> None:
        """Disabled providers stay hidden from install suggestions."""
        from deepagents_code import config_manifest, model_config
        from deepagents_code.tui.widgets import model_selector

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[models.providers.baseten]
enabled = false
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
        model_config.clear_caches()
        request.addfinalizer(model_config.clear_caches)

        monkeypatch.setattr(
            model_selector, "get_available_models", lambda: {"openai": ["gpt-5.5"]}
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

        specs = {spec for spec, _ in all_models}
        assert not any(spec.startswith("baseten:") for spec in specs)
        assert "baseten" not in install_extras

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

    async def test_confirm_install_sets_pending_extra_and_dismisses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirming install records the extra and dismisses with the model."""
        import importlib.util

        if importlib.util.find_spec("langchain_baseten") is not None:
            pytest.skip("langchain_baseten is installed in this environment")

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

            screen._prompt_install_provider(
                "baseten:moonshotai/Kimi-K2.7-Code", "baseten", "baseten"
            )
            on_confirm = pushed[0][1]
            assert on_confirm is not None
            on_confirm(True)
            await pilot.pause()

        assert screen.pending_install_extra == "baseten"
        assert app.dismissed is True
        assert app.result == ("baseten:moonshotai/Kimi-K2.7-Code", "baseten")

    async def test_decline_install_stays_on_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declining install keeps the selector open and records no extra."""
        import importlib.util

        if importlib.util.find_spec("langchain_baseten") is not None:
            pytest.skip("langchain_baseten is installed in this environment")

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

            screen._prompt_install_provider(
                "baseten:moonshotai/Kimi-K2.7-Code", "baseten", "baseten"
            )
            on_confirm = pushed[0][1]
            assert on_confirm is not None
            on_confirm(False)
            await pilot.pause()

            assert screen.pending_install_extra is None
            assert app.dismissed is False

    async def test_fuzzy_ranks_installed_above_uninstalled(self) -> None:
        """Installed providers outrank install-required suggestions in search."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Both specs fuzzy-match "gpt"; only baseten needs an install, so
            # openai must rank first despite baseten being an equal/better match.
            screen._curated = False
            screen._install_extras = {"baseten": "baseten"}
            screen._unfiltered_models = [
                ("baseten:gpt-thing", "baseten"),
                ("openai:gpt-5.5", "openai"),
            ]
            screen._all_models = list(screen._unfiltered_models)
            screen._filter_text = "gpt"
            screen._update_filtered_list()

            providers = [provider for _spec, provider in screen._filtered_models]
            assert "openai" in providers
            assert "baseten" in providers
            assert providers.index("openai") < providers.index("baseten")

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
