"""Tests for ModelSelectorScreen."""

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_cli.widgets.model_selector import ModelSelectorScreen


class ModelSelectorTestApp(App):
    """Test app for ModelSelectorScreen."""

    def __init__(self) -> None:
        super().__init__()
        self.result: tuple[str, str] | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the model selector screen."""

        def handle_result(result: tuple[str, str] | None) -> None:
            self.result = result
            self.dismissed = True

        screen = ModelSelectorScreen(
            current_model="claude-sonnet-4-5",
            current_provider="anthropic",
        )
        self.push_screen(screen, handle_result)


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

    async def test_custom_model_spec_entry(self) -> None:
        """User can enter a custom provider:model spec."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Type a custom model spec
            for char in "custom:my-model":
                await pilot.press(char)
            await pilot.pause()

            # Press enter to select
            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result == ("custom:my-model", "custom")

    async def test_enter_selects_highlighted_model_not_filter_text(self) -> None:
        """Enter selects highlighted model, not raw filter text."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type a partial spec with colon that matches existing models
            for char in "anthropic:claude":
                await pilot.press(char)
            await pilot.pause()

            # Should have filtered results
            assert len(screen._filtered_models) > 0

            # Press enter - should select the highlighted model, not raw text
            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None
            # Result should be a full model spec from the list, not "anthropic:claude"
            model_spec, provider = app.result
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

    async def test_fuzzy_exact_substring_still_works(self) -> None:
        """Exact substring matches should still work with fuzzy matching."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert any("claude" in s for s in specs), (
                f"'claude' substring should match. Got: {specs}"
            )

    async def test_fuzzy_subsequence_match(self) -> None:
        """Subsequence queries like 'cs45' should match 'claude-sonnet-4-5'."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "cs45":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert any("claude-sonnet-4-5" in s for s in specs), (
                f"'cs45' should fuzzy-match claude-sonnet-4-5. Got: {specs}"
            )

    async def test_fuzzy_across_hyphen(self) -> None:
        """Queries should match across hyphens (e.g., 'gpt4' matches 'gpt-4o')."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "gpt4":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert any("gpt-4" in s for s in specs), (
                f"'gpt4' should fuzzy-match gpt-4 models. Got: {specs}"
            )

    async def test_fuzzy_case_insensitive(self) -> None:
        """Fuzzy matching should be case-insensitive."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Type uppercase "CLAUDE"
            for char in "CLAUDE":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert any("claude" in s for s in specs), (
                f"'CLAUDE' should case-insensitively match claude models. Got: {specs}"
            )

    async def test_fuzzy_no_match(self) -> None:
        """A query that matches nothing should produce an empty filtered list."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "xyz999qqq":
                await pilot.press(char)
            await pilot.pause()

            assert len(screen._filtered_models) == 0

    async def test_fuzzy_ranking_better_match_first(self) -> None:
        """Better fuzzy matches should rank higher than weaker matches."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert len(specs) > 0
            # First result should be a strong match containing the query
            assert "claude" in specs[0].lower()

    async def test_empty_filter_shows_all(self) -> None:
        """Empty filter should show all models in original order."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            total = len(screen._filtered_models)
            assert total == len(screen._all_models)

    async def test_whitespace_filter_shows_all(self) -> None:
        """Whitespace-only filter should be treated as empty."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            await pilot.press("space", "space", "space")
            await pilot.pause()

            assert len(screen._filtered_models) == len(screen._all_models)

    async def test_selection_clamped_on_filter(self) -> None:
        """Selected index should stay valid when filter results shrink."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # Move selection down several times
            for _ in range(5):
                await pilot.press("down")
            await pilot.pause()

            # Now type a filter that produces fewer results
            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            assert screen._filtered_models, "Filter should match claude models"
            assert screen._selected_index == 0, (
                "Fuzzy filter should reset selection to best match (index 0)"
            )

    async def test_enter_selects_fuzzy_result(self) -> None:
        """Pressing Enter after fuzzy filtering should select the top result."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            for char in "claude":
                await pilot.press(char)
            await pilot.pause()

            assert len(screen._filtered_models) > 0

            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None
            model_spec, _ = app.result
            assert "claude" in model_spec.lower()

    async def test_fuzzy_space_separated_tokens(self) -> None:
        """Space-separated tokens should each fuzzy-match independently."""
        app = ModelSelectorTestApp()
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectorScreen)

            # "claude sonnet" should match models containing both subsequences
            for char in "claude sonnet":
                await pilot.press(char)
            await pilot.pause()

            specs = [spec for spec, _ in screen._filtered_models]
            assert any("claude" in s and "sonnet" in s for s in specs), (
                f"'claude sonnet' should match claude-sonnet models. Got: {specs}"
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


class TestModelDetailFooter:
    """Tests for the model detail footer in the selector."""

    def test_format_footer_full_profile(self) -> None:
        """Full profile renders token counts, modalities, and capabilities."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

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
        assert "200.0K" in result
        assert "64.0K" in result
        assert "text" in result
        assert "image" in result
        assert "tool calling" in result
        assert "reasoning" in result
        # No override marker
        assert "* =" not in result

    def test_format_footer_no_profile(self) -> None:
        """None profile shows 'No profile data available'."""
        from deepagents_cli.config import UNICODE_GLYPHS

        result = ModelSelectorScreen._format_footer(None, UNICODE_GLYPHS)
        assert "No profile data available" in result

    def test_format_footer_overridden_fields(self) -> None:
        """Overridden fields show yellow * marker and override legend."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={
                "max_input_tokens": 100000,
                "max_output_tokens": 64000,
                "tool_calling": True,
            },
            overridden_keys=frozenset({"max_input_tokens"}),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "[yellow]*" in result
        assert "= override" in result

    def test_format_footer_partial_profile(self) -> None:
        """Profile with only token counts still renders without crash."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": 4096},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "4096" in result or "4.1K" in result or "4.0K" in result
        # Should not crash and should have content
        assert "No profile data" not in result

    def test_format_footer_empty_profile(self) -> None:
        """Empty profile dict shows 'Context: unknown' fallback."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "Context: unknown" in result
        assert "No profile data" not in result

    def test_format_footer_override_on_non_displayed_key(self) -> None:
        """Override on a non-displayed key should not show legend."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": 4096, "supports_thinking": True},
            overridden_keys=frozenset({"supports_thinking"}),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "= override" not in result

    def test_format_footer_non_numeric_tokens(self) -> None:
        """Non-numeric token values render gracefully instead of crashing."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={"max_input_tokens": "unlimited", "max_output_tokens": 64000},
            overridden_keys=frozenset(),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "unlimited" in result
        assert "64.0K" in result

    def test_format_footer_cli_override(self) -> None:
        """CLI-overridden key shows yellow * marker and override legend."""
        from deepagents_cli.config import UNICODE_GLYPHS
        from deepagents_cli.model_config import ModelProfileEntry

        entry = ModelProfileEntry(
            profile={
                "max_input_tokens": 4096,
                "max_output_tokens": 64000,
                "tool_calling": True,
            },
            overridden_keys=frozenset({"max_input_tokens"}),
        )
        result = ModelSelectorScreen._format_footer(entry, UNICODE_GLYPHS)
        assert "[yellow]*" in result
        assert "= override" in result

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
