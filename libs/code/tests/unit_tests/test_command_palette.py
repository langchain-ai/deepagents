"""Tests for the Textual command palette integration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from deepagents_code.app import DeepAgentsApp, ModelProvider

if TYPE_CHECKING:
    import pytest


class TestCommandPalette:
    """Tests for the Textual command palette integration."""

    def test_command_palette_is_enabled_with_model_provider(self) -> None:
        """The app should expose the model-switching provider to Textual."""
        assert DeepAgentsApp.ENABLE_COMMAND_PALETTE is True
        assert ModelProvider in DeepAgentsApp.COMMANDS

    async def test_model_provider_keeps_existing_input_when_switching(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Palette model switches should not submit or clear draft input."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")

        monkeypatch.setattr(
            "deepagents_code.widgets.model_selector.ModelSelectorScreen._load_model_data",
            staticmethod(
                lambda _override: (
                    [("openai:test-gpt", "openai")],
                    None,
                    {},
                    [],
                )
            ),
        )

        async with app.run_test():
            assert app._chat_input is not None
            app._chat_input.value = "draft prompt"
            provider = ModelProvider(app.screen)
            await provider.startup()
            hits = [hit async for hit in provider.search("gpt")]

            hits[0].command()

            assert app._chat_input.value == "draft prompt"

    async def test_model_provider_search_switches_matching_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Choosing a model palette hit should request that model switch."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        requested: list[str] = []

        monkeypatch.setattr(
            "deepagents_code.widgets.model_selector.ModelSelectorScreen._load_model_data",
            staticmethod(
                lambda _override: (
                    [
                        ("anthropic:test-sonnet", "anthropic"),
                        ("openai:test-gpt", "openai"),
                    ],
                    None,
                    {},
                    [],
                )
            ),
        )

        def record_model_switch(
            model_spec: str,
            *,
            extra_kwargs: dict[str, object] | None = None,
        ) -> None:
            del extra_kwargs
            requested.append(model_spec)

        monkeypatch.setattr(app, "_request_model_switch", record_model_switch)

        async with app.run_test():
            provider = ModelProvider(app.screen)
            await provider.startup()
            hits = [hit async for hit in provider.search("gpt")]

            assert [hit.text for hit in hits] == ["Switch model: openai:test-gpt"]
            hits[0].command()

        assert requested == ["openai:test-gpt"]

    async def test_model_provider_discovery_lists_models(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty palette query should discover switchable models."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")

        monkeypatch.setattr(
            "deepagents_code.widgets.model_selector.ModelSelectorScreen._load_model_data",
            staticmethod(
                lambda _override: (
                    [("openai:test-gpt", "openai")],
                    None,
                    {},
                    [],
                )
            ),
        )

        async with app.run_test():
            provider = ModelProvider(app.screen)
            await provider.startup()
            hits = [hit async for hit in provider.discover()]

        assert [hit.text for hit in hits] == ["Switch model: openai:test-gpt"]
