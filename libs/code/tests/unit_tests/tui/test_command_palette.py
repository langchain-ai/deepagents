"""Tests for ModelProvider and CommandPalette integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.command import DiscoveryHit, Hit

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.command_palette import ModelProvider


class CommandPaletteTestApp(App):
    """Test app for CommandPalette registration and functionality."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.switched_model: str | None = None
        self.current_spec: str | None = "openai:gpt-4"

    def _effective_model_spec(self) -> str | None:
        """Mock _effective_model_spec."""
        return self.current_spec

    async def _switch_model(
        self,
        model_spec: str,
        *,
        extra_kwargs: dict[str, Any] | None = None,
        announce_unchanged: bool = True,
        persist: bool = True,
        from_resume: bool = False,
    ) -> None:
        """Mock _switch_model for testing."""
        del extra_kwargs, announce_unchanged, persist, from_resume
        self.switched_model = model_spec


@pytest.mark.asyncio
async def test_model_provider_logic() -> None:
    """Test that ModelProvider correctly discovers and searches models."""

    mock_available = {
        "anthropic": ["claude-3-sonnet", "claude-3-opus"],
        "openai": ["gpt-4"],
    }
    mock_profiles = {
        "anthropic:claude-3-sonnet": {"profile": {"name": "Claude 3 Sonnet"}},
        "anthropic:claude-3-opus": {"profile": {"name": "Claude 3 Opus"}},
        "openai:gpt-4": {"profile": {"name": "GPT-4"}},
    }

    # Patch model_config functions that ModelProvider calls.
    with patch(
        "deepagents_code.tui.command_palette.get_available_models",
        return_value=mock_available,
    ), patch(
        "deepagents_code.tui.command_palette.get_model_profiles",
        return_value=mock_profiles,
    ):
        app = CommandPaletteTestApp()
        from textual.app import active_app

        active_app.set(app)
        provider = ModelProvider(app)

        # Test discover()
        discovery_hits = [hit async for hit in provider.discover()]
        assert len(discovery_hits) == 3
        assert all(isinstance(hit, DiscoveryHit) for hit in discovery_hits)

        # DiscoveryHit uses 'display' for the label
        display_names = {str(hit.display) for hit in discovery_hits}
        assert "Claude 3 Sonnet" in display_names
        assert "Claude 3 Opus" in display_names
        # Check that current model is indicated
        assert "GPT-4 (current)" in display_names

        # Test search() with a query
        search_hits = [hit async for hit in provider.search("claude")]
        assert len(search_hits) == 2
        assert all(isinstance(hit, Hit) for hit in search_hits)

        # Verify help text includes provider
        assert any("anthropic" in hit.help.lower() for hit in search_hits)


@pytest.mark.asyncio
async def test_command_palette_registration() -> None:
    """Test that the command palette is enabled and provider is registered."""
    assert DeepAgentsApp.ENABLE_COMMAND_PALETTE is True
    assert "deepagents_code.tui.command_palette:ModelProvider" in DeepAgentsApp.COMMANDS
