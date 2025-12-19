"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch
from chatlas_agents.cli import show_splash, SPLASH_ART


def test_splash_art_defined():
    """Test that the splash art is properly defined."""
    assert SPLASH_ART is not None
    assert isinstance(SPLASH_ART, str)
    # Splash art uses ASCII art representation and mentions "AI Agents for ATLAS"
    assert "AI Agents for ATLAS" in SPLASH_ART or "Powered by DeepAgents" in SPLASH_ART
    assert "v." in SPLASH_ART  # Version number is included


def test_show_splash():
    """Test that show_splash prints the splash art."""
    with patch("chatlas_agents.cli.console") as mock_console:
        show_splash()
        # Verify that print was called at least once
        assert mock_console.print.call_count >= 1
        # Verify the splash art was printed
        mock_console.print.assert_any_call(SPLASH_ART)
