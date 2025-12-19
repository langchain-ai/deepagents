"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch
from chatlas_agents.cli import app, version, init, setup_logging, CHATLAS_AGENTS_VERSION


def test_cli_app_exists():
    """Test that the CLI app is properly defined."""
    assert app is not None
    assert hasattr(app, 'command')


def test_version_command_exists():
    """Test that version command exists."""
    assert version is not None
    assert callable(version)


def test_init_command_exists():
    """Test that init command exists."""
    assert init is not None
    assert callable(init)


def test_setup_logging():
    """Test that setup_logging works correctly."""
    # Should not raise an exception
    setup_logging(verbose=False)
    setup_logging(verbose=True)


def test_version_info():
    """Test that version info is available."""
    assert CHATLAS_AGENTS_VERSION is not None
    assert isinstance(CHATLAS_AGENTS_VERSION, str)
