"""Tests for the CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from typer.testing import CliRunner

from chatlas_agents.cli import app, version, init, setup_logging, CHATLAS_AGENTS_VERSION


runner = CliRunner()


def test_cli_app_exists():
    """Test that the CLI app is properly defined."""
    assert app is not None
    assert hasattr(app, 'registered_commands')


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


def test_version_command():
    """Test the version command output."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "ChATLAS Agents" in result.stdout
    assert CHATLAS_AGENTS_VERSION in result.stdout


def test_init_command():
    """Test the init command creates a config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test-config.env"
        result = runner.invoke(app, ["init", "--output", str(config_path)])
        
        assert result.exit_code == 0
        assert config_path.exists()
        
        # Verify config file content
        content = config_path.read_text()
        assert "CHATLAS_MCP_URL" in content
        assert "CHATLAS_MCP_TIMEOUT" in content
        assert "OPENAI_API_KEY" in content


def test_help_command():
    """Test that help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    
    # Strip ANSI color codes for easier assertion
    import re
    output = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    
    assert "ChATLAS AI agents" in output
    assert "--agent" in output or "-a" in output
    assert "--sandbox" in output
