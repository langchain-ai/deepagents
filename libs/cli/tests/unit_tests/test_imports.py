"""Test importing files."""


def test_imports() -> None:
    """Test importing deepagents modules."""
    from deepagents_cli import (
        agent,
        integrations,
    )
    from deepagents_cli.main import cli_main
