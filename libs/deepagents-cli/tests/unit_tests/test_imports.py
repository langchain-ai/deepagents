"""Test importing files."""


def test_imports() -> None:
    """Test importing deepagents modules."""
    from deepagents_cli import (
        agent,  # noqa: F401
        integrations,  # noqa: F401
    )
    from deepagents_cli.main import cli_main  # noqa: F401
