"""Tests for lazy package imports."""


def test_cli_main_via_package() -> None:
    """Package-level `__getattr__` resolves `cli_main` lazily."""
    from deepagents_code import cli_main

    assert callable(cli_main)
