"""Tests for the __main__ module."""


def test_main_module_imports():
    """Test that __main__ module can be imported."""
    # This test ensures the __main__.py file is properly structured
    import chatlas_agents.__main__

    assert hasattr(chatlas_agents.__main__, 'main')


def test_main_module_calls_cli_main():
    """Test that __main__ module imports main from cli."""
    import chatlas_agents.__main__
    import chatlas_agents.cli

    # Verify that __main__.main is the same as cli.main
    assert chatlas_agents.__main__.main is chatlas_agents.cli.main
