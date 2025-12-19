"""Test package initialization."""


def test_version():
    """Test version is importable."""
    from chatlas_agents import __version__
    assert __version__ == "0.1.0"
