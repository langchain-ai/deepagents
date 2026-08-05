"""Unit tests for langchain_cloud_run package import."""

from langchain_cloud_run import CloudRunSandbox, __version__


def test_import_exports() -> None:
    """Test that public symbols are exported at package root."""
    assert CloudRunSandbox is not None
    assert isinstance(__version__, str)
