"""Compatibility tests for the `fetch_url` tool."""

from deepagents.tools import fetch_url as sdk_fetch_url

from deepagents_code.tools import fetch_url


def test_fetch_url_reexports_sdk_tool() -> None:
    """Existing dcode imports resolve to the SDK-owned implementation."""
    assert fetch_url is sdk_fetch_url
