"""Integration tests for Codex OAuth.

These tests require local Codex credentials. Run
``deepagents auth login --provider codex`` first.
"""

import pytest

from deepagents_codex.store import CodexAuthStore


@pytest.fixture
def codex_credentials():
    store = CodexAuthStore()
    creds = store.load()
    if creds is None:
        pytest.skip(
            "No local Codex credentials found. "
            "Run 'deepagents auth login --provider codex' first."
        )
    return creds


def test_credentials_loadable(codex_credentials) -> None:
    assert codex_credentials.access_token
    assert codex_credentials.refresh_token
