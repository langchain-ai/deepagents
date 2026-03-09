"""Tests for the optional E2B CLI provider."""

from __future__ import annotations

import importlib

import pytest


def test_provider_module_import_is_safe_without_e2b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the module should not require the optional dependency."""
    module = importlib.import_module("deepagents_cli.integrations.e2b")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ImportError, match="e2b package is required"):
        module.E2BProvider(api_key="test-key")
