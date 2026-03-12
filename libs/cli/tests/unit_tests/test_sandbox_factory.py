"""Tests for sandbox factory provider routing."""

import sys
import types
from typing import Any, cast

import pytest

from deepagents_cli.integrations.sandbox_factory import (
    _get_available_sandbox_types,
    _get_provider,
    get_default_working_dir,
)


def test_available_sandbox_types_include_e2b() -> None:
    """The sandbox type list should expose the E2B provider."""
    assert "e2b" in _get_available_sandbox_types()


def test_get_default_working_dir_for_e2b() -> None:
    """E2B should use the documented default working directory."""
    assert get_default_working_dir("e2b") == "/home/user"


def test_get_provider_returns_e2b_provider() -> None:
    """Provider lookup should instantiate `E2BProvider` for `e2b`."""
    sentinel = object()
    fake_module = cast("Any", types.ModuleType("deepagents_cli.integrations.e2b"))
    fake_module.E2BProvider = lambda: sentinel

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(
            sys.modules,
            "deepagents_cli.integrations.e2b",
            fake_module,
        )
        assert _get_provider("e2b") is sentinel


def test_unknown_provider_still_errors() -> None:
    """Unknown providers should still raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        get_default_working_dir("missing-provider")
