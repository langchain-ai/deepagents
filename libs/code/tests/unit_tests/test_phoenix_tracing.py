"""Tests for optional Phoenix tracing setup."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from deepagents_code import phoenix_tracing
from deepagents_code._env_vars import PHOENIX_TRACING


@pytest.fixture(autouse=True)
def _reset_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset process-global tracing state between tests."""
    monkeypatch.setattr(phoenix_tracing, "_provider", None)


def _stub_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock, object]:
    """Install lightweight Phoenix and OpenInference module doubles."""
    provider = object()
    register = MagicMock(return_value=provider)
    instrument = MagicMock()
    instrumentor = MagicMock(return_value=MagicMock(instrument=instrument))

    phoenix = ModuleType("phoenix")
    phoenix_otel = ModuleType("phoenix.otel")
    setattr(phoenix_otel, "register", register)  # noqa: B010  # module test double
    openinference = ModuleType("openinference")
    instrumentation = ModuleType("openinference.instrumentation")
    langchain = ModuleType("openinference.instrumentation.langchain")
    setattr(langchain, "LangChainInstrumentor", instrumentor)  # noqa: B010  # module test double

    monkeypatch.setitem(sys.modules, "phoenix", phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", phoenix_otel)
    monkeypatch.setitem(sys.modules, "openinference", openinference)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation", instrumentation)
    monkeypatch.setitem(
        sys.modules, "openinference.instrumentation.langchain", langchain
    )
    return register, instrument, provider


def test_disabled_does_not_import_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path must not require or import Phoenix packages."""
    monkeypatch.delenv(PHOENIX_TRACING, raising=False)
    sys.modules.pop("phoenix.otel", None)
    sys.modules.pop("openinference.instrumentation.langchain", None)

    assert phoenix_tracing.configure_phoenix_tracing() is False
    assert "phoenix.otel" not in sys.modules
    assert "openinference.instrumentation.langchain" not in sys.modules


def test_enabled_registers_and_instruments_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled tracing should batch-export LangChain spans to Phoenix once."""
    register, instrument, provider = _stub_dependencies(monkeypatch)
    monkeypatch.setenv(PHOENIX_TRACING, "true")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "dcode-debug")

    assert phoenix_tracing.configure_phoenix_tracing() is True
    assert phoenix_tracing.configure_phoenix_tracing() is True

    register.assert_called_once_with(
        project_name="dcode-debug",
        protocol="http/protobuf",
        batch=True,
        verbose=False,
    )
    instrument.assert_called_once_with(tracer_provider=provider)
    assert phoenix_tracing._provider is provider
    assert os.environ["PHOENIX_DISCOVER_CONFIG"] == "false"


def test_enabled_uses_default_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """The integration should group traces under a recognizable project."""
    register, _, _ = _stub_dependencies(monkeypatch)
    monkeypatch.setenv(PHOENIX_TRACING, "1")
    monkeypatch.delenv("PHOENIX_PROJECT", raising=False)
    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)

    assert phoenix_tracing.configure_phoenix_tracing() is True

    assert register.call_args.kwargs["project_name"] == "deepagents-code"


def test_enabled_without_extra_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit opt-in should explain how to install missing packages."""
    monkeypatch.setenv(PHOENIX_TRACING, "yes")
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.langchain", None)

    with pytest.raises(RuntimeError, match=r"deepagents-code\[phoenix\]"):
        phoenix_tracing.configure_phoenix_tracing()
