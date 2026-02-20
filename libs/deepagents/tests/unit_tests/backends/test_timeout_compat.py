"""Tests for timeout compatibility guards.

Verifies that `execute_accepts_timeout` correctly detects whether a backend's
`execute` method accepts a `timeout` keyword argument, and that callers handle
the result appropriately.
"""

import logging

import pytest

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    SandboxBackendProtocol,
    execute_accepts_timeout,
)

# ---------------------------------------------------------------------------
# Test backends
# ---------------------------------------------------------------------------


class ModernBackend(SandboxBackendProtocol):
    """Backend whose `execute` accepts `timeout` (current SDK signature)."""

    @property
    def id(self) -> str:
        return "modern"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return ExecuteResponse(output=command, exit_code=0)


class LegacyBackend(SandboxBackendProtocol):
    """Backend whose `execute` does NOT accept `timeout` (pre-0.4.3)."""

    @property
    def id(self) -> str:
        return "legacy"

    def execute(self, command: str) -> ExecuteResponse:  # type: ignore[override]
        return ExecuteResponse(output=command, exit_code=0)


class KwargsBackend(SandboxBackendProtocol):
    """Backend whose `execute` uses **kwargs."""

    @property
    def id(self) -> str:
        return "kwargs"

    def execute(self, command: str, **kwargs: object) -> ExecuteResponse:  # type: ignore[override]
        return ExecuteResponse(output=command, exit_code=0)


# ---------------------------------------------------------------------------
# execute_accepts_timeout
# ---------------------------------------------------------------------------


class TestExecuteAcceptsTimeout:
    def setup_method(self) -> None:
        execute_accepts_timeout.cache_clear()

    def test_modern_backend_returns_true(self) -> None:
        assert execute_accepts_timeout(ModernBackend) is True

    def test_legacy_backend_returns_false(self) -> None:
        assert execute_accepts_timeout(LegacyBackend) is False

    def test_kwargs_backend_returns_false(self) -> None:
        """A backend with **kwargs does not have a named `timeout` param."""
        assert execute_accepts_timeout(KwargsBackend) is False

    def test_result_is_cached(self) -> None:
        execute_accepts_timeout(ModernBackend)
        execute_accepts_timeout(ModernBackend)
        info = execute_accepts_timeout.cache_info()
        assert info.hits >= 1

    def test_logs_warning_on_inspect_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """If inspect.signature raises, a warning is logged and False returned."""

        class BadBackend(SandboxBackendProtocol):
            @property
            def id(self) -> str:
                return "bad"

        # Forcibly set execute to something un-inspectable
        BadBackend.execute = property(lambda _self: None)  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            result = execute_accepts_timeout(BadBackend)

        assert result is False
        assert "Could not inspect signature" in caplog.text


# ---------------------------------------------------------------------------
# SandboxBackendProtocol.aexecute
# ---------------------------------------------------------------------------


class TestAexecuteTimeoutGuard:
    @pytest.mark.asyncio
    async def test_modern_backend_forwards_timeout(self) -> None:
        class RecordingBackend(SandboxBackendProtocol):
            received_timeout: int | None = None

            @property
            def id(self) -> str:
                return "recording"

            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                self.received_timeout = timeout
                return ExecuteResponse(output="ok", exit_code=0)

        execute_accepts_timeout.cache_clear()
        backend = RecordingBackend()
        await backend.aexecute("ls", timeout=42)
        assert backend.received_timeout == 42

    @pytest.mark.asyncio
    async def test_legacy_backend_drops_timeout_silently(self) -> None:
        """Timeout is silently dropped for legacy backends (middleware handles user-facing errors)."""
        execute_accepts_timeout.cache_clear()
        backend = LegacyBackend()
        result = await backend.aexecute("ls", timeout=30)
        assert result.output == "ls"

    @pytest.mark.asyncio
    async def test_no_timeout_skips_check(self) -> None:
        execute_accepts_timeout.cache_clear()
        backend = ModernBackend()
        result = await backend.aexecute("echo hi")
        assert result.output == "echo hi"


# ---------------------------------------------------------------------------
# CompositeBackend.execute / .aexecute timeout guard
# ---------------------------------------------------------------------------


class TestCompositeTimeoutGuard:
    def setup_method(self) -> None:
        execute_accepts_timeout.cache_clear()

    def test_execute_forwards_timeout_to_modern(self) -> None:
        class RecordingModern(SandboxBackendProtocol):
            received_timeout: int | None = None

            @property
            def id(self) -> str:
                return "rec"

            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                self.received_timeout = timeout
                return ExecuteResponse(output="ok", exit_code=0)

        inner = RecordingModern()
        comp = CompositeBackend(default=inner, routes={})
        comp.execute("ls", timeout=60)
        assert inner.received_timeout == 60

    def test_execute_omits_timeout_for_legacy(self) -> None:
        """Legacy backend is called without timeout (no TypeError)."""
        inner = LegacyBackend()
        comp = CompositeBackend(default=inner, routes={})
        result = comp.execute("ls", timeout=60)
        assert result.output == "ls"

    @pytest.mark.asyncio
    async def test_aexecute_forwards_timeout_to_modern(self) -> None:
        class RecordingModern(SandboxBackendProtocol):
            received_timeout: int | None = None

            @property
            def id(self) -> str:
                return "rec"

            def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
                self.received_timeout = timeout
                return ExecuteResponse(output="ok", exit_code=0)

        inner = RecordingModern()
        comp = CompositeBackend(default=inner, routes={})
        await comp.aexecute("ls", timeout=60)
        assert inner.received_timeout == 60

    @pytest.mark.asyncio
    async def test_aexecute_omits_timeout_for_legacy(self) -> None:
        inner = LegacyBackend()
        comp = CompositeBackend(default=inner, routes={})
        result = await comp.aexecute("ls", timeout=60)
        assert result.output == "ls"
