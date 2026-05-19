"""Tests for the transient-error retry layer on chat models."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from deepagents_code._model_retry import (
    MAX_ATTEMPTS,
    _is_transient_error,
    install_transient_retry,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeHTTPStatusError(Exception):
    """Stand-in for ``httpx.HTTPStatusError`` without the import."""

    def __init__(self, message: str, response: _FakeResponse) -> None:
        super().__init__(message)
        self.response = response


class _FakeServiceUnavailableError(Exception):
    """Stand-in for ``fireworks.client.error.ServiceUnavailableError``."""


class _FakeInvalidRequestError(Exception):
    """Stand-in for a deterministic 4xx error that must NOT be retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


class TestIsTransientError:
    def test_matches_503_status(self) -> None:
        exc = _FakeHTTPStatusError(
            "service overloaded, please try again later",
            _FakeResponse(503, "service overloaded, please try again later"),
        )
        assert _is_transient_error(exc) is True

    def test_matches_service_unavailable_name(self) -> None:
        assert _is_transient_error(_FakeServiceUnavailableError("overloaded")) is True

    def test_matches_overloaded_body_text(self) -> None:
        # An SDK that raises a bare exception with the overload body still
        # gets retried.
        exc = RuntimeError("Service overloaded, please try again later")
        assert _is_transient_error(exc) is True

    def test_does_not_match_400(self) -> None:
        assert _is_transient_error(_FakeInvalidRequestError("bad json")) is False

    def test_does_not_match_404(self) -> None:
        exc = _FakeHTTPStatusError("not found", _FakeResponse(404))
        assert _is_transient_error(exc) is False

    def test_matches_httpx_read_error(self) -> None:
        httpx = pytest.importorskip("httpx")
        assert _is_transient_error(httpx.ReadError("connection dropped")) is True

    def test_matches_httpx_read_timeout(self) -> None:
        httpx = pytest.importorskip("httpx")
        assert _is_transient_error(httpx.ReadTimeout("slow")) is True


class _FakeChatModel:
    """Minimal chat model surface: enough to install retry against."""

    def __init__(self, side_effects: list[Any]) -> None:
        # ``side_effects`` is consumed FIFO; each item is either a value to
        # return or an exception to raise.
        self._side_effects = list(side_effects)
        self.invoke_calls = 0
        self.agenerate_calls = 0

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        self.invoke_calls += 1
        outcome = self._side_effects.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _agenerate(self, *_args: Any, **_kwargs: Any) -> Any:
        self.agenerate_calls += 1
        outcome = self._side_effects.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tenacity's backoff instantaneous.

    Each retry helper in ``_model_retry`` re-imports
    ``wait_exponential_jitter`` from tenacity inside its closure, so
    monkey-patching the symbol on the ``tenacity`` module is sufficient
    to neutralize backoff without touching the production code path.
    """
    import tenacity

    monkeypatch.setattr(
        tenacity,
        "wait_exponential_jitter",
        lambda **_: tenacity.wait_none(),
    )


class TestInstallTransientRetry:
    def test_retries_503_then_succeeds(self) -> None:
        """Mock a 503 then a 200; the call eventually succeeds."""
        first = _FakeHTTPStatusError("overloaded", _FakeResponse(503))
        model = _FakeChatModel([first, "ok"])
        install_transient_retry(model)

        result = model.invoke("hello")
        assert result == "ok"
        assert model.invoke_calls == 2

    def test_does_not_retry_400(self) -> None:
        """Deterministic 4xx surfaces immediately — no extra attempts."""
        model = _FakeChatModel([_FakeInvalidRequestError("bad json")])
        install_transient_retry(model)

        with pytest.raises(_FakeInvalidRequestError):
            model.invoke("hello")
        assert model.invoke_calls == 1

    def test_gives_up_after_max_attempts(self) -> None:
        """A persistently-failing transient error eventually raises."""
        errors = [
            _FakeServiceUnavailableError("overloaded") for _ in range(MAX_ATTEMPTS)
        ]
        model = _FakeChatModel(errors)
        install_transient_retry(model)

        with pytest.raises(_FakeServiceUnavailableError):
            model.invoke("hello")
        assert model.invoke_calls == MAX_ATTEMPTS

    def test_idempotent(self) -> None:
        """Calling install twice does not double-wrap the methods."""
        model = _FakeChatModel(
            [_FakeServiceUnavailableError("overloaded"), "ok"]
        )
        install_transient_retry(model)
        wrapped_once = model.invoke
        install_transient_retry(model)
        assert model.invoke is wrapped_once
        assert model.invoke("hello") == "ok"
        assert model.invoke_calls == 2

    def test_async_retry(self) -> None:
        """``_agenerate`` is also wrapped and retries transient failures."""
        import asyncio

        model = _FakeChatModel(
            [_FakeServiceUnavailableError("overloaded"), "ok"]
        )
        install_transient_retry(model)

        result = asyncio.run(model._agenerate("hello"))
        assert result == "ok"
        assert model.agenerate_calls == 2

    def test_survives_slotted_model(self) -> None:
        """Models that forbid attribute assignment fail open, not crash."""

        class _SlottedModel:
            __slots__ = ()

            def invoke(self, _: Any) -> str:
                return "ok"

        model = _SlottedModel()
        # Must not raise even though ``setattr`` will fail on the slot-only
        # instance.
        install_transient_retry(model)  # type: ignore[arg-type]
        assert model.invoke("hi") == "ok"


class TestCreateModelInstallsRetry:
    """Smoke test: create_model() wires the retry layer onto the result."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials",
            lambda _: True,
        )

    def test_create_model_wraps_with_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code import config as config_mod

        mock_model = MagicMock()
        mock_model.profile = None

        monkeypatch.setattr(
            config_mod,
            "_create_model_via_init",
            lambda *a, **k: mock_model,
        )

        installed: list[Any] = []

        def _spy(model: Any) -> Any:
            installed.append(model)
            return model

        monkeypatch.setattr(
            "deepagents_code._model_retry.install_transient_retry", _spy
        )

        config_mod.create_model("anthropic:claude-sonnet-4-5")
        assert installed == [mock_model]
