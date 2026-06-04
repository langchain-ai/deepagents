"""Tests for ProviderErrorRecoveryMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._cli_context import CLIContext
from deepagents_code.provider_error_recovery import (
    ProviderErrorRecoveryMiddleware,
    _classify,
)


def _make_model(name: str) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    return model


def _make_request(
    model: BaseChatModel,
    context: CLIContext | None = None,
) -> ModelRequest:
    runtime = SimpleNamespace(context=context)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=[],
        runtime=cast("Any", runtime),
        model_settings=None,
    )


def _ok_response() -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content="ok")])


# Stand-ins for openai exceptions; we set __module__/__name__ so
# `_matches_class` recognizes them without importing the SDK.
class _FakePermissionDenied(Exception):  # noqa: N818
    pass


_FakePermissionDenied.__module__ = "openai"
_FakePermissionDenied.__name__ = "PermissionDeniedError"


class _FakeConnectionError(Exception):
    pass


_FakeConnectionError.__module__ = "openai"
_FakeConnectionError.__name__ = "APIConnectionError"


_AUTH_MSG = "Error code: 403 Forbidden"
_FIVEXX_MSG = "Error code: 503 Service Unavailable"
_FOUROHONE_MSG = "Error code: 401 Unauthorized"
_CONN_MSG = "connection reset"
_OTHER_MSG = "not a provider error"


_mw = ProviderErrorRecoveryMiddleware()


class TestClassify:
    def test_permission_denied_class(self) -> None:
        assert _classify(_FakePermissionDenied(_AUTH_MSG)) == "auth"

    def test_message_403(self) -> None:
        assert _classify(RuntimeError(_AUTH_MSG)) == "auth"

    def test_message_401(self) -> None:
        assert _classify(RuntimeError(_FOUROHONE_MSG)) == "auth"

    def test_connection_error_class(self) -> None:
        assert _classify(_FakeConnectionError(_CONN_MSG)) == "transient"

    def test_message_5xx(self) -> None:
        assert _classify(RuntimeError(_FIVEXX_MSG)) == "transient"

    def test_unrelated_propagates(self) -> None:
        assert _classify(ValueError(_OTHER_MSG)) == "other"


class TestAuthSync:
    def test_auth_error_surfaces_message_without_retry(self) -> None:
        request = _make_request(_make_model("gpt-5"))
        calls = 0

        def handler(_: ModelRequest) -> ModelResponse[Any]:
            nonlocal calls
            calls += 1
            raise _FakePermissionDenied(_AUTH_MSG)

        response = _mw.wrap_model_call(request, handler)

        assert calls == 1
        assert "403" in response.result[0].content
        assert "gpt-5" in response.result[0].content

    def test_auth_error_with_fallback_swaps_once(self) -> None:
        primary = _make_model("gpt-5")
        fallback = _make_model("claude-sonnet-4-6")
        request = _make_request(
            primary, context=CLIContext(fallback_model="anthropic:claude-sonnet-4-6")
        )
        seen: list[Any] = []

        def handler(r: ModelRequest) -> ModelResponse[Any]:
            seen.append(r.model)
            if r.model is primary:
                raise _FakePermissionDenied(_AUTH_MSG)
            return _ok_response()

        result = SimpleNamespace(
            model=fallback,
            model_name="claude-sonnet-4-6",
            provider="anthropic",
            context_limit=None,
            unsupported_modalities=frozenset(),
        )
        with patch("deepagents_code.config.create_model", return_value=result):
            response = _mw.wrap_model_call(request, handler)

        assert seen == [primary, fallback]
        assert response.result[0].content == "ok"


class TestTransientSync:
    def test_transient_retries_once_and_succeeds(self) -> None:
        request = _make_request(_make_model("gpt-5"))
        calls = 0

        def handler(_: ModelRequest) -> ModelResponse[Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeConnectionError(_CONN_MSG)
            return _ok_response()

        with patch("deepagents_code.provider_error_recovery.time.sleep"):
            response = _mw.wrap_model_call(request, handler)

        assert calls == 2
        assert response.result[0].content == "ok"

    def test_transient_retry_failure_surfaces_message(self) -> None:
        request = _make_request(_make_model("gpt-5"))
        calls = 0

        def handler(_: ModelRequest) -> ModelResponse[Any]:
            nonlocal calls
            calls += 1
            raise RuntimeError(_FIVEXX_MSG)

        with patch("deepagents_code.provider_error_recovery.time.sleep"):
            response = _mw.wrap_model_call(request, handler)

        assert calls == 2
        assert "temporarily unavailable" in response.result[0].content


class TestPassThroughSync:
    def test_unrelated_exception_propagates(self) -> None:
        request = _make_request(_make_model("gpt-5"))

        def handler(_: ModelRequest) -> ModelResponse[Any]:
            raise ValueError(_OTHER_MSG)

        with pytest.raises(ValueError, match=_OTHER_MSG):
            _mw.wrap_model_call(request, handler)

    def test_success_passes_through(self) -> None:
        request = _make_request(_make_model("gpt-5"))
        response = _mw.wrap_model_call(request, lambda _: _ok_response())
        assert response.result[0].content == "ok"


class TestAuthAsync:
    async def test_auth_error_surfaces_message(self) -> None:
        request = _make_request(_make_model("gpt-5"))

        async def handler(_: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            raise _FakePermissionDenied(_AUTH_MSG)

        response = await _mw.awrap_model_call(request, handler)
        assert "403" in response.result[0].content


class TestTransientAsync:
    async def test_transient_retries_once(self) -> None:
        request = _make_request(_make_model("gpt-5"))
        calls = 0

        async def handler(_: ModelRequest) -> ModelResponse[Any]:  # noqa: RUF029
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeConnectionError(_CONN_MSG)
            return _ok_response()

        async def _noop(_: float) -> None:  # noqa: RUF029
            return None

        with patch("deepagents_code.provider_error_recovery.asyncio.sleep", _noop):
            response = await _mw.awrap_model_call(request, handler)

        assert calls == 2
        assert response.result[0].content == "ok"
