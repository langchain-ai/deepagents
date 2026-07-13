"""Unit tests for `FireworksPromptCachingMiddleware`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from deepagents.middleware.fireworks import (
    _SESSION_AFFINITY_HEADER,
    FireworksPromptCachingMiddleware,
)
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    import pytest

_THREAD_ID = "thread-abc-123"


def _make_model(provider: str | None) -> GenericFakeChatModel:
    """Build a fake chat model reporting `provider` via `_get_ls_params`."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    ls_params: dict[str, Any] = {} if provider is None else {"ls_provider": provider}
    model._get_ls_params = MagicMock(return_value=ls_params)  # type: ignore[method-assign]
    return model


def _make_request(model: GenericFakeChatModel, model_settings: dict[str, Any] | None = None) -> ModelRequest:
    """Build a minimal `ModelRequest` for the given model."""
    return ModelRequest(
        model=model,
        messages=[],
        model_settings=model_settings if model_settings is not None else {},
    )


def _run(request: ModelRequest, thread_id: str | None = _THREAD_ID) -> ModelRequest:
    """Run the middleware and return the request seen by the handler."""
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    config = {"configurable": {"thread_id": thread_id}} if thread_id is not None else {}
    with patch("deepagents.middleware.fireworks.get_config", return_value=config):
        middleware.wrap_model_call(request, handler)
    return captured["request"]


def test_fireworks_model_injects_session_affinity() -> None:
    """Fireworks model + thread ID injects both prompt_cache_key and header."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request)
    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID


def test_non_fireworks_model_is_unchanged() -> None:
    """A non-Fireworks model passes through untouched."""
    request = _make_request(_make_model("openai"))
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


def test_missing_thread_id_is_unchanged() -> None:
    """No configured thread ID leaves the request unchanged."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request, thread_id=None)
    assert result is request
    assert result.model_settings == {}


def test_empty_thread_id_is_unchanged() -> None:
    """An empty-string thread ID leaves the request unchanged."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request, thread_id="")
    assert result is request
    assert result.model_settings == {}


def test_existing_user_setting_causes_no_injection() -> None:
    """A caller-supplied `user` setting is treated as user-managed affinity."""
    request = _make_request(_make_model("fireworks"), {"user": "caller"})
    result = _run(request)
    assert result is request
    assert "prompt_cache_key" not in result.model_settings
    assert "extra_headers" not in result.model_settings


def test_existing_prompt_cache_key_causes_no_injection() -> None:
    """A caller-supplied `prompt_cache_key` is left untouched."""
    request = _make_request(_make_model("fireworks"), {"prompt_cache_key": "mine"})
    result = _run(request)
    assert result is request
    assert result.model_settings == {"prompt_cache_key": "mine"}


def test_null_user_setting_injects_session_affinity() -> None:
    """A null `user` setting does not disable session affinity."""
    request = _make_request(_make_model("fireworks"), {"user": None})
    result = _run(request)
    assert result.model_settings["user"] is None
    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID


def test_null_prompt_cache_key_injects_session_affinity() -> None:
    """A null `prompt_cache_key` is replaced with the active thread ID."""
    request = _make_request(_make_model("fireworks"), {"prompt_cache_key": None})
    result = _run(request)
    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID


def test_existing_session_affinity_header_causes_no_injection() -> None:
    """An existing x-session-affinity header (any case) is left untouched."""
    request = _make_request(
        _make_model("fireworks"),
        {"extra_headers": {"X-Session-Affinity": "existing"}},
    )
    result = _run(request)
    assert result is request
    assert result.model_settings["extra_headers"] == {"X-Session-Affinity": "existing"}


def test_unrelated_headers_are_preserved_and_not_mutated() -> None:
    """Existing unrelated headers are copied and preserved, not mutated in place."""
    original_headers = {"Authorization": "Bearer token"}
    original_settings = {"temperature": 0.5, "extra_headers": original_headers}
    request = _make_request(_make_model("fireworks"), original_settings)
    result = _run(request)

    assert result.model_settings["temperature"] == 0.5
    assert result.model_settings["extra_headers"]["Authorization"] == "Bearer token"
    assert result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID

    # Caller-provided dicts must not be mutated.
    assert original_headers == {"Authorization": "Bearer token"}
    assert original_settings == {"temperature": 0.5, "extra_headers": original_headers}


def test_non_mapping_extra_headers_is_unchanged_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-mapping extra_headers leaves the request unchanged and logs a warning."""
    request = _make_request(_make_model("fireworks"), {"extra_headers": ["not", "a", "mapping"]})
    with caplog.at_level(logging.WARNING, logger="deepagents.middleware.fireworks"):
        result = _run(request)
    assert result is request
    assert result.model_settings == {"extra_headers": ["not", "a", "mapping"]}
    assert any("extra_headers" in record.message for record in caplog.records)


def test_thread_id_not_logged() -> None:
    """The thread ID is treated as sensitive and never written to logs."""
    request = _make_request(_make_model("fireworks"))
    middleware = FireworksPromptCachingMiddleware()

    def handler(_req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="ok")])

    with (
        patch(
            "deepagents.middleware.fireworks.get_config",
            return_value={"configurable": {"thread_id": _THREAD_ID}},
        ),
        patch("deepagents.middleware.fireworks.logger") as mock_logger,
    ):
        middleware.wrap_model_call(request, handler)

    logged = " ".join(str(arg) for call in mock_logger.debug.call_args_list + mock_logger.warning.call_args_list for arg in call.args)
    assert _THREAD_ID not in logged


async def test_async_fireworks_model_injects_session_affinity() -> None:
    """The async path injects session affinity the same way as the sync path."""
    request = _make_request(_make_model("fireworks"))
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    async def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    with patch(
        "deepagents.middleware.fireworks.get_config",
        return_value={"configurable": {"thread_id": _THREAD_ID}},
    ):
        await middleware.awrap_model_call(request, handler)

    result = captured["request"]
    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID
