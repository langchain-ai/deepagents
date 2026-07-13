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


def _set_model_kwargs(model: GenericFakeChatModel, settings: dict[str, Any]) -> None:
    """Set constructor-style passthrough settings on a fake chat model."""
    object.__setattr__(model, "model_kwargs", settings)


def _make_request(model: GenericFakeChatModel, model_settings: dict[str, Any] | None = None) -> ModelRequest:
    """Build a minimal `ModelRequest` for the given model."""
    return ModelRequest(
        model=model,
        messages=[],
        model_settings=model_settings if model_settings is not None else {},
    )


def _run(
    request: ModelRequest,
    thread_id: str | None = _THREAD_ID,
    config: dict[str, Any] | None = None,
) -> ModelRequest:
    """Run the middleware and return the request seen by the handler.

    `config` overrides the patched `get_config()` return value outright; when it
    is omitted, a config carrying `thread_id` is synthesized (or `{}` when
    `thread_id` is `None`).
    """
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    if config is None:
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


def test_missing_ls_provider_is_unchanged() -> None:
    """A model reporting no `ls_provider` is treated as non-Fireworks."""
    request = _make_request(_make_model(None))
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


def test_ls_params_raising_is_unchanged_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model whose `_get_ls_params` raises degrades to a no-op with a warning."""
    model = _make_model("fireworks")
    model._get_ls_params = MagicMock(side_effect=NotImplementedError)  # type: ignore[method-assign]
    request = _make_request(model)
    with caplog.at_level(logging.WARNING, logger="deepagents.middleware.fireworks"):
        result = _run(request)
    assert result is request
    assert result.model_settings == {}
    assert any("provider detection failed" in record.message for record in caplog.records)


def test_model_without_ls_params_is_unchanged() -> None:
    """A model lacking a callable `_get_ls_params` is treated as non-Fireworks."""
    request = ModelRequest(model=object(), messages=[], model_settings={})
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


def test_non_dict_ls_params_is_unchanged() -> None:
    """A model whose `_get_ls_params` returns a non-dict is treated as non-Fireworks."""
    model = _make_model("fireworks")
    model._get_ls_params = MagicMock(return_value=None)  # type: ignore[method-assign]
    request = _make_request(model)
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


def test_missing_thread_id_is_unchanged() -> None:
    """No configured thread ID leaves the request unchanged."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request, thread_id=None)
    assert result is request
    assert result.model_settings == {}


def test_null_configurable_is_unchanged() -> None:
    """A null `configurable` in the config leaves the request unchanged."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request, config={"configurable": None})
    assert result is request
    assert result.model_settings == {}


def test_empty_thread_id_is_unchanged() -> None:
    """An empty-string thread ID leaves the request unchanged."""
    request = _make_request(_make_model("fireworks"))
    result = _run(request, thread_id="")
    assert result is request
    assert result.model_settings == {}


def test_no_runnable_context_is_unchanged() -> None:
    """Outside a runnable context (`get_config` raises) the request is unchanged."""
    request = _make_request(_make_model("fireworks"))
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    with patch("deepagents.middleware.fireworks.get_config", side_effect=RuntimeError):
        middleware.wrap_model_call(request, handler)

    assert captured["request"] is request
    assert captured["request"].model_settings == {}


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


def test_model_prompt_cache_key_causes_no_injection() -> None:
    """A constructor-level `prompt_cache_key` is left untouched."""
    model = _make_model("fireworks")
    _set_model_kwargs(model, {"prompt_cache_key": "mine"})
    request = _make_request(model)
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


def test_model_user_causes_no_injection() -> None:
    """A constructor-level `user` is treated as caller-managed affinity."""
    model = _make_model("fireworks")
    _set_model_kwargs(model, {"user": "caller"})
    request = _make_request(model)
    result = _run(request)
    assert result is request
    assert result.model_settings == {}


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


def test_model_headers_are_preserved_when_affinity_is_injected() -> None:
    """Constructor-level headers survive binding the session affinity settings."""
    model = _make_model("fireworks")
    original_headers = {"Authorization": "Bearer token"}
    model_settings = {"extra_headers": original_headers}
    _set_model_kwargs(model, model_settings)

    result = _run(_make_request(model))

    assert result.model_settings["extra_headers"] == {
        "Authorization": "Bearer token",
        _SESSION_AFFINITY_HEADER: _THREAD_ID,
    }
    assert original_headers == {"Authorization": "Bearer token"}
    assert model_settings == {"extra_headers": original_headers}


def test_model_and_request_headers_are_merged() -> None:
    """Per-request headers extend rather than replace constructor-level headers."""
    model = _make_model("fireworks")
    _set_model_kwargs(model, {"extra_headers": {"Authorization": "Bearer token"}})
    request = _make_request(model, {"extra_headers": {"X-Request-ID": "request-1"}})

    result = _run(request)

    assert result.model_settings["extra_headers"] == {
        "Authorization": "Bearer token",
        "X-Request-ID": "request-1",
        _SESSION_AFFINITY_HEADER: _THREAD_ID,
    }


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
