"""Tests for carrying `/effort` changes in OpenAI `configuration_update` blocks."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from deepagents_code._cli_context import CLIContext
from deepagents_code.app import _user_message_text
from deepagents_code.cold_cache import (
    cache_identity_params_changed,
    configuration_update_supported,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware


def _block(effort: str) -> dict[str, Any]:
    return {"type": "configuration_update", "reasoning": {"effort": effort}}


def _model(name: str, provider: str = "openai", *, responses: bool = True) -> Any:  # noqa: ANN401
    model = MagicMock(spec=BaseChatModel)
    model.model_name = name
    model._get_ls_params.return_value = {"ls_provider": provider}
    model._default_params = {}
    model._use_responses_api = MagicMock(return_value=responses)
    return model


def _recorded(spec: str, effort: str | None) -> dict[str, Any]:
    params = {"reasoning_effort": effort} if effort else None
    return {"_last_cache_model_spec": spec, "_last_cache_params": params}


async def _call(
    model: Any,  # noqa: ANN401
    messages: list[AnyMessage],
    *,
    effort: str,
    state: dict[str, Any] | None = None,
    asynchronous: bool = False,
) -> tuple[ModelRequest, dict[str, Any]]:
    """Run one model call and return the sent request and checkpoint update."""
    sent: list[ModelRequest] = []
    request = ModelRequest(
        model=model,
        messages=messages,
        tools=[],
        runtime=cast(
            "Any",
            SimpleNamespace(
                context=CLIContext(model_params={"reasoning_effort": effort})
            ),
        ),
        state=cast("Any", {"messages": messages, **(state or {})}),
    )
    middleware = ConfigurableModelMiddleware(openai_prompt_cache_key=False)

    def handler(req: ModelRequest) -> ModelResponse[Any]:
        sent.append(req)
        return ModelResponse(result=[AIMessage(content="ok")])

    if asynchronous:
        result = await middleware.awrap_model_call(
            request, AsyncMock(side_effect=handler)
        )
    else:
        result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    return sent[0], cast("dict[str, Any]", result.command.update)


def _request_effort(request: ModelRequest) -> object:
    return request.model_settings.get("reasoning_effort")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_effort_change_keeps_request_effort_and_persists_block(
    asynchronous: bool,
) -> None:
    messages: list[AnyMessage] = [
        HumanMessage("first", id="h1"),
        AIMessage("done", id="a1"),
        HumanMessage("second", id="h2"),
    ]

    sent, update = await _call(
        _model("gpt-6"),
        messages,
        effort="high",
        state=_recorded("openai:gpt-6", "medium"),
        asynchronous=asynchronous,
    )

    assert _request_effort(sent) == "medium"
    assert sent.messages[-1].content == [
        _block("high"),
        {"type": "text", "text": "second"},
    ]
    assert sent.messages[0].content == "first"
    assert update["messages"] == [sent.messages[-1]]
    assert update["messages"][0].id == "h2"
    assert update["_last_cache_params"] == {"reasoning_effort": "medium"}
    assert update["_model_params"] == {"reasoning_effort": "high"}


async def test_same_effort_on_later_turn_adds_nothing() -> None:
    messages: list[AnyMessage] = [
        HumanMessage([_block("high"), {"type": "text", "text": "a"}], id="h1"),
        AIMessage("done", id="a1"),
        HumanMessage("b", id="h2"),
    ]

    sent, update = await _call(
        _model("gpt-6"),
        messages,
        effort="high",
        state=_recorded("openai:gpt-6", "medium"),
    )

    assert _request_effort(sent) == "medium"
    assert sent.messages == messages
    assert "messages" not in update


async def test_changing_effort_again_adds_a_new_block() -> None:
    first = HumanMessage([_block("high"), {"type": "text", "text": "a"}], id="h1")
    messages: list[AnyMessage] = [
        first,
        AIMessage("done", id="a1"),
        HumanMessage("b", id="h2"),
    ]

    sent, update = await _call(
        _model("gpt-6"),
        messages,
        effort="medium",
        state=_recorded("openai:gpt-6", "medium"),
    )

    assert _request_effort(sent) == "medium"
    assert sent.messages[0] is first
    assert update["messages"][0].content[0] == _block("medium")


async def test_first_call_sends_effort_at_request_level() -> None:
    messages: list[AnyMessage] = [HumanMessage("hi", id="h1")]

    sent, update = await _call(_model("gpt-6"), messages, effort="high")

    assert _request_effort(sent) == "high"
    assert sent.messages == messages
    assert "messages" not in update
    assert update["_last_cache_params"] == {"reasoning_effort": "high"}


@pytest.mark.parametrize(("name", "responses"), [("gpt-5.6", True), ("gpt-6", False)])
async def test_unsupported_openai_requests_keep_request_level_effort(
    name: str,
    responses: bool,
) -> None:
    messages: list[AnyMessage] = [HumanMessage("hi", id="h1")]

    sent, update = await _call(
        _model(name, responses=responses),
        messages,
        effort="high",
        state=_recorded(f"openai:{name}", "medium"),
    )

    assert _request_effort(sent) == "high"
    assert sent.messages == messages
    assert "messages" not in update
    assert update["_last_cache_params"] == {"reasoning_effort": "high"}


async def test_other_models_do_not_receive_stored_blocks() -> None:
    stored = HumanMessage([_block("high"), {"type": "text", "text": "a"}], id="h1")

    sent, update = await _call(
        _model("claude-opus-5", provider="anthropic"), [stored], effort="high"
    )

    assert sent.messages[0].content == [{"type": "text", "text": "a"}]
    assert stored.content[0] == _block("high")
    assert "messages" not in update


@pytest.mark.parametrize(
    ("spec", "last_spec", "changed"),
    [
        ("openai:gpt-6", "openai:gpt-6", False),
        ("openai:gpt-6.1-mini", "openai:gpt-6.1-mini", False),
        ("openai:gpt-6", "openai:gpt-6-mini", True),
        ("openai:gpt-5.6", "openai:gpt-5.6", True),
        ("anthropic:claude-opus-5", "anthropic:claude-opus-5", True),
    ],
)
def test_effort_change_moves_cache_identity_only_without_configuration_update(
    spec: str,
    last_spec: str,
    changed: bool,
) -> None:
    assert (
        cache_identity_params_changed(
            {"reasoning_effort": "high"},
            {"reasoning_effort": "medium"},
            model_spec=spec,
            last_spec=last_spec,
        )
        is changed
    )


def test_configuration_update_respects_chat_completions_opt_out() -> None:
    assert configuration_update_supported("openai:gpt-6")
    assert not configuration_update_supported(
        "openai:gpt-6", {"use_responses_api": False}
    )
    assert not configuration_update_supported("openai:anthropic/claude-opus-5")


def test_cache_params_other_than_effort_still_move_identity() -> None:
    assert cache_identity_params_changed(
        {"reasoning_effort": "high", "prompt_cache_retention": "24h"},
        {"reasoning_effort": "medium"},
        model_spec="openai:gpt-6",
        last_spec="openai:gpt-6",
    )


def test_resumed_user_message_hides_configuration_update() -> None:
    assert _user_message_text([_block("high"), {"type": "text", "text": "hi"}]) == "hi"
