"""Tests for the GLM-5.2 Deep Agents Code harness profile."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_code import _glm_5p2_profile as glm_profile
from deepagents_code._glm_5p2_profile import _GlmTerminalStallRecovery

if TYPE_CHECKING:
    from deepagents.profiles import HarnessProfile
    from langchain_core.tools import BaseTool


_FIREWORKS_GLM = "fireworks:accounts/fireworks/models/glm-5p2"
_OPENROUTER_GLM = "openrouter:z-ai/glm-5.2"
_BASETEN_GLM = "baseten:zai-org/GLM-5.2"
_NON_GLM = "openai:gpt-5.5"

_PROVIDER_BY_IDENTIFIER = {
    "accounts/fireworks/models/glm-5p2": "fireworks",
    "z-ai/glm-5.2": "openrouter",
    "zai-org/GLM-5.2": "baseten",
    "gpt-5.5": "openai",
}


def _model(identifier: str, *, provider: str | None = None) -> BaseChatModel:
    model = MagicMock(spec=BaseChatModel)
    model.model_name = identifier
    model._get_ls_params.return_value = {
        "ls_provider": provider or _PROVIDER_BY_IDENTIFIER[identifier]
    }
    return cast("BaseChatModel", model)


def _model_request(
    identifier: str,
    *,
    prompt: str = "base prompt",
    provider: str | None = None,
) -> ModelRequest:
    runtime = SimpleNamespace(context={"model": None})
    return ModelRequest(
        model=_model(identifier, provider=provider),
        messages=[HumanMessage(content="run")],
        tools=[],
        system_prompt=prompt,
        state={"messages": []},
        runtime=cast("Any", runtime),
    )


def _model_response(
    *,
    content: str = "done",
    finish_reason: str = "stop",
    with_tool_call: bool = False,
) -> ModelResponse[Any]:
    tool_calls = (
        [
            {
                "id": "call-write",
                "name": "write_file",
                "args": {"file_path": "/app/result.txt", "content": "done"},
                "type": "tool_call",
            }
        ]
        if with_tool_call
        else []
    )
    return ModelResponse(
        result=[
            AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata={"finish_reason": finish_reason},
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        ]
    )


def test_registration_is_exact_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepagents.profiles.harness.harness_profiles as core_profiles

    calls: list[tuple[str, HarnessProfile]] = []

    def register(key: str, profile: HarnessProfile) -> None:
        calls.append((key, profile))

    monkeypatch.setattr(glm_profile, "_glm_5p2_profile_registered", False)
    monkeypatch.setattr(glm_profile, "register_harness_profile", register)
    monkeypatch.setattr(core_profiles, "_ensure_harness_profiles_loaded", lambda: None)
    monkeypatch.setattr(core_profiles, "_HARNESS_PROFILES", {})

    glm_profile._ensure_glm_5p2_profile_registered()
    glm_profile._ensure_glm_5p2_profile_registered()

    assert tuple(key for key, _ in calls) == (
        _FIREWORKS_GLM,
        _OPENROUTER_GLM,
        _BASETEN_GLM,
    )
    assert all(profile is glm_profile._GLM_5P2_PROFILE for _, profile in calls)


def test_registration_defers_to_existing_suffix_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepagents.profiles.harness.harness_profiles as core_profiles
    from deepagents.profiles import HarnessProfile as RuntimeHarnessProfile

    calls: list[str] = []

    monkeypatch.setattr(glm_profile, "_glm_5p2_profile_registered", False)
    monkeypatch.setattr(
        glm_profile,
        "register_harness_profile",
        lambda key, _profile: calls.append(key),
    )
    monkeypatch.setattr(core_profiles, "_ensure_harness_profiles_loaded", lambda: None)
    monkeypatch.setattr(
        core_profiles,
        "_HARNESS_PROFILES",
        {_FIREWORKS_GLM: RuntimeHarnessProfile(system_prompt_suffix="user override")},
    )

    glm_profile._ensure_glm_5p2_profile_registered()

    assert calls == [_OPENROUTER_GLM, _BASETEN_GLM]


def test_prompt_is_concise_and_execution_focused() -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX

    assert 240 <= len(suffix.split()) <= 360
    for omitted in (
        "write_todos",
        "<todo_rules>",
        "<tool_preferences>",
        "Prefer specialized tools",
    ):
        assert omitted not in suffix


def test_prompt_tells_glm_to_keep_media_out_of_context() -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX

    assert "text-only model" in suffix
    assert "Do not call `read_file` on images, PDFs, audio, or video" in suffix
    assert "Never place binary or encoded media in model context" in suffix


def test_headless_glm_retries_length_truncated_turn() -> None:
    middleware = _GlmTerminalStallRecovery()
    tools: list[BaseTool | dict[str, Any]] = [{"name": "write_file"}]
    request = _model_request("accounts/fireworks/models/glm-5p2").override(
        tools=tools,
        tool_choice="auto",
        model_settings={
            "model_kwargs": {"reasoning_effort": "max"},
            "temperature": 0.25,
        },
    )
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _model_response(content="unfinished design", finish_reason="length"),
            _model_response(content="recovered"),
        ]
    )

    def handler(actual: ModelRequest) -> ModelResponse[Any]:
        requests.append(actual)
        return next(responses)

    result = middleware.wrap_model_call(request, handler)

    assert len(requests) == 2
    assert requests[0].tool_choice == "auto"
    assert requests[0].model_settings == {
        "model_kwargs": {"reasoning_effort": "max"},
        "temperature": 0.25,
    }
    assert requests[0].tools == tools
    assert requests[1].system_prompt is not None
    assert "call a tool now" in requests[1].system_prompt
    assert requests[1].tool_choice == "any"
    assert requests[1].model_settings == {
        "model_kwargs": {"reasoning_effort": "none"},
        "temperature": 0.25,
    }
    assert requests[1].tools == tools
    assert request.tool_choice == "auto"
    assert request.model_settings == {
        "model_kwargs": {"reasoning_effort": "max"},
        "temperature": 0.25,
    }
    assert request.tools == tools
    assert result.result[0].text == "recovered"


async def test_async_headless_glm_retries_at_most_once() -> None:
    middleware = _GlmTerminalStallRecovery()
    calls = 0

    async def handler(_request: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        await asyncio.sleep(0)
        calls += 1
        return _model_response(content="still stalled", finish_reason="length")

    result = await middleware.awrap_model_call(
        _model_request("accounts/fireworks/models/glm-5p2"),
        handler,
    )

    assert calls == 2
    assert result.result[0].text == "still stalled"


def test_terminal_stall_recovery_rejects_fireworks_identifier_from_other_provider() -> (
    None
):
    middleware = _GlmTerminalStallRecovery()
    calls = 0

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _model_response(finish_reason="length")

    middleware.wrap_model_call(
        _model_request(
            "accounts/fireworks/models/glm-5p2",
            provider="custom_gateway",
        ),
        handler,
    )

    assert calls == 1


@pytest.mark.parametrize(
    ("identifier", "finish_reason", "with_tool_call"),
    [
        pytest.param("gpt-5.5", "length", False, id="non-glm"),
        pytest.param("z-ai/glm-5.2", "length", False, id="openrouter"),
        pytest.param("zai-org/GLM-5.2", "length", False, id="baseten"),
        pytest.param(
            "accounts/fireworks/models/glm-5p2",
            "stop",
            False,
            id="not-truncated",
        ),
        pytest.param(
            "accounts/fireworks/models/glm-5p2",
            "length",
            True,
            id="tool-call",
        ),
    ],
)
def test_terminal_stall_recovery_ignores_near_misses(
    identifier: str,
    finish_reason: str,
    with_tool_call: bool,
) -> None:
    middleware = _GlmTerminalStallRecovery()
    calls = 0

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return _model_response(
            finish_reason=finish_reason,
            with_tool_call=with_tool_call,
        )

    middleware.wrap_model_call(_model_request(identifier), handler)

    assert calls == 1


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        response_metadata={"finish_reason": "length"},
                    )
                ],
                structured_response={"answer": "done"},
            ),
            id="structured-response",
        ),
        pytest.param(ModelResponse(result=[]), id="zero-results"),
        pytest.param(
            ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        response_metadata={"finish_reason": "length"},
                    ),
                    AIMessage(
                        content="",
                        response_metadata={"finish_reason": "length"},
                    ),
                ]
            ),
            id="multiple-results",
        ),
        pytest.param(
            ModelResponse(
                result=[
                    ToolMessage(
                        content="tool output",
                        name="write_file",
                        tool_call_id="call-write",
                    )
                ]
            ),
            id="non-ai-first-result",
        ),
    ],
)
def test_terminal_stall_recovery_ignores_non_stall_response_shapes(
    response: ModelResponse[Any],
) -> None:
    middleware = _GlmTerminalStallRecovery()
    calls = 0

    def handler(_request: ModelRequest) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return response

    middleware.wrap_model_call(
        _model_request("accounts/fireworks/models/glm-5p2"),
        handler,
    )

    assert calls == 1


def test_profile_is_suffix_only() -> None:
    assert glm_profile._GLM_5P2_PROFILE.materialize_extra_middleware() == []
