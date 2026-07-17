"""Tests for the GLM-5.2 Deep Agents Code harness profile."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints
from unittest.mock import MagicMock

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.factory import (
    _chain_model_call_handlers,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagents_code import _glm_5p2_profile as glm_profile
from deepagents_code._glm_5p2_profile import (
    _GlmReadFileMediaGuard,
    _GlmTerminalStallRecovery,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware

if TYPE_CHECKING:
    from deepagents.profiles import HarnessProfile
    from langchain_core.tools import BaseTool


_FIREWORKS_GLM = "fireworks:accounts/fireworks/models/glm-5p2"
_OPENROUTER_GLM = "openrouter:z-ai/glm-5.2"
_BASETEN_GLM = "baseten:zai-org/GLM-5.2"
_NON_GLM = "openai:gpt-5.5"
_MISSING = object()

_PROVIDER_BY_IDENTIFIER = {
    "accounts/fireworks/models/glm-5p2": "fireworks",
    "z-ai/glm-5.2": "openrouter",
    "zai-org/GLM-5.2": "baseten",
    "gpt-5.5": "openai",
}


@pytest.fixture(autouse=True)
def _dcode_owns_all_glm_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: the dcode profile owns every GLM spec, as after registration.

    The suffix transition only manages a spec the dcode profile actually owns, so
    guard tests need the registry populated. Tests that model a user/built-in
    override override `_HARNESS_PROFILES` themselves.
    """
    import deepagents.profiles.harness.harness_profiles as core_profiles

    monkeypatch.setattr(
        core_profiles,
        "_HARNESS_PROFILES",
        dict.fromkeys(glm_profile._GLM_5P2_MODEL_SPECS, glm_profile._GLM_5P2_PROFILE),
    )


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
    requested_model: str | None = None,
) -> ModelRequest:
    runtime = SimpleNamespace(context={"model": requested_model})
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


def _model_update(result: ModelResponse[Any] | ExtendedModelResponse[Any]) -> bool:
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert isinstance(result.command.update, dict)
    active = result.command.update["_glm_5p2_active"]
    assert isinstance(active, bool)
    return active


def _request(
    name: str = "read_file",
    *,
    active: object = True,
) -> ToolCallRequest:
    state = {} if active is _MISSING else {"_glm_5p2_active": active}
    return ToolCallRequest(
        runtime=cast("Any", None),
        tool_call={
            "id": "call-media",
            "name": name,
            "args": {"file_path": "/private/input/secret.png"},
        },
        state=state,
        tool=None,
    )


def _media_message(
    *, block_type: str = "image", mime: str = "image/png"
) -> ToolMessage:
    return ToolMessage(
        content=[
            {"type": "text", "text": "confidential caption"},
            {
                "type": block_type,
                "base64": "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
                "mime_type": mime,
            },
        ],
        name="read_file",
        tool_call_id="call-media",
        additional_kwargs={"read_file_path": "/private/input/secret.png"},
        status="success",
    )


def _assert_generic_media_error(result: ToolMessage | Command[Any]) -> None:
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.name == "read_file"
    assert result.tool_call_id == "call-media"
    assert result.additional_kwargs == {}

    text = result.text.lower()
    assert "shell" in text
    assert "script" in text
    for sensitive in (
        "/private/input/secret.png",
        "image/png",
        "application/pdf",
        "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
        "confidential caption",
    ):
        assert sensitive not in str(result)


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
    # A pre-existing suffix profile (user override or built-in) must win.
    monkeypatch.setattr(
        core_profiles,
        "_HARNESS_PROFILES",
        {_FIREWORKS_GLM: RuntimeHarnessProfile(system_prompt_suffix="user override")},
    )

    glm_profile._ensure_glm_5p2_profile_registered()

    assert calls == [_OPENROUTER_GLM, _BASETEN_GLM]


def test_deferred_override_spec_keeps_its_own_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepagents.profiles.harness.harness_profiles as core_profiles
    from deepagents.profiles import HarnessProfile as RuntimeHarnessProfile

    # A user override owns the Fireworks GLM spec; the dcode profile deferred to
    # it at registration, so the runtime guard must not append the dcode suffix.
    monkeypatch.setattr(
        core_profiles,
        "_HARNESS_PROFILES",
        {_FIREWORKS_GLM: RuntimeHarnessProfile(system_prompt_suffix="CUSTOM")},
    )
    middleware = _GlmReadFileMediaGuard(_NON_GLM)

    model_result, actual = _run_model(
        middleware,
        _model_request(
            "accounts/fireworks/models/glm-5p2",
            prompt="base prompt\n\nCUSTOM",
        ),
    )

    # Media gating still applies (text-only model), but the override's suffix is
    # left untouched rather than getting the dcode suffix bolted on.
    assert _model_update(model_result) is True
    assert actual.system_prompt == "base prompt\n\nCUSTOM"
    assert glm_profile._SYSTEM_PROMPT_SUFFIX not in actual.system_prompt


def test_prompt_is_concise_and_execution_focused() -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX

    # Guard conciseness (against prompt bloat) without mirroring exact wording
    # or tag names, which would be brittle to reword.
    assert 240 <= len(suffix.split()) <= 360

    # The execution suffix must not absorb the todo-oriented base agent prompt.
    for omitted in (
        "write_todos",
        "<todo_rules>",
        "<tool_preferences>",
        "Prefer specialized tools",
    ):
        assert omitted not in suffix


def _run_model(
    middleware: _GlmReadFileMediaGuard,
    request: ModelRequest,
) -> tuple[ModelResponse[Any] | ExtendedModelResponse[Any], ModelRequest]:
    captured: list[ModelRequest] = []

    def handler(actual: ModelRequest) -> ModelResponse[Any]:
        captured.append(actual)
        return _model_response()

    result = middleware.wrap_model_call(request, handler)
    return result, captured[0]


def test_non_glm_to_glm_adds_suffix_and_blocks_following_media() -> None:
    middleware = _GlmReadFileMediaGuard(_NON_GLM)
    model_result, actual = _run_model(
        middleware,
        _model_request("accounts/fireworks/models/glm-5p2"),
    )

    active = _model_update(model_result)
    assert active is True
    assert actual.system_prompt == f"base prompt\n\n{glm_profile._SYSTEM_PROMPT_SUFFIX}"

    media = _media_message()
    tool_result = middleware.wrap_tool_call(
        _request(active=active),
        lambda _request: media,
    )
    _assert_generic_media_error(tool_result)


def test_glm_to_non_glm_removes_suffix_and_passes_following_media() -> None:
    middleware = _GlmReadFileMediaGuard(_OPENROUTER_GLM)
    prompt = f"base prompt\n\n{glm_profile._SYSTEM_PROMPT_SUFFIX}"
    model_result, actual = _run_model(
        middleware,
        _model_request("gpt-5.5", prompt=prompt),
    )

    active = _model_update(model_result)
    assert active is False
    assert actual.system_prompt == "base prompt"

    media = _media_message()
    tool_result = middleware.wrap_tool_call(
        _request(active=active),
        lambda _request: media,
    )
    assert tool_result is media


@pytest.mark.parametrize(
    ("actual_identifier", "requested_model", "expected_active"),
    [
        ("z-ai/glm-5.2", _NON_GLM, True),
        ("gpt-5.5", _BASETEN_GLM, False),
    ],
)
def test_actual_model_wins_over_conflicting_requested_context(
    actual_identifier: str,
    requested_model: str,
    expected_active: bool,
) -> None:
    middleware = _GlmReadFileMediaGuard(_NON_GLM)
    result, actual = _run_model(
        middleware,
        _model_request(
            actual_identifier,
            requested_model=requested_model,
        ),
    )

    assert _model_update(result) == expected_active
    assert actual.system_prompt is not None
    assert (
        glm_profile._SYSTEM_PROMPT_SUFFIX in actual.system_prompt
    ) == expected_active


@pytest.mark.parametrize(
    "identifier",
    [
        "accounts/fireworks/models/glm-5p2",
        "z-ai/glm-5.2",
        "zai-org/GLM-5.2",
    ],
)
def test_model_wrapper_recognizes_provider_native_glm_identifiers(
    identifier: str,
) -> None:
    middleware = _GlmReadFileMediaGuard(_NON_GLM)

    result, actual = _run_model(middleware, _model_request(identifier))

    assert _model_update(result) is True
    assert actual.system_prompt is not None
    assert actual.system_prompt.endswith(glm_profile._SYSTEM_PROMPT_SUFFIX)


@pytest.mark.parametrize(
    "identifier",
    [
        "accounts/fireworks/models/glm-5p2",
        "z-ai/glm-5.2",
        "zai-org/GLM-5.2",
    ],
)
def test_model_wrapper_rejects_glm_identifier_from_other_provider(
    identifier: str,
) -> None:
    middleware = _GlmReadFileMediaGuard(_NON_GLM)

    result, actual = _run_model(
        middleware,
        _model_request(identifier, provider="custom_gateway"),
    )

    assert _model_update(result) is False
    assert actual.system_prompt == "base prompt"


@pytest.mark.parametrize(
    "identifier",
    [
        "accounts/fireworks/models/glm-5p2",
        "z-ai/glm-5.2",
        "zai-org/GLM-5.2",
    ],
)
def test_construction_spec_rejects_glm_identifier_from_other_provider(
    identifier: str,
) -> None:
    middleware = _GlmReadFileMediaGuard(f"custom_gateway:{identifier}")
    media = _media_message()

    result = middleware.wrap_tool_call(
        _request(active=_MISSING),
        lambda _request: media,
    )

    assert result is media


def test_model_wrapper_deduplicates_trusted_suffix() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    prompt = f"base prompt\n\n{suffix}\n\n{suffix}"

    result, actual = _run_model(
        middleware,
        _model_request("accounts/fireworks/models/glm-5p2", prompt=prompt),
    )

    assert _model_update(result) is True
    assert actual.system_prompt is not None
    assert actual.system_prompt.count(suffix) == 1


def test_active_model_moves_embedded_trusted_suffix_to_tail() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    prompt = f"base prompt\n\n{suffix}\n\ndownstream middleware appendix"

    result, actual = _run_model(
        middleware,
        _model_request("accounts/fireworks/models/glm-5p2", prompt=prompt),
    )

    assert _model_update(result) is True
    assert actual.system_prompt == (
        f"base prompt\n\ndownstream middleware appendix\n\n{suffix}"
    )
    assert actual.system_prompt.count(suffix) == 1


def test_inactive_model_removes_embedded_trusted_suffix() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    prompt = f"base prompt\n\n{suffix}\n\ndownstream middleware appendix"

    result, actual = _run_model(
        middleware,
        _model_request("gpt-5.5", prompt=prompt),
    )

    assert _model_update(result) is False
    assert actual.system_prompt == "base prompt\n\ndownstream middleware appendix"
    assert suffix not in actual.system_prompt


@pytest.mark.parametrize(
    ("prompt_template", "expected"),
    [
        ("{suffix}", ""),
        ("{suffix}\n\ndownstream appendix", "downstream appendix"),
    ],
)
def test_inactive_model_removes_leading_trusted_suffix(
    prompt_template: str,
    expected: str,
) -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)

    _, actual = _run_model(
        middleware,
        _model_request(
            "gpt-5.5",
            prompt=prompt_template.format(suffix=suffix),
        ),
    )

    assert actual.system_prompt == expected


@pytest.mark.parametrize(
    "prompt_template",
    [
        "user prefix{suffix}\n\ndownstream appendix",
        "base prompt\n\n{suffix}user suffix",
    ],
)
def test_inactive_model_preserves_undelimited_suffix_text(
    prompt_template: str,
) -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    prompt = prompt_template.format(suffix=suffix)
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)

    _, actual = _run_model(
        middleware,
        _model_request("gpt-5.5", prompt=prompt),
    )

    assert actual.system_prompt == prompt


def _run_composed_model_stack(
    middleware: list[AgentMiddleware[Any, Any]],
    request: ModelRequest,
) -> ModelRequest:
    """Return the request seen after LangChain composes the middleware stack."""
    composed = _chain_model_call_handlers([item.wrap_model_call for item in middleware])
    assert composed is not None
    captured: list[ModelRequest] = []

    def handler(actual: ModelRequest) -> ModelResponse[Any]:
        captured.append(actual)
        return _model_response()

    composed(request, handler)
    return captured[0]


@pytest.mark.parametrize(
    (
        "construction_model",
        "initial_identifier",
        "requested_model",
        "target_identifier",
        "persist_model_state",
        "expected_active",
    ),
    [
        pytest.param(
            _FIREWORKS_GLM,
            "accounts/fireworks/models/glm-5p2",
            None,
            None,
            True,
            True,
            id="initial-glm",
        ),
        pytest.param(
            _FIREWORKS_GLM,
            "accounts/fireworks/models/glm-5p2",
            _NON_GLM,
            "gpt-5.5",
            True,
            False,
            id="glm-to-non-glm",
        ),
        pytest.param(
            _NON_GLM,
            "gpt-5.5",
            _OPENROUTER_GLM,
            "z-ai/glm-5.2",
            False,
            True,
            id="inherited-subagent-non-glm-to-glm",
        ),
    ],
)
def test_composed_filesystem_stack_keeps_trusted_suffix_model_scoped(
    monkeypatch: pytest.MonkeyPatch,
    construction_model: str,
    initial_identifier: str,
    requested_model: str | None,
    target_identifier: str | None,
    persist_model_state: bool,
    expected_active: bool,
) -> None:
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    prompt = (
        "base prompt" if construction_model == _NON_GLM else f"base prompt\n\n{suffix}"
    )
    if target_identifier is not None:
        assert requested_model is not None
        target_model = _model(target_identifier)
        model_result = SimpleNamespace(
            model=target_model,
            model_name=target_identifier,
            provider=requested_model.partition(":")[0],
            context_limit=None,
            unsupported_modalities=frozenset(),
        )
        monkeypatch.setattr(
            "deepagents_code.config.create_model",
            lambda _spec: model_result,
        )

    actual = _run_composed_model_stack(
        [
            FilesystemMiddleware(backend=StateBackend()),
            ConfigurableModelMiddleware(
                persist_model_state=persist_model_state,
            ),
            _GlmReadFileMediaGuard(construction_model),
        ],
        _model_request(
            initial_identifier,
            prompt=prompt,
            requested_model=requested_model,
        ),
    )

    assert actual.system_prompt is not None
    assert "base prompt\n\n## Following Conventions" in actual.system_prompt
    assert actual.system_prompt.count(suffix) == int(expected_active)
    assert actual.system_prompt.endswith(suffix) is expected_active


async def test_async_model_wrapper_matches_sync_transition() -> None:
    middleware = _GlmReadFileMediaGuard(_NON_GLM)
    request = _model_request("z-ai/glm-5.2")
    captured: list[ModelRequest] = []

    async def handler(actual: ModelRequest) -> ModelResponse[Any]:
        await asyncio.sleep(0)
        captured.append(actual)
        return _model_response()

    result = await middleware.awrap_model_call(request, handler)

    assert _model_update(result) is True
    assert captured[0].system_prompt is not None
    assert captured[0].system_prompt.endswith(glm_profile._SYSTEM_PROMPT_SUFFIX)


def test_headless_glm_retries_length_truncated_turn() -> None:
    middleware = _GlmTerminalStallRecovery()
    tools: list[BaseTool | dict[str, Any]] = [{"name": "write_file"}]
    # Fireworks carries reasoning effort nested under `model_kwargs` (see dcode
    # `reasoning_effort._fireworks_model_params`), so the request the recovery
    # sees uses that shape.
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
    # Reasoning is disabled in the nested `model_kwargs` form Fireworks reads,
    # not at the top level, and other model kwargs are preserved.
    assert requests[1].model_settings == {
        "model_kwargs": {"reasoning_effort": "none"},
        "temperature": 0.25,
    }
    assert requests[1].tools == tools
    # The original request is left unmutated.
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
    """A length-capped but non-stall response shape must not trigger recovery.

    Covers the `_is_terminal_stall` guards beyond `finish_reason`/`tool_calls`:
    a structured response, a result list that is not exactly one message, and a
    first result that is not an `AIMessage`.
    """
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


@pytest.mark.parametrize(
    "active",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(None, id="none"),
        pytest.param("true", id="string"),
        pytest.param(1, id="integer"),
    ],
)
@pytest.mark.parametrize(
    ("construction_model", "expected_blocked"),
    [(_BASETEN_GLM, True), (_NON_GLM, False)],
)
def test_tool_wrapper_uses_construction_default_for_invalid_state(
    active: object,
    construction_model: str,
    expected_blocked: bool,
) -> None:
    middleware = _GlmReadFileMediaGuard(construction_model)
    media = _media_message()

    result = middleware.wrap_tool_call(
        _request(active=active),
        lambda _request: media,
    )

    if expected_blocked:
        _assert_generic_media_error(result)
    else:
        assert result is media


def test_guard_state_is_private() -> None:
    hints = get_type_hints(
        _GlmReadFileMediaGuard.state_schema,
        include_extras=True,
    )

    assert PrivateStateAttr in get_args(hints["_glm_5p2_active"])


@pytest.mark.parametrize(
    "content",
    [
        "1\tplain text",
        ["plain text"],
        [{"type": "text", "text": "plain text"}],
    ],
)
def test_media_guard_passes_text_read_unchanged(
    content: str | list[str] | list[dict[str, str]],
) -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    # LangChain's mutable-list overload is invariant even though these dicts
    # are valid text content blocks at runtime.
    tool_content = cast("str | list[str | dict[Any, Any]]", content)
    original = ToolMessage(
        content=tool_content,
        name="read_file",
        tool_call_id="call-media",
        status="success",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    assert result is original


def test_media_guard_passes_error_read_unchanged() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    original = ToolMessage(
        content="Error: path validation rejected /private/input/secret.png",
        name="read_file",
        tool_call_id="call-media",
        status="error",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    assert result is original


def test_media_guard_replaces_error_media_without_payload() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    original = _media_message().model_copy(update={"status": "error"})

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    _assert_generic_media_error(result)


@pytest.mark.parametrize(
    ("block_type", "mime"),
    [("image", "image/png"), ("file", "application/pdf")],
)
def test_media_guard_replaces_media_tool_message_without_payload(
    block_type: str,
    mime: str,
) -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    original = _media_message(block_type=block_type, mime=mime)

    result = middleware.wrap_tool_call(_request(), lambda _request: original)

    _assert_generic_media_error(result)


def test_media_guard_fails_closed_on_malformed_content_blocks() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    malformed = ToolMessage(
        content=[{"type": "text"}],
        name="read_file",
        tool_call_id="call-media",
        status="success",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: malformed)

    _assert_generic_media_error(result)


def test_media_guard_fails_closed_on_empty_list_content() -> None:
    """An empty content list is not positively text, so it is replaced.

    Guards the asymmetry with an empty string (which passes as text): an empty
    list falls through `_has_only_text_content` and must fail closed.
    """
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    empty = ToolMessage(
        content=[],
        name="read_file",
        tool_call_id="call-media",
        status="success",
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: empty)

    _assert_generic_media_error(result)


def test_media_guard_replaces_video_command() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    video = Command(
        update={
            "messages": [
                ToolMessage(
                    content="Read video /private/input/secret.png",
                    name="read_file",
                    tool_call_id="call-media",
                    status="success",
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "confidential caption"},
                        {
                            "type": "image",
                            "base64": "c2VjcmV0LW1lZGlhLXBheWxvYWQ=",
                            "mime_type": "image/png",
                        },
                    ]
                ),
            ]
        }
    )

    result = middleware.wrap_tool_call(_request(), lambda _request: video)

    _assert_generic_media_error(result)


async def test_media_guard_async_matches_sync_behavior() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    original = _media_message()

    async def handler(_request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        return original

    sync_result = middleware.wrap_tool_call(_request(), lambda _request: original)
    async_result = await middleware.awrap_tool_call(_request(), handler)

    assert async_result == sync_result
    _assert_generic_media_error(async_result)


def test_media_guard_ignores_other_tools() -> None:
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)
    original = _media_message()

    result = middleware.wrap_tool_call(
        _request("custom_media_tool"),
        lambda _request: original,
    )

    assert result is original


def _unresolvable_model_request(prompt: str = "base prompt") -> ModelRequest:
    """A request whose model exposes no derivable provider/identifier."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = None
    model.model = None
    model._get_ls_params.return_value = {}
    runtime = SimpleNamespace(context={"model": None})
    return ModelRequest(
        model=cast("BaseChatModel", model),
        messages=[HumanMessage(content="run")],
        tools=[],
        system_prompt=prompt,
        state={"messages": []},
        runtime=cast("Any", runtime),
    )


def test_media_guard_fails_closed_when_runtime_model_unresolvable() -> None:
    """An unresolvable runtime spec falls back to the GLM construction default.

    A text-only guard must not treat "cannot resolve the model" as "not GLM"
    and let media through; it reuses the construction-time classification.
    """
    middleware = _GlmReadFileMediaGuard(_FIREWORKS_GLM)

    model_result, actual = _run_model(middleware, _unresolvable_model_request())

    active = _model_update(model_result)
    assert active is True
    # The prompt is left untouched: the guard cannot know which profile owns any
    # suffix present, so it neither adds nor removes one.
    assert actual.system_prompt == "base prompt"

    media = _media_message()
    tool_result = middleware.wrap_tool_call(
        _request(active=active),
        lambda _request: media,
    )
    _assert_generic_media_error(tool_result)


def test_media_guard_unresolvable_model_stays_inactive_for_non_glm_construction() -> (
    None
):
    """Fail-closed means the construction default, not "always block".

    A non-GLM stack whose runtime spec is unresolvable must keep passing media
    through rather than start blocking it.
    """
    middleware = _GlmReadFileMediaGuard(_NON_GLM)

    model_result, actual = _run_model(middleware, _unresolvable_model_request())

    active = _model_update(model_result)
    assert active is False
    assert actual.system_prompt == "base prompt"

    media = _media_message()
    tool_result = middleware.wrap_tool_call(
        _request(active=active),
        lambda _request: media,
    )
    assert tool_result is media


def test_media_guard_and_stall_recovery_compose_prompt_order() -> None:
    """Recovery (inner) appends its suffix after the guard's execution suffix.

    Exercises the two middlewares composed as they are wired in `create_cli_agent`
    (media guard outer, recovery inner): on a real stall the retry must keep the
    guard's transitioned prompt with the execution suffix intact at the tail, and
    disable reasoning in the nested `model_kwargs` form Fireworks reads.
    """
    suffix = glm_profile._SYSTEM_PROMPT_SUFFIX
    recovery_suffix = glm_profile._TERMINAL_STALL_RECOVERY_SUFFIX
    stack: list[AgentMiddleware[Any, Any]] = [
        _GlmReadFileMediaGuard(_FIREWORKS_GLM),
        _GlmTerminalStallRecovery(),
    ]
    composed = _chain_model_call_handlers([mw.wrap_model_call for mw in stack])
    assert composed is not None

    captured: list[ModelRequest] = []
    responses = iter(
        [
            _model_response(content="unfinished", finish_reason="length"),
            _model_response(content="recovered", with_tool_call=True),
        ]
    )

    def handler(actual: ModelRequest) -> ModelResponse[Any]:
        captured.append(actual)
        return next(responses)

    result = composed(
        _model_request("accounts/fireworks/models/glm-5p2", prompt="base prompt"),
        handler,
    )

    assert len(captured) == 2
    # First call: the outer media guard appended the execution suffix at the tail.
    assert captured[0].system_prompt == f"base prompt\n\n{suffix}"
    # Retry: recovery appended its suffix AFTER the execution suffix, leaving the
    # execution suffix intact (count == 1) rather than displacing it.
    assert captured[1].system_prompt == f"base prompt\n\n{suffix}\n\n{recovery_suffix}"
    assert captured[1].system_prompt.count(suffix) == 1
    assert captured[1].model_settings == {"model_kwargs": {"reasoning_effort": "none"}}
    # The recovered turn propagates out, and the guard still marks GLM active.
    assert result.model_response.result[0].text == "recovered"
    assert any(
        isinstance(command, Command) and command.update == {"_glm_5p2_active": True}
        for command in result.commands
    )


def test_profile_is_suffix_only() -> None:
    assert glm_profile._GLM_5P2_PROFILE.materialize_extra_middleware() == []
