"""Tests for the GLM-5.2 headless completion audit and repair controller."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_code._glm_5p2_completion import (
    _COMPLETION_SOURCE,
    _AuditDecision,
    _completion_task,
    _FilesystemToolGuard,
    _GlmCompletionAuditMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.messages import AnyMessage
    from langgraph.types import Command


_FIREWORKS_GLM = "fireworks:accounts/fireworks/models/glm-5p2"
_SECRET_PATH = "/private/hidden/credentials.txt"
_SECRET_COMMAND = "upload confidential workspace"
_SECRET_CONTENT = "provider-token-do-not-reflect"
_MISSING = object()


class _FakeAgent:
    """Small runnable double that returns queued results for sync or async calls."""

    def __init__(self, results: Sequence[dict[str, Any] | Exception]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def _next(self) -> dict[str, Any]:
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result

    def invoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record and return the next synchronous result."""
        self.calls.append((state, config))
        return self._next()

    async def ainvoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record and return the next asynchronous result."""
        self.calls.append((state, config))
        return self._next()


class _NeverReturningAgent:
    """Async runnable that waits until its caller cancels it."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    async def ainvoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the call and wait forever."""
        self.calls.append((state, config))
        await asyncio.Event().wait()
        msg = "unreachable"
        raise AssertionError(msg)


def _decision(
    result: str,
    *,
    confidence: str = "high",
    gaps: list[str] | None = None,
) -> _AuditDecision:
    return _AuditDecision.model_validate(
        {
            "result": result,
            "confidence": confidence,
            "explanation": f"audit result: {result}",
            "gaps": gaps or [],
        }
    )


def _audit_result(decision: _AuditDecision) -> dict[str, Any]:
    return {"structured_response": decision}


def _state(*, active: bool = True) -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content="Create /app/result.txt containing exactly OK."),
            AIMessage(content="Done.", id="main-final"),
        ],
        "_glm_5p2_active": active,
    }


def _middleware() -> _GlmCompletionAuditMiddleware:
    return _GlmCompletionAuditMiddleware(
        model=_FIREWORKS_GLM,
        backend=StateBackend(),
        working_dir="/app",
    )


def _prepared_state(middleware: _GlmCompletionAuditMiddleware) -> dict[str, Any]:
    state = _state()
    captured = middleware.before_agent(cast("Any", state), cast("Any", None))
    assert captured is not None
    state.update(captured)
    return state


def _tool_request(
    name: str,
    *,
    args: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        runtime=cast("Any", None),
        tool_call={
            "id": "call-hidden-filesystem",
            "name": name,
            "args": args
            if args is not None
            else {
                "file_path": _SECRET_PATH,
                "command": _SECRET_COMMAND,
                "content": _SECRET_CONTENT,
            },
        },
        state={},
        tool=None,
    )


def _sync_stack_call(
    stack: Sequence[AgentMiddleware[Any, Any]],
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    wrappers = [
        middleware
        for middleware in stack
        if middleware.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    def invoke(index: int, current: ToolCallRequest) -> ToolMessage | Command[Any]:
        if index == len(wrappers):
            return handler(current)
        return wrappers[index].wrap_tool_call(
            current,
            lambda nested: invoke(index + 1, nested),
        )

    return invoke(0, request)


async def _async_stack_call(
    stack: Sequence[AgentMiddleware[Any, Any]],
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    wrappers = [
        middleware
        for middleware in stack
        if middleware.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

    async def invoke(
        index: int, current: ToolCallRequest
    ) -> ToolMessage | Command[Any]:
        if index == len(wrappers):
            return await handler(current)
        return await wrappers[index].awrap_tool_call(
            current,
            lambda nested: invoke(index + 1, nested),
        )

    return await invoke(0, request)


def _completion_stack(
    middleware: _GlmCompletionAuditMiddleware,
    stack_name: str,
) -> list[AgentMiddleware[Any, Any]]:
    if stack_name == "auditor":
        return middleware._auditor_middleware()
    return middleware._repair_middleware()


def _assert_generic_denial(result: ToolMessage | Command[Any], name: str) -> None:
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.name == name
    assert result.tool_call_id == "call-hidden-filesystem"
    reflected = str(result)
    assert _SECRET_PATH not in reflected
    assert _SECRET_COMMAND not in reflected
    assert _SECRET_CONTENT not in reflected
    assert "sensitive handler output" not in reflected


def _media_result() -> ToolMessage:
    return ToolMessage(
        content=[
            {"type": "text", "text": _SECRET_CONTENT},
            {
                "type": "image",
                "base64": "c2Vuc2l0aXZlLWltYWdl",
                "mime_type": "image/png",
            },
        ],
        name="read_file",
        tool_call_id="call-hidden-filesystem",
        additional_kwargs={"path": _SECRET_PATH},
    )


def test_needs_repair_requires_concrete_gaps() -> None:
    with pytest.raises(ValueError, match="at least one concrete gap"):
        _decision("needs_repair")


def test_completion_task_uses_latest_real_human_message() -> None:
    messages: list[AnyMessage] = [
        HumanMessage(content="first task"),
        HumanMessage(
            content="synthetic feedback",
            additional_kwargs={"lc_source": _COMPLETION_SOURCE},
        ),
        HumanMessage(content="second task"),
        AIMessage(content="done"),
    ]

    task = _completion_task(messages)
    repeated = _completion_task(messages)
    first = _completion_task(messages[:1])

    assert task is not None
    assert repeated is not None
    assert first is not None
    assert task.text == "second task"
    assert task.key == repeated.key
    assert task.key != first.key


def test_before_agent_captures_exact_task_and_resets_one_shot_state() -> None:
    middleware = _middleware()
    state = _state()
    state["messages"] = [
        HumanMessage(content="  preserve my exact whitespace  \n"),
        HumanMessage(
            content="synthetic feedback",
            additional_kwargs={"lc_source": _COMPLETION_SOURCE},
        ),
        AIMessage(content="done"),
    ]
    state["_glm_completion_status"] = "passed"

    update = middleware.before_agent(cast("Any", state), cast("Any", None))

    assert update is not None
    assert update["_glm_completion_task"] == "  preserve my exact whitespace  \n"
    assert update["_glm_completion_status"] == "pending"
    assert update["_glm_completion_audits"] == 0
    assert update["_glm_completion_repairs"] == 0
    assert update["_glm_completion_gaps"] == []


def test_auditor_is_read_only_and_repairer_has_bounded_non_delete_tools() -> None:
    middleware = _middleware()

    auditor_stack = middleware._auditor_middleware()
    repair_stack = middleware._repair_middleware()
    auditor_fs = next(
        item for item in auditor_stack if isinstance(item, FilesystemMiddleware)
    )
    repair_fs = next(
        item for item in repair_stack if isinstance(item, FilesystemMiddleware)
    )

    assert auditor_fs._enabled_tools == frozenset({"ls", "read_file", "glob", "grep"})
    assert repair_fs._enabled_tools == frozenset(
        {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
    )
    assert repair_fs._max_execute_timeout == 300
    assert any(
        item.__class__.__name__ == "_CompletionReadFileMediaGuard"
        for item in auditor_stack
    )
    assert any(
        item.__class__.__name__ == "_CompletionReadFileMediaGuard"
        for item in repair_stack
    )


@pytest.mark.parametrize(
    ("stack_name", "tool_name"),
    [
        ("auditor", "write_file"),
        ("auditor", "edit_file"),
        ("auditor", "delete"),
        ("auditor", "execute"),
        ("repairer", "delete"),
    ],
)
def test_hidden_filesystem_tool_is_blocked_before_sync_handler(
    stack_name: str,
    tool_name: str,
) -> None:
    stack = _completion_stack(_middleware(), stack_name)
    handler_called = False

    def handler(_request: ToolCallRequest) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return ToolMessage(
            content="sensitive handler output",
            name=tool_name,
            tool_call_id="call-hidden-filesystem",
        )

    result = _sync_stack_call(stack, _tool_request(tool_name), handler)

    assert handler_called is False
    _assert_generic_denial(result, tool_name)


@pytest.mark.parametrize(
    ("stack_name", "tool_name"),
    [
        ("auditor", "write_file"),
        ("auditor", "edit_file"),
        ("auditor", "delete"),
        ("auditor", "execute"),
        ("repairer", "delete"),
    ],
)
async def test_hidden_filesystem_tool_is_blocked_before_async_handler(
    stack_name: str,
    tool_name: str,
) -> None:
    stack = _completion_stack(_middleware(), stack_name)
    handler_called = False

    async def handler(  # noqa: RUF029  # Async callback shape is under test.
        _request: ToolCallRequest,
    ) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return ToolMessage(
            content="sensitive handler output",
            name=tool_name,
            tool_call_id="call-hidden-filesystem",
        )

    result = await _async_stack_call(stack, _tool_request(tool_name), handler)

    assert handler_called is False
    _assert_generic_denial(result, tool_name)


@pytest.mark.parametrize(
    ("stack_name", "tool_name"),
    [
        ("auditor", "read_file"),
        ("repairer", "write_file"),
        ("auditor", "_AuditDecision"),
    ],
)
def test_allowed_and_non_filesystem_tools_delegate_unchanged(
    stack_name: str,
    tool_name: str,
) -> None:
    stack = _completion_stack(_middleware(), stack_name)
    expected = ToolMessage(
        content="delegated",
        name=tool_name,
        tool_call_id="call-hidden-filesystem",
    )
    calls: list[ToolCallRequest] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request)
        return expected

    result = _sync_stack_call(stack, _tool_request(tool_name), handler)

    assert calls == [_tool_request(tool_name)]
    assert result is expected


async def test_allowed_and_non_filesystem_tools_delegate_unchanged_async() -> None:
    stack = _completion_stack(_middleware(), "auditor")
    expected = ToolMessage(
        content="delegated",
        name="_AuditDecision",
        tool_call_id="call-hidden-filesystem",
    )
    calls: list[ToolCallRequest] = []

    async def handler(  # noqa: RUF029  # Async callback shape is under test.
        request: ToolCallRequest,
    ) -> ToolMessage:
        calls.append(request)
        return expected

    request = _tool_request("_AuditDecision")
    result = await _async_stack_call(stack, request, handler)

    assert calls == [request]
    assert result is expected


def test_completion_stacks_receive_fresh_immutable_tool_guards() -> None:
    middleware = _middleware()
    first_auditor = middleware._auditor_middleware()
    second_auditor = middleware._auditor_middleware()
    repairer = middleware._repair_middleware()

    first_guards = [
        item for item in first_auditor if isinstance(item, _FilesystemToolGuard)
    ]
    second_guards = [
        item for item in second_auditor if isinstance(item, _FilesystemToolGuard)
    ]
    repair_guards = [
        item for item in repairer if isinstance(item, _FilesystemToolGuard)
    ]

    assert len(first_guards) == len(second_guards) == len(repair_guards) == 1
    assert first_guards[0] is not second_guards[0]
    assert first_guards[0]._allowed_tools == frozenset(
        {"ls", "read_file", "glob", "grep"}
    )
    assert repair_guards[0]._allowed_tools == frozenset(
        {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
    )


@pytest.mark.parametrize(
    ("timeout_value", "expected"),
    [
        pytest.param(_MISSING, 300, id="absent"),
        pytest.param(None, 300, id="none"),
        pytest.param(True, 300, id="bool"),
        pytest.param("30", 300, id="string"),
        pytest.param(0, 300, id="zero"),
        pytest.param(-1, 300, id="negative"),
        pytest.param(301, 300, id="above-cap"),
        pytest.param(1, 1, id="minimum"),
        pytest.param(180, 180, id="within-cap"),
        pytest.param(300, 300, id="at-cap"),
    ],
)
def test_repair_execute_timeout_is_bounded_before_sync_handler(
    timeout_value: object,
    expected: int,
) -> None:
    args: dict[str, Any] = {"command": "run bounded check"}
    if timeout_value is not _MISSING:
        args["timeout"] = timeout_value
    original = args.copy()
    request = _tool_request("execute", args=args)
    seen: list[ToolCallRequest] = []

    def handler(actual: ToolCallRequest) -> ToolMessage:
        seen.append(actual)
        return ToolMessage(
            content="ok",
            name="execute",
            tool_call_id="call-hidden-filesystem",
        )

    _sync_stack_call(_middleware()._repair_middleware(), request, handler)

    assert len(seen) == 1
    valid = (
        isinstance(timeout_value, int)
        and not isinstance(timeout_value, bool)
        and 1 <= timeout_value <= 300
    )
    assert (seen[0] is request) is valid
    assert seen[0].tool_call["args"]["timeout"] == expected
    assert request.tool_call["args"] == original


@pytest.mark.parametrize(
    ("timeout_value", "expected"),
    [
        pytest.param(_MISSING, 300, id="absent"),
        pytest.param(None, 300, id="none"),
        pytest.param(False, 300, id="bool"),
        pytest.param(30.5, 300, id="float"),
        pytest.param(0, 300, id="zero"),
        pytest.param(-1, 300, id="negative"),
        pytest.param(301, 300, id="above-cap"),
        pytest.param(1, 1, id="minimum"),
        pytest.param(180, 180, id="within-cap"),
        pytest.param(300, 300, id="at-cap"),
    ],
)
async def test_repair_execute_timeout_is_bounded_before_async_handler(
    timeout_value: object,
    expected: int,
) -> None:
    args: dict[str, Any] = {"command": "run bounded check"}
    if timeout_value is not _MISSING:
        args["timeout"] = timeout_value
    original = args.copy()
    request = _tool_request("execute", args=args)
    seen: list[ToolCallRequest] = []

    async def handler(  # noqa: RUF029  # Async callback shape is under test.
        actual: ToolCallRequest,
    ) -> ToolMessage:
        seen.append(actual)
        return ToolMessage(
            content="ok",
            name="execute",
            tool_call_id="call-hidden-filesystem",
        )

    await _async_stack_call(_middleware()._repair_middleware(), request, handler)

    assert len(seen) == 1
    valid = (
        isinstance(timeout_value, int)
        and not isinstance(timeout_value, bool)
        and 1 <= timeout_value <= 300
    )
    assert (seen[0] is request) is valid
    assert seen[0].tool_call["args"]["timeout"] == expected
    assert request.tool_call["args"] == original


def test_child_media_guard_has_no_model_prompt_hooks() -> None:
    for stack in (
        _middleware()._auditor_middleware(),
        _middleware()._repair_middleware(),
    ):
        guards = [
            item
            for item in stack
            if item.__class__.__name__ == "_CompletionReadFileMediaGuard"
        ]
        assert len(guards) == 1
        guard_type = guards[0].__class__
        assert guard_type.wrap_model_call is AgentMiddleware.wrap_model_call
        assert guard_type.awrap_model_call is AgentMiddleware.awrap_model_call


def test_child_media_guard_blocks_media_without_reflection_sync() -> None:
    result = _sync_stack_call(
        _middleware()._auditor_middleware(),
        _tool_request("read_file"),
        lambda _request: _media_result(),
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    for secret in (_SECRET_PATH, _SECRET_CONTENT, "image/png", "c2Vuc2l0aXZlLWltYWdl"):
        assert secret not in str(result)


async def test_child_media_guard_blocks_media_without_reflection_async() -> None:
    async def handler(  # noqa: RUF029  # Async callback shape is under test.
        _request: ToolCallRequest,
    ) -> ToolMessage:
        return _media_result()

    result = await _async_stack_call(
        _middleware()._repair_middleware(),
        _tool_request("read_file"),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    for secret in (_SECRET_PATH, _SECRET_CONTENT, "image/png", "c2Vuc2l0aXZlLWltYWdl"):
        assert secret not in str(result)


def test_pass_records_audit_without_running_repair() -> None:
    middleware = _middleware()
    auditor = _FakeAgent([_audit_result(_decision("pass"))])
    repairer = _FakeAgent([])
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    update = middleware.after_agent(
        cast("Any", _prepared_state(middleware)), cast("Any", None)
    )

    assert update is not None
    assert update["_glm_completion_status"] == "passed"
    assert update["_glm_completion_audits"] == 1
    assert update["_glm_completion_repairs"] == 0
    assert update["_glm_completion_gaps"] == []
    assert "messages" not in update
    assert len(auditor.calls) == 1
    assert repairer.calls == []


@pytest.mark.parametrize("confidence", ["medium", "low"])
def test_only_high_confidence_failure_can_run_repair(confidence: str) -> None:
    middleware = _middleware()
    auditor = _FakeAgent(
        [
            _audit_result(
                _decision(
                    "needs_repair",
                    confidence=confidence,
                    gaps=["/app/result.txt is missing"],
                )
            )
        ]
    )
    repairer = _FakeAgent([])
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    update = middleware.after_agent(
        cast("Any", _prepared_state(middleware)), cast("Any", None)
    )

    assert update is not None
    assert update["_glm_completion_status"] == "cannot_determine"
    assert update["_glm_completion_repairs"] == 0
    assert update["_glm_completion_gaps"] == ["/app/result.txt is missing"]
    assert repairer.calls == []


def test_high_confidence_failure_runs_one_fresh_repair_and_reaudit() -> None:
    middleware = _middleware()
    auditor = _FakeAgent(
        [
            _audit_result(
                _decision(
                    "needs_repair",
                    gaps=["/app/result.txt is missing"],
                )
            ),
            _audit_result(_decision("pass")),
        ]
    )
    repairer = _FakeAgent(
        [
            {
                "messages": [
                    HumanMessage(content="repair"),
                    AIMessage(content="Created and checked /app/result.txt."),
                ]
            }
        ]
    )
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    update = middleware.after_agent(
        cast("Any", _prepared_state(middleware)), cast("Any", None)
    )

    assert update is not None
    assert update["_glm_completion_status"] == "repaired"
    assert update["_glm_completion_audits"] == 2
    assert update["_glm_completion_repairs"] == 1
    assert update["_glm_completion_gaps"] == []
    assert len(auditor.calls) == 2
    assert len(repairer.calls) == 1
    assert [config for _, config in auditor.calls] == [
        {"recursion_limit": 200},
        {"recursion_limit": 200},
    ]
    assert repairer.calls[0][1] == {"recursion_limit": 200}
    messages = update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].id == "main-final"
    assert messages[0].content == "Created and checked /app/result.txt."
    assert messages[0].additional_kwargs["lc_source"] == _COMPLETION_SOURCE


def test_repair_stops_after_one_pass_when_reaudit_still_fails() -> None:
    middleware = _middleware()
    first_gap = "/app/result.txt is missing"
    second_gap = "/app/result.txt contains the wrong value"
    auditor = _FakeAgent(
        [
            _audit_result(_decision("needs_repair", gaps=[first_gap])),
            _audit_result(_decision("needs_repair", gaps=[second_gap])),
        ]
    )
    repairer = _FakeAgent(
        [{"messages": [AIMessage(content="Attempted the bounded repair.")]}]
    )
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    update = middleware.after_agent(
        cast("Any", _prepared_state(middleware)), cast("Any", None)
    )

    assert update is not None
    assert update["_glm_completion_status"] == "repair_incomplete"
    assert update["_glm_completion_audits"] == 2
    assert update["_glm_completion_repairs"] == 1
    assert update["_glm_completion_gaps"] == [second_gap]
    assert len(repairer.calls) == 1


@pytest.mark.parametrize("reaudit", ["pass", "fail", "error"])
def test_repair_exception_still_reaudits_and_replaces_sync_final(
    reaudit: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_gap = "/app/result.txt is missing"
    second_gap = "/app/result.txt still has the wrong content"
    if reaudit == "pass":
        second: dict[str, Any] | Exception = _audit_result(_decision("pass"))
    elif reaudit == "fail":
        second = _audit_result(_decision("needs_repair", gaps=[second_gap]))
    else:
        second = RuntimeError("reaudit-provider-secret")
    auditor = _FakeAgent(
        [_audit_result(_decision("needs_repair", gaps=[first_gap])), second]
    )
    repairer = _FakeAgent([RuntimeError("repair-provider-secret")])
    middleware = _middleware()
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    with caplog.at_level("WARNING", logger="deepagents_code._glm_5p2_completion"):
        update = middleware.after_agent(
            cast("Any", _prepared_state(middleware)), cast("Any", None)
        )

    assert update is not None
    assert update["_glm_completion_status"] == (
        "repaired" if reaudit == "pass" else "repair_incomplete"
    )
    assert update["_glm_completion_audits"] == 2
    assert update["_glm_completion_repairs"] == 1
    assert len(auditor.calls) == 2
    assert len(repairer.calls) == 1
    replacement = update["messages"][0]
    assert isinstance(replacement, AIMessage)
    assert replacement.id == "main-final"
    assert replacement.additional_kwargs["lc_source"] == _COMPLETION_SOURCE
    assert replacement.content != "Done."
    assert ("verified" if reaudit == "pass" else "incomplete") in replacement.text
    assert "repair-provider-secret" not in caplog.text
    assert "reaudit-provider-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("reaudit", ["pass", "fail", "error"])
async def test_repair_exception_still_reaudits_and_replaces_async_final(
    reaudit: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_gap = "/app/result.txt is missing"
    second_gap = "/app/result.txt still has the wrong content"
    if reaudit == "pass":
        second: dict[str, Any] | Exception = _audit_result(_decision("pass"))
    elif reaudit == "fail":
        second = _audit_result(_decision("needs_repair", gaps=[second_gap]))
    else:
        second = RuntimeError("async-reaudit-provider-secret")
    auditor = _FakeAgent(
        [_audit_result(_decision("needs_repair", gaps=[first_gap])), second]
    )
    repairer = _FakeAgent([RuntimeError("async-repair-provider-secret")])
    middleware = _middleware()
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    with caplog.at_level("WARNING", logger="deepagents_code._glm_5p2_completion"):
        update = await middleware.aafter_agent(
            cast("Any", _prepared_state(middleware)), cast("Any", None)
        )

    assert update is not None
    assert update["_glm_completion_status"] == (
        "repaired" if reaudit == "pass" else "repair_incomplete"
    )
    assert update["_glm_completion_audits"] == 2
    assert update["_glm_completion_repairs"] == 1
    assert len(auditor.calls) == 2
    assert len(repairer.calls) == 1
    replacement = update["messages"][0]
    assert isinstance(replacement, AIMessage)
    assert replacement.id == "main-final"
    assert replacement.additional_kwargs["lc_source"] == _COMPLETION_SOURCE
    assert replacement.content != "Done."
    assert ("verified" if reaudit == "pass" else "incomplete") in replacement.text
    assert "async-repair-provider-secret" not in caplog.text
    assert "async-reaudit-provider-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_repair_final_rejects_invalid_tool_calls() -> None:
    invalid = AIMessage(
        content="not a natural final",
        invalid_tool_calls=[
            {
                "id": "call-truncated",
                "name": "write_file",
                "args": '{"file_path": "/app/result.txt"',
                "error": "truncated",
                "type": "invalid_tool_call",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="final AIMessage"):
        _GlmCompletionAuditMiddleware._extract_repair_final(
            {
                "messages": [
                    AIMessage(content="stale earlier response"),
                    invalid,
                ]
            }
        )


def test_partial_repair_generic_final_preserves_only_safe_identity() -> None:
    original = AIMessage(
        content="stale",
        id="main-final",
        additional_kwargs={"provider_detail": "sensitive"},
    )

    replacement = _GlmCompletionAuditMiddleware._repair_failure_message(
        original,
        verified=False,
    )

    assert replacement.id == "main-final"
    assert replacement.additional_kwargs == {"lc_source": _COMPLETION_SOURCE}
    assert "sensitive" not in str(replacement)


def test_same_task_is_not_audited_twice() -> None:
    middleware = _middleware()
    auditor = _FakeAgent([_audit_result(_decision("pass"))])
    middleware._auditor = cast("Any", auditor)
    state = _prepared_state(middleware)

    first = middleware.after_agent(cast("Any", state), cast("Any", None))
    assert first is not None
    state.update(first)
    second = middleware.after_agent(cast("Any", state), cast("Any", None))

    assert second is None
    assert len(auditor.calls) == 1


@pytest.mark.parametrize(
    ("state_update", "expected"),
    [
        ({"_glm_5p2_active": False}, None),
        ({"rubric": "explicit caller rubric"}, None),
        ({"messages": [AIMessage(content="no task")]}, None),
    ],
)
def test_inactive_explicit_rubric_and_missing_task_are_noops(
    state_update: dict[str, Any], expected: None
) -> None:
    middleware = _middleware()
    auditor = _FakeAgent([])
    middleware._auditor = cast("Any", auditor)
    state = (
        {"_glm_5p2_active": True, "_glm_completion_status": "pending"}
        if "messages" in state_update
        else _prepared_state(middleware)
    )
    state.update(state_update)

    result = middleware.after_agent(cast("Any", state), cast("Any", None))

    assert result is expected
    assert auditor.calls == []


def test_audit_error_is_contained_without_running_repair(
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware = _middleware()
    auditor = _FakeAgent([RuntimeError("sensitive provider detail")])
    repairer = _FakeAgent([])
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    with caplog.at_level("WARNING", logger="deepagents_code._glm_5p2_completion"):
        update = middleware.after_agent(
            cast("Any", _prepared_state(middleware)), cast("Any", None)
        )

    assert update is not None
    assert update["_glm_completion_status"] == "audit_error"
    assert update["_glm_completion_audits"] == 1
    assert update["_glm_completion_repairs"] == 0
    assert "sensitive provider detail" not in str(update)
    assert "sensitive provider detail" not in caplog.text
    assert "audit" in caplog.text
    assert "RuntimeError" in caplog.text
    assert repairer.calls == []


async def test_async_audit_error_log_excludes_exception_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware = _middleware()
    middleware._auditor = cast(
        "Any", _FakeAgent([RuntimeError("async-sensitive-provider-detail")])
    )

    with caplog.at_level("WARNING", logger="deepagents_code._glm_5p2_completion"):
        update = await middleware.aafter_agent(
            cast("Any", _prepared_state(middleware)), cast("Any", None)
        )

    assert update is not None
    assert update["_glm_completion_status"] == "audit_error"
    assert "async-sensitive-provider-detail" not in caplog.text
    assert "audit" in caplog.text
    assert "RuntimeError" in caplog.text


async def test_async_path_runs_one_repair_and_reaudit() -> None:
    middleware = _middleware()
    auditor = _FakeAgent(
        [
            _audit_result(
                _decision("needs_repair", gaps=["/app/result.txt is missing"])
            ),
            _audit_result(_decision("pass")),
        ]
    )
    repairer = _FakeAgent([{"messages": [AIMessage(content="Async repair complete.")]}])
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)

    update = await middleware.aafter_agent(
        cast("Any", _prepared_state(middleware)), cast("Any", None)
    )

    assert update is not None
    assert update["_glm_completion_status"] == "repaired"
    assert update["_glm_completion_audits"] == 2
    assert update["_glm_completion_repairs"] == 1
    assert len(auditor.calls) == 2
    assert len(repairer.calls) == 1
    assert [config for _, config in auditor.calls] == [
        {"recursion_limit": 200},
        {"recursion_limit": 200},
    ]
    assert repairer.calls[0][1] == {"recursion_limit": 200}


async def test_async_audit_phase_timeout_is_contained(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    middleware = _middleware()
    auditor = _NeverReturningAgent()
    repairer = _FakeAgent([])
    middleware._auditor = cast("Any", auditor)
    middleware._repairer = cast("Any", repairer)
    monkeypatch.setattr(
        "deepagents_code._glm_5p2_completion._COMPLETION_PHASE_TIMEOUT_SECONDS",
        0.001,
        raising=False,
    )

    with caplog.at_level("WARNING", logger="deepagents_code._glm_5p2_completion"):
        update = await asyncio.wait_for(
            middleware.aafter_agent(
                cast("Any", _prepared_state(middleware)), cast("Any", None)
            ),
            timeout=0.1,
        )

    assert update is not None
    assert update["_glm_completion_status"] == "audit_error"
    assert update["_glm_completion_audits"] == 1
    assert update["_glm_completion_repairs"] == 0
    assert "TimeoutError" not in str(update)
    assert "Traceback" not in caplog.text
    assert repairer.calls == []
    assert auditor.calls[0][1] == {"recursion_limit": 200}


def test_graph_replaces_the_natural_final_with_the_fresh_repair() -> None:
    main_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="premature final", id="main-final")])
    )
    middleware = _middleware()
    middleware._construction_active = True
    middleware._auditor = cast(
        "Any",
        _FakeAgent(
            [
                _audit_result(
                    _decision(
                        "needs_repair",
                        gaps=["/app/result.txt is missing"],
                    )
                ),
                _audit_result(_decision("pass")),
            ]
        ),
    )
    middleware._repairer = cast(
        "Any",
        _FakeAgent(
            [{"messages": [AIMessage(content="fresh repair", id="repair-final")]}]
        ),
    )
    graph = create_agent(model=main_model, middleware=[middleware])

    result = graph.invoke(
        {"messages": [HumanMessage(content="Create /app/result.txt")]}
    )

    ai_messages = [
        message for message in result["messages"] if isinstance(message, AIMessage)
    ]
    assert len(ai_messages) == 1
    assert ai_messages[0].id == "main-final"
    assert ai_messages[0].content == "fresh repair"
    assert ai_messages[0].additional_kwargs["lc_source"] == _COMPLETION_SOURCE
