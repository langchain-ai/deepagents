"""Unit tests for CodeInterpreterMiddleware and its backing REPL wrapper."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import (
    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY,
    SubAgentMiddleware,
    _build_task_tool,
)
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.structured_output import AutoStrategy
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt
from pydantic import BaseModel, Field
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._format import format_outcome
from langchain_quickjs._repl import _clear_exception_references, _Registry, _ThreadREPL
from langchain_quickjs._subagent import (
    _ensure_schema_title,
    _runtime_with_response_format,
    call_subagent_task_tool,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def worker() -> ThreadWorker:
    w = ThreadWorker()
    try:
        yield w
    finally:
        w.close()


@pytest.fixture
def runtime(worker: ThreadWorker) -> Runtime:
    """A fresh QuickJS Runtime for tests that drive _ThreadREPL directly."""

    async def _make() -> Runtime:
        return Runtime()

    rt = worker.run_sync(_make())
    try:
        yield rt
    finally:

        async def _close() -> None:
            rt.close()

        worker.run_sync(_close())


@pytest.fixture
def repl(worker: ThreadWorker, runtime: Runtime) -> _ThreadREPL:
    return _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )


# ---------------------------------------------------------------------------
# Registration + system prompt
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal chat model that records the last request and returns a stock reply.

    We don't actually invoke it; we only need something create_agent accepts
    and whose tools binding we can introspect.
    """

    def bind_tools(self, tools, **_: object) -> _StubModel:
        self._tools = tools
        return self

    def invoke(self, *_a, **_k):  # pragma: no cover — not exercised
        from langchain_core.messages import AIMessage

        return AIMessage(content="ok")


def test_tool_registered_with_default_name() -> None:
    mw = CodeInterpreterMiddleware()
    # langchain's create_agent accepts a model string; we use a cheap local
    # fake to avoid any provider import. Any string maps through init_chat_model,
    # but we want to avoid network/config; go direct via tools=[] + our middleware.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        middleware=[mw],
    )
    tools = agent.nodes["tools"].bound._tools_by_name
    assert "eval" in tools
    assert "persistent" in tools["eval"].description.lower()
    assert tools["eval"].metadata["ls_code_input_language"] == "javascript"


def test_tool_registered_with_custom_name() -> None:
    mw = CodeInterpreterMiddleware(tool_name="js")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        middleware=[mw],
    )
    tools = agent.nodes["tools"].bound._tools_by_name
    assert "js" in tools
    assert "eval" not in tools
    assert tools["js"].metadata["ls_code_input_language"] == "javascript"


def test_legacy_system_prompt_alias_removed() -> None:
    mw = CodeInterpreterMiddleware()
    assert not hasattr(mw, "system_prompt")


def test_rejects_invalid_max_ptc_calls() -> None:
    with pytest.raises(ValueError, match="must be >= 1 or None"):
        CodeInterpreterMiddleware(max_ptc_calls=0)


def test_rejects_invalid_max_snapshot_bytes() -> None:
    with pytest.raises(ValueError, match="must be >= 1 or None"):
        CodeInterpreterMiddleware(max_snapshot_bytes=0)


def test_system_prompt_injected_once() -> None:
    """wrap_model_call appends exactly one snippet per call, idempotent in content."""
    mw = CodeInterpreterMiddleware(timeout=7.0, memory_limit=32 * 1024 * 1024)
    seen: list[ModelRequest] = []

    def handler(req: ModelRequest):
        seen.append(req)
        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage

        return ModelResponse(result=[AIMessage(content="ok")])

    req = MagicMock(spec=ModelRequest)
    req.system_message = SystemMessage(content="base")

    # override() returns a new ModelRequest with the given fields replaced;
    # emulate that with a MagicMock-returning-self pattern.
    def _override(**kwargs):
        new = MagicMock(spec=ModelRequest)
        new.system_message = kwargs.get("system_message", req.system_message)
        return new

    req.override = _override

    mw.wrap_model_call(req, handler)
    assert len(seen) == 1
    sys_text = "\n".join(
        block["text"]
        for block in seen[0].system_message.content_blocks
        if block["type"] == "text"
    )
    assert "`eval` tool" in sys_text
    assert "7.0s per call" in sys_text
    assert "32 MB total" in sys_text
    assert "across multiple turns for this conversation thread" in sys_text
    assert "### Dispatching Subagents with `task`" not in sys_text


def test_system_prompt_includes_subagent_guidance_when_specs_configured() -> None:
    mw = CodeInterpreterMiddleware()
    seen: list[ModelRequest] = []

    def handler(req: ModelRequest):
        seen.append(req)
        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage

        return ModelResponse(result=[AIMessage(content="ok")])

    req = MagicMock(spec=ModelRequest)
    req.system_message = SystemMessage(content="base")
    req.tools = [
        _task_tool_for_runnable(
            RunnableLambda(
                lambda _state, _config: {"messages": [AIMessage(content="ok")]}
            )
        )
    ]

    def _override(**kwargs):
        new = MagicMock(spec=ModelRequest)
        new.system_message = kwargs.get("system_message", req.system_message)
        return new

    req.override = _override

    mw.wrap_model_call(req, handler)

    assert len(seen) == 1
    sys_text = "\n".join(
        block["text"]
        for block in seen[0].system_message.content_blocks
        if block["type"] == "text"
    )
    assert "### Dispatching Subagents with `task`" in sys_text
    assert "await task({" in sys_text


def test_system_prompt_omits_subagent_guidance_when_disabled() -> None:
    mw = CodeInterpreterMiddleware(subagents=False)
    seen: list[ModelRequest] = []

    def handler(req: ModelRequest):
        seen.append(req)
        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage

        return ModelResponse(result=[AIMessage(content="ok")])

    req = MagicMock(spec=ModelRequest)
    req.system_message = SystemMessage(content="base")
    req.tools = [
        _task_tool_for_runnable(
            RunnableLambda(
                lambda _state, _config: {"messages": [AIMessage(content="ok")]}
            )
        )
    ]

    def _override(**kwargs):
        new = MagicMock(spec=ModelRequest)
        new.system_message = kwargs.get("system_message", req.system_message)
        return new

    req.override = _override

    mw.wrap_model_call(req, handler)

    assert len(seen) == 1
    sys_text = "\n".join(
        block["text"]
        for block in seen[0].system_message.content_blocks
        if block["type"] == "text"
    )
    assert "### Dispatching Subagents with `task`" not in sys_text
    assert "await task({" not in sys_text


def test_system_prompt_mentions_single_turn_for_mode_turn() -> None:
    mw = CodeInterpreterMiddleware(mode="turn")
    assert "DO NOT persist across multiple turns" in mw._base_prompt(ptc_attached=False)


def test_system_prompt_mentions_mode_call() -> None:
    mw = CodeInterpreterMiddleware(mode="call")
    base_prompt = mw._base_prompt(ptc_attached=False)
    assert "fresh sandboxed REPL for each invocation" in base_prompt
    assert "does not persist across tool calls" in base_prompt


def test_default_mode_is_thread() -> None:
    mw = CodeInterpreterMiddleware()
    assert mw._mode == "thread"


def test_mode_turn_is_stored() -> None:
    mw = CodeInterpreterMiddleware(mode="turn")
    assert mw._mode == "turn"


def test_mode_call_is_stored() -> None:
    mw = CodeInterpreterMiddleware(mode="call")
    assert mw._mode == "call"


def test_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        CodeInterpreterMiddleware(mode="session")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Persistence + isolation
# ---------------------------------------------------------------------------


def test_state_persists_across_evals(repl: _ThreadREPL) -> None:
    first = repl.eval_sync("let x = 40")
    assert first.error_type is None
    second = repl.eval_sync("x + 2")
    assert second.error_type is None
    assert second.result == "42"


def test_threads_are_isolated(worker: ThreadWorker, runtime: Runtime) -> None:
    a = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    b = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    a.eval_sync("let shared = 'from_a'")
    outcome = b.eval_sync("typeof shared")
    # QuickJS returns "undefined" for missing globals — an isolated context
    # must not see A's binding.
    assert outcome.result == "undefined"


def test_registry_reset_repl_clears_state_without_recreating_runtime() -> None:
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        repl = reg.get("thread-a")
        old_runtime = reg._slots["thread-a"].runtime
        repl.eval_sync("globalThis.answer = 42")
        reg.reset_repl("thread-a")
        replaced = reg.get("thread-a")
        outcome = replaced.eval_sync("typeof answer")
        assert replaced is not repl
        assert reg._slots["thread-a"].runtime is old_runtime
    finally:
        reg.close()
    assert outcome.result == "undefined"


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def test_runtime_throw_becomes_error_block(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("throw new TypeError('bad')")
    assert outcome.error_type == "TypeError"
    assert "bad" in outcome.error_message
    formatted = format_outcome(outcome, max_result_chars=1000)
    assert '<error type="TypeError">' in formatted
    assert "bad" in formatted


def test_syntax_error_surfaces(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("1 +")
    assert outcome.error_type == "SyntaxError"


def test_timeout(worker: ThreadWorker, runtime: Runtime) -> None:
    tight = _ThreadREPL(
        worker,
        runtime,
        timeout=0.1,
        capture_console=True,
        max_stdout_chars=4000,
    )
    outcome = tight.eval_sync("while(true){}")
    assert outcome.error_type == "Timeout"


# ---------------------------------------------------------------------------
# Console capture
# ---------------------------------------------------------------------------


def test_console_log_is_captured(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("console.log('hi', 2); 1 + 1")
    assert outcome.result == "2"
    assert "hi 2" in outcome.stdout
    formatted = format_outcome(outcome, max_result_chars=1000)
    assert "<stdout>" in formatted
    assert "hi 2" in formatted
    assert "<result>2</result>" in formatted


def test_console_can_be_disabled(worker: ThreadWorker, runtime: Runtime) -> None:
    quiet = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=False,
        max_stdout_chars=4000,
    )
    outcome = quiet.eval_sync("typeof console")
    # With the bridge off, the global is absent.
    assert outcome.result == "undefined"


def test_console_capture_is_bounded_at_append_time(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    bounded = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=64,
    )
    outcome = bounded.eval_sync(
        "console.log('x'.repeat(80)); console.log('y'.repeat(80)); 1"
    )
    assert outcome.result == "1"
    assert len(outcome.stdout) <= 64
    assert outcome.stdout_truncated_chars > 0
    formatted = format_outcome(outcome, max_result_chars=64)
    assert "truncated" in formatted


def test_console_overflow_preserves_prefix(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    bounded = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=10,
    )
    outcome = bounded.eval_sync("console.log('abcdef'); console.log('ghij');")
    assert outcome.stdout == "abcdef\nghi"
    assert outcome.stdout_truncated_chars == 1


def test_console_truncation_state_resets_between_evals(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    bounded = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=10,
    )
    first = bounded.eval_sync("console.log('abcdef'); console.log('ghij');")
    assert first.stdout_truncated_chars == 1
    second = bounded.eval_sync("console.log('ok'); 2")
    assert second.result == "2"
    assert second.stdout == "ok"
    assert second.stdout_truncated_chars == 0


def test_console_truncation_marker_emits_with_zero_budget(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    bounded = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=0,
    )
    outcome = bounded.eval_sync("console.log('hello'); 1")
    assert outcome.stdout == ""
    assert outcome.stdout_truncated_chars > 0
    formatted = format_outcome(outcome, max_result_chars=60)
    assert "<stdout>" in formatted
    assert "truncated" in formatted


# ---------------------------------------------------------------------------
# Function return (MarshalError fallback)
# ---------------------------------------------------------------------------


def test_function_return_falls_back_to_handle_description(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("(a, b) => a + b")
    assert outcome.error_type is None
    assert outcome.result_kind == "handle"
    assert "Function" in (outcome.result or "")
    assert "arity=2" in (outcome.result or "")
    formatted = format_outcome(outcome, max_result_chars=1000)
    assert '<result kind="handle">' in formatted


# ---------------------------------------------------------------------------
# Final-expression Promise unwrapping (issue #3424)
# ---------------------------------------------------------------------------


def test_async_iife_returning_promise_is_unwrapped(repl: _ThreadREPL) -> None:
    """Issue #3424: a final expression that is a Promise (e.g. a bare async
    IIFE) must surface its resolved value instead of the Promise object.
    """
    outcome = repl.eval_sync(
        "(async () => { const v = await Promise.resolve(456); return v; })();"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result_kind is None
    assert outcome.result == "456"


def test_top_level_promise_expression_is_unwrapped(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("Promise.resolve(7)")
    assert outcome.error_type is None
    assert outcome.result_kind is None
    assert outcome.result == "7"


def test_top_level_await_marshals_resolved_value(repl: _ThreadREPL) -> None:
    """A `const x = await ...; x;` script must still marshal the resolved
    value, not the Promise — the existing top-level-await path is unaffected
    by the new handle-based eval flow.
    """
    outcome = repl.eval_sync("const v1 = await Promise.resolve(123); v1;")
    assert outcome.error_type is None
    assert outcome.result_kind is None
    assert outcome.result == "123"


def test_rejected_promise_surfaces_as_jserror(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync("(async () => { throw new Error('iife-rejection'); })();")
    assert outcome.result is None
    assert outcome.error_type == "Error"
    assert "iife-rejection" in (outcome.error_message or "")


def test_unwrapping_does_not_double_user_side_effects(repl: _ThreadREPL) -> None:
    """The user program (and its console.log side effects) must run exactly
    once even when the final expression is a Promise that needs unwrapping.
    """
    outcome = repl.eval_sync("(async () => { console.log('hit'); return 1; })();")
    assert outcome.error_type is None
    assert outcome.result == "1"
    assert outcome.stdout.count("hit") == 1


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_large_result_is_truncated(repl: _ThreadREPL) -> None:
    outcome = repl.eval_sync('"x".repeat(5000)')
    formatted = format_outcome(outcome, max_result_chars=100)
    assert "truncated" in formatted
    # Bound ourselves a bit: tags add overhead, but body should be close to the limit.
    assert len(formatted) < 300


# ---------------------------------------------------------------------------
# Registry / cleanup
# ---------------------------------------------------------------------------


def test_registry_reuses_thread_repl() -> None:
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        r1 = reg.get("thread-a")
        r2 = reg.get("thread-a")
        r3 = reg.get("thread-b")
        assert r1 is r2
        assert r1 is not r3
    finally:
        reg.close()


def test_registry_get_if_exists_does_not_create_slot() -> None:
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        assert reg.get_if_exists("missing") is None
        assert reg._slots == {}
        created = reg.get("thread-a")
        assert reg.get_if_exists("thread-a") is created
    finally:
        reg.close()


def test_middleware_del_closes_runtime() -> None:
    mw = CodeInterpreterMiddleware()
    # Force a slot to exist
    _ = mw._registry.get("t")
    slots = list(mw._registry._slots.values())
    assert len(slots) == 1
    rt = slots[0].runtime
    with patch.object(rt, "close", wraps=rt.close) as close_spy:
        mw.__del__()
        assert close_spy.called


def test_clear_exception_references_removes_traceback_links() -> None:
    """Clears traceback/context/cause to avoid cross-thread GC cycles."""
    outer_msg = "outer"
    first_msg = "first"
    second_msg = "second"

    def _raise_outer() -> None:
        raise ValueError(outer_msg)

    def _raise_with_links() -> None:
        try:
            _raise_outer()
        except ValueError:
            raise RuntimeError(first_msg) from RuntimeError(second_msg)

    with pytest.raises(RuntimeError) as exc_info:
        _raise_with_links()
    caught = exc_info.value
    assert caught.__traceback__ is not None
    assert caught.__context__ is not None
    assert caught.__cause__ is not None
    _clear_exception_references(caught)
    assert caught.__traceback__ is None
    assert caught.__context__ is None
    assert caught.__cause__ is None


def test_per_thread_slot_has_own_worker_and_runtime() -> None:
    """Each thread_id gets its own ThreadWorker and Runtime — not shared."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        reg.get("thread-a")
        reg.get("thread-b")
        slot_a = reg._slots["thread-a"]
        slot_b = reg._slots["thread-b"]
        assert slot_a.worker is not slot_b.worker
        assert slot_a.runtime is not slot_b.runtime
        assert slot_a.worker._name == "quickjs-worker-thread-a"
        assert slot_b.worker._name == "quickjs-worker-thread-b"
    finally:
        reg.close()


def test_evict_closes_and_removes_slot() -> None:
    """`evict` closes the runtime and drops the slot from the registry."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        reg.get("thread-a")
        rt = reg._slots["thread-a"].runtime
        repl = reg._slots["thread-a"].repl
        with (
            patch.object(repl, "close", wraps=repl.close) as repl_close_spy,
            patch.object(rt, "close", wraps=rt.close) as close_spy,
        ):
            reg.evict("thread-a")
        assert repl_close_spy.called
        assert close_spy.called
        assert "thread-a" not in reg._slots
    finally:
        reg.close()


def test_evict_returns_fresh_slot_on_next_get() -> None:
    """After eviction, `get` rebuilds a new slot for the same thread_id."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        first = reg.get("thread-a")
        first_runtime = reg._slots["thread-a"].runtime
        reg.evict("thread-a")
        second = reg.get("thread-a")
        assert second is not first
        assert reg._slots["thread-a"].runtime is not first_runtime
    finally:
        reg.close()


def test_evict_unknown_thread_id_is_noop() -> None:
    """Evicting a thread_id that was never registered does not raise."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        reg.evict("never-existed")
        assert reg._slots == {}
    finally:
        reg.close()


async def test_aevict_closes_and_removes_slot() -> None:
    """`aevict` closes the runtime via the worker loop and drops the slot."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        reg.get("thread-a")
        rt = reg._slots["thread-a"].runtime
        repl = reg._slots["thread-a"].repl
        with (
            patch.object(repl, "aclose", wraps=repl.aclose) as repl_close_spy,
            patch.object(rt, "close", wraps=rt.close) as close_spy,
        ):
            await reg.aevict("thread-a")
        assert repl_close_spy.called
        assert close_spy.called
        assert "thread-a" not in reg._slots
    finally:
        reg.close()


def test_after_agent_evicts_current_thread_slot() -> None:
    """`after_agent` snapshots state and evicts the resolved thread slot."""
    mw = CodeInterpreterMiddleware()
    try:
        # Force a slot to exist for the middleware's fallback thread id.
        repl = mw._registry.get(mw._fallback_thread_id)
        repl.eval_sync("globalThis.counter = 10")
        assert mw._fallback_thread_id in mw._registry._slots
        update = mw.after_agent(state={}, runtime=MagicMock())
        assert isinstance(update, dict)
        # First write (no prior snapshot) is a full anchor record.
        kind, blob = update["_quickjs_snapshot_payload"]
        assert kind == "snap"
        assert isinstance(blob, bytes)
        assert mw._fallback_thread_id not in mw._registry._slots
    finally:
        mw._registry.close()


async def test_aafter_agent_evicts_current_thread_slot() -> None:
    """`aafter_agent` snapshots state and evicts the resolved thread slot."""
    mw = CodeInterpreterMiddleware()
    try:
        repl = mw._registry.get(mw._fallback_thread_id)
        repl.eval_sync("globalThis.counter = 10")
        assert mw._fallback_thread_id in mw._registry._slots
        update = await mw.aafter_agent(state={}, runtime=MagicMock())
        assert isinstance(update, dict)
        kind, blob = update["_quickjs_snapshot_payload"]
        assert kind == "snap"
        assert isinstance(blob, bytes)
        assert mw._fallback_thread_id not in mw._registry._slots
    finally:
        mw._registry.close()


async def test_mode_call_resets_state_between_tool_calls() -> None:
    mw = CodeInterpreterMiddleware(mode="call")
    try:
        tool = mw.tools[0]
        runtime = ToolRuntime(
            state={},
            context={},
            config={},
            stream_writer=lambda _chunk: None,
            tools=[tool],
            tool_call_id="outer_eval_call",
            store=None,
        )
        assert tool.coroutine is not None
        first_repl = mw._registry.get(mw._fallback_thread_id)
        first_runtime = mw._registry._slots[mw._fallback_thread_id].runtime
        first = await tool.coroutine(
            runtime=runtime,
            code="globalThis.answer = 42; answer",
        )
        assert "<result>42</result>" in first.content
        after_first = mw._registry.get_if_exists(mw._fallback_thread_id)
        assert after_first is not None
        assert after_first is not first_repl
        assert mw._registry._slots[mw._fallback_thread_id].runtime is first_runtime

        second = await tool.coroutine(runtime=runtime, code="typeof answer")
        assert "<result>undefined</result>" in second.content
        after_second = mw._registry.get_if_exists(mw._fallback_thread_id)
        assert after_second is not None
        assert after_second is not after_first
        assert mw._registry._slots[mw._fallback_thread_id].runtime is first_runtime
    finally:
        mw._registry.close()


# ---------------------------------------------------------------------------
# Async path (v0.2 native `eval_async`)
# ---------------------------------------------------------------------------


async def test_async_eval_basic(repl: _ThreadREPL) -> None:
    outcome = await repl.eval_async("1 + 1")
    assert outcome.error_type is None
    assert outcome.result == "2"


async def test_async_state_persists(repl: _ThreadREPL) -> None:
    """v0.2 module-with-async mode keeps realm-level bindings across calls."""
    first = await repl.eval_async("globalThis.counter = 10")
    assert first.error_type is None
    second = await repl.eval_async("counter + 5")
    assert second.result == "15"


async def test_async_top_level_await(repl: _ThreadREPL) -> None:
    """The feature this whole upgrade is about — awaiting a Promise works."""
    outcome = await repl.eval_async("await new Promise(resolve => resolve(42))")
    assert outcome.error_type is None
    assert outcome.result == "42"


def _task_tool_for_runnable(
    runnable: Any,
) -> BaseTool:
    spec: dict[str, Any] = {
        "name": "worker",
        "description": "Does work.",
        "runnable": runnable,
    }
    return _build_task_tool([spec])


def _subagent_runtime_from_task_tool(task_tool: BaseTool) -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={},
        config={"configurable": {}},
        stream_writer=lambda _chunk: None,
        tools=[task_tool],
        tool_call_id="outer_eval_call",
        store=None,
    )


def _subagent_runtime(
    runnable: Any,
) -> ToolRuntime:
    return _subagent_runtime_from_task_tool(_task_tool_for_runnable(runnable))


async def test_call_subagent_task_tool_forwards_config_and_tool_call_id() -> None:
    calls: list[dict[str, Any]] = []

    class _TaskTool:
        name = "task"

        async def arun(
            self,
            tool_input: dict[str, Any],
            *,
            config: dict[str, Any] | None = None,
            tool_call_id: str | None = None,
        ) -> str:
            calls.append(
                {
                    "tool_input": tool_input,
                    "config": config,
                    "tool_call_id": tool_call_id,
                }
            )
            return "ok"

    runtime = ToolRuntime(
        state={},
        context={},
        config={"configurable": {"thread_id": "parent-thread"}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="outer_eval_call",
        store=None,
    )

    result = await call_subagent_task_tool(
        cast("BaseTool", _TaskTool()),
        description="work",
        subagent_type="worker",
        response_schema=None,
        runtime=runtime,
    )

    assert result == "ok"
    assert calls
    assert calls[0]["config"] == runtime.config
    assert calls[0]["tool_call_id"].startswith("ptc_task_")
    assert calls[0]["tool_input"]["runtime"].tool_call_id == calls[0]["tool_call_id"]


async def test_async_task_global_propagates_graph_interrupt(repl: _ThreadREPL) -> None:
    interrupt = GraphInterrupt([Interrupt(value={"action_requests": []})])

    async def _async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        raise interrupt

    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
        afunc=_async,
    )

    with pytest.raises(GraphInterrupt) as exc_info:
        await repl.eval_async(
            "await task({description: 'work', subagentType: 'worker'})",
            outer_runtime=_subagent_runtime(runnable),
        )

    assert exc_info.value is interrupt


def test_runtime_with_response_format_uses_configurable() -> None:
    runtime = _subagent_runtime(
        RunnableLambda(lambda _state, _config: {"messages": [AIMessage(content="ok")]})
    )
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    updated = _runtime_with_response_format(runtime, schema)

    assert updated.context is runtime.context
    assert SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY not in runtime.config["configurable"]
    strategy = updated.config["configurable"][SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY]
    assert isinstance(strategy, AutoStrategy)
    assert strategy.schema == schema


def test_ensure_schema_title_injects_default_when_missing() -> None:
    schema = {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }

    updated = _ensure_schema_title(schema)

    assert updated["title"] == "subagent_response"
    # Original schema is not mutated.
    assert "title" not in schema
    # Other keys are preserved unchanged.
    assert updated["properties"] == schema["properties"]
    assert updated["required"] == schema["required"]


def test_ensure_schema_title_preserves_existing_non_empty_title() -> None:
    schema = {"title": "MyWord", "type": "object"}

    updated = _ensure_schema_title(schema)

    assert updated is schema
    assert updated["title"] == "MyWord"


@pytest.mark.parametrize(
    "title",
    ["", "   ", 0, None],
)
def test_ensure_schema_title_replaces_blank_or_invalid_title(title: Any) -> None:
    schema = {"title": title, "type": "object"}

    updated = _ensure_schema_title(schema)

    assert updated["title"] == "subagent_response"


class _StructuredSubagentModel(GenericFakeChatModel):
    """Fake subagent model that calls the currently bound response-format tool."""

    messages: Any = Field(exclude=True)
    tools: Any = ()

    def bind_tools(
        self,
        tools: Any,
        **_: Any,
    ) -> _StructuredSubagentModel:
        self.tools = tools
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_name = getattr(self.tools[-1], "name", None)
        if not isinstance(tool_name, str):
            msg = "Expected a response-format tool to be bound"
            raise TypeError(msg)
        self.messages = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"label": "positive", "score": 0.95},
                            "id": "call_structured",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def _declarative_subagent_runtime() -> ToolRuntime:
    middleware = SubAgentMiddleware(
        backend=StateBackend(),
        subagents=[
            {
                "name": "worker",
                "description": "Does work.",
                "system_prompt": "Return structured data.",
                "model": _StructuredSubagentModel(messages=iter(())),
                "tools": [],
            }
        ],
        system_prompt=None,
    )
    return _subagent_runtime_from_task_tool(middleware.tools[0])


async def test_eval_tool_does_not_install_subagent_when_disabled() -> None:
    mw = CodeInterpreterMiddleware(subagents=False)
    try:
        tool = mw.tools[0]
        assert tool.coroutine is not None
        runtime = _subagent_runtime(
            RunnableLambda(
                lambda _state, _config: {"messages": [AIMessage(content="ok")]}
            )
        )

        result = await tool.coroutine(runtime=runtime, code="typeof task")

        assert "<result>undefined</result>" in result.content
    finally:
        mw._registry.close()


async def test_async_task_global_invokes_runner(repl: _ThreadREPL) -> None:
    calls: list[dict[str, Any]] = []

    def _sync(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="sync")]}

    async def _async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        calls.append({"state": state, "config": config})
        return {"messages": [AIMessage(content="ok")]}

    runnable = RunnableLambda(_sync, afunc=_async)

    outcome = await repl.eval_async(
        # The `task` global is bound writable:false / configurable:false, so under
        # strict-mode eval every tamper attempt must THROW (a stronger guarantee
        # than the old sloppy-mode silent no-op). Each attempt is required to
        # raise, and `task` must survive intact and remain callable afterwards.
        "const mustThrow = (label, fn) => {"
        "  try { fn(); } catch (e) { return; }"
        "  throw new Error('task binding is mutable: ' + label);"
        "};"
        "mustThrow('assign', () => { globalThis.task = null; });"
        "mustThrow('delete', () => { delete globalThis.task; });"
        "mustThrow('addProp', () => { task.extra = 1; });"
        "if (!Object.isFrozen(task) || task.extra !== undefined) {"
        "  throw new Error('task binding is mutable');"
        "}"
        "JSON.stringify(await task({"
        "description: 'work', "
        "subagentType: 'worker'"
        "}))",
        outer_runtime=_subagent_runtime(runnable),
    )

    assert outcome.error_type is None
    assert outcome.result == '"ok"'
    assert calls
    assert calls[0]["state"]["messages"][0].content == "work"
    assert calls[0]["config"]["configurable"]["ls_agent_type"] == "subagent"


async def test_async_task_global_not_installed_when_disabled(
    worker: ThreadWorker,
    runtime: Runtime,
) -> None:
    disabled = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        subagents_enabled=False,
    )

    outcome = await disabled.eval_async("typeof task")

    assert outcome.error_type is None
    assert outcome.result == "undefined"


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            "await task({subagentType: 'worker', label: 'lbl'})",
            "task() requires non-empty string field `description`",
        ),
        (
            "await task({description: 'work', label: 'lbl'})",
            "task() requires non-empty string field `subagentType`",
        ),
        (
            "await task({description: 'work', subagentType: 'worker', label: 123})",
            "task() field `label` must be a string when provided",
        ),
        (
            "await task({"
            "description: 'work', "
            "subagentType: 'worker', "
            "label: 'lbl', "
            "responseSchema: 'bad'"
            "})",
            "task() field `responseSchema` must be an object when provided",
        ),
    ],
)
async def test_async_task_global_validation_errors_surface_as_eval_errors(
    repl: _ThreadREPL, code: str, message: str
) -> None:
    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
    )

    outcome = await repl.eval_async(code, outer_runtime=_subagent_runtime(runnable))

    assert outcome.error_type == "ValueError"
    assert message in outcome.error_message
    formatted = format_outcome(outcome, max_result_chars=1000)
    assert '<error type="ValueError">' in formatted
    assert message in formatted


async def test_async_task_global_missing_task_tool_surfaces_as_eval_error(
    repl: _ThreadREPL,
) -> None:
    runtime = ToolRuntime(
        state={},
        context={},
        config={"configurable": {}},
        stream_writer=lambda _chunk: None,
        tools=[],
        tool_call_id="outer_eval_call",
        store=None,
    )

    outcome = await repl.eval_async(
        "await task({description: 'work', subagentType: 'worker', label: 'lbl'})",
        outer_runtime=runtime,
    )

    assert outcome.error_type == "RuntimeError"
    assert "task tool not configured for this eval" in outcome.error_message


async def test_async_task_global_returns_declarative_structured_response_object(
    repl: _ThreadREPL,
) -> None:
    outcome = await repl.eval_async(
        "const schema = {"
        "type: 'object', "
        "properties: {label: {type: 'string'}, score: {type: 'number'}}, "
        "required: ['label', 'score']"
        "};"
        "const first = await task({"
        "description: 'work', "
        "subagentType: 'worker', "
        "label: 'first', "
        "responseSchema: schema"
        "});"
        "const second = await task({"
        "description: 'work again', "
        "subagentType: 'worker', "
        "label: 'second', "
        "responseSchema: schema"
        "});"
        "JSON.stringify({"
        "label: first.label, "
        "score: first.score, "
        "type: typeof first, "
        "secondLabel: second.label"
        "})",
        outer_runtime=_declarative_subagent_runtime(),
    )

    assert outcome.error_type is None
    assert outcome.result == (
        '{"label":"positive","score":0.95,"type":"object","secondLabel":"positive"}'
    )


async def test_async_task_global_rejects_compiled_response_schema(
    repl: _ThreadREPL,
) -> None:
    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
    )

    outcome = await repl.eval_async(
        "await task({"
        "description: 'work', "
        "subagentType: 'worker', "
        "label: 'lbl', "
        "responseSchema: {"
        "type: 'object', "
        "properties: {ok: {type: 'boolean'}}, "
        "required: ['ok']"
        "}"
        "})",
        outer_runtime=_subagent_runtime(runnable),
    )

    assert outcome.error_type == "ValueError"
    assert (
        'response_schema cannot be used with compiled subagent "worker"'
        in outcome.error_message
    )


async def test_async_task_global_uses_last_non_empty_ai_message(
    repl: _ThreadREPL,
) -> None:
    async def _async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="final"), AIMessage(content="")]}

    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
        afunc=_async,
    )

    outcome = await repl.eval_async(
        "JSON.stringify(await task({"
        "description: 'work', subagentType: 'worker', label: 'lbl'"
        "}))",
        outer_runtime=_subagent_runtime(runnable),
    )

    assert outcome.error_type is None
    assert outcome.result == '"final"'


async def test_async_task_global_rejects_state_without_messages(
    repl: _ThreadREPL,
) -> None:
    async def _async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"structured_response": {"ok": True}}

    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
        afunc=_async,
    )

    outcome = await repl.eval_async(
        "await task({description: 'work', subagentType: 'worker', label: 'lbl'})",
        outer_runtime=_subagent_runtime(runnable),
    )

    assert outcome.error_type == "ValueError"
    assert "messages" in outcome.error_message


async def test_async_task_global_limits_concurrency_per_repl(
    repl: _ThreadREPL,
) -> None:
    class _CountingRunner:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        async def ainvoke(
            self,
            *,
            description: str,
            subagent_type: str,
            response_schema: dict[str, Any] | None = None,
            runtime: ToolRuntime | None = None,
        ) -> str:
            del subagent_type, response_schema, runtime
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            with self.lock:
                self.active -= 1
            return description

    runner = _CountingRunner()

    async def _counting_async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del config
        content = await runner.ainvoke(
            description=state["messages"][0].content,
            subagent_type="worker",
        )
        return {"messages": [AIMessage(content=content)]}

    runnable = RunnableLambda(
        lambda state, config: {"messages": [AIMessage(content="sync")]},
        afunc=_counting_async,
    )

    outcome = await repl.eval_async(
        "const calls = [];"
        "for (let i = 0; i < 64; i++) {"
        "  calls.push(task({"
        "description: String(i), subagentType: 'worker', label: 'c' + i"
        "}));"
        "}"
        "(await Promise.all(calls)).length",
        outer_runtime=_subagent_runtime(runnable),
    )

    assert outcome.error_type is None
    assert outcome.result == "64"
    assert runner.max_active <= 32
    assert runner.max_active > 1


async def test_async_promise_chain(repl: _ThreadREPL) -> None:
    outcome = await repl.eval_async(
        "await Promise.resolve(1).then(x => x + 2).then(x => x * 10)"
    )
    assert outcome.result == "30"


async def test_async_rejected_promise_surfaces_as_error(repl: _ThreadREPL) -> None:
    outcome = await repl.eval_async("await Promise.reject(new TypeError('nope'))")
    assert outcome.error_type == "TypeError"
    assert "nope" in outcome.error_message


async def test_async_sync_code_still_works(repl: _ThreadREPL) -> None:
    """Pure-sync code runs fine on the async path — no await needed."""
    outcome = await repl.eval_async("[1, 2, 3].map(x => x * 2)")
    assert outcome.result == "[2, 4, 6]"


async def test_async_deadlock_detection(repl: _ThreadREPL) -> None:
    """A top-level Promise with no resolver surfaces as a Deadlock error."""
    outcome = await repl.eval_async("await new Promise(() => {})")
    assert outcome.error_type == "Deadlock"


async def test_async_concurrent_calls_surface_error(repl: _ThreadREPL) -> None:
    """Overlapping async evals on the same context surface as
    `ConcurrentEvalError` rather than silently serialising.

    A model issuing overlapping evals against shared state is almost
    always a prompting bug; a loud failure is a better signal than
    silent queueing. The slow tool forces a yield so the two evals
    actually overlap (pure sync code takes the non-promise fast path).
    """

    class _NoArgs(BaseModel):
        pass

    async def _slow(**_: Any) -> str:
        await asyncio.sleep(0.05)
        return "ok"

    slow_tool = StructuredTool.from_function(
        name="slow",
        description="Sleeps briefly.",
        coroutine=_slow,
        args_schema=_NoArgs,
    )
    repl.install_tools([slow_tool])

    a, b = await asyncio.gather(
        repl.eval_async("await tools.slow({})"),
        repl.eval_async("await tools.slow({})"),
    )
    assert "ConcurrentEval" in {a.error_type, b.error_type}, (a, b)


def test_sync_path_still_works(repl: _ThreadREPL) -> None:
    """After the v0.2 split, the sync path continues to use `ctx.eval`."""
    repl.eval_sync("let n = 7")
    assert repl.eval_sync("n * 6").result == "42"
