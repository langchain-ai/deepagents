"""Unit tests for REPLMiddleware and its backing REPL wrapper."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import REPLMiddleware
from langchain_quickjs._format import format_outcome
from langchain_quickjs._repl import _clear_exception_references, _Registry, _ThreadREPL

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
    mw = REPLMiddleware()
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


def test_tool_registered_with_custom_name() -> None:
    mw = REPLMiddleware(tool_name="js")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        middleware=[mw],
    )
    tools = agent.nodes["tools"].bound._tools_by_name
    assert "js" in tools
    assert "eval" not in tools


def test_legacy_system_prompt_alias_removed() -> None:
    mw = REPLMiddleware()
    assert not hasattr(mw, "system_prompt")


def test_rejects_invalid_max_ptc_calls() -> None:
    with pytest.raises(ValueError, match="must be >= 1 or None"):
        REPLMiddleware(max_ptc_calls=0)


def test_system_prompt_injected_once() -> None:
    """wrap_model_call appends exactly one snippet per call, idempotent in content."""
    mw = REPLMiddleware(timeout=7.0, memory_limit=32 * 1024 * 1024)
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


def test_middleware_del_closes_runtime() -> None:
    mw = REPLMiddleware()
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
    """``evict`` closes the runtime and drops the slot from the registry."""
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
    """After eviction, ``get`` rebuilds a new slot for the same thread_id."""
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
    """``aevict`` closes the runtime via the worker loop and drops the slot."""
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
    """``after_agent`` evicts the slot for the resolved thread_id."""
    mw = REPLMiddleware()
    try:
        # Force a slot to exist for the middleware's fallback thread id.
        mw._registry.get(mw._fallback_thread_id)
        assert mw._fallback_thread_id in mw._registry._slots
        mw.after_agent(state={}, runtime=MagicMock())
        assert mw._fallback_thread_id not in mw._registry._slots
    finally:
        mw._registry.close()


async def test_aafter_agent_evicts_current_thread_slot() -> None:
    """``aafter_agent`` evicts the slot for the resolved thread_id."""
    mw = REPLMiddleware()
    try:
        mw._registry.get(mw._fallback_thread_id)
        assert mw._fallback_thread_id in mw._registry._slots
        await mw.aafter_agent(state={}, runtime=MagicMock())
        assert mw._fallback_thread_id not in mw._registry._slots
    finally:
        mw._registry.close()


# ---------------------------------------------------------------------------
# Async path (v0.2 native ``eval_async``)
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
    ``ConcurrentEvalError`` rather than silently serialising.

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
    """After the v0.2 split, the sync path continues to use ``ctx.eval``."""
    repl.eval_sync("let n = 7")
    assert repl.eval_sync("n * 6").result == "42"
