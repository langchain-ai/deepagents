"""Unit tests for REPLMiddleware and its backing REPL wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage
from quickjs_rs import Runtime

from deepagents_repl import REPLMiddleware
from deepagents_repl._repl import _Registry, _ThreadREPL, format_outcome

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> Runtime:
    """A fresh QuickJS Runtime for tests that drive _ThreadREPL directly."""
    rt = Runtime()
    try:
        yield rt
    finally:
        rt.close()


@pytest.fixture
def repl(runtime: Runtime) -> _ThreadREPL:
    return _ThreadREPL(runtime, timeout=5.0, capture_console=True)


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
    assert "TypeScript/JavaScript REPL (`eval`)" in sys_text
    assert "Execution timeout per call: 7 s." in sys_text
    # No-swarm variant of the large-file rule appears when swarm is off.
    assert "Check file size before processing" in sys_text
    # The swarm-variant directive should NOT leak into non-swarm REPLs.
    assert "Check inputs before processing" not in sys_text


def test_system_prompt_large_file_rule_points_to_swarm_when_configured() -> None:
    """When swarm is configured, the large-file hard rule directs to
    swarm.create/swarm.execute instead of generic chunking advice."""
    from deepagents.backends import StateBackend
    from deepagents.middleware.subagents import CompiledSubAgent
    from langchain_core.runnables import RunnableLambda

    # Pre-compiled graph — avoids needing a real model in this unit test.
    compiled: CompiledSubAgent = {
        "name": "general-purpose",
        "description": "d",
        "runnable": RunnableLambda(lambda x: x),
    }
    mw = REPLMiddleware(backend=StateBackend(), subagents=[compiled])
    prompt = mw._base_system_prompt
    assert "Check inputs before processing" in prompt
    assert "swarm.create`/`swarm.execute" in prompt
    # The no-swarm variant must not appear in this configuration.
    assert "Check file size before processing" not in prompt


# ---------------------------------------------------------------------------
# Persistence + isolation
# ---------------------------------------------------------------------------


def test_state_persists_across_evals(repl: _ThreadREPL) -> None:
    first = repl.eval_sync("let x = 40")
    assert first.error_type is None
    second = repl.eval_sync("x + 2")
    assert second.error_type is None
    assert second.result == "42"


def test_threads_are_isolated(runtime: Runtime) -> None:
    a = _ThreadREPL(runtime, timeout=5.0, capture_console=True)
    b = _ThreadREPL(runtime, timeout=5.0, capture_console=True)
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


def test_timeout(runtime: Runtime) -> None:
    tight = _ThreadREPL(runtime, timeout=0.1, capture_console=True)
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


def test_console_can_be_disabled(runtime: Runtime) -> None:
    quiet = _ThreadREPL(runtime, timeout=5.0, capture_console=False)
    outcome = quiet.eval_sync("typeof console")
    # With the bridge off, the global is absent.
    assert outcome.result == "undefined"


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
    reg = _Registry(memory_limit=32 * 1024 * 1024, timeout=5.0, capture_console=True)
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
    # Force Runtime creation
    _ = mw._registry.get("t")
    rt = mw._registry._runtime
    assert rt is not None
    with patch.object(rt, "close", wraps=rt.close) as close_spy:
        mw.__del__()
        assert close_spy.called


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
    outcome = await repl.eval_async(
        "await new Promise(resolve => resolve(42))"
    )
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


async def test_async_concurrent_calls_serialize(repl: _ThreadREPL) -> None:
    """Two concurrent eval_async on the same thread should queue, not raise.

    If the internal lock is removed this test will flakily raise
    ``ConcurrentEvalError`` rather than returning two clean outcomes.
    """
    import asyncio as _asyncio

    a, b = await _asyncio.gather(
        repl.eval_async("1 + 1"),
        repl.eval_async("2 + 2"),
    )
    assert a.result == "2"
    assert b.result == "4"
    assert a.error_type is None
    assert b.error_type is None


def test_sync_path_still_works(repl: _ThreadREPL) -> None:
    """After the v0.2 split, the sync path continues to use ``ctx.eval``."""
    repl.eval_sync("let n = 7")
    assert repl.eval_sync("n * 6").result == "42"
