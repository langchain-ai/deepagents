"""Tests for programmatic tool calling (PTC).

PTC exposes agent tools as ``tools.<camelCase>`` async functions inside
the REPL so one ``eval`` can orchestrate many tool invocations.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import REPLMiddleware
from langchain_quickjs._ptc import (
    filter_tools_for_ptc,
    render_ptc_prompt,
    to_camel_case,
)
from langchain_quickjs._repl import _ThreadREPL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GreetInput(BaseModel):
    name: str = Field(description="Who to greet")
    times: int = Field(default=1, description="Repeat count")


def _greet_tool(record: list[dict] | None = None) -> BaseTool:
    """A synchronous tool that records its invocations.

    Async-capable by default because ``StructuredTool.from_function``
    synthesises a coroutine wrapper when only ``func=`` is passed.
    """
    calls = record if record is not None else []

    def _fn(name: str, times: int = 1) -> str:
        calls.append({"name": name, "times": times})
        return f"hi {name} x{times}"

    return StructuredTool.from_function(
        name="greet",
        description="Greet a person.",
        func=_fn,
        args_schema=_GreetInput,
    )


def _echo_tool(name: str = "echo") -> BaseTool:
    """A minimal tool that echoes its input."""

    class _In(BaseModel):
        msg: str = Field(description="Message to echo back")

    def _fn(msg: str) -> str:
        return msg

    return StructuredTool.from_function(
        name=name,
        description=f"Echo back its input ({name}).",
        func=_fn,
        args_schema=_In,
    )


@pytest.fixture
def worker() -> ThreadWorker:
    w = ThreadWorker()
    try:
        yield w
    finally:
        w.close()


@pytest.fixture
def runtime(worker: ThreadWorker) -> Runtime:
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
    return _ThreadREPL(worker, runtime, timeout=5.0, capture_console=True)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_filter_rejects_boolean_config() -> None:
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        filter_tools_for_ptc([_greet_tool()], True, self_tool_name="eval")
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        filter_tools_for_ptc([_greet_tool()], False, self_tool_name="eval")


def test_filter_rejects_dict_config() -> None:
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        filter_tools_for_ptc(  # type: ignore[arg-type]
            [_greet_tool()],
            {"include": ["greet"]},
            self_tool_name="eval",
        )


def test_filter_list_include_empty_returns_empty() -> None:
    assert filter_tools_for_ptc([_greet_tool()], [], self_tool_name="eval") == []


def test_filter_list_include_excludes_self_tool() -> None:
    greet = _greet_tool()
    eval_tool = _echo_tool("eval")  # same name as the REPL tool
    out = filter_tools_for_ptc([greet, eval_tool], ["greet"], self_tool_name="eval")
    names = [t.name for t in out]
    assert names == ["greet"]


def test_filter_list_include() -> None:
    a, b, c = _echo_tool("a"), _echo_tool("b"), _echo_tool("c")
    out = filter_tools_for_ptc([a, b, c], ["a", "c"], self_tool_name="eval")
    assert [t.name for t in out] == ["a", "c"]


def test_filter_list_of_tools_uses_them_directly() -> None:
    """`list[BaseTool]` ignores agent tools and uses the supplied list."""
    greet = _greet_tool()
    out = filter_tools_for_ptc([], [greet], self_tool_name="eval")
    assert out == [greet]


def test_filter_list_of_tools_excludes_self_tool() -> None:
    greet = _greet_tool()
    eval_tool = _echo_tool("eval")
    out = filter_tools_for_ptc([], [greet, eval_tool], self_tool_name="eval")
    assert [t.name for t in out] == ["greet"]


def test_filter_accepts_mixed_str_and_tool_list() -> None:
    agent_echo = _echo_tool("echo")
    direct_greet = _greet_tool()
    out = filter_tools_for_ptc(
        [agent_echo],
        [direct_greet, "echo"],
        self_tool_name="eval",
    )
    assert [t.name for t in out] == ["greet", "echo"]


def test_filter_rejects_invalid_mixed_entry_type() -> None:
    with pytest.raises(TypeError, match="ptc list entries must be str or BaseTool"):
        filter_tools_for_ptc(  # type: ignore[list-item]
            [_echo_tool()],
            ["echo", 42],
            self_tool_name="eval",
        )


def test_filter_rejects_invalid_js_identifiers() -> None:
    valid = _echo_tool("good_tool")
    invalid = _echo_tool("123bad")
    with pytest.raises(ValueError, match="cannot be exposed as JavaScript identifier"):
        filter_tools_for_ptc(
            [valid, invalid], ["good_tool", "123bad"], self_tool_name="eval"
        )


# ---------------------------------------------------------------------------
# Camel case + prompt rendering
# ---------------------------------------------------------------------------


def test_camel_case() -> None:
    assert to_camel_case("http_request") == "httpRequest"
    assert to_camel_case("tool-name") == "toolName"
    assert to_camel_case("alreadyCamel") == "alreadyCamel"


def test_render_ptc_prompt_empty() -> None:
    assert render_ptc_prompt([]) == ""


def test_render_ptc_prompt_uses_signatures() -> None:
    prompt = render_ptc_prompt([_greet_tool()])
    assert "`tools` namespace" in prompt
    assert "async tools.greet(input:" in prompt
    # Fields come through
    assert "name: string" in prompt
    assert "times?: number" in prompt
    # Descriptions from Field(description=...) appear on the fields
    assert "Who to greet" in prompt


def test_render_ptc_prompt_rejects_invalid_js_identifiers() -> None:
    with pytest.raises(ValueError, match="cannot be exposed as JavaScript identifier"):
        render_ptc_prompt([_echo_tool("123bad")])


# ---------------------------------------------------------------------------
# In-REPL invocation
# ---------------------------------------------------------------------------


async def test_tool_invocation_from_repl(repl: _ThreadREPL) -> None:
    calls: list[dict] = []
    repl.install_tools([_greet_tool(calls)])
    outcome = await repl.eval_async('await tools.greet({name: "world", times: 2})')
    assert outcome.error_type is None, outcome.error_message
    # tool returned the string "hi world x2"
    assert outcome.result == "hi world x2"
    assert calls == [{"name": "world", "times": 2}]


async def test_promise_all_runs_tools_concurrently(repl: _ThreadREPL) -> None:
    """``Promise.all`` on two tool calls resolves both before returning."""
    calls: list[dict] = []
    repl.install_tools([_greet_tool(calls)])
    outcome = await repl.eval_async(
        "const results = await Promise.all([\n"
        '  tools.greet({name: "a"}),\n'
        '  tools.greet({name: "b"}),\n'
        "]);\n"
        "results.join('|')"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "hi a x1|hi b x1"
    assert {c["name"] for c in calls} == {"a", "b"}


@pytest.mark.xfail
async def test_tool_failure_surfaces_as_js_error(repl: _ThreadREPL) -> None:
    def _boom(**_: object) -> str:
        msg = "tool exploded"
        raise RuntimeError(msg)

    class _In(BaseModel):
        x: int = Field(description="unused")

    tool = StructuredTool.from_function(
        name="boom",
        description="Always fails.",
        func=_boom,
        args_schema=_In,
    )
    repl.install_tools([tool])
    outcome = await repl.eval_async(
        "try { await tools.boom({x: 1}); 'no-throw' } catch (e) { e.message }"
    )
    # Host errors surface into JS as a catchable error. The exact class
    # name is implementation detail; we only care that the message is
    # reachable.
    assert outcome.error_type is None, outcome.error_message
    assert any(
        msg in (outcome.result or "")
        for msg in ("tool exploded", "Host function failed")
    )


async def test_install_tools_skips_invalid_js_identifier_names(
    repl: _ThreadREPL,
) -> None:
    repl.install_tools([_echo_tool("123bad")])
    outcome = await repl.eval_async('typeof tools["123bad"]')
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "undefined"


async def test_tools_namespace_assignment_escapes_malicious_tool_name(
    repl: _ThreadREPL,
) -> None:
    from langchain_quickjs._repl import _render_tools_namespace_assignment

    malicious = 'x"]; globalThis.__ptc_pwned = "yes"; //'
    quoted = json.dumps(malicious)
    js = _render_tools_namespace_assignment({malicious: "__console_log"})

    outcome = await repl.eval_async(f"{js}; typeof globalThis.__ptc_pwned")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "undefined"

    outcome2 = await repl.eval_async(f"{js}; typeof globalThis.tools[{quoted}]")
    assert outcome2.error_type is None, outcome2.error_message
    assert outcome2.result == "function"


async def test_install_tools_is_idempotent(repl: _ThreadREPL) -> None:
    """Calling install_tools twice with the same set is a no-op for the guest."""
    calls: list[dict] = []
    tool = _greet_tool(calls)
    repl.install_tools([tool])
    repl.install_tools([tool])
    outcome = await repl.eval_async('await tools.greet({name: "x"})')
    assert outcome.result == "hi x x1"
    assert len(calls) == 1


async def test_install_tools_shrinks_namespace(repl: _ThreadREPL) -> None:
    """Dropping a tool removes it from ``globalThis.tools`` on next install."""
    repl.install_tools([_greet_tool(), _echo_tool("echo")])
    repl.install_tools([_greet_tool()])
    outcome = await repl.eval_async("typeof tools.echo")
    assert outcome.result == "undefined"
    outcome2 = await repl.eval_async("typeof tools.greet")
    assert outcome2.result == "function"


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------


def test_middleware_ptc_default_off_omits_prompt_block() -> None:
    mw = REPLMiddleware()
    # Calling _prepare_for_call directly is fine — pass a minimal request
    # stand-in. We don't need a full ModelRequest for this check.
    from types import SimpleNamespace

    req = SimpleNamespace(tools=[_greet_tool()])
    prompt = mw._prepare_for_call(req)
    assert "`tools` namespace" not in prompt


def test_middleware_ptc_list_includes_prompt_block() -> None:
    from types import SimpleNamespace

    mw = REPLMiddleware(ptc=["greet", "eval"])
    req = SimpleNamespace(tools=[_greet_tool(), _echo_tool("eval")])
    prompt = mw._prepare_for_call(req)
    # Greet included
    assert "async tools.greet(" in prompt
    # The REPL's own tool never appears
    assert "tools.eval(" not in prompt


def test_middleware_ptc_list_of_tools_exposes_without_agent_tools() -> None:
    """`ptc=[tool]` installs the tool in the REPL even when the agent has none."""
    from types import SimpleNamespace

    mw = REPLMiddleware(ptc=[_greet_tool()])
    req = SimpleNamespace(tools=[])
    prompt = mw._prepare_for_call(req)
    assert "async tools.greet(" in prompt


async def test_ptc_install_and_eval_resolve_to_same_repl() -> None:
    """PTC install and the eval tool must see the same REPL instance.

    Regression: without a stable fallback thread id, each call to
    ``_resolve_thread_id`` minted a fresh UUID, so ``wrap_model_call``
    installed tools on one REPL and the eval ran on another — JS saw
    ``ReferenceError: tools is not defined``.
    """
    from types import SimpleNamespace

    mw = REPLMiddleware(ptc=["greet", "eval"])
    # Simulate a model-call turn without any langgraph config present.
    req = SimpleNamespace(tools=[_greet_tool(), _echo_tool("eval")])
    mw._prepare_for_call(req)
    # Now invoke the eval tool directly via the middleware-owned registry.
    # The resolver should return the *same* REPL instance.
    first = mw._registry.get(mw._fallback_thread_id)
    outcome = await first.eval_async("typeof tools.greet")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "function"


def test_middleware_rejects_boolean_ptc_config() -> None:
    with pytest.raises(TypeError, match="Boolean `ptc` is no longer supported"):
        REPLMiddleware(ptc=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Boolean `ptc` is no longer supported"):
        REPLMiddleware(ptc=False)  # type: ignore[arg-type]


def test_middleware_rejects_dict_ptc_config() -> None:
    with pytest.raises(TypeError, match="Dict `ptc` configs are no longer supported"):
        REPLMiddleware(ptc={"include": ["greet"]})  # type: ignore[arg-type]
