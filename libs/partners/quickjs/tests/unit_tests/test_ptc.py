"""Tests for programmatic tool calling (PTC).

PTC exposes agent tools as `tools.<camelCase>` async functions inside
the REPL so one `eval` can orchestrate many tool invocations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import (
    BaseTool,
    InjectedToolCallId,
    StructuredTool,
    ToolException,
    tool,
)
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from quickjs_rs import Runtime, ThreadWorker
from typing_extensions import TypedDict

from langchain_quickjs import CodeInterpreterMiddleware
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


class _Status(BaseModel):
    """Module-scope BaseModel used as a return annotation in PTC tests."""

    status: str
    count: int


class _UserLookup(TypedDict):
    """Module-scope TypedDict used as a return annotation in PTC tests."""

    id: int
    name: str


class _TreeNode(BaseModel):
    """A self-referencing model."""

    label: str
    children: list[_TreeNode] = []


class _Cat(BaseModel):
    kind: Literal["cat"]


class _Dog(BaseModel):
    kind: Literal["dog"]


def _greet_tool(record: list[dict] | None = None) -> BaseTool:
    """A synchronous tool that records its invocations.

    Async-capable by default because `StructuredTool.from_function`
    synthesises a coroutine wrapper when only `func=` is passed.
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


def _command_tool(name: str = "emit_command") -> BaseTool:
    """A tool that returns a LangGraph `Command` update."""

    class _In(BaseModel):
        value: int = Field(description="Integer marker to emit")

    def _fn(value: int) -> Command:
        return Command(
            update={
                "ptc_values": [value],
                "messages": [
                    ToolMessage(
                        content=f"value={value}",
                        tool_call_id=f"ptc_call_{value}",
                        name=name,
                    )
                ],
            }
        )

    return StructuredTool.from_function(
        name=name,
        description="Return a Command update carrying the input value.",
        func=_fn,
        args_schema=_In,
    )


def _tool_message_list_tool(name: str = "emit_messages") -> BaseTool:
    """A tool that returns a list of `ToolMessage` values."""

    class _In(BaseModel):
        value: int = Field(description="Integer marker to emit")

    def _fn(value: int) -> list[ToolMessage]:
        return [
            ToolMessage(
                content=f"first={value}",
                tool_call_id=f"ptc_call_{value}_first",
                name=name,
            ),
            ToolMessage(
                content=f"second={value}",
                tool_call_id=f"ptc_call_{value}_second",
                name=name,
            ),
        ]

    return StructuredTool.from_function(
        name=name,
        description="Return a list of ToolMessage values.",
        func=_fn,
        args_schema=_In,
    )


def _mixed_list_tool(name: str = "emit_mixed") -> BaseTool:
    """A tool that returns a mixed list of `Command` and `ToolMessage`."""

    class _In(BaseModel):
        value: int = Field(description="Integer marker to emit")

    def _fn(value: int) -> list[Command | ToolMessage]:
        return [
            Command(
                update={
                    "ptc_values": [value],
                    "messages": [
                        ToolMessage(
                            content=f"from-command={value}",
                            tool_call_id=f"ptc_call_{value}_command",
                            name=name,
                        )
                    ],
                }
            ),
            ToolMessage(
                content=f"from-list-tail={value}",
                tool_call_id=f"ptc_call_{value}_tail",
                name=name,
            ),
        ]

    return StructuredTool.from_function(
        name=name,
        description="Return a mixed list of Command and ToolMessage values.",
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
    return _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
    )


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


def test_filter_rejects_task_by_name() -> None:
    task = _echo_tool("task")
    with pytest.raises(ValueError, match="task` tool cannot be exposed"):
        filter_tools_for_ptc([task], ["task"], self_tool_name="eval")


def test_filter_rejects_task_by_instance() -> None:
    task = _echo_tool("task")
    with pytest.raises(ValueError, match="task` tool cannot be exposed"):
        filter_tools_for_ptc([], [task], self_tool_name="eval")


def test_filter_allows_non_task_tools() -> None:
    # Sanity: the reservation is specific to the name "task".
    out = filter_tools_for_ptc(
        [_echo_tool("tasks"), _echo_tool("subtask")],
        ["tasks", "subtask"],
        self_tool_name="eval",
    )
    assert [t.name for t in out] == ["tasks", "subtask"]


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
    assert "globalThis.tools" in prompt
    assert "tools.greet(input:" in prompt
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


async def test_undefined_object_property_uses_schema_default(
    repl: _ThreadREPL,
) -> None:
    """A property passed as JS ``undefined`` is omitted so defaults apply."""

    class _In(BaseModel):
        name: str = Field(default="", description="Optional name")

    def _fn(name: str = "") -> str:
        return f"ok:{name!r}"

    tool = StructuredTool.from_function(
        name="myTool",
        description="Echo the name with its repr.",
        func=_fn,
        args_schema=_In,
    )
    repl.install_tools([tool])

    empty = await repl.eval_async("await tools.myTool({})")
    assert empty.error_type is None, empty.error_message
    assert empty.result == "ok:''"

    explicit = await repl.eval_async('await tools.myTool({ name: "" })')
    assert explicit.error_type is None, explicit.error_message
    assert explicit.result == "ok:''"

    undefined = await repl.eval_async("await tools.myTool({ name: undefined })")
    assert undefined.error_type is None, undefined.error_message
    assert undefined.result == "ok:''"


async def test_promise_all_runs_tools_concurrently(repl: _ThreadREPL) -> None:
    """`Promise.all` on two tool calls resolves both before returning."""
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


async def test_command_tool_output_is_visible_to_js(
    repl: _ThreadREPL,
) -> None:
    repl.install_tools([_command_tool()])
    outcome = await repl.eval_async(
        "await tools.emitCommand({value: 1});\n"
        "await tools.emitCommand({value: 2});\n"
        "'done'"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "done"


async def test_command_tool_does_not_change_eval_outcome_shape(
    repl: _ThreadREPL,
) -> None:
    repl.install_tools([_command_tool()])
    first = await repl.eval_async("await tools.emitCommand({value: 7})")
    assert first.error_type is None, first.error_message
    assert first.result == "value=7"
    assert not hasattr(first, "commands")
    second = await repl.eval_async("1 + 1")
    assert second.error_type is None, second.error_message
    assert second.result == "2"
    assert not hasattr(second, "commands")


async def test_toolmessage_list_uses_last_message_content(repl: _ThreadREPL) -> None:
    repl.install_tools([_tool_message_list_tool()])
    outcome = await repl.eval_async("await tools.emitMessages({value: 9})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "second=9"


async def test_mixed_list_uses_tail_message(
    repl: _ThreadREPL,
) -> None:
    repl.install_tools([_mixed_list_tool()])
    outcome = await repl.eval_async("await tools.emitMixed({value: 11})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "from-list-tail=11"


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
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "Host function failed"


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
    """Dropping a tool removes it from `globalThis.tools` on next install."""
    repl.install_tools([_greet_tool(), _echo_tool("echo")])
    repl.install_tools([_greet_tool()])
    outcome = await repl.eval_async("typeof tools.echo")
    assert outcome.result == "undefined"
    outcome2 = await repl.eval_async("typeof tools.greet")
    assert outcome2.result == "function"


async def test_ptc_host_call_budget_exceeded_surfaces_error(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    limited = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=2,
    )
    limited.install_tools([_greet_tool()])
    outcome = await limited.eval_async(
        "await tools.greet({name: 'a'});\n"
        "await tools.greet({name: 'b'});\n"
        "await tools.greet({name: 'c'});\n"
        "'done'"
    )
    assert outcome.error_type == "PTCCallBudgetExceeded"
    assert "limit=2" in outcome.error_message
    assert "attempted=3" in outcome.error_message
    assert "function=tools.greet" in outcome.error_message


async def test_ptc_host_call_budget_catch_surfaces_generic_host_error(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    limited = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=1,
    )
    limited.install_tools([_greet_tool()])
    outcome = await limited.eval_async(
        "try {\n"
        "  await tools.greet({name: 'ok'});\n"
        "  await tools.greet({name: 'overflow'});\n"
        "  'not-caught';\n"
        "} catch (e) {\n"
        "  `${e.name}:${e.message}`;\n"
        "}"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "HostError:Host function failed"


async def test_ptc_host_call_budget_resets_each_eval(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    limited = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=1,
    )
    limited.install_tools([_greet_tool()])
    first = await limited.eval_async("await tools.greet({name: 'a'})")
    second = await limited.eval_async("await tools.greet({name: 'b'})")
    assert first.error_type is None, first.error_message
    assert second.error_type is None, second.error_message


async def test_ptc_host_call_budget_none_disables_limit(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    unlimited = _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=None,
    )
    unlimited.install_tools([_greet_tool()])
    outcome = await unlimited.eval_async(
        "await tools.greet({name: 'a'});\n"
        "await tools.greet({name: 'b'});\n"
        "await tools.greet({name: 'c'});\n"
        "'done'"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "done"


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------


def test_middleware_ptc_default_off_omits_prompt_block() -> None:
    mw = CodeInterpreterMiddleware()
    # Calling _prepare_for_call directly is fine — pass a minimal request
    # stand-in. We don't need a full ModelRequest for this check.
    from types import SimpleNamespace

    req = SimpleNamespace(tools=[_greet_tool()])
    prompt = mw._prepare_for_call(req)
    assert "`tools` namespace" not in prompt


def test_middleware_ptc_list_includes_prompt_block() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=["greet", "eval"])
    state = {"_quickjs_slot_id": "ptc-prompt"}
    req = SimpleNamespace(tools=[_greet_tool(), _echo_tool("eval")], state=state)
    prompt = mw._prepare_for_call(req)
    # Greet included
    assert "tools.greet(" in prompt
    # The REPL's own tool never appears
    assert "tools.eval(" not in prompt


def test_middleware_ptc_list_of_tools_exposes_without_agent_tools() -> None:
    """`ptc=[tool]` installs the tool in the REPL even when the agent has none."""
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=[_greet_tool()])
    state = {"_quickjs_slot_id": "ptc-tool-only"}
    req = SimpleNamespace(tools=[], state=state)
    prompt = mw._prepare_for_call(req)
    assert "tools.greet(" in prompt


def test_middleware_ptc_prompt_follows_each_slots_own_tools() -> None:
    """Same-named tools with different schemas never share a rendered prompt.

    One middleware serves every run of a compiled graph, so concurrent runs
    (slots) can expose different `lookup` tools, and a run can swap one in.
    """
    from types import SimpleNamespace

    def lookup_tool(description: str) -> BaseTool:
        def lookup(query: str) -> str:
            return query

        return StructuredTool.from_function(
            lookup, name="lookup", description=description
        )

    mw = CodeInterpreterMiddleware(ptc=["lookup"])
    acme = {"_quickjs_slot_id": "run-acme"}
    globex = {"_quickjs_slot_id": "run-globex"}

    acme_prompt = mw._prepare_for_call(
        SimpleNamespace(tools=[lookup_tool("Search acme issues.")], state=acme)
    )
    globex_prompt = mw._prepare_for_call(
        SimpleNamespace(tools=[lookup_tool("Search globex issues.")], state=globex)
    )
    swapped = mw._prepare_for_call(
        SimpleNamespace(tools=[lookup_tool("Search acme pull requests.")], state=acme)
    )

    assert "Search acme issues." in acme_prompt
    assert "Search globex issues." in globex_prompt
    assert "acme" not in globex_prompt
    assert "Search acme pull requests." in swapped


def test_middleware_ptc_requires_private_slot_state() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=["greet"])
    with pytest.raises(ValueError, match="_quickjs_slot_id"):
        mw._prepare_for_call(SimpleNamespace(tools=[_greet_tool()]))


async def test_ptc_install_and_eval_resolve_to_same_repl() -> None:
    """PTC install and the eval tool must see the same REPL instance.

    Regression: PTC installation must use the private slot id that the eval
    tool reads from runtime state, not a shared middleware fallback.
    """
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=["greet", "eval"])
    state = {"_quickjs_slot_id": "ptc-shared-slot"}
    req = SimpleNamespace(tools=[_greet_tool(), _echo_tool("eval")], state=state)
    mw._prepare_for_call(req)
    assert mw._registry.get_if_exists(state["_quickjs_slot_id"]) is None
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tools=mw.tools,
        tool_call_id="lazy-ptc",
        store=None,
    )
    try:
        result = await mw.tools[0].ainvoke(
            {"runtime": runtime, "code": "typeof tools.greet"}
        )
        assert result.content == "<result>function</result>"
    finally:
        mw._registry.close()


async def test_middleware_eval_tool_returns_tool_message_only() -> None:
    from types import SimpleNamespace

    command_tool = _command_tool()
    mw = CodeInterpreterMiddleware(ptc=[command_tool])
    tool = mw.tools[0]
    state = {"_quickjs_slot_id": "ptc-command-slot"}
    mw._prepare_for_call(SimpleNamespace(tools=[command_tool, tool], state=state))
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _chunk: None,
        tools=[tool],
        tool_call_id="outer_eval_call",
        store=None,
    )
    assert tool.coroutine is not None
    result = await tool.coroutine(
        runtime=runtime,
        code="await tools.emitCommand({value: 5})",
    )
    assert isinstance(result, ToolMessage)
    assert result.name == "eval"
    assert "<result>value=5</result>" in result.content


async def test_mode_call_reinstalls_ptc_tools_for_each_eval_call() -> None:
    from types import SimpleNamespace

    greet_tool = _greet_tool()
    mw = CodeInterpreterMiddleware(ptc=[greet_tool], mode="call")
    tool = mw.tools[0]
    state = {"_quickjs_slot_id": "ptc-call-slot"}
    mw._prepare_for_call(SimpleNamespace(tools=[greet_tool, tool], state=state))
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _chunk: None,
        tools=[tool],
        tool_call_id="outer_eval_call",
        store=None,
    )
    assert tool.coroutine is not None

    first = await tool.coroutine(
        runtime=runtime,
        code="await tools.greet({name: 'Ada'})",
    )
    second = await tool.coroutine(
        runtime=runtime,
        code="await tools.greet({name: 'Bob'})",
    )
    assert "<result>hi Ada x1</result>" in first.content
    assert "<result>hi Bob x1</result>" in second.content


def test_middleware_rejects_boolean_ptc_config_during_prepare() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        mw._prepare_for_call(SimpleNamespace(tools=[_greet_tool()]))
    mw = CodeInterpreterMiddleware(ptc=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        mw._prepare_for_call(SimpleNamespace(tools=[_greet_tool()]))


def test_middleware_rejects_dict_ptc_config_during_prepare() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc={"include": ["greet"]})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported `ptc` config type"):
        mw._prepare_for_call(SimpleNamespace(tools=[_greet_tool()]))


# ---------------------------------------------------------------------------
# Return-type rendering
# ---------------------------------------------------------------------------


def test_render_ptc_prompt_renders_concrete_primitive_return_types() -> None:
    """`render_ptc_prompt` renders Promise<T> from primitive annotations."""

    def get_service_id() -> int:
        """Return a service id."""
        return 1

    def get_service_name() -> str:
        """Return a service name."""
        return "svc"

    async def list_ids() -> list[int]:
        """List ids."""
        return [1, 2, 3]

    tools = [
        StructuredTool.from_function(
            name="get_service_id",
            description="Return a service id.",
            func=get_service_id,
        ),
        StructuredTool.from_function(
            name="get_service_name",
            description="Return a service name.",
            func=get_service_name,
        ),
        StructuredTool.from_function(
            name="list_ids",
            description="List ids.",
            coroutine=list_ids,
        ),
    ]
    prompt = render_ptc_prompt(tools)
    assert "Promise<integer>" in prompt or "Promise<number>" in prompt
    assert "Promise<string>" in prompt
    assert "Promise<integer[]>" in prompt or "Promise<number[]>" in prompt


def test_render_ptc_prompt_falls_back_to_unknown_for_unannotated_returns() -> None:
    """Tools without a return annotation render as `Promise<unknown>`."""

    def no_annotation():
        """Return something."""
        return 1

    tool = StructuredTool.from_function(
        name="no_annotation",
        description="Return something.",
        func=no_annotation,
    )
    prompt = render_ptc_prompt([tool])
    assert "Promise<unknown>" in prompt


def _stub() -> None:
    """Stub function used as a tool callable in parametrized return-type tests."""
    return


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        # Primitives.
        (int, "Promise<number>"),
        (float, "Promise<number>"),
        (str, "Promise<string>"),
        (bool, "Promise<boolean>"),
        (type(None), "Promise<null>"),
        # Containers of primitives.
        (list[int], "Promise<number[]>"),
        (list[int | None], "Promise<(number | null)[]>"),
        (dict[str, int], "Promise<Record<string, number>>"),
        (dict[str, Any], "Promise<Record<string, unknown>>"),
        # Optional / Literal / unions all flow through `anyOf`, `enum`, or
        # `const`.
        (int | None, "Promise<number | null>"),
        (Literal["active", "resolved"], 'Promise<"active" | "resolved">'),
        (Literal["only"], 'Promise<"only">'),
        (int | str, "Promise<number | string>"),
        # Top-level TypedDict / BaseModel — Pydantic inlines the schema.
        (_UserLookup, "Promise<{ id: number; name: string }>"),
        (_Status, "Promise<{ status: string; count: number }>"),
        # Nested TypedDict / BaseModel are declared once as named types.
        (
            list[_UserLookup],
            "/** Module-scope TypedDict used as a return annotation in PTC "
            "tests. */\n"
            "type _UserLookup = {\n  id: number;\n  name: string;\n};\n\n"
            "/** Stub tool. */\n"
            "tools.t(input: Record<string, unknown>): Promise<_UserLookup[]>",
        ),
        (list[_Status], "Promise<_Status[]>"),
        (dict[str, _UserLookup], "Promise<Record<string, _UserLookup>>"),
        (_UserLookup | None, "Promise<_UserLookup | null>"),
        # Recursive types refer to their own name instead of expanding.
        (
            _TreeNode,
            "type _TreeNode = {\n  label: string;\n  children?: _TreeNode[];\n};"
            "\n\n/** Stub tool. */\n"
            "tools.t(input: Record<string, unknown>): Promise<_TreeNode>",
        ),
        # Discriminated unions render as `oneOf` over named types.
        (
            Annotated[_Cat | _Dog, Field(discriminator="kind")],
            'type _Cat = {\n  kind: "cat";\n};',
        ),
        (
            Annotated[_Cat | _Dog, Field(discriminator="kind")],
            "Promise<_Cat | _Dog>",
        ),
        (
            list[Annotated[_Cat | _Dog, Field(discriminator="kind")]],
            "Promise<(_Cat | _Dog)[]>",
        ),
    ],
)
def test_render_ptc_prompt_return_types(annotation: Any, expected: str) -> None:
    """Return-type rendering covers each supported annotation shape."""

    # Build a fresh callable so the parametrized annotation is bound at runtime
    # rather than at import (`from __future__ import annotations` would
    # otherwise leave the annotation as a string).
    def _fn() -> None:
        """Tool stub."""
        return

    _fn.__annotations__["return"] = annotation
    tool = StructuredTool.from_function(
        name="t",
        description="Stub tool.",
        func=_fn,
    )
    prompt = render_ptc_prompt([tool])
    assert expected in prompt, prompt


def _returning(name: str, annotation: Any) -> BaseTool:
    """Build a no-argument tool whose function returns `annotation`."""

    def _fn() -> None:
        """Tool stub."""
        return

    _fn.__annotations__["return"] = annotation
    return StructuredTool.from_function(name=name, description="Stub.", func=_fn)


def test_render_ptc_prompt_resolves_nested_input_types() -> None:
    """Nested models and maps in input args render as concrete types."""

    class _Filter(BaseModel):
        tag: str

    class _Query(BaseModel):
        filters: list[_Filter]
        weights: dict[str, float]

    tool = StructuredTool.from_function(
        name="search", description="Search.", func=_stub, args_schema=_Query
    )
    prompt = render_ptc_prompt([tool])
    assert "type _Filter = {\n  tag: string;\n};" in prompt, prompt
    assert "  filters: _Filter[];\n  weights: Record<string, number>;" in prompt


def test_render_ptc_prompt_unwraps_single_item_all_of() -> None:
    """A `$ref` wrapped in a one-item `allOf` renders as the referenced type.

    Pydantic before 2.9 wraps a `$ref` that carries a description this way.
    """

    def _wrap_ref(schema: dict[str, Any]) -> None:
        schema["properties"]["status"] = {
            "allOf": [schema["properties"]["status"]],
            "description": "Current status",
        }

    class _Report(BaseModel):
        model_config = ConfigDict(json_schema_extra=_wrap_ref)
        status: _Status

    tool = StructuredTool.from_function(
        name="report", description="Report.", func=_stub, args_schema=_Report
    )
    prompt = render_ptc_prompt([tool])
    assert "*/ status: _Status;" in prompt, prompt


def test_render_ptc_prompt_declares_shared_types_once() -> None:
    """Tools returning the same nested type share one declaration."""
    prompt = render_ptc_prompt(
        [_returning("first", list[_Status]), _returning("second", list[_Status])]
    )
    assert prompt.count("type _Status = {") == 1, prompt
    assert prompt.count("Promise<_Status[]>") == 2, prompt


def test_render_ptc_prompt_prefixes_conflicting_type_names() -> None:
    """Distinct types that share a name get the tool name as a prefix."""

    def _hit_type(field: str) -> type:
        return TypedDict("Hit", {field: int})

    prompt = render_ptc_prompt(
        [
            _returning("find_docs", list[_hit_type("page")]),
            _returning("find_images", list[_hit_type("width")]),
        ]
    )
    assert "type Hit = {\n  page: number;\n};" in prompt, prompt
    assert "type FindImagesHit = {\n  width: number;\n};" in prompt, prompt
    assert "tools.findDocs(input: Record<string, unknown>): Promise<Hit[]>" in prompt
    assert (
        "tools.findImages(input: Record<string, unknown>): Promise<FindImagesHit[]>"
        in prompt
    )


def _get_status_record() -> _Status:
    """Module-level helper.

    Defined at module scope so `get_type_hints` can resolve the return
    annotation under `from __future__ import annotations`.
    """
    return _Status(status="ok", count=3)


async def test_pydantic_return_arrives_as_object_matching_schema(
    repl: _ThreadREPL,
) -> None:
    """BaseModel returns are dumped at the bridge so the JS shape matches the schema."""
    tool = StructuredTool.from_function(
        name="get_status",
        description="Return a status record.",
        func=_get_status_record,
    )
    # The prompt advertises a structured object (Pydantic JSON Schema inlined).
    prompt = render_ptc_prompt([tool])
    assert "status: string" in prompt
    assert "count: number" in prompt

    # And the bridge delivers an object with those fields, not a string.
    repl.install_tools([tool])
    outcome = await repl.eval_async(
        "const r = await tools.getStatus({});\n`${typeof r}:${r.status}:${r.count}`"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "object:ok:3"


# ---------------------------------------------------------------------------
# Tool data reaching JS
# ---------------------------------------------------------------------------


class _SearchArtifact(TypedDict):
    """Module-scope TypedDict used as an artifact annotation."""

    total: int
    query: str


@tool(response_format="content_and_artifact")
def _search(query: str) -> tuple[str, _SearchArtifact]:
    """Search, returning model text and a separate data value."""
    return f"Found 2 results for {query}", {"total": 2, "query": query}


@dataclass
class _Event:
    """Module-scope dataclass returned by a PTC tool."""

    name: str
    when: datetime


def _next_event() -> _Event:
    """Return the next event."""
    return _Event(name="launch", when=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))


def _thumbnail() -> bytes:
    """Return image bytes."""
    return b"\x89PNG\x00\xff"


def _message_with_artifact() -> ToolMessage:
    """Return a ToolMessage carrying an artifact."""
    return ToolMessage(
        content="2 rows", artifact={"rows": [1, 2]}, tool_call_id="ignored"
    )


def _failed_message() -> ToolMessage:
    """Return an error ToolMessage that also carries an artifact."""
    return ToolMessage(
        content="not found",
        artifact={"code": "missing"},
        tool_call_id="ignored",
        status="error",
    )


@tool
def _whoami(tool_call_id: Annotated[str, InjectedToolCallId]) -> dict[str, bool]:
    """Report whether the call has an id."""
    return {"has_id": bool(tool_call_id)}


async def test_artifact_tool_gives_js_its_artifact(repl: _ThreadREPL) -> None:
    repl.install_tools([_search.model_copy(update={"name": "search"})])
    outcome = await repl.eval_async(
        "const r = await tools.search({query: 'ada'});\n`${r.total}:${r.query}`"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "2:ada"


def test_artifact_tool_return_type_is_its_artifact() -> None:
    prompt = render_ptc_prompt([_search.model_copy(update={"name": "search"})])
    assert "Promise<{ total: number; query: string }>" in prompt, prompt


async def test_artifact_tool_without_artifact_gives_js_its_content(
    repl: _ThreadREPL,
) -> None:
    @tool(response_format="content_and_artifact")
    def lookup(key: str) -> tuple[str, dict[str, int] | None]:
        """Look up a key."""
        return f"no entry for {key}", None

    assert "Promise<unknown>" in render_ptc_prompt([lookup])
    repl.install_tools([lookup])
    outcome = await repl.eval_async("await tools.lookup({key: 'k'})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "no entry for k"


async def test_base_tool_subclass_return_reaches_js_as_object(
    repl: _ThreadREPL,
) -> None:
    class _Inventory(BaseTool):
        name: str = "inventory"
        description: str = "Count items."

        def _run(self, item: str) -> dict[str, int]:
            return {item: 3}

    repl.install_tools([_Inventory()])
    outcome = await repl.eval_async("(await tools.inventory({item: 'pens'})).pens")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "3"


async def test_handled_tool_error_reaches_js_as_its_text(repl: _ThreadREPL) -> None:
    def _fail(query: str) -> tuple[str, dict[str, int]]:
        msg = f"no index for {query}"
        raise ToolException(msg)

    tool_ = StructuredTool.from_function(
        name="search",
        description="Search.",
        func=_fail,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
    repl.install_tools([tool_])
    outcome = await repl.eval_async("await tools.search({query: 'q'})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "no index for q"


async def test_dataclass_return_arrives_as_json_shaped_object(
    repl: _ThreadREPL,
) -> None:
    tool_ = StructuredTool.from_function(
        name="next_event", description="Next event.", func=_next_event
    )
    repl.install_tools([tool_])
    outcome = await repl.eval_async(
        "const e = await tools.nextEvent({});\n`${e.name}@${e.when}`"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "launch@2026-01-02T03:04:05Z"


async def test_bytes_return_arrives_as_uint8array(repl: _ThreadREPL) -> None:
    tool_ = StructuredTool.from_function(
        name="thumbnail", description="Thumbnail.", func=_thumbnail
    )
    assert "Promise<Uint8Array>" in render_ptc_prompt([tool_])
    repl.install_tools([tool_])
    outcome = await repl.eval_async(
        "const b = await tools.thumbnail({});\n"
        "`${b instanceof Uint8Array}:${b.length}:${b[5]}`"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "true:6:255"


async def test_injected_tool_call_id_tool_returns_an_object(
    repl: _ThreadREPL,
) -> None:
    repl.install_tools([_whoami.model_copy(update={"name": "whoami"})])
    outcome = await repl.eval_async("(await tools.whoami({})).has_id")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "true"


async def test_returned_tool_message_gives_js_its_artifact(
    repl: _ThreadREPL,
) -> None:
    tool_ = StructuredTool.from_function(
        name="rows", description="Rows.", func=_message_with_artifact
    )
    repl.install_tools([tool_])
    outcome = await repl.eval_async("(await tools.rows({})).rows.length")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "2"
    assert "Promise<unknown>" in render_ptc_prompt([tool_])


async def test_error_tool_message_gives_js_its_content(repl: _ThreadREPL) -> None:
    tool_ = StructuredTool.from_function(
        name="fetch_row", description="Fetch a row.", func=_failed_message
    )
    repl.install_tools([tool_])
    outcome = await repl.eval_async("await tools.fetchRow({})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "not found"
