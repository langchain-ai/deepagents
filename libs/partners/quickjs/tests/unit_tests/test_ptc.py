"""Tests for programmatic tool calling (PTC).

PTC exposes agent tools as `tools.<camelCase>` async functions inside
the REPL so one `eval` can orchestrate many tool invocations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._prompt import _tool_block_key
from langchain_quickjs._ptc import (
    filter_tools_for_ptc,
    render_ptc_prompt,
    to_camel_case,
)
from tests.unit_tests._slots import drop_slots, install_tools, slot_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_quickjs._repl import _Slot

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


async def test_tool_invocation_from_repl(repl: _Slot) -> None:
    calls: list[dict] = []
    install_tools(repl, [_greet_tool(calls)])
    outcome = await repl.eval_async('await tools.greet({name: "world", times: 2})')
    assert outcome.error_type is None, outcome.error_message
    # tool returned the string "hi world x2"
    assert outcome.result == "hi world x2"
    assert calls == [{"name": "world", "times": 2}]


async def test_undefined_object_property_uses_schema_default(
    repl: _Slot,
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
    install_tools(repl, [tool])

    empty = await repl.eval_async("await tools.myTool({})")
    assert empty.error_type is None, empty.error_message
    assert empty.result == "ok:''"

    explicit = await repl.eval_async('await tools.myTool({ name: "" })')
    assert explicit.error_type is None, explicit.error_message
    assert explicit.result == "ok:''"

    undefined = await repl.eval_async("await tools.myTool({ name: undefined })")
    assert undefined.error_type is None, undefined.error_message
    assert undefined.result == "ok:''"


async def test_promise_all_runs_tools_concurrently(repl: _Slot) -> None:
    """`Promise.all` on two tool calls resolves both before returning."""
    calls: list[dict] = []
    install_tools(repl, [_greet_tool(calls)])
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
    repl: _Slot,
) -> None:
    install_tools(repl, [_command_tool()])
    outcome = await repl.eval_async(
        "await tools.emitCommand({value: 1});\n"
        "await tools.emitCommand({value: 2});\n"
        "'done'"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "done"


async def test_command_tool_does_not_change_eval_outcome_shape(
    repl: _Slot,
) -> None:
    install_tools(repl, [_command_tool()])
    first = await repl.eval_async("await tools.emitCommand({value: 7})")
    assert first.error_type is None, first.error_message
    assert first.result == "value=7"
    assert not hasattr(first, "commands")
    second = await repl.eval_async("1 + 1")
    assert second.error_type is None, second.error_message
    assert second.result == "2"
    assert not hasattr(second, "commands")


async def test_toolmessage_list_uses_last_message_content(repl: _Slot) -> None:
    install_tools(repl, [_tool_message_list_tool()])
    outcome = await repl.eval_async("await tools.emitMessages({value: 9})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "second=9"


async def test_mixed_list_uses_tail_message(
    repl: _Slot,
) -> None:
    install_tools(repl, [_mixed_list_tool()])
    outcome = await repl.eval_async("await tools.emitMixed({value: 11})")
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "from-list-tail=11"


async def test_tool_failure_surfaces_as_js_error(repl: _Slot) -> None:
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
    install_tools(repl, [tool])
    outcome = await repl.eval_async(
        "try { await tools.boom({x: 1}); 'no-throw' } catch (e) { e.message }"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "Host function failed"


async def test_install_tools_skips_invalid_js_identifier_names(
    repl: _Slot,
) -> None:
    install_tools(repl, [_echo_tool("123bad")])
    outcome = await repl.eval_async('typeof tools["123bad"]')
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "undefined"


async def test_dropped_tool_kept_by_code_fails_the_eval(repl: _Slot) -> None:
    install_tools(repl, [_echo_tool()])
    kept = await repl.eval_async("globalThis.keep = tools.echo; typeof keep")
    assert kept.result == "function"
    install_tools(repl, [_echo_tool("other")])
    outcome = await repl.eval_async("await keep({msg: 'x'})")
    assert outcome.error_type == "TypeError"
    assert outcome.error_message == "tools.echo is no longer available"


async def test_tools_proxy_behaves_like_a_plain_object(repl: _Slot) -> None:
    install_tools(repl, [_echo_tool()])
    outcome = await repl.eval_async(
        "[typeof tools.echo, typeof tools.missing, 'echo' in tools,"
        " Object.keys(tools).join(','), JSON.stringify(tools),"
        " tools.hasOwnProperty('echo'), tools.hasOwnProperty('x'), `${tools}`]"
        ".join('|')"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == (
        "function|undefined|true|echo|{}|true|false|[object Object]"
    )


async def test_install_tools_is_idempotent(repl: _Slot) -> None:
    """Calling install_tools twice with the same set is a no-op for the guest."""
    calls: list[dict] = []
    tool = _greet_tool(calls)
    install_tools(repl, [tool])
    install_tools(repl, [tool])
    outcome = await repl.eval_async('await tools.greet({name: "x"})')
    assert outcome.result == "hi x x1"
    assert len(calls) == 1


async def test_install_tools_shrinks_namespace(repl: _Slot) -> None:
    """Dropping a tool removes it from `globalThis.tools` on next install."""
    install_tools(repl, [_greet_tool(), _echo_tool("echo")])
    install_tools(repl, [_greet_tool()])
    outcome = await repl.eval_async("typeof tools.echo")
    assert outcome.result == "undefined"
    outcome2 = await repl.eval_async("typeof tools.greet")
    assert outcome2.result == "function"


async def test_ptc_host_call_budget_exceeded_surfaces_error(
    make_repl: Callable[..., _Slot],
) -> None:
    limited = make_repl(
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=2,
    )
    install_tools(limited, [_greet_tool()])
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
    make_repl: Callable[..., _Slot],
) -> None:
    limited = make_repl(
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=1,
    )
    install_tools(limited, [_greet_tool()])
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
    make_repl: Callable[..., _Slot],
) -> None:
    limited = make_repl(
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=1,
    )
    install_tools(limited, [_greet_tool()])
    first = await limited.eval_async("await tools.greet({name: 'a'})")
    second = await limited.eval_async("await tools.greet({name: 'b'})")
    assert first.error_type is None, first.error_message
    assert second.error_type is None, second.error_message


async def test_ptc_host_call_budget_none_disables_limit(
    make_repl: Callable[..., _Slot],
) -> None:
    unlimited = make_repl(
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4_000,
        max_ptc_calls=None,
    )
    install_tools(unlimited, [_greet_tool()])
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


def _deferred(tool: BaseTool) -> BaseTool:
    """Copy `tool` the way `ProviderToolSearchMiddleware` defers it."""
    return tool.model_copy(update={"extras": {"defer_loading": True}})


def test_middleware_ptc_omits_deferred_tools_from_prompt() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=["greet", "echo"])
    tools = [_deferred(_greet_tool()), _echo_tool()]
    state = {"_quickjs_slot_id": "ptc-deferred"}
    prompt = mw._prepare_for_call(SimpleNamespace(tools=tools, state=state))
    assert "tools.echo(" in prompt
    assert "tools.greet(" not in prompt
    assert "found through tool search" in prompt
    assert [t.name for t in mw._ptc_tools_by_slot["ptc-deferred"]] == [
        "greet",
        "echo",
    ]


def test_middleware_ptc_prompt_has_no_search_line_without_deferred_tools() -> None:
    from types import SimpleNamespace

    mw = CodeInterpreterMiddleware(ptc=["greet"])
    state = {"_quickjs_slot_id": "ptc-not-deferred"}
    prompt = mw._prepare_for_call(SimpleNamespace(tools=[_greet_tool()], state=state))
    assert "tools.greet(" in prompt
    assert "tool search" not in prompt


def test_middleware_ptc_reads_deferral_from_request_tools() -> None:
    """An explicit `ptc` tool is deferred when the request carries it deferred.

    The request also holds the provider's tool search spec, a plain dict.
    """
    from types import SimpleNamespace

    greet = _greet_tool()
    mw = CodeInterpreterMiddleware(ptc=[greet])
    tools = [_deferred(greet), {"type": "tool_search"}]
    state = {"_quickjs_slot_id": "ptc-deferred-request"}
    prompt = mw._prepare_for_call(SimpleNamespace(tools=tools, state=state))
    assert "### API Reference" not in prompt
    assert "found through tool search" in prompt


async def test_deferred_tool_is_callable_from_eval() -> None:
    from types import SimpleNamespace

    calls: list[dict] = []
    mw = CodeInterpreterMiddleware(ptc=["greet"])
    state = {"_quickjs_slot_id": "ptc-deferred-eval"}
    tools = [_deferred(_greet_tool(calls))]
    mw._prepare_for_call(SimpleNamespace(tools=tools, state=state))
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tools=mw.tools,
        tool_call_id="deferred-ptc",
        store=None,
    )
    try:
        result = await mw.tools[0].ainvoke(
            {"runtime": runtime, "code": 'await tools.greet({ name: "Ada" })'}
        )
        assert result.content == "<result>hi Ada x1</result>"
        assert calls == [{"name": "Ada", "times": 1}]
    finally:
        drop_slots(mw)


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
    assert slot_of(mw, state["_quickjs_slot_id"]) is None
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
        drop_slots(mw)


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
        (dict[str, int], "Promise<Record<string, number>>"),
        (list[int | None], "Promise<(number | null)[]>"),
        # Optional / Literal / unions all flow through `anyOf` or `enum`.
        (int | None, "Promise<number | null>"),
        (Literal["active", "resolved"], 'Promise<"active" | "resolved">'),
        (int | str, "Promise<number | string>"),
        # Top-level TypedDict / BaseModel — Pydantic inlines the schema.
        (_UserLookup, "Promise<{ id: number; name: string }>"),
        (_Status, "Promise<{ status: string; count: number }>"),
        # Collections of TypedDict / BaseModel resolve their `$ref` inline.
        (list[_UserLookup], "Promise<{ id: number; name: string }[]>"),
        (list[_Status], "Promise<{ status: string; count: number }[]>"),
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


def _get_status_record() -> _Status:
    """Module-level helper.

    Defined at module scope so `get_type_hints` can resolve the return
    annotation under `from __future__ import annotations`.
    """
    return _Status(status="ok", count=3)


async def test_pydantic_return_arrives_as_object_matching_schema(
    repl: _Slot,
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
    install_tools(repl, [tool])
    outcome = await repl.eval_async(
        "const r = await tools.getStatus({});\n`${typeof r}:${r.status}:${r.count}`"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "object:ok:3"


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class _Address(BaseModel):
    city: str


class _User(BaseModel):
    name: str
    address: _Address
    home: _Address | None = None


class _UsersInput(BaseModel):
    users: list[_User]


class _TreeNode(BaseModel):
    value: int
    children: list[_TreeNode] = []


def _normalize_users(users: list[_User]) -> list[_User]:
    """Normalize users."""
    return users


def _walk(value: int, children: list[_TreeNode] | None = None) -> _TreeNode:
    """Walk a tree."""
    return _TreeNode(value=value, children=children or [])


def _ptc_body(tools: list[BaseTool]) -> str:
    """Return the fenced API reference block of the rendered PTC prompt."""
    return render_ptc_prompt(tools).rsplit("```typescript\n", 1)[1]


def test_nested_models_render_inline() -> None:
    """Nested, list-of-model and Optional fields render inline, not `unknown`."""
    user = "{ name: string; address: { city: string }; home?: { city: string } | null }"
    tool = StructuredTool.from_function(
        _normalize_users, name="normalize_users", args_schema=_UsersInput
    )
    body = _ptc_body([tool])
    assert f"  users: {user}[];\n}}): Promise<{user}[]>" in body


def test_recursive_model_back_reference_renders_unknown() -> None:
    """A model's reference to itself is cut off as `unknown`."""
    tool = StructuredTool.from_function(_walk, name="walk", args_schema=_TreeNode)
    body = _ptc_body([tool])
    assert "  children?: { value: number; children?: unknown[] }[];" in body
    assert "): Promise<{ value: number; children?: unknown[] }>" in body


def _dict_schema_tool(name: str, schema: dict[str, Any]) -> BaseTool:
    """A tool whose `args_schema` is raw JSON Schema, as MCP adapters build."""
    return StructuredTool(
        name=name, description="MCP tool.", args_schema=schema, func=lambda **_: None
    )


def test_dict_schema_renders_fields_with_refs_inline() -> None:
    """MCP-style dict schemas render their fields, with `$defs` inlined."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
            "regions": {"type": "array", "items": {"$ref": "#/$defs/Region"}},
        },
        "required": ["service"],
        "$defs": {
            "Region": {"type": "object", "properties": {"name": {"type": "string"}}}
        },
    }
    body = _ptc_body([_dict_schema_tool("request_rollout_check", schema)])
    assert "Record<string, unknown>" not in body
    assert "  service: string;\n  regions?: { name?: string }[];" in body


def test_dict_schema_recursive_ref_renders_unknown() -> None:
    """A recursive `$ref` in a dict schema is cut off as `unknown`."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                },
            }
        },
    }
    body = _ptc_body([_dict_schema_tool("walk", schema)])
    assert "  root?: { children?: unknown[] };" in body


class _LookupArgs(BaseModel):
    key: str = Field(description="Key to look up.")


def _lookup_tool() -> BaseTool:
    def _fn(key: str) -> str:
        return key

    return StructuredTool.from_function(
        name="lookup",
        description="Look up a key.",
        func=_fn,
        args_schema=_LookupArgs,
    )


def test_render_ptc_prompt_reuses_blocks_across_tool_instances() -> None:
    """Tools rebuilt per run (same name, schema, return type) render once."""
    first = render_ptc_prompt([_lookup_tool()])
    with patch(
        "langchain_quickjs._prompt._safe_json_schema", side_effect=AssertionError
    ):
        assert render_ptc_prompt([_lookup_tool()]) == first


def test_render_ptc_prompt_keys_dict_schemas_by_content() -> None:
    """MCP-style dict schemas share the `dict` class but must not collide."""

    def _mcp_tool(prop: str) -> BaseTool:
        return StructuredTool(
            name="mcp",
            description="An MCP tool.",
            args_schema={
                "type": "object",
                "properties": {prop: {"type": "string"}},
                "required": [prop],
            },
            func=lambda **_: "ok",
        )

    assert _tool_block_key(_mcp_tool("query")) == _tool_block_key(_mcp_tool("query"))
    assert _tool_block_key(_mcp_tool("query")) != _tool_block_key(_mcp_tool("path"))


def test_dict_schema_boolean_subschemas_render_unknown() -> None:
    """Boolean subschemas (`true` = any value) render as `unknown`."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "value": True,
            "either": {"anyOf": [True, {"type": "string"}]},
            "nested": {"type": "object", "properties": {"a": True}},
        },
    }
    body = _ptc_body([_dict_schema_tool("mcp", schema)])
    assert "  value?: unknown;" in body
    assert "  either?: unknown | string;" in body
    assert "  nested?: { a?: unknown };" in body
