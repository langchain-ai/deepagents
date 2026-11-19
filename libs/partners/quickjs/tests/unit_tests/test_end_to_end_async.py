"""Async end-to-end tests for `CodeInterpreterMiddleware` with a fake LLM.

Covers the same integration surfaces as the prior quickjs e2e suite:
agent wiring, REPL execution, PTC tool calls, runtime propagation,
error surfacing, and concurrent agent runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import Any, NotRequired

import pytest
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import write_state
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel


@tool
def list_user_ids() -> list[str]:
    """Return example user identifiers for QuickJS bridging tests."""
    return ["user_1", "user_2", "user_3"]


@tool("sync_label")
def sync_label_tool(value: str) -> str:
    """Return a labeled value from a synchronous LangChain tool."""
    return f"sync:{value}"


@tool("async_label")
async def async_label_tool(value: str) -> str:
    """Return a labeled value from an asynchronous LangChain tool."""
    await asyncio.sleep(0)
    return f"async:{value}"


@tool("runtime_configurable")
def runtime_configurable(value: str, runtime: ToolRuntime) -> str:
    """Return configurable runtime data for testing ToolRuntime context propagation."""
    return f"{value}:{runtime.config['configurable']['user_id']}"


@tool("always_fails")
async def always_fails(value: str) -> str:
    """Raise to verify host-call failures surface to the model."""
    await asyncio.sleep(0)
    msg = f"boom:{value}"
    raise RuntimeError(msg)


def _script(code: str, *, final_message: str = "done") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _make_agent(
    code: str,
    middleware: CodeInterpreterMiddleware,
    *,
    final_message: str = "done",
) -> Any:
    return create_deep_agent(
        model=FakeChatModel(messages=_script(code, final_message=final_message)),
        middleware=[middleware],
    )


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _assert_result_contains(content: str, expected: str) -> None:
    assert "<error" not in content, content
    assert "<result" in content, content
    assert expected in content, content


async def test_deepagent_with_quickjs_interpreter() -> None:
    """Basic async test with QuickJS interpreter."""
    result = await _make_agent(
        "6 * 7",
        CodeInterpreterMiddleware(),
        final_message="The answer is 42.",
    ).ainvoke(
        {"messages": [HumanMessage(content="Use the eval tool to calculate 6 * 7")]}
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "42")
    assert result["messages"][-1].content == "The answer is 42."


async def test_deepagent_with_quickjs_list_returning_foreign_function() -> None:
    """A PTC tool returning a Python `list` surfaces as a native JS Array."""
    code = "const ids = await tools.listUserIds({});\nids.join(',');"
    result = await _make_agent(
        code, CodeInterpreterMiddleware(ptc=[list_user_ids])
    ).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to print the available user ids"
                )
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "user_1,user_2,user_3")


async def test_deepagent_with_quickjs_async_foreign_function() -> None:
    """Verify the eval tool can call sync and async LangChain tools in one run."""
    code = (
        "const [left, right] = await Promise.all([\n"
        "  tools.syncLabel({value: 'left'}),\n"
        "  tools.asyncLabel({value: 'right'}),\n"
        "]);\n"
        "`${left}|${right}`;"
    )
    result = await _make_agent(
        code,
        CodeInterpreterMiddleware(ptc=[sync_label_tool, async_label_tool]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool to call sync and async tools")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "sync:left|async:right")


async def test_quickjs_async_timeout_error() -> None:
    """Verify the async eval path surfaces QuickJS eval timeouts."""
    result = await _make_agent(
        "while (true) {}",
        CodeInterpreterMiddleware(timeout=1),
        final_message="timeout hit",
    ).ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool and keep looping until timeout")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    assert '<error type="Timeout">' in tool_message.content
    assert result["messages"][-1].content == "timeout hit"


async def test_quickjs_async_tool_exception_propagates() -> None:
    """Tool exceptions propagate as the original Python exception so
    ToolNode's default handler reraises and the agent crashes — same
    semantics as a non-quickjs tool that raises."""
    agent = _make_agent(
        "await tools.alwaysFails({value: 'x'})",
        CodeInterpreterMiddleware(ptc=[always_fails]),
    )
    with pytest.raises(RuntimeError, match="boom:x"):
        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Use the eval tool to call the async tool that raises"
                    )
                ]
            }
        )


async def test_quickjs_async_host_call_budget_exceeded() -> None:
    """Verify async eval surfaces per-eval PTC call budget exhaustion."""
    result = await _make_agent(
        "await tools.syncLabel({value: 'a'});\n"
        "await tools.syncLabel({value: 'b'});\n"
        "'done'",
        CodeInterpreterMiddleware(ptc=[sync_label_tool], max_ptc_calls=1),
    ).ainvoke(
        {"messages": [HumanMessage(content="Use eval and call sync_label twice.")]}
    )
    tool_message = _eval_tool_message(result)
    assert '<error type="PTCCallBudgetExceeded">' in tool_message.content
    assert "limit=1" in tool_message.content


async def test_quickjs_async_toolruntime_configurable_foreign_function() -> None:
    """Verify async PTC tool calls see configurable runtime data."""
    result = await _make_agent(
        "await tools.runtimeConfigurable({value: 'value'})",
        CodeInterpreterMiddleware(ptc=[runtime_configurable]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to inspect configurable runtime"
                )
            ]
        },
        config={"configurable": {"user_id": "user-123"}},
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "value:user-123")


async def test_quickjs_async_parallel_agents() -> None:
    """Verify many async agents can run in parallel with REPL middleware."""

    async def _run_agent(index: int) -> tuple[int, dict[str, object]]:
        result = await _make_agent(
            f"{index} * 10",
            CodeInterpreterMiddleware(),
            final_message=f"done-{index}",
        ).ainvoke(
            {
                "messages": [
                    HumanMessage(content=f"Use the eval tool to multiply {index} by 10")
                ]
            }
        )
        return index, result

    runs = await asyncio.gather(*(_run_agent(index) for index in range(25)))

    assert len(runs) == 25
    for index, result in runs:
        tool_message = _eval_tool_message(result)
        _assert_result_contains(tool_message.content, str(index * 10))
        assert result["messages"][-1].content == f"done-{index}"


class _ClaimState(AgentState):
    claims: NotRequired[dict[str, str]]


class _ClaimMiddleware(AgentMiddleware):
    state_schema = _ClaimState


@tool("write_claim")
def write_claim(claim_id: str, value: str, runtime: ToolRuntime) -> Command:
    """Store a claim in graph state."""
    claims = {**(runtime.state.get("claims") or {}), claim_id: value}
    message = ToolMessage(content="stored", tool_call_id=runtime.tool_call_id)
    return Command(update={"claims": claims, "messages": [message]})


@tool("read_claim")
def read_claim(claim_id: str, runtime: ToolRuntime) -> str:
    """Read a claim from graph state."""
    return (runtime.state.get("claims") or {}).get(claim_id, "missing")


@tool("route_claim")
def route_claim() -> Command:
    """Store a claim and route to another node."""
    return Command(goto="elsewhere", update={"claims": {"routed": "x"}})


@tool("resume_claim")
def resume_claim() -> Command:
    """Store a claim on the parent graph and resume it."""
    return Command(graph=Command.PARENT, resume="r", update={"claims": {"r": "x"}})


@tool("list_claim")
def list_claim() -> Command:
    """Store a claim as a list of key-value pairs."""
    return Command(update=[("claims", {"listed": "x"})])


@tool("mixed_claim")
def mixed_claim() -> list[Command]:
    """Store a claim, then route to another node."""
    return [Command(update={"claims": {"mixed": "x"}}), Command(goto="elsewhere")]


@tool("reroute_claim")
def reroute_claim() -> Command:
    """Store a claim and write the channel that triggers the model node."""
    return Command(update={"claims": {"rerouted": "x"}, "branch:to:model": None})


@tool("replace_claims")
def replace_claims(claim_id: str) -> str:
    """Replace every claim from inside the tool, without returning a `Command`."""
    write_state({"claims": {claim_id: "direct"}})
    return "replaced"


@tool("ask_human")
def ask_human() -> str:
    """Pause the run for a human answer."""
    return str(interrupt("ok?"))


_CLAIM_TOOLS = [
    write_claim,
    read_claim,
    route_claim,
    resume_claim,
    list_claim,
    mixed_claim,
    reroute_claim,
    replace_claims,
    ask_human,
]


def _claim_agent(code: str, **kwargs: Any) -> Any:
    return create_agent(
        model=FakeChatModel(messages=_script(code)),
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(),
            _ClaimMiddleware(),
            CodeInterpreterMiddleware(ptc=["write_todos", "write_file", *_CLAIM_TOOLS]),
        ],
        **kwargs,
    )


_GO = {"messages": [HumanMessage(content="go")]}
_WRITE_C1 = "await tools.writeClaim({claim_id: 'c1', value: 'v1'});\n"
_WRITE_C2 = "await tools.writeClaim({claim_id: 'c2', value: 'v2'});\n"


async def test_ptc_state_updates_apply_and_are_readable_in_same_eval() -> None:
    """`tools.*` state updates land in the graph and later calls see them."""
    code = (
        "await tools.writeTodos({todos: [{content: 'a', status: 'pending'}]});\n"
        + _WRITE_C1
        + "await tools.writeFile({file_path: '/one.txt', content: '1'});\n"
        "await tools.writeFile({file_path: '/two.txt', content: '2'});\n"
        "await tools.readClaim({claim_id: 'c1'})"
    )
    result = await _claim_agent(code).ainvoke(_GO)

    _assert_result_contains(_eval_tool_message(result).content, "v1")
    assert result["todos"] == [{"content": "a", "status": "pending"}]
    assert result["claims"] == {"c1": "v1"}
    assert {"/one.txt", "/two.txt"} <= set(result["files"])
    assert [m.name for m in result["messages"] if isinstance(m, ToolMessage)] == [
        "eval"
    ]


def test_ptc_state_updates_apply_under_sync_invoke() -> None:
    result = _claim_agent(_WRITE_C1 + "'ok'").invoke(_GO)

    assert result["claims"] == {"c1": "v1"}


async def test_ptc_second_write_to_one_value_key_fails_eval_and_keeps_first() -> None:
    """A key that takes one value per step rejects the second write when made."""
    result = await _claim_agent(_WRITE_C1 + _WRITE_C2 + "'unreached'").ainvoke(_GO)

    content = _eval_tool_message(result).content
    assert '<error type="InvalidUpdate">' in content, content
    assert "tools.writeClaim could not update state" in content
    assert result["claims"] == {"c1": "v1"}
    assert result["messages"][-1].content == "done"


async def test_ptc_update_cannot_write_a_key_that_is_not_state() -> None:
    """A `Command` update cannot route the graph through a trigger channel."""
    result = await _claim_agent("await tools.rerouteClaim({}); 'unreached'").ainvoke(
        _GO
    )

    content = _eval_tool_message(result).content
    assert '<error type="InvalidUpdate">' in content, content
    assert "claims" not in result
    assert [m.content for m in result["messages"] if isinstance(m, AIMessage)][
        -1
    ] == "done"


async def test_ptc_second_write_inside_a_tool_fails_eval_and_keeps_first() -> None:
    """A tool that writes state itself is rejected the same way as a `Command`."""
    code = (
        "await tools.replaceClaims({claim_id: 'c1'});\n"
        "await tools.replaceClaims({claim_id: 'c2'});\n"
        "'unreached'"
    )
    result = await _claim_agent(code).ainvoke(_GO)

    content = _eval_tool_message(result).content
    assert '<error type="InvalidUpdate">' in content, content
    assert "tools.replaceClaims could not update state" in content
    assert result["claims"] == {"c1": "direct"}
    assert result["messages"][-1].content == "done"


async def test_ptc_script_can_catch_a_rejected_write_and_continue() -> None:
    code = (
        _WRITE_C1
        + "try { "
        + _WRITE_C2.strip()
        + " } catch (e) {}\n"
        + "await tools.readClaim({claim_id: 'c2'})"
    )
    result = await _claim_agent(code).ainvoke(_GO)

    _assert_result_contains(_eval_tool_message(result).content, "missing")
    assert result["claims"] == {"c1": "v1"}


async def test_ptc_concurrent_writes_to_one_value_key_have_one_winner() -> None:
    code = (
        "const settled = await Promise.allSettled([\n"
        "  tools.writeClaim({claim_id: 'c1', value: 'v1'}),\n"
        "  tools.writeClaim({claim_id: 'c2', value: 'v2'}),\n"
        "]);\n"
        "settled.map((s) => s.status).sort().join(',')"
    )
    result = await _claim_agent(code).ainvoke(_GO)

    _assert_result_contains(_eval_tool_message(result).content, "fulfilled,rejected")
    assert result["claims"] in ({"c1": "v1"}, {"c2": "v2"})


@pytest.mark.parametrize(
    ("call", "returned"),
    [
        ("routeClaim", "Command(goto=...)"),
        ("resumeClaim", "Command(graph=..., resume=...)"),
        ("listClaim", "Command(update=...) as list"),
        ("mixedClaim", "Command(goto=...)"),
    ],
    ids=["goto", "graph-resume", "list-update", "mixed-list"],
)
async def test_ptc_unsupported_command_fails_eval_and_writes_nothing(
    call: str, returned: str
) -> None:
    """A `Command` eval cannot apply fails the eval; none of its result is written."""
    result = await _claim_agent(f"await tools.{call}({{}})").ainvoke(_GO)

    content = _eval_tool_message(result).content
    assert '<error type="UnsupportedCommand">' in content, content
    assert f"tools.{call} returned {returned}" in content
    assert "claims" not in result
    assert result["messages"][-1].content == "done"


async def test_ptc_state_updates_apply_once_after_interrupt_and_resume() -> None:
    code = _WRITE_C1 + "await tools.askHuman({})"
    agent = _claim_agent(code, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    await agent.ainvoke(_GO, config=config)
    assert "claims" not in (await agent.aget_state(config)).values
    result = await agent.ainvoke(Command(resume="yes"), config=config)

    _assert_result_contains(_eval_tool_message(result).content, "yes")
    assert result["claims"] == {"c1": "v1"}
