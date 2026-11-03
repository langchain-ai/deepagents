"""Human approval for `tools.*` calls made from eval code."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel, answer_pauses

CALLS: list[tuple[str, str]] = []

APPROVE = {"type": "approve"}


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


@tool
def danger(x: str) -> str:
    """Do something that needs approval."""
    CALLS.append(("danger", x))
    return f"danger:{x}"


@tool
def note(x: str) -> str:
    """Record something without approval."""
    CALLS.append(("note", x))
    return f"note:{x}"


@tool
def flaky(x: str) -> str:
    """Fail every time it runs."""
    CALLS.append(("flaky", x))
    msg = "flaky failed"
    raise RuntimeError(msg)


@tool
def cyclic(x: str) -> dict[str, Any]:
    """Return a result that cannot be handed to code."""
    CALLS.append(("cyclic", x))
    result: dict[str, Any] = {}
    result["self"] = result
    return result


@tool
async def slow(x: str) -> str:
    """Take a while."""
    await asyncio.sleep(0.2)
    return f"slow:{x}"


@tool
def ask(question: str) -> str:
    """Ask a human directly."""
    return f"answer:{interrupt({'question': question})}"


@tool(response_format="content_and_artifact")
def search(query: str) -> tuple[str, dict[str, Any]]:
    """Search, returning content and an artifact."""
    return f"found {query}", {"docs": [1, 2, 3]}


@tool
def write_then_read(path: str) -> str:
    """Write a file in graph state, then read it back in the same call."""
    files = StateBackend()
    files.write(path, "mine")
    return files.read(path).file_data["content"]


@tool
async def task(description: str, subagent_type: str) -> str:
    """Stand in for the Deep Agents subagent tool behind JS `task()`."""
    del subagent_type
    CALLS.append(("subagent", description))
    return f"sub:{description}"


def _agent(
    *turns: str | tuple[str, ...],
    checkpointer: InMemorySaver | None = None,
    resuming: bool = False,
) -> Any:
    """Build an agent whose model runs one turn (an eval or a tuple of them) per arg."""
    script = []
    ids = iter(range(1, 100))
    for turn in turns:
        codes = (turn,) if isinstance(turn, str) else turn
        calls = [
            {"name": "eval", "args": {"code": c}, "id": f"call_{next(ids)}"}
            for c in codes
        ]
        script.append(AIMessage(content="", tool_calls=calls))
    script.append(AIMessage("done"))
    return create_agent(
        model=FakeChatModel(messages=iter(script[1:] if resuming else script)),
        tools=[danger, note, flaky, cyclic, slow, ask, search, write_then_read, task],
        middleware=[
            CodeInterpreterMiddleware(
                ptc=[
                    *("danger", "note", "flaky", "cyclic", "slow", "ask", "search"),
                    *("write_then_read", "write_file", "read_file"),
                ]
            ),
            FilesystemMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on={"danger": True}, interrupt_mode="per_call"
            ),
        ],
        checkpointer=checkpointer or InMemorySaver(),
    )


def _eval_results(out: Any) -> list[str]:
    return [m.text for m in out.value["messages"] if isinstance(m, ToolMessage)]


async def test_gated_call_pauses_then_runs_once_in_a_rebuilt_agent() -> None:
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    code = "await tools.note({x: 'a'}); await tools.danger({x: 'b'})"
    paused = await _agent(code, checkpointer=checkpointer).ainvoke(
        {"messages": [HumanMessage("go")]}, config, version="v2"
    )
    (interrupt,) = paused.interrupts
    assert interrupt.value["name"] == "danger"
    assert interrupt.value["args"] == {"x": "b"}
    assert interrupt.value["tool_call_id"] == "call_1.ptc1"

    out = await _agent(code, checkpointer=checkpointer, resuming=True).ainvoke(
        Command(resume={interrupt.id: APPROVE}), config, version="v2"
    )

    assert CALLS == [("note", "a"), ("danger", "b")]
    assert _eval_results(out) == ["<result>danger:b</result>"]


async def test_rejected_call_throws_in_js_and_does_not_run() -> None:
    code = (
        "try { await tools.danger({x: 'b'}) }"
        " catch (e) { `${e.name} ${e.code}: ${e.message}` }"
    )
    out, _ = await answer_pauses(
        _agent(code), lambda _: {"type": "reject", "message": "no"}
    )

    assert CALLS == []
    assert _eval_results(out) == [
        "<result>ToolError EACCES: User rejected the tool call for `danger` "
        "with reason: no</result>"
    ]


async def test_edited_call_runs_with_edited_args_and_notes_the_edit() -> None:
    def edit(_: dict[str, Any]) -> dict[str, Any]:
        return {"type": "edit", "edited_action": {"name": "danger", "args": {"x": "c"}}}

    out, _ = await answer_pauses(_agent("await tools.danger({x: 'b'})"), edit)

    assert CALLS == [("danger", "c")]
    (result,) = _eval_results(out)
    assert result.startswith("<result>danger:c</result>\n<notes>\ntools.danger call 0:")
    assert 'Executed instead: danger with arguments {"x": "c"}' in result


@pytest.mark.parametrize(
    ("code", "paused", "calls", "result"),
    [
        pytest.param(
            "let r; try { r = await tools.danger({x: 'b'}) } catch (e) { r = 'caught' }"
            " await tools.note({x: r}); r",
            1,
            [("danger", "b"), ("note", "danger:b")],
            "danger:b",
            id="catching-the-pause-does-not-skip-review",
        ),
        pytest.param(
            "let r; try { r = await tools.danger({x: 'b'}) } catch (e) { r = 'caught' }"
            " await task({description: 'after-' + r, subagentType: 'general-purpose'})",
            1,
            [("danger", "b"), ("subagent", "after-danger:b")],
            "sub:after-danger:b",
            id="subagent-dispatch-after-a-caught-pause-does-not-run",
        ),
        pytest.param(
            "globalThis.n = (globalThis.n ?? 0) + 1; await tools.danger({x: 'b'}); n",
            1,
            [("danger", "b")],
            "1",
            id="heap-changes-from-a-paused-run-are-rolled-back",
        ),
        pytest.param(
            "await tools.search({query: 'x'})",
            0,
            [],
            "found x",
            id="content-and-artifact-tool-returns-its-content",
        ),
    ],
)
async def test_approved_eval_result(
    code: str, paused: int, calls: list[tuple[str, str]], result: str
) -> None:
    out, pauses = await answer_pauses(_agent(code), lambda _: APPROVE)

    assert len(pauses) == paused
    assert calls == CALLS
    assert _eval_results(out) == [f"<result>{result}</result>"]


async def test_concurrent_gated_calls_pause_together_in_issue_order() -> None:
    code = (
        "(await Promise.all([tools.danger({x: 'a'}), tools.danger({x: 'b'})])).join()"
    )
    out, pauses = await answer_pauses(_agent(code), lambda _: APPROVE)

    assert [[v["args"]["x"] for v in p] for p in pauses] == [["a", "b"]]
    assert sorted(CALLS) == [("danger", "a"), ("danger", "b")]
    assert _eval_results(out) == ["<result>danger:a,danger:b</result>"]


async def test_decision_is_not_applied_to_a_call_with_different_args() -> None:
    # The wall clock moves between runs, so each re-run issues a new call.
    def approve_later(_: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.01)
        return APPROVE

    code = "await tools.danger({x: String(Date.now())})"
    out, pauses = await answer_pauses(_agent(code), approve_later, max_pauses=3)

    assert out.interrupts
    assert len({p[0]["args"]["x"] for p in pauses}) == 3
    assert CALLS == []


@pytest.mark.xfail(
    strict=True,
    reason="LangGraph gives a resume without an interrupt id to whichever "
    "interrupt asks first on the re-run. Model tool calls never hit this because "
    "their args are fixed in state; a re-run of eval code can issue new args",
)
async def test_bare_resume_is_not_applied_to_a_call_with_different_args() -> None:
    config = {"configurable": {"thread_id": "t"}}
    agent = _agent("await tools.danger({x: String(Date.now())})")
    paused = await agent.ainvoke(
        {"messages": [HumanMessage("go")]}, config, version="v2"
    )
    assert paused.interrupts
    await asyncio.sleep(0.01)

    out = await agent.ainvoke(Command(resume=APPROVE), config, version="v2")

    assert out.interrupts
    assert CALLS == []


@pytest.mark.parametrize("durability", ["async", "exit"])
@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
@pytest.mark.parametrize("name", ["flaky", "cyclic"])
async def test_call_that_raised_before_a_pause_does_not_run_again(
    name: str,
    invoke_mode: Literal["invoke", "ainvoke"],
    durability: Literal["async", "exit"],
) -> None:
    code = f"""
    let r;
    try {{ await tools.{name}({{x: 'a'}}) }} catch (e) {{ r = e.name }}
    await tools.danger({{x: r}});
    r
    """
    out, pauses = await answer_pauses(
        _agent(code), lambda _: APPROVE, invoke_mode=invoke_mode, durability=durability
    )

    assert len(pauses) == 1
    assert [(name, "a"), ("danger", "HostError")] == CALLS
    assert _eval_results(out) == ["<result>HostError</result>"]


@pytest.mark.parametrize("durability", ["async", "exit"])
@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
async def test_state_backend_writes_survive_a_pause_and_are_read_back(
    invoke_mode: Literal["invoke", "ainvoke"], durability: Literal["async", "exit"]
) -> None:
    code = """
    await tools.writeFile({file_path: '/a.txt', content: 'hello'});
    await tools.danger({x: 'b'});
    await tools.writeFile({file_path: '/c.txt', content: 'after'});
    const own = await tools.writeThenRead({path: '/own.txt'});
    own + '|' + await tools.readFile({file_path: '/a.txt'})
    """
    out, pauses = await answer_pauses(
        _agent(code), lambda _: APPROVE, invoke_mode=invoke_mode, durability=durability
    )

    assert len(pauses) == 1
    (result,) = _eval_results(out)
    assert result.startswith("<result>mine|")
    assert "hello" in result
    assert out.value["files"]["/a.txt"]["content"] == "hello"
    assert out.value["files"]["/c.txt"]["content"] == "after"


async def test_interrupts_inside_a_tool_keep_their_own_answers() -> None:
    answers = iter(["yes1", "yes2"])
    code = (
        "const a = await tools.ask({question: 'q1'});"
        " const b = await tools.ask({question: 'q2'}); a + '|' + b"
    )
    out, pauses = await answer_pauses(_agent(code), lambda _: next(answers))

    assert [p[0]["question"] for p in pauses] == ["q1", "q2"]
    assert _eval_results(out) == ["<result>answer:yes1|answer:yes2</result>"]


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
async def test_evals_run_while_another_eval_or_an_abandoned_call_is_in_flight(
    invoke_mode: Literal["invoke", "ainvoke"],
) -> None:
    out, _ = await answer_pauses(
        _agent(
            ("await tools.slow({x: 'a'}); 'A'", "'B'"),
            "tools.slow({x: 's'}); null.boom",
            "1 + 1",
        ),
        lambda _: APPROVE,
        invoke_mode=invoke_mode,
    )

    overlapping, parallel, abandoning, after = _eval_results(out)
    assert overlapping == "<result>A</result>"
    assert parallel == "<result>B</result>" or parallel.startswith(
        '<error type="ConcurrentEval">'
    )
    assert abandoning.startswith('<error type="TypeError">')
    assert after == "<result>2</result>"
