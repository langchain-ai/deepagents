"""Human approval for `tools.*` calls made from eval code."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """Build an agent whose model runs one turn of evals per argument.

    A tuple is several evals in one model message. A `resuming` agent's model
    has already issued its first turn.
    """
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
        tools=[danger, note, slow, ask, search, task],
        middleware=[
            CodeInterpreterMiddleware(ptc=["danger", "note", "slow", "ask", "search"]),
            HumanInTheLoopMiddleware(
                interrupt_on={"danger": True}, interrupt_mode="per_call"
            ),
        ],
        checkpointer=checkpointer or InMemorySaver(),
    )


async def _run(
    agent: Any,
    answer: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    invoke_mode: Literal["invoke", "ainvoke"] = "ainvoke",
    max_pauses: int = 5,
) -> tuple[Any, list[list[dict[str, Any]]]]:
    """Run to completion, answering every pause; return the output and pauses."""
    config = {"configurable": {"thread_id": "t"}}

    async def step(payload: Any) -> Any:
        if invoke_mode == "ainvoke":
            return await agent.ainvoke(payload, config, version="v2")
        return agent.invoke(payload, config, version="v2")

    out = await step({"messages": [HumanMessage("go")]})
    pauses = []
    while out.interrupts and len(pauses) < max_pauses:
        pauses.append([i.value for i in out.interrupts])
        out = await step(
            Command(resume={i.id: answer(i.value) for i in out.interrupts})
        )
    return out, pauses


def _eval_result(out: Any) -> str:
    return next(m.text for m in out.value["messages"] if isinstance(m, ToolMessage))


def _eval_results(out: Any) -> list[str]:
    return [m.text for m in out.value["messages"] if isinstance(m, ToolMessage)]


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
async def test_gated_call_from_code_pauses_and_runs_once_after_approval(
    invoke_mode: Literal["invoke", "ainvoke"],
) -> None:
    agent = _agent("await tools.note({x: 'a'}); await tools.danger({x: 'b'})")
    out, pauses = await _run(agent, lambda _: APPROVE, invoke_mode=invoke_mode)

    assert [[(v["name"], v["args"], v["tool_call_id"]) for v in p] for p in pauses] == [
        [("danger", {"x": "b"}, "call_1.ptc1")]
    ]
    assert CALLS == [("note", "a"), ("danger", "b")]
    assert _eval_result(out) == "<result>danger:b</result>"


async def test_rejected_call_throws_in_js_and_does_not_run() -> None:
    code = (
        "try { await tools.danger({x: 'b'}) } catch (e) { e.name + ': ' + e.message }"
    )
    out, _ = await _run(_agent(code), lambda _: {"type": "reject", "message": "no"})

    assert CALLS == []
    assert _eval_result(out) == (
        "<result>ToolError: User rejected the tool call for `danger` "
        "with reason: no</result>"
    )


async def test_edited_call_runs_with_edited_args_and_notes_the_edit() -> None:
    def edit(_: dict[str, Any]) -> dict[str, Any]:
        return {"type": "edit", "edited_action": {"name": "danger", "args": {"x": "c"}}}

    out, _ = await _run(_agent("await tools.danger({x: 'b'})"), edit)

    assert CALLS == [("danger", "c")]
    result = _eval_result(out)
    assert result.startswith("<result>danger:c</result>\n<notes>\ntools.danger call 0:")
    assert 'Executed instead: danger with arguments {"x": "c"}' in result


async def test_catching_the_pause_does_not_skip_review() -> None:
    code = """
    let r;
    try { r = await tools.danger({x: 'b'}) } catch (e) { r = 'caught' }
    await tools.note({x: r});
    r
    """
    out, pauses = await _run(_agent(code), lambda _: APPROVE)

    assert len(pauses) == 1
    assert CALLS == [("danger", "b"), ("note", "danger:b")]
    assert _eval_result(out) == "<result>danger:b</result>"


async def test_concurrent_gated_calls_pause_together_in_issue_order() -> None:
    code = (
        "(await Promise.all([tools.danger({x: 'a'}), tools.danger({x: 'b'})])).join()"
    )
    out, pauses = await _run(_agent(code), lambda _: APPROVE)

    assert [[v["args"]["x"] for v in p] for p in pauses] == [["a", "b"]]
    assert sorted(CALLS) == [("danger", "a"), ("danger", "b")]
    assert _eval_result(out) == "<result>danger:a,danger:b</result>"


async def test_heap_changes_from_a_paused_run_are_rolled_back() -> None:
    code = "globalThis.n = (globalThis.n ?? 0) + 1; await tools.danger({x: 'b'}); n"
    out, pauses = await _run(_agent(code), lambda _: APPROVE)

    assert len(pauses) == 1
    assert _eval_result(out) == "<result>1</result>"


async def test_decision_is_not_applied_to_a_call_with_different_args() -> None:
    # The wall clock moves between runs, so each re-run issues a new call.
    def approve_later(_: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.01)
        return APPROVE

    code = "await tools.danger({x: String(Date.now())})"
    out, pauses = await _run(_agent(code), approve_later, max_pauses=3)

    assert out.interrupts
    assert len({p[0]["args"]["x"] for p in pauses}) == 3
    assert CALLS == []


async def test_rebuilt_agent_resumes_without_rerunning_finished_calls() -> None:
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    code = "await tools.note({x: 'a'}); await tools.danger({x: 'b'})"
    paused = await _agent(code, checkpointer=checkpointer).ainvoke(
        {"messages": [HumanMessage("go")]}, config, version="v2"
    )
    (interrupt,) = paused.interrupts

    out = await _agent(code, checkpointer=checkpointer, resuming=True).ainvoke(
        Command(resume={interrupt.id: APPROVE}), config, version="v2"
    )

    assert CALLS == [("note", "a"), ("danger", "b")]
    assert _eval_result(out) == "<result>danger:b</result>"


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
async def test_overlapping_evals_report_concurrent_eval_instead_of_crashing(
    invoke_mode: Literal["invoke", "ainvoke"],
) -> None:
    out, _ = await _run(
        _agent(("await tools.slow({x: 'a'}); 'A'", "'B'")),
        lambda _: APPROVE,
        invoke_mode=invoke_mode,
    )

    first, second = _eval_results(out)
    assert first == "<result>A</result>"
    assert second == "<result>B</result>" or second.startswith(
        '<error type="ConcurrentEval">'
    )


async def test_eval_runs_while_an_abandoned_call_is_still_pending() -> None:
    out, _ = await _run(
        _agent("tools.slow({x: 's'}); null.boom", "1 + 1"), lambda _: APPROVE
    )

    first, second = _eval_results(out)
    assert first.startswith('<error type="TypeError">')
    assert second == "<result>2</result>"


async def test_interrupts_inside_a_tool_keep_their_own_answers() -> None:
    answers = iter(["yes1", "yes2"])
    code = (
        "const a = await tools.ask({question: 'q1'});"
        " const b = await tools.ask({question: 'q2'}); a + '|' + b"
    )
    out, pauses = await _run(_agent(code), lambda _: next(answers))

    assert [p[0]["question"] for p in pauses] == ["q1", "q2"]
    assert _eval_result(out) == "<result>answer:yes1|answer:yes2</result>"


async def test_subagent_dispatch_after_a_caught_pause_does_not_run() -> None:
    code = """
    let r;
    try { r = await tools.danger({x: 'b'}) } catch (e) { r = 'caught' }
    await task({description: 'after-' + r, subagentType: 'general-purpose'})
    """
    out, _ = await _run(_agent(code), lambda _: APPROVE)

    assert CALLS == [("danger", "b"), ("subagent", "after-danger:b")]
    assert _eval_result(out) == "<result>sub:after-danger:b</result>"


async def test_content_and_artifact_tool_returns_its_content() -> None:
    out, _ = await _run(_agent("await tools.search({query: 'x'})"), lambda _: APPROVE)

    assert _eval_result(out) == "<result>found x</result>"
