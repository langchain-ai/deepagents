"""Skill tools supplied by a resolver, through `create_deep_agent`.

`GenericFakeChatModel` takes the path for models without mid-conversation tool
changes, so disclosed tools show up in the tools each model call was bound
with. The resolver records every name it is asked for.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware, disclosed_skill_tool_names
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import (
    CREATE_ISSUE,
    LINEAR_PATH,
    LIST_ISSUES,
    SKILLS_SOURCE,
    RecordingResolver,
    ai,
    bound_tool_names,
    call,
    create_issue,
    invoke,
    linear_resolver,
    list_issues,
    logged_create_issue,
    read,
    search_tickets,
    skills_agent,
    skills_backend,
    tool_messages,
    write_skill,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig

    from deepagents.middleware.subagents import SubAgent


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    """Run each test through both `invoke` and `ainvoke`."""
    return request.param


def _model(*turns: AIMessage) -> GenericFakeChatModel:
    """Return a fake model that plays `turns`, then answers "done"."""
    return GenericFakeChatModel(messages=iter([*turns, AIMessage(content="done")]))


def test_one_name_discloses_a_family_of_tools(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="bug")))

    result = invoke(skills_agent(tmp_path, model, skill_tools=linear_resolver()), {"messages": [HumanMessage("file a bug")]}, mode)

    assert not {LIST_ISSUES, CREATE_ISSUE} & set(bound_tool_names(model.call_history[0]))
    assert bound_tool_names(model.call_history[1])[-2:] == [CREATE_ISSUE, LIST_ISSUES]
    [ran] = tool_messages(result, CREATE_ISSUE)
    assert ran.content == "issue bug (c1)"


def _bound(entry: dict[str, Any], name: str) -> list[BaseTool]:
    """Return the `BaseTool`s named `name` that one model call was bound with."""
    return [t for t in entry["tools"] if not isinstance(t, dict) and t.name == name]


def test_exact_names_are_claimed_without_calling_the_resolver(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "search_tickets ls linear")
    resolver = linear_resolver()
    model = _model(ai(read("r1")))

    invoke(skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    [deferred] = _bound(model.call_history[1], "search_tickets")
    assert "defer_loading" not in (deferred.extras or {})
    assert bound_tool_names(model.call_history[1]).count("ls") == 1
    assert resolver.calls == ["linear"]


def test_resolver_returning_a_deferred_request_tool_discloses_it_ungated(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "support")
    resolver = RecordingResolver({"support": [search_tickets]})
    model = _model(ai(call("search_tickets", "s1", query="before")), ai(read("r1")), ai(call("search_tickets", "s2", query="after")))

    result = invoke(skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    [before] = _bound(model.call_history[0], "search_tickets")
    [after] = _bound(model.call_history[2], "search_tickets")
    assert before.extras == {"defer_loading": True}
    assert "defer_loading" not in (after.extras or {})
    assert [m.content for m in tool_messages(result, "search_tickets")] == ["tickets for before", "tickets for after"]
    # One resolution per model call after the read, and none for the registered tool's call.
    assert resolver.calls == ["support", "support"]


class _RecordsDisclosedNames(AgentMiddleware):
    """A gate outside `SkillsMiddleware`, recording which skill tools it would admit at each tool call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[frozenset[str]] = []

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        self.seen.append(disclosed_skill_tool_names(request.state))
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        self.seen.append(disclosed_skill_tool_names(request.state))
        return await handler(request)


@tool("search_tickets")
def other_search_tickets(query: str) -> str:
    """Search tickets somewhere else."""
    return f"other tickets for {query}"


@tool("ls")
def other_ls(path: str) -> str:
    """List files somewhere else."""
    return f"other files in {path}"


def test_resolved_tool_whose_name_is_taken_stands_for_the_request_tool(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "support")
    resolver = RecordingResolver({"support": [other_search_tickets, other_ls, list_issues]})
    gate = _RecordsDisclosedNames()
    model = _model(ai(read("r1")), ai(call("search_tickets", "s1", query="x"), call("ls", "l1", path="/")))
    agent = skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=resolver, middleware=[gate])

    result = invoke(agent, {"messages": [HumanMessage("go")]}, mode)

    # The deferred `search_tickets` is disclosed, and the bound `ls` left alone.
    [disclosed] = _bound(model.call_history[1], "search_tickets")
    assert disclosed.description == search_tickets.description
    assert "defer_loading" not in (disclosed.extras or {})
    assert _bound(model.call_history[1], "ls") == _bound(model.call_history[0], "ls")
    assert bound_tool_names(model.call_history[1])[-1] == LIST_ISSUES
    assert tool_messages(result, "search_tickets")[0].content == "tickets for x"
    assert tool_messages(result, "ls")[0].content == "['/skills/']"
    # At the read, then at each call of the second turn: only `LIST_ISSUES` is gated.
    assert gate.seen == [frozenset(), frozenset({LIST_ISSUES}), frozenset({LIST_ISSUES})]


def _ran_via(calls: list[str], call_id: str) -> str:
    """Return the name the resolver was asked for just before the tool call `call_id` ran."""
    return calls[calls.index(f"ran {call_id}") - 1]


def test_tool_two_names_produce_is_disclosed_once_and_looked_up_by_the_first(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "tracker linear")
    resolver = RecordingResolver({})
    logged = logged_create_issue(resolver.calls)
    resolver.families = {"tracker": [logged], "linear": [list_issues, logged]}
    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x")))

    invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    assert bound_tool_names(model.call_history[1]).count(CREATE_ISSUE) == 1
    assert _ran_via(resolver.calls, "c1") == "tracker"


def test_compacting_the_anchoring_read_moves_the_lookup_to_the_next_producer(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    tracker_path = write_skill(tmp_path, "tracker", "tracker")
    resolver = RecordingResolver({})
    logged = logged_create_issue(resolver.calls)
    resolver.families = {"linear": [logged], "tracker": [logged]}
    model = _model(
        ai(read("r1", path=LINEAR_PATH)),
        ai(read("r2", path=tracker_path)),
        ai(call(CREATE_ISSUE, "c1", title="a")),
        # Seven messages: compaction keeps the second read onward, dropping the first.
        ai(call(CREATE_ISSUE, "c2", title="b")),
    )
    summarization = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter(["summary"] * 10)),
        backend=skills_backend(tmp_path),
        trigger=("messages", 7),
        keep=("messages", 4),
    )

    invoke(skills_agent(tmp_path, model, skill_tools=resolver, middleware=[summarization]), {"messages": [HumanMessage("go")]}, mode)

    assert _ran_via(resolver.calls, "c1") == "linear"
    assert _ran_via(resolver.calls, "c2") == "tracker"


def test_each_name_is_resolved_once_per_model_call_and_only_for_read_skills(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    triage_path = write_skill(tmp_path, "triage", "linear notion")
    write_skill(tmp_path, "docs", "confluence")
    resolver = linear_resolver()
    model = _model(ai(read("r1", path=LINEAR_PATH), read("r2", path=triage_path)), ai(call("ls", "l1", path="/")))

    invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    # Two model calls follow the reads; the registered `ls` call resolves nothing.
    assert resolver.calls == ["linear", "notion", "linear", "notion"]


def test_call_in_the_same_turn_as_the_read_gets_the_invalid_tool_error(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    resolver = linear_resolver()
    model = _model(ai(read("r1", path=LINEAR_PATH), call(CREATE_ISSUE, "c1", title="early")))

    result = invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    [rejected] = tool_messages(result, CREATE_ISSUE)
    assert rejected.content.startswith(f"Error: {CREATE_ISSUE} is not a valid tool")
    assert rejected.status == "error"
    # Only the model call after the read resolved; the rejected call looked nothing up.
    assert resolver.calls == ["linear"]


def test_tool_the_resolver_stops_returning_gets_the_invalid_tool_error(tmp_path: Path, mode: str, caplog: pytest.LogCaptureFixture) -> None:
    write_skill(tmp_path, "linear", "linear")
    calls: list[str] = []

    def forgetful(name: str, _runtime: Runtime[Any]) -> list[BaseTool]:
        calls.append(name)
        return [create_issue] if len(calls) == 1 else []

    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x")))

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware"):
        result = invoke(skills_agent(tmp_path, model, skill_tools=forgetful), {"messages": [HumanMessage("go")]}, mode)

    [rejected] = tool_messages(result, CREATE_ISSUE)
    assert rejected.content.startswith(f"Error: {CREATE_ISSUE} is not a valid tool")
    assert f"Skill tool '{CREATE_ISSUE}' was disclosed via 'linear', but the resolver no longer returns it" in caplog.messages


@dataclass
class _Tenant:
    name: str


def test_resolver_receives_the_graph_runtime_at_model_and_tool_time(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    seen: list[tuple[type, object]] = []

    def per_tenant(_name: str, runtime: Runtime[_Tenant]) -> list[BaseTool]:
        seen.append((type(runtime), runtime.context))
        return [list_issues]

    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(LIST_ISSUES, "c1")))
    agent = skills_agent(tmp_path, model, skill_tools=per_tenant, context_schema=_Tenant)

    result = invoke(agent, {"messages": [HumanMessage("go")]}, mode, context=_Tenant("acme"))

    assert tool_messages(result, LIST_ISSUES)[0].content == "no issues"
    # Model call, tool call, model call.
    assert seen == [(Runtime, _Tenant("acme"))] * 3


def test_tool_runs_when_its_step_resumes_in_a_fresh_agent(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "notion linear")
    checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "fresh"}}
    approval = {CREATE_ISSUE: True}
    first = RecordingResolver({"linear": [list_issues, create_issue]})
    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x")))
    agent = skills_agent(tmp_path, model, skill_tools=first, interrupt_on=approval, checkpointer=checkpointer)

    paused = invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)

    [interrupt] = paused["__interrupt__"]
    assert interrupt.value["action_requests"][0]["name"] == CREATE_ISSUE
    assert not tool_messages(paused, CREATE_ISSUE)

    second = RecordingResolver({})
    second.families = {"linear": [list_issues, logged_create_issue(second.calls)]}
    rebuilt = skills_agent(tmp_path, _model(), skill_tools=second, interrupt_on=approval, checkpointer=checkpointer)
    resumed = invoke(rebuilt, Command(resume={"decisions": [{"type": "approve"}]}), mode, config)

    assert tool_messages(resumed, CREATE_ISSUE)[0].content == "issue x (c1)"
    assert second.calls[:2] == ["linear", "ran c1"]


async def _async_linear(name: str, _runtime: Runtime[Any]) -> list[BaseTool]:
    await asyncio.sleep(0)
    return [list_issues, create_issue] if name == "linear" else []


def test_async_resolver_works_on_the_async_entry_point(tmp_path: Path) -> None:
    write_skill(tmp_path, "linear", "linear")
    model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x")))

    result = invoke(skills_agent(tmp_path, model, skill_tools=_async_linear), {"messages": [HumanMessage("go")]}, "async")

    assert bound_tool_names(model.call_history[1])[-2:] == [CREATE_ISSUE, LIST_ISSUES]
    assert tool_messages(result, CREATE_ISSUE)[0].content == "issue x (c1)"


@contextmanager
def _no_warnings() -> Iterator[None]:
    """Fail if anything inside warns, including a coroutine collected without being awaited."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    assert [str(w.message) for w in caught] == []


def test_async_resolver_on_the_sync_entry_point_raises(tmp_path: Path) -> None:
    write_skill(tmp_path, "linear", "linear")
    agent = skills_agent(tmp_path, _model(ai(read("r1", path=LINEAR_PATH))), skill_tools=_async_linear)
    msg = r"^skill_tools resolver returned an awaitable for 'linear'; an async resolver needs the agent's async entry point \(e\.g\. `ainvoke`\)"

    with _no_warnings(), pytest.raises(TypeError, match=msg):
        invoke(agent, {"messages": [HumanMessage("go")]}, "sync")


def test_resolver_error_at_model_call_time_propagates(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")

    def failing(_name: str, _runtime: Runtime[Any]) -> list[BaseTool]:
        msg = "linear is down"
        raise RuntimeError(msg)

    agent = skills_agent(tmp_path, _model(ai(read("r1", path=LINEAR_PATH))), skill_tools=failing)

    with pytest.raises(RuntimeError, match=r"^linear is down"):
        invoke(agent, {"messages": [HumanMessage("go")]}, mode)


def test_resolver_error_at_tool_time_propagates(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    calls: list[str] = []

    def flaky(name: str, _runtime: Runtime[Any]) -> list[BaseTool]:
        calls.append(name)
        if len(calls) > 1:
            msg = "linear is down"
            raise RuntimeError(msg)
        return [create_issue]

    agent = skills_agent(tmp_path, _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x"))), skill_tools=flaky)

    with pytest.raises(RuntimeError, match=r"^linear is down"):
        invoke(agent, {"messages": [HumanMessage("go")]}, mode)
    assert calls == ["linear", "linear"]


@pytest.mark.parametrize(
    ("output", "msg"),
    [
        pytest.param([{"name": "x"}], r"^skill_tools resolver returned a dict for 'linear'; expected BaseTool instances", id="non-tool-item"),
        pytest.param(list_issues, r"^skill_tools resolver must return a sequence of BaseTool for 'linear', got StructuredTool", id="bare-tool"),
    ],
)
def test_resolver_returning_something_other_than_tools_raises(tmp_path: Path, mode: str, *, output: object, msg: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    agent = skills_agent(tmp_path, _model(ai(read("r1", path=LINEAR_PATH))), skill_tools=lambda _name, _runtime: output)

    with pytest.raises(TypeError, match=msg):
        invoke(agent, {"messages": [HumanMessage("go")]}, mode)


class TestConstruction:
    """`skill_tools` mistakes in the resolver form raise when the agent or middleware is built."""

    def test_a_bare_tool_raises(self) -> None:
        msg = r"^skill_tools must be a list of tools or a resolver function, got StructuredTool; wrap a single tool in a list$"
        with pytest.raises(TypeError, match=msg):
            SkillsMiddleware(backend=StateBackend(), sources=[SKILLS_SOURCE], skill_tools=create_issue)  # ty: ignore[invalid-argument-type]


def _task(subagent_type: str, call_id: str = "t1") -> AIMessage:
    return ai(call("task", call_id, description="file an issue", subagent_type=subagent_type))


class TestSubagents:
    """General-purpose subagents and forks resolve through the parent's resolver; declarative ones through their own."""

    def test_general_purpose_subagent_inherits_the_resolver(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "linear", "linear")
        resolver = linear_resolver()
        # Parent and general-purpose subagent share the model, so turns interleave.
        model = _model(
            _task("general-purpose"),
            ai(read("r1", path=LINEAR_PATH)),
            ai(call(CREATE_ISSUE, "c1", title="x")),
            AIMessage(content="subagent done"),
        )

        invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

        assert model.call_history[3]["messages"][-1].content == "issue x (c1)"
        assert "linear" in resolver.calls

    def test_fork_resolves_through_the_parents_resolver(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "linear", "linear")
        resolver = linear_resolver()
        worker_model = _model(ai(call(CREATE_ISSUE, "c1", title="x")))
        worker: SubAgent = {"name": "worker", "description": "Continues.", "model": worker_model, "mode": "fork"}
        agent = skills_agent(tmp_path, _model(ai(read("r1", path=LINEAR_PATH)), _task("worker")), skill_tools=resolver, subagents=[worker])

        invoke(agent, {"messages": [HumanMessage("go")]}, mode)

        assert CREATE_ISSUE in bound_tool_names(worker_model.call_history[0])
        assert worker_model.call_history[1]["messages"][-1].content == "issue x (c1)"

    def test_declarative_subagent_resolves_only_through_its_own(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "linear", "linear")
        parent = linear_resolver()
        own = RecordingResolver({"linear": [list_issues]})
        worker_model = _model(ai(read("r1", path=LINEAR_PATH)), ai(call(CREATE_ISSUE, "c1", title="x")))
        worker: SubAgent = {
            "name": "worker",
            "description": "d",
            "model": worker_model,
            "skills": [SKILLS_SOURCE],
            "middleware": [SkillsMiddleware(backend=skills_backend(tmp_path), sources=[SKILLS_SOURCE], skill_tools=own)],
        }

        invoke(skills_agent(tmp_path, _model(_task("worker")), skill_tools=parent, subagents=[worker]), {"messages": [HumanMessage("go")]}, mode)

        assert bound_tool_names(worker_model.call_history[1])[-1] == LIST_ISSUES
        assert CREATE_ISSUE not in bound_tool_names(worker_model.call_history[1])
        assert "is not a valid tool" in worker_model.call_history[2]["messages"][-1].content
        assert own.calls
        assert parent.calls == []


def test_other_gates_see_exactly_the_skill_tools_the_latest_model_call_was_shown(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear search_tickets")
    gate = _RecordsDisclosedNames()
    model = _model(
        ai(call("ls", "l1", path="/")),
        ai(read("r1", path=LINEAR_PATH)),
        ai(call("ls", "l2", path="/")),
        # Seven messages: compaction keeps only the l2 exchange, dropping the read.
        ai(call("ls", "l3", path="/")),
    )
    summarization = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter(["summary"] * 10)),
        backend=skills_backend(tmp_path),
        trigger=("messages", 7),
        keep=("messages", 2),
    )
    agent = skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=linear_resolver(), middleware=[gate, summarization])

    invoke(agent, {"messages": [HumanMessage("go")]}, mode)

    # At `ls`, at the read itself, at `ls` after it (the deferred `search_tickets` isn't gated), and at `ls` after compaction.
    assert gate.seen == [frozenset(), frozenset(), frozenset({CREATE_ISSUE, LIST_ISSUES}), frozenset()]


def test_async_resolver_resolves_names_concurrently(tmp_path: Path) -> None:
    write_skill(tmp_path, "linear", "linear")
    triage_path = write_skill(tmp_path, "triage", "notion")
    both_in_flight = asyncio.Barrier(2)

    async def rendezvous(name: str, _runtime: Runtime[Any]) -> list[BaseTool]:
        # Returns only once both names are in flight, so resolving one at a time times out.
        async with asyncio.timeout(1):
            await both_in_flight.wait()
        return [list_issues] if name == "linear" else []

    model = _model(ai(read("r1", path=LINEAR_PATH), read("r2", path=triage_path)))

    invoke(skills_agent(tmp_path, model, skill_tools=rendezvous), {"messages": [HumanMessage("go")]}, "async")

    assert bound_tool_names(model.call_history[1])[-1] == LIST_ISSUES
