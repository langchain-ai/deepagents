"""Skill tool disclosure through `create_deep_agent`, observed at the model's call history.

`GenericFakeChatModel` is neither `ChatAnthropic` nor `ChatOpenAI`, so these
tests exercise the path for models without mid-conversation tool changes,
where disclosed tools are appended to `tools`. The gate is the same on every
path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents.backends.state import StateBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.profiles import HarnessProfile, register_harness_profile
from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import (
    SKILLS_SOURCE,
    ai,
    bound_tool_names,
    call,
    create_customer_request,
    invoke,
    list_customer_requests,
    read,
    search_tickets,
    skill_md,
    skills_agent,
    skills_backend,
    tool_messages,
    write_skill,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from langchain_core.messages import ToolCall
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.middleware.subagents import SubAgent


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    """Run each test through both `invoke` and `ainvoke`."""
    return request.param


def _model(*turns: AIMessage) -> GenericFakeChatModel:
    """Return a fake model that plays `turns`, then answers "done"."""
    return GenericFakeChatModel(messages=iter([*turns, AIMessage(content="done")]))


def test_skill_tool_is_bound_only_after_its_skill_is_read(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="refund")))

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("file a refund request")]}, mode)

    assert "create_customer_request" not in bound_tool_names(model.call_history[0])
    assert bound_tool_names(model.call_history[1])[-1] == "create_customer_request"
    [message] = tool_messages(result, "create_customer_request")
    assert message.content == "created refund (c1)"


def test_call_in_the_same_turn_as_the_read_is_rejected(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(
        ai(read("r1"), call("create_customer_request", "c1", title="early")),
        ai(call("create_customer_request", "c2", title="late")),
    )

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    rejected, ran = tool_messages(result, "create_customer_request")
    assert rejected.content == "Error: create_customer_request is not available yet. Read the crm skill (/skills/crm/SKILL.md) to make it available."
    assert rejected.status == "error"
    assert rejected.tool_call_id == "c1"
    assert ran.content == "created late (c2)"


def test_call_before_any_read_is_rejected_without_running(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    write_skill(tmp_path, "support", "create_customer_request")
    model = _model(ai(call("create_customer_request", "c1", title="x")))

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    [rejected] = tool_messages(result, "create_customer_request")
    assert rejected.content == (
        "Error: create_customer_request is not available yet. Read one of these skills to make it available: "
        "crm (/skills/crm/SKILL.md), support (/skills/support/SKILL.md)."
    )
    assert rejected.status == "error"
    assert "create_customer_request" not in bound_tool_names(model.call_history[1])


def test_unknown_tool_error_never_lists_skill_tools(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(ai(read("r1")), ai(call("no_such_tool", "c1")))

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    [error] = tool_messages(result, "no_such_tool")
    assert error.status == "error"
    assert "read_file" in error.content
    assert "create_customer_request" not in error.content


def test_skill_tool_no_skill_names_gets_the_invalid_tool_error(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "list_customer_requests")
    model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="x")))

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    [error] = tool_messages(result, "create_customer_request")
    assert error.status == "error"
    assert "is not a valid tool" in error.content
    assert "not available yet" not in error.content


def _compacting(tmp_path: Path, trigger: int) -> SummarizationMiddleware:
    """Summarize once the effective conversation reaches `trigger` messages, keeping the last two."""
    summary_model = GenericFakeChatModel(messages=iter(["summary"] * 10))
    return SummarizationMiddleware(model=summary_model, backend=skills_backend(tmp_path), trigger=("messages", trigger), keep=("messages", 2))


def test_compaction_that_drops_the_read_withdraws_the_tool(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(
        ai(read("r1")),
        ai(call("create_customer_request", "c1", title="a")),
        # Five messages: compaction keeps only the c1 exchange, dropping the read.
        ai(call("create_customer_request", "c2", title="b")),
    )
    agent = skills_agent(tmp_path, model, middleware=[_compacting(tmp_path, trigger=5)], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "withdrawal"}}

    result = invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)

    assert "create_customer_request" in bound_tool_names(model.call_history[1])
    assert "create_customer_request" not in bound_tool_names(model.call_history[2])
    ran, rejected = tool_messages(result, "create_customer_request")
    assert ran.content == "created a (c1)"
    assert rejected.content.startswith("Error: create_customer_request is not available yet.")
    assert agent.get_state(config).values["_skill_tools_disclosed"] == []


def test_disclosed_record_follows_each_model_call(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request list_customer_requests")
    model = _model(ai(read("r1")), AIMessage(content="first run done"))
    agent = skills_agent(tmp_path, model, skill_tools=[create_customer_request, list_customer_requests], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "record"}}

    invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)

    assert agent.get_state(config).values["_skill_tools_disclosed"] == ["create_customer_request", "list_customer_requests"]
    assert "_skill_tools_disclosed" not in invoke(agent, {"messages": [HumanMessage("again")]}, mode, config)


def _discloses_after(tmp_path: Path, mode: str, history: list[AIMessage | ToolMessage]) -> bool:
    """Return whether one model call over `history` is shown `create_customer_request`."""
    model = _model()
    invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go"), *history]}, mode)
    return "create_customer_request" in bound_tool_names(model.call_history[0])


@pytest.mark.parametrize(
    ("read_call", "status", "expected"),
    [
        pytest.param(read("r1"), "success", True, id="plain-read"),
        pytest.param(read("r1", offset=10, limit=5), "success", True, id="offset-and-limit"),
        pytest.param(read("r1", path="/skills//crm/./SKILL.md"), "success", True, id="path-normalizes-to-skill"),
        pytest.param(read("r1"), "error", False, id="failed-read"),
        pytest.param(read("r1", path="/skills/crm/notes.md"), "success", False, id="other-file"),
        pytest.param(read("r1", path="/skills/../skills/crm/SKILL.md"), "success", False, id="invalid-path"),
        pytest.param(call("ls", "r1", path="/skills/crm/SKILL.md"), "success", False, id="not-read-file"),
    ],
)
def test_what_counts_as_a_skill_load(tmp_path: Path, mode: str, *, read_call: ToolCall, status: str, expected: bool) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    history = [ai(read_call), ToolMessage("# crm", tool_call_id="r1", name=read_call["name"], status=status)]

    assert _discloses_after(tmp_path, mode, history) is expected


class _Script:
    """Model turns that may include an exception to raise in place of a reply."""

    def __init__(self, *turns: AIMessage | Exception) -> None:
        self._turns = list(turns)

    def __iter__(self) -> _Script:
        return self

    def __next__(self) -> AIMessage:
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def test_read_clipped_by_the_overflow_retry_still_counts(tmp_path: Path, mode: str) -> None:
    body = "\n".join(f"Step {i}: " + "x" * 100 for i in range(60))
    write_skill(tmp_path, "crm", content=skill_md("crm", "create_customer_request") + body)
    model = GenericFakeChatModel(
        messages=_Script(
            ai(read("r1", limit=1000)),
            ContextOverflowError("prompt is too long"),
            ai(call("create_customer_request", "c1", title="x")),
            AIMessage(content="done"),
        )
    )

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    clipped_read = model.call_history[2]["messages"][-1]
    assert "Output was truncated due to context window size limits" in clipped_read.text
    assert "create_customer_request" in bound_tool_names(model.call_history[2])
    [ran] = tool_messages(result, "create_customer_request")
    assert ran.content == "created x (c1)"


def _bound(entry: dict[str, Any], name: str) -> list[BaseTool]:
    return [t for t in entry["tools"] if not isinstance(t, dict) and t.name == name]


def test_deferred_tool_named_by_a_skill_is_undeferred_but_never_gated(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "search_tickets")
    model = _model(ai(call("search_tickets", "s1", query="before")), ai(read("r1")))

    result = invoke(skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=None), {"messages": [HumanMessage("go")]}, mode)

    [before] = _bound(model.call_history[0], "search_tickets")
    [after] = _bound(model.call_history[2], "search_tickets")
    assert before.extras == {"defer_loading": True}
    assert "defer_loading" not in (after.extras or {})
    assert tool_messages(result, "search_tickets")[0].content == "tickets for before"


def test_skill_naming_an_already_bound_tool_changes_nothing(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "ls")
    model = _model(ai(read("r1")))

    invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    assert bound_tool_names(model.call_history[1]) == bound_tool_names(model.call_history[0])


@tool("create_customer_request")
def registered_create_customer_request(title: str) -> str:
    """Create a customer request (registered)."""
    return f"registered {title}"


def test_registered_tool_wins_over_a_skill_tool_of_the_same_name(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(ai(call("create_customer_request", "c1", title="x")), ai(read("r1")))

    result = invoke(skills_agent(tmp_path, model, tools=[registered_create_customer_request]), {"messages": [HumanMessage("go")]}, mode)

    [ran] = tool_messages(result, "create_customer_request")
    assert ran.content == "registered x"
    assert bound_tool_names(model.call_history[2]).count("create_customer_request") == 1


class _AddsCreateCustomerRequest(AgentMiddleware):
    """Exposes and runs its own `create_customer_request`, as a dynamic-tool middleware would."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(tools=[*request.tools, registered_create_customer_request]))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(request.override(tools=[*request.tools, registered_create_customer_request]))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        if request.tool_call["name"] == "create_customer_request":
            request = request.override(tool=registered_create_customer_request)
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        if request.tool_call["name"] == "create_customer_request":
            request = request.override(tool=registered_create_customer_request)
        return await handler(request)


def test_tool_another_middleware_exposes_wins(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="x")))

    result = invoke(skills_agent(tmp_path, model, middleware=[_AddsCreateCustomerRequest()]), {"messages": [HumanMessage("go")]}, mode)

    assert _bound(model.call_history[1], "create_customer_request") == [registered_create_customer_request]
    [ran] = tool_messages(result, "create_customer_request")
    assert ran.content == "registered x"


def test_unresolved_names_and_unreferenced_skill_tools_log_at_debug(tmp_path: Path, mode: str, caplog: pytest.LogCaptureFixture) -> None:
    write_skill(tmp_path, "crm", "create_customer_request missing_tool")
    model = _model(ai(read("r1")))
    agent = skills_agent(tmp_path, model, skill_tools=[create_customer_request, list_customer_requests])

    with caplog.at_level(logging.DEBUG, logger="deepagents.middleware"):
        invoke(agent, {"messages": [HumanMessage("go")]}, mode)

    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG and r.name.startswith("deepagents.middleware")]
    assert "Skill 'crm' names tool 'missing_tool', which is not available in this request" in debug
    assert debug.count("Skill tool 'list_customer_requests' is not named by any loaded skill") == 1
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG and "missing_tool" in r.getMessage()]


def test_interrupt_on_applies_to_a_disclosed_skill_tool(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="x")))
    agent = skills_agent(tmp_path, model, interrupt_on={"create_customer_request": True}, checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "hitl"}}

    paused = invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)
    [interrupt] = paused["__interrupt__"]
    assert interrupt.value["action_requests"][0]["name"] == "create_customer_request"
    assert not tool_messages(paused, "create_customer_request")

    resumed = invoke(agent, Command(resume={"decisions": [{"type": "approve"}]}), mode, config)

    [ran] = tool_messages(resumed, "create_customer_request")
    assert ran.content == "created x (c1)"


def test_include_tools_splits_on_any_whitespace(tmp_path: Path, mode: str) -> None:
    content = '---\nname: crm\ndescription: CRM\nmetadata:\n  include_tools: "create_customer_request\\n\\t list_customer_requests"\n---\n'
    write_skill(tmp_path, "crm", content=content)
    model = _model(ai(read("r1")))

    invoke(skills_agent(tmp_path, model, skill_tools=[create_customer_request, list_customer_requests]), {"messages": [HumanMessage("go")]}, mode)

    assert bound_tool_names(model.call_history[1])[-2:] == ["create_customer_request", "list_customer_requests"]


def test_include_tools_written_as_a_yaml_list_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    content = "---\nname: crm\ndescription: CRM\nmetadata:\n  include_tools: [create_customer_request, list_customer_requests]\n---\n"
    write_skill(tmp_path, "crm", content=content)
    model = _model()

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware.skills"):
        result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, "sync")

    assert (
        "Skill 'crm' (/skills/crm/SKILL.md): metadata.include_tools should be a space-separated string of tool names; "
        "got \"['create_customer_request', 'list_customer_requests']\""
    ) in caplog.messages
    assert "include_tools" not in model.call_history[0]["messages"][0].text
    assert result["messages"][-1].content == "done"


class TestConstruction:
    """`skill_tools` mistakes raise when the agent or middleware is built."""

    def test_duplicate_names_raise(self) -> None:
        with pytest.raises(ValueError, match=r"^skill_tools contains duplicate tool name\(s\): create_customer_request$"):
            create_deep_agent(model=_model(), skills=[SKILLS_SOURCE], skill_tools=[create_customer_request, registered_create_customer_request])
        with pytest.raises(ValueError, match=r"^skill_tools contains duplicate tool name\(s\): create_customer_request$"):
            SkillsMiddleware(backend=StateBackend(), sources=[SKILLS_SOURCE], skill_tools=[create_customer_request, create_customer_request])

    def test_dict_entries_raise(self) -> None:
        msg = r"^skill_tools entries must be BaseTool instances or callables; provider-native tool dicts are not supported$"
        web_search = {"type": "web_search_20250305", "name": "web_search"}
        with pytest.raises(TypeError, match=msg):
            create_deep_agent(model=_model(), skills=[SKILLS_SOURCE], skill_tools=[web_search])  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=msg):
            SkillsMiddleware(backend=StateBackend(), sources=[SKILLS_SOURCE], skill_tools=[web_search])  # ty: ignore[invalid-argument-type]

    def test_skill_tools_without_skills_raise(self) -> None:
        with pytest.raises(ValueError, match=r"^skill_tools requires skills$"):
            create_deep_agent(model=_model(), skill_tools=[create_customer_request])
        spec: SubAgent = {"name": "worker", "description": "d", "skill_tools": [create_customer_request]}
        with pytest.raises(ValueError, match=r"^skill_tools requires skills$"):
            create_deep_agent(model=_model(), subagents=[spec])

    def test_empty_skills_list_still_mounts_skills(self) -> None:
        create_deep_agent(model=_model(), skills=[], skill_tools=[create_customer_request])

    def test_fork_with_its_own_skill_tools_raises(self) -> None:
        spec: SubAgent = {"name": "worker", "description": "d", "mode": "fork", "skill_tools": [create_customer_request]}
        msg = r"^SubAgent 'worker' cannot set skill_tools under mode='fork'; the parent's skill tools are inherited instead\.$"
        with pytest.raises(ValueError, match=msg):
            create_deep_agent(model=_model(), skills=[SKILLS_SOURCE], subagents=[spec])

    def test_skill_tools_are_never_registered(self) -> None:
        middleware = SkillsMiddleware(backend=StateBackend(), sources=[SKILLS_SOURCE], skill_tools=[create_customer_request])
        agent = create_deep_agent(model=_model(), skills=[SKILLS_SOURCE], skill_tools=[create_customer_request])

        # `create_agent` collects `getattr(middleware, "tools", [])` into the tool node.
        assert getattr(middleware, "tools", []) == []
        assert "create_customer_request" not in agent.nodes["tools"].bound._tools_by_name

    def test_callables_are_converted_like_ordinary_tools(self, tmp_path: Path) -> None:
        def lookup_account(account_id: str) -> str:
            """Look up an account."""
            return f"account {account_id}"

        write_skill(tmp_path, "crm", "lookup_account")
        model = _model(ai(read("r1")), ai(call("lookup_account", "c1", account_id="42")))

        result = invoke(skills_agent(tmp_path, model, skill_tools=[lookup_account]), {"messages": [HumanMessage("go")]}, "sync")

        assert tool_messages(result, "lookup_account")[0].content == "account 42"


def test_profile_excluded_tools_drop_a_skill_tool(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request list_customer_requests")
    model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="x")))
    original = dict(_HARNESS_PROFILES)
    try:
        register_harness_profile("skilltoolsprov", HarnessProfile(excluded_tools=frozenset({"create_customer_request"})))
        with patch("deepagents.graph.resolve_model", return_value=model):
            agent = skills_agent(tmp_path, "skilltoolsprov:model", skill_tools=[create_customer_request, list_customer_requests])
        result = invoke(agent, {"messages": [HumanMessage("go")]}, mode)
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)

    assert "create_customer_request" not in bound_tool_names(model.call_history[1])
    assert "list_customer_requests" in bound_tool_names(model.call_history[1])
    assert tool_messages(result, "create_customer_request")[0].content == "Error: create_customer_request is not available."


class _RecordsInputState(AgentMiddleware):
    """Records the state keys a (sub)agent starts with."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: set[str] = set()

    def before_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.keys = set(state)

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.keys = set(state)


def _task(subagent_type: str, call_id: str = "t1") -> AIMessage:
    return ai(call("task", call_id, description="file a request", subagent_type=subagent_type))


class TestSubagents:
    """Which subagents get skill tools, and that the disclosed record never crosses into one."""

    def test_general_purpose_subagent_inherits_skill_tools(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "crm", "create_customer_request")
        # Parent and general-purpose subagent share the model, so turns interleave.
        model = _model(
            _task("general-purpose"),
            ai(read("r1")),
            ai(call("create_customer_request", "c1", title="x")),
            AIMessage(content="subagent done"),
        )

        invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

        subagent_final_call = model.call_history[3]["messages"]
        assert subagent_final_call[-1].content == "created x (c1)"

    def test_declarative_subagent_uses_only_its_own_skill_tools(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "crm", "create_customer_request list_customer_requests")
        worker_model = _model(ai(read("r1")), ai(call("create_customer_request", "c1", title="x")))
        worker: SubAgent = {
            "name": "worker",
            "description": "Files requests.",
            "model": worker_model,
            "skills": [SKILLS_SOURCE],
            "skill_tools": [list_customer_requests],
        }

        invoke(skills_agent(tmp_path, _model(_task("worker")), subagents=[worker]), {"messages": [HumanMessage("go")]}, mode)

        disclosed = bound_tool_names(worker_model.call_history[1])
        assert "list_customer_requests" in disclosed
        assert "create_customer_request" not in disclosed
        assert "is not a valid tool" in worker_model.call_history[2]["messages"][-1].content

    def test_declarative_subagent_without_skill_tools_gets_none(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "crm", "create_customer_request")
        worker_model = _model(ai(read("r1")))
        worker: SubAgent = {"name": "worker", "description": "Files requests.", "model": worker_model, "skills": [SKILLS_SOURCE]}

        invoke(skills_agent(tmp_path, _model(_task("worker")), subagents=[worker]), {"messages": [HumanMessage("go")]}, mode)

        assert "create_customer_request" not in bound_tool_names(worker_model.call_history[1])

    def test_fork_mirrors_the_parents_skill_tools_without_their_record(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "crm", "create_customer_request")
        worker_model = _model(ai(call("create_customer_request", "c1", title="x")))
        recorder = _RecordsInputState()
        worker: SubAgent = {"name": "worker", "description": "Continues.", "model": worker_model, "mode": "fork", "middleware": [recorder]}
        agent = skills_agent(tmp_path, _model(ai(read("r1")), _task("worker")), subagents=[worker])

        invoke(agent, {"messages": [HumanMessage("go")]}, mode)

        assert "skills_metadata" in recorder.keys
        assert "_skill_tools_disclosed" not in recorder.keys
        assert "create_customer_request" in bound_tool_names(worker_model.call_history[0])
        assert worker_model.call_history[1]["messages"][-1].content == "created x (c1)"

    def test_isolated_subagent_input_omits_the_record(self, tmp_path: Path, mode: str) -> None:
        write_skill(tmp_path, "crm", "create_customer_request")
        recorder = _RecordsInputState()
        worker: SubAgent = {"name": "worker", "description": "d", "model": _model(), "middleware": [recorder]}
        agent = skills_agent(tmp_path, _model(ai(read("r1")), _task("worker")), subagents=[worker])

        invoke(agent, {"messages": [HumanMessage("go")]}, mode)

        assert recorder.keys
        assert "_skill_tools_disclosed" not in recorder.keys
