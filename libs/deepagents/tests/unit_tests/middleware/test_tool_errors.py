"""Unit tests for default tool-error handling in `create_deep_agent`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError, NodeCancelledError, NodeTimeoutError
from langgraph.types import interrupt

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent, SubAgentMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import ChatResult


@tool
def raises_tool_exception(q: str) -> str:
    """Raise the exception `langchain_mcp_adapters` raises for an `isError` result."""
    msg = f"upstream server returned an error for {q!r}"
    raise ToolException(msg)


@tool
def raises_value_error(q: str) -> str:
    """Raise an unplanned exception carrying detail the model shouldn't see."""
    msg = f"connection to postgres://user:hunter2@internal-db/{q} refused"
    raise ValueError(msg)


@tool
def interrupts(q: str) -> str:
    """Raise a `GraphBubbleUp` from inside the tool body."""
    return str(interrupt(f"approve {q}?"))


@tool
def ok(q: str) -> str:
    """Succeed."""
    return f"ok: {q}"


_GRAPH_ERRORS: dict[str, Exception] = {
    "recursion": GraphRecursionError("limit reached"),
    "cancelled": NodeCancelledError("tools", "cancelled"),
    "timeout": NodeTimeoutError("tools", 1.0, kind="run", run_timeout=1.0),
}


@tool
def raises_graph_error(q: str) -> str:
    """Raise the graph-level error named by `q`."""
    raise _GRAPH_ERRORS[q]


def _propagate(_exc: Exception, _request: object) -> None:
    """`on_error` handler that hands every exception back to the caller."""


def _tool_call(name: str, call_id: str = "call-1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {"q": "x"}, "id": call_id}])


def _scripted(*messages: AIMessage) -> GenericFakeChatModel:
    """Fake model that plays `messages` in order."""
    return GenericFakeChatModel(messages=iter(messages))


def _calls_then_answers(tool_name: str) -> GenericFakeChatModel:
    """Fake model that calls `tool_name` once, then answers."""
    return _scripted(_tool_call(tool_name), AIMessage(content="done"))


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


class TestToolExceptionsReachTheModel:
    """A raising tool costs one step, not the whole run (deepagents#5356)."""

    def test_tool_exception_becomes_error_tool_message(self) -> None:
        agent = create_deep_agent(model=_calls_then_answers("raises_tool_exception"), tools=[raises_tool_exception])

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert tool_msg.status == "error"
        # ToolException is written *for* the model, so its message is passed through.
        assert tool_msg.content == "upstream server returned an error for 'x'"
        assert result["messages"][-1].content == "done", "loop did not continue past the failure"

    def test_arbitrary_exception_becomes_error_tool_message(self) -> None:
        agent = create_deep_agent(model=_calls_then_answers("raises_value_error"), tools=[raises_value_error])

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert tool_msg.status == "error"
        assert "ValueError" in tool_msg.content
        assert "raises_value_error" in tool_msg.content
        assert result["messages"][-1].content == "done"

    def test_arbitrary_exception_message_is_not_leaked_to_the_model(self) -> None:
        """Only the exception *type* is surfaced; the message may hold secrets."""
        agent = create_deep_agent(model=_calls_then_answers("raises_value_error"), tools=[raises_value_error])

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert "hunter2" not in tool_msg.content
        assert "internal-db" not in tool_msg.content

    def test_arbitrary_exception_detail_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Detail withheld from the model still reaches whoever owns the tool."""
        agent = create_deep_agent(model=_calls_then_answers("raises_value_error"), tools=[raises_value_error])

        with caplog.at_level(logging.ERROR, logger="deepagents.middleware._tool_errors"):
            agent.invoke({"messages": [HumanMessage(content="go")]})

        (record,) = caplog.records
        assert record.exc_info is not None
        assert "hunter2" in caplog.text

    async def test_tool_exception_is_handled_on_the_async_path(self) -> None:
        agent = create_deep_agent(model=_calls_then_answers("raises_tool_exception"), tools=[raises_tool_exception])

        result = await agent.ainvoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert tool_msg.status == "error"
        assert tool_msg.content == "upstream server returned an error for 'x'"

    def test_successful_tool_calls_are_untouched(self) -> None:
        agent = create_deep_agent(model=_calls_then_answers("ok"), tools=[ok])

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert tool_msg.status == "success"
        assert tool_msg.content == "ok: x"


class TestSubagentContainment:
    """A tool failure inside a subagent no longer takes the parent down."""

    def test_subagent_tool_failure_does_not_kill_the_parent(self) -> None:
        subagent: SubAgent = {
            "name": "searcher",
            "description": "searches",
            "system_prompt": "search",
            "tools": [raises_tool_exception],
            "model": _calls_then_answers("raises_tool_exception"),
        }
        agent = create_deep_agent(
            model=_scripted(
                _tool_call("task", description="search", subagent_type="searcher"),
                AIMessage(content="parent done"),
            ),
            subagents=[subagent],
        )

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        assert result["messages"][-1].content == "parent done"

    def test_subagent_run_failures_still_propagate(self) -> None:
        """`task` nests a run; a structural failure there must surface, not become a `ToolMessage`."""

        class _FailingModel(GenericFakeChatModel):
            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                msg = "research failed"
                raise RuntimeError(msg)

        subagent: SubAgent = {
            "name": "searcher",
            "description": "searches",
            "system_prompt": "search",
            "tools": [ok],
            "model": _FailingModel(messages=iter([])),
        }
        agent = create_deep_agent(
            model=_scripted(
                _tool_call("task", description="search", subagent_type="searcher"),
                AIMessage(content="parent done"),
            ),
            subagents=[subagent],
        )

        with pytest.raises(RuntimeError, match="research failed"):
            agent.invoke({"messages": [HumanMessage(content="go")]})


class TestExceptionsThatMustPropagate:
    """Control-flow and structural signals are not tool failures."""

    def test_interrupt_from_a_tool_body_still_interrupts(self) -> None:
        """`GraphBubbleUp` must not become an error `ToolMessage`, or HITL breaks."""
        agent = create_deep_agent(
            model=_scripted(_tool_call("interrupts"), AIMessage(content="done")),
            tools=[interrupts],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        assert result.get("__interrupt__"), "interrupt was swallowed by tool-error handling"
        assert not _tool_messages(result)

    @pytest.mark.parametrize("kind", list(_GRAPH_ERRORS))
    def test_graph_level_errors_propagate(self, kind: str) -> None:
        agent = create_deep_agent(
            model=_scripted(_tool_call("raises_graph_error", q=kind), AIMessage(content="done")),
            tools=[raises_graph_error],
        )

        with pytest.raises(type(_GRAPH_ERRORS[kind])):
            agent.invoke({"messages": [HumanMessage(content="go")]})


class TestOptOut:
    """Callers keep control via the usual name-based middleware override."""

    def test_caller_middleware_replaces_the_default(self) -> None:
        agent = create_deep_agent(
            model=_calls_then_answers("raises_tool_exception"),
            tools=[raises_tool_exception],
            middleware=[ToolErrorMiddleware(lambda _exc, _request: "handled by the caller")],
        )

        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        (tool_msg,) = _tool_messages(result)
        assert tool_msg.content == "handled by the caller"

    def test_caller_can_restore_strict_propagation(self) -> None:
        agent = create_deep_agent(
            model=_calls_then_answers("raises_tool_exception"),
            tools=[raises_tool_exception],
            middleware=[ToolErrorMiddleware(_propagate)],
        )

        with pytest.raises(ToolException, match="upstream server returned an error"):
            agent.invoke({"messages": [HumanMessage(content="go")]})

    def test_override_does_not_stack_two_instances(self) -> None:
        override = ToolErrorMiddleware(_propagate)
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = "compiled-agent"

        with patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create:
            create_deep_agent(model=_scripted(AIMessage(content="ok")), middleware=[override])

        stack = mock_create.call_args.kwargs["middleware"]
        instances = [m for m in stack if isinstance(m, ToolErrorMiddleware)]
        assert instances == [override]


class TestStackPlacement:
    """The middleware is outermost in every stack `create_deep_agent` assembles."""

    def _main_stack(self, **kwargs: Any) -> list[Any]:
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = "compiled-agent"
        with patch("deepagents.graph.create_agent", return_value=fake_agent) as mock_create:
            create_deep_agent(model=_scripted(AIMessage(content="ok")), **kwargs)
        return list(mock_create.call_args.kwargs["middleware"])

    def test_first_in_the_main_stack(self) -> None:
        stack = self._main_stack()
        assert isinstance(stack[0], ToolErrorMiddleware), [m.name for m in stack]

    def test_first_in_the_main_stack_with_skills(self) -> None:
        """Skills middleware normally leads the stack; error handling still precedes it."""
        stack = self._main_stack(skills=[])
        assert isinstance(stack[0], ToolErrorMiddleware), [m.name for m in stack]

    def test_first_in_the_general_purpose_subagent_stack(self) -> None:
        stack = self._main_stack()
        sub_mw = next(m for m in stack if isinstance(m, SubAgentMiddleware))
        gp_spec = next(s for s in sub_mw._subagents if s["name"] == "general-purpose")
        assert isinstance(gp_spec["middleware"][0], ToolErrorMiddleware)

    def test_first_in_an_inline_subagent_stack(self) -> None:
        subagent: SubAgent = {
            "name": "worker",
            "description": "a worker",
            "system_prompt": "work",
            "model": _scripted(AIMessage(content="ok")),
        }
        stack = self._main_stack(subagents=[subagent])
        sub_mw = next(m for m in stack if isinstance(m, SubAgentMiddleware))
        worker_spec = next(s for s in sub_mw._subagents if s.get("name") == "worker")
        assert isinstance(worker_spec["middleware"][0], ToolErrorMiddleware)
