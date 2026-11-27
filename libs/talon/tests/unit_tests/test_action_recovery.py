"""Recover interrupted objective writes through the execution middleware."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from deepagents_talon.cron.actions import reconcile_job_action
from deepagents_talon.cron.journal import JobJournal
from deepagents_talon.cron.scope import JobScopeMiddleware
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import _job_scope

if TYPE_CHECKING:
    from pathlib import Path


def request(name: str, call_id: str) -> SimpleNamespace:
    tool = StructuredTool.from_function(lambda: "", name=name, description=name)
    return SimpleNamespace(tool=tool, tool_call={"name": name, "id": call_id, "args": {}})


def _interrupt_write(directory: Path, *, applied: bool) -> None:
    async def interrupted(_request):
        if applied:
            (directory / "destination").write_text("published\n")
        os._exit(23)

    async def run():
        async with AsyncSqliteStore.from_conn_string(str(directory / "history.db")) as store:
            await store.setup()
            with JobJournal(store, "test").capture_actions("goal", "first"):
                await JobScopeMiddleware().awrap_tool_call(
                    request("publish", "original"), interrupted
                )

    asyncio.run(run())


@pytest.mark.parametrize("applied", [True, False])
async def test_interrupted_write_requires_destination_receipt(
    tmp_path: Path, *, applied: bool
) -> None:
    destination = tmp_path / "destination"
    worker = multiprocessing.get_context("spawn").Process(
        target=_interrupt_write, args=(tmp_path,), kwargs={"applied": applied}
    )
    worker.start()
    worker.join(20)
    assert worker.exitcode == 23
    middleware = JobScopeMiddleware()

    async def publish(_request):
        with destination.open("a") as stream:
            stream.write("published\n")
        return ToolMessage(content="published", tool_call_id="retry")

    async def lookup(_request):
        return ToolMessage(
            content=destination.read_text() if destination.exists() else "absent",
            tool_call_id="lookup",
        )

    async with AsyncSqliteStore.from_conn_string(str(tmp_path / "history.db")) as store:
        await store.setup()
        reopened = JobJournal(store, "test")
        with reopened.capture_actions("goal", "second"):
            assert (
                await middleware.awrap_tool_call(request("publish", "retry"), publish)
            ).status == "error"

            question = await middleware.awrap_tool_call(request("ask_user", "question"), lookup)
            assert question.status == "success"

            observed = await middleware.awrap_tool_call(request("read_file", "lookup"), lookup)
            args = {
                "call_id": "original",
                "read_call_id": "missing",
                "outcome": "absent" if observed.content == "absent" else "applied",
                "reference": "publication log",
            }
            assert "read" in await reconcile_job_action.ainvoke(args)
            args["read_call_id"] = "lookup"
            assert "Recorded" in await reconcile_job_action.ainvoke(args)
            if applied:
                assert (
                    await middleware.awrap_tool_call(request("publish", "retry"), publish)
                ).status == "error"
            else:
                assert (
                    await middleware.awrap_tool_call(request("publish", "retry"), publish)
                ).status == "success"
    assert destination.read_text().splitlines() == ["published"]


@pytest.mark.parametrize(
    "result",
    [
        ToolMessage(content="invalid argument", status="error", tool_call_id="failed"),
        ToolMessage(content="Error: not found", tool_call_id="failed"),
    ],
)
async def test_returned_failure_clears_write_intent(result: ToolMessage) -> None:
    middleware = JobScopeMiddleware()
    with JobJournal(InMemoryStore(), "test").capture_actions("goal", "first") as actions:
        assert (
            await asyncio.to_thread(
                middleware.wrap_tool_call, request("publish", "failed"), lambda _: result
            )
            == result
        )
        assert await actions.unresolved() == []


async def test_interrupted_history_search_leaves_no_write_intent() -> None:
    async def interrupted(_request):
        raise asyncio.CancelledError

    with JobJournal(InMemoryStore(), "test").capture_actions("goal", "run") as actions:
        with pytest.raises(asyncio.CancelledError):
            await JobScopeMiddleware().awrap_tool_call(
                request("search_conversations", "lookup"), interrupted
            )
        assert await actions.unresolved() == []


@pytest.mark.parametrize("receipt_id", ["task-call", "other-call"])
async def test_command_receipt_matches_dispatched_action(receipt_id: str) -> None:
    message = ToolMessage("research", tool_call_id=receipt_id)
    result = Command(update={"messages": [message]})
    with JobJournal(InMemoryStore(), "test").capture_actions("goal", "run") as actions:
        await asyncio.to_thread(
            JobScopeMiddleware().wrap_tool_call, request("task", "task-call"), lambda _: result
        )
        assert [row["call_id"] for row in await actions.unresolved()] == (
            [] if receipt_id == "task-call" else ["task-call"]
        )


@pytest.mark.parametrize("has_scope", [True, False])
def test_scheduled_search_requires_chat_scope(*, has_scope: bool) -> None:
    metadata = {"trigger": "cron"}
    if has_scope:
        metadata.update(history_channel="telegram", history_chat="123")
    tools = [request(name, name).tool for name in ("search_conversations", "execute")]
    with _job_scope(AgentRequest(conversation_id="run", text="", metadata=metadata)):
        visible = (
            JobScopeMiddleware()
            .wrap_model_call(ModelRequest(model=None, messages=[], tools=tools), lambda req: req)
            .tools
        )
    assert visible == (tools if has_scope else tools[1:])
