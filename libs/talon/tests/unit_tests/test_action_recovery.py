"""Recover interrupted objective writes through the execution middleware."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents_talon.cron.actions import reconcile_job_action
from deepagents_talon.cron.journal import JobJournal
from deepagents_talon.cron.scope import JobScopeMiddleware

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
