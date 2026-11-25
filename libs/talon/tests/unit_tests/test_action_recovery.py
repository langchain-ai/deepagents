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

    with JobJournal(directory).capture_actions("goal", "first"):
        asyncio.run(
            JobScopeMiddleware().awrap_tool_call(request("publish", "original"), interrupted)
        )


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

    reopened = JobJournal(tmp_path)
    with reopened.capture_actions("goal", "second"):
        assert (
            await middleware.awrap_tool_call(request("publish", "retry"), publish)
        ).status == "error"

        observed = await middleware.awrap_tool_call(request("read_file", "lookup"), lookup)
        args = {
            "call_id": "original",
            "read_call_id": "missing",
            "outcome": "absent" if observed.content == "absent" else "applied",
            "reference": "publication log",
        }
        assert "read" in reconcile_job_action.invoke(args)
        args["read_call_id"] = "lookup"
        assert "Recorded" in reconcile_job_action.invoke(args)
        if applied:
            assert (
                await middleware.awrap_tool_call(request("publish", "retry"), publish)
            ).status == "error"
        else:
            assert (
                await middleware.awrap_tool_call(request("publish", "retry"), publish)
            ).status == "success"
    assert destination.read_text().splitlines() == ["published"]
