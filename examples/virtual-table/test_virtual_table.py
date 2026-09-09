"""Focused tests for the virtual-table prototype."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel

from virtual_table import VirtualTableMiddleware


class TaskInput(BaseModel):
    description: str
    subagent_type: str


def _runtime(state: dict[str, Any], *, tools: list | None = None) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="outer-call",
        store=None,
        tools=tools or [],
    )


def _tool(middleware: VirtualTableMiddleware, name: str):
    return next(tool for tool in middleware.tools if tool.name == name)


def test_create_describe_and_query() -> None:
    middleware = VirtualTableMiddleware()
    runtime = _runtime({"messages": []})
    create = _tool(middleware, "virtual_table_create")

    result = create.invoke(
        {
            "name": "docs",
            "rows": [{"team": "a", "score": 2}, {"team": "a", "score": 3}, {"team": "b", "score": 4}],
            "runtime": runtime,
        }
    )

    assert isinstance(result, Command)
    tables = result.update["_virtual_tables"]
    assert [row["_row_id"] for row in tables["docs"]] == [1, 2, 3]
    query = _tool(middleware, "virtual_table_query")
    output = query.invoke(
        {
            "name": "docs",
            "sql": "SELECT team, SUM(score) AS total FROM docs GROUP BY team ORDER BY team",
            "parameters": [],
            "runtime": _runtime({"messages": [], "_virtual_tables": tables}),
        }
    )
    assert json.loads(output) == {"rows": [{"team": "a", "total": 5}, {"team": "b", "total": 4}], "count": 2}


def test_query_rejects_writes_and_unapproved_functions() -> None:
    middleware = VirtualTableMiddleware(initial_tables={"docs": [{"text": "hello"}]})
    state = {"messages": [], "_virtual_tables": middleware._initial_tables}
    query = _tool(middleware, "virtual_table_query")

    with pytest.raises(ValueError, match="Only SELECT"):
        query.invoke({"name": "docs", "sql": "DELETE FROM docs", "parameters": [], "runtime": _runtime(state)})
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        query.invoke({"name": "docs", "sql": "SELECT random() FROM docs", "parameters": [], "runtime": _runtime(state)})


@pytest.mark.asyncio
async def test_enrich_materializes_structured_columns_and_errors() -> None:
    async def task(description: str, subagent_type: str, runtime: ToolRuntime) -> Command:
        assert subagent_type == "analyst"
        assert "<row_data>" in description
        row = json.loads(description.split("<row_data>\n", 1)[1].split("\n</row_data>", 1)[0])
        if row["text"] == "bad":
            raise RuntimeError("provider failed")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps({"sentiment": "positive", "length": len(row["text"])}),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    task_tool = StructuredTool.from_function(name="task", description="Run a subagent.", coroutine=task, infer_schema=False, args_schema=TaskInput)
    middleware = VirtualTableMiddleware(initial_tables={"docs": [{"text": "great"}, {"text": "bad"}]})
    state = {"messages": [], "_virtual_tables": middleware._initial_tables}
    enrich = _tool(middleware, "virtual_table_enrich")
    runtime = _runtime(state, tools=[task_tool, enrich])

    assert enrich.coroutine is not None
    result = await enrich.coroutine(
        name="docs",
        enrichment_name="classification",
        instruction="Classify sentiment.",
        subagent_type="analyst",
        output_schema={
            "type": "object",
            "properties": {"sentiment": {"type": "string"}, "length": {"type": "integer"}},
            "required": ["sentiment", "length"],
        },
        input_columns=["text"],
        row_ids=None,
        concurrency=2,
        overwrite=False,
        runtime=runtime,
    )

    assert isinstance(result, Command)
    rows = result.update["_virtual_tables"]["docs"]
    assert rows[0] == {
        "text": "great",
        "_row_id": 1,
        "sentiment": "positive",
        "length": 5,
        "classification_status": "succeeded",
        "classification_error": None,
    }
    assert rows[1]["classification_status"] == "error"
    assert "provider failed" in rows[1]["classification_error"]


def test_initial_tables_are_private_state() -> None:
    middleware = VirtualTableMiddleware(initial_tables={"docs": [{"text": "hello"}]})
    update = middleware.before_agent({"messages": []}, None)
    assert update == {"_virtual_tables": {"docs": [{"text": "hello", "_row_id": 1}]}}
    assert middleware.before_agent({"messages": [], **update}, None) is None
