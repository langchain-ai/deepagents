"""Focused tests for the virtual-table prototype."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command

from virtual_table import VirtualTableMiddleware


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


async def test_enrich_defines_a_temporary_row_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class Worker:
        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            message = payload["messages"][0]
            assert "<row_data>" in message.content
            row = json.loads(message.content.split("<row_data>\n", 1)[1].split("\n</row_data>", 1)[0])
            calls.append({"row": row, "config": config})
            return {"structured_response": {"sentiment": "positive", "length": len(row["text"])}}

    def worker(model: Any, prompt: str, schema: dict[str, Any]) -> Worker:
        assert isinstance(model, FakeListChatModel)
        assert prompt == "Classify sentiment."
        assert schema["required"] == ["sentiment", "length"]
        return Worker()

    monkeypatch.setattr("virtual_table._worker", worker)
    middleware = VirtualTableMiddleware(initial_tables={"docs": [{"text": "great"}, {"text": "good"}]})
    middleware._model = FakeListChatModel(responses=["unused"])
    state = {"messages": [], "_virtual_tables": middleware._initial_tables}
    enrich = _tool(middleware, "virtual_table_enrich")

    assert enrich.coroutine is not None
    result = await enrich.coroutine(
        name="docs",
        enrichment_name="classification",
        worker_prompt="Classify sentiment.",
        output_schema={
            "type": "object",
            "properties": {"sentiment": {"type": "string"}, "length": {"type": "integer"}},
            "required": ["sentiment", "length"],
        },
        worker_model=None,
        input_columns=["text"],
        row_ids=None,
        concurrency=2,
        overwrite=False,
        runtime=_runtime(state),
    )

    assert isinstance(result, Command)
    rows = result.update["_virtual_tables"]["docs"]
    assert rows[0]["sentiment"] == "positive"
    assert rows[0]["length"] == 5
    assert rows[0]["classification_status"] == "succeeded"
    assert rows[1]["classification_status"] == "succeeded"
    assert {call["row"]["text"] for call in calls} == {"great", "good"}


def test_middleware_adds_table_instructions() -> None:
    middleware = VirtualTableMiddleware()
    model = FakeListChatModel(responses=["unused"])
    request = ModelRequest(model=model, messages=[], system_message=SystemMessage("Host instructions."))

    def handler(updated: ModelRequest[Any]) -> ModelResponse[Any]:
        assert updated.model is model
        assert "Host instructions." in updated.system_message.text
        assert "virtual_table_enrich" in updated.system_message.text
        assert "Define the\n  row worker" in updated.system_message.text
        return ModelResponse(result=[AIMessage("done")])

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == "done"
    assert middleware._model is model


def test_initial_tables_are_private_state() -> None:
    middleware = VirtualTableMiddleware(initial_tables={"docs": [{"text": "hello"}]})
    update = middleware.before_agent({"messages": []}, None)
    assert update == {"_virtual_tables": {"docs": [{"text": "hello", "_row_id": 1}]}}
    assert middleware.before_agent({"messages": [], **update}, None) is None
