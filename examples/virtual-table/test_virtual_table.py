"""Focused tests for the virtual-table prototype."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command

from virtual_table import VirtualTableMiddleware


def _backend(files: dict[str, bytes] | None = None) -> Any:
    files = files or {}

    async def download(paths: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(path=path, content=files.get(path), error=None if path in files else "file_not_found") for path in paths]

    return SimpleNamespace(adownload_files=AsyncMock(side_effect=download))


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


def test_create_and_query() -> None:
    middleware = VirtualTableMiddleware(backend=_backend())
    runtime = _runtime({"messages": []})
    create = _tool(middleware, "virtual_table_create")

    result = create.invoke(
        {
            "name": "docs",
            "rows": [
                {"file": "/docs/1.txt", "team": "a", "score": 2},
                {"file": "/docs/2.txt", "team": "a", "score": 3},
                {"file": "/docs/3.txt", "team": "b", "score": 4},
            ],
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
    middleware = VirtualTableMiddleware(backend=_backend(), initial_tables={"docs": [{"file": "/docs/hello.txt"}]})
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
            return {"structured_response": {"sentiment": "positive", "length": len(row["file_content"])}}

    def worker(model: Any, prompt: str, schema: dict[str, Any]) -> Worker:
        assert isinstance(model, FakeListChatModel)
        assert prompt == "Classify sentiment."
        assert schema["required"] == ["sentiment", "length"]
        return Worker()

    monkeypatch.setattr("virtual_table._worker", worker)
    backend = _backend({"/docs/great.txt": b"great", "/docs/good.txt": b"good"})
    middleware = VirtualTableMiddleware(
        backend=backend,
        initial_tables={"docs": [{"file": "/docs/great.txt", "team": "a"}, {"file": "/docs/good.txt", "team": "b"}]},
    )
    middleware._model = FakeListChatModel(responses=["unused"])
    state = {"messages": [], "_virtual_tables": middleware._initial_tables}
    enrich = _tool(middleware, "virtual_table_enrich")

    assert enrich._injected_args_keys == frozenset({"runtime"})
    assert "runtime" not in enrich.get_input_schema().model_json_schema()["properties"]
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
        input_columns=["file", "team"],
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
    assert {call["row"]["file_content"] for call in calls} == {"great", "good"}
    assert {call["row"]["team"] for call in calls} == {"a", "b"}
    assert backend.adownload_files.await_count == 2


def test_create_requires_a_file_reference() -> None:
    middleware = VirtualTableMiddleware(backend=_backend())
    create = _tool(middleware, "virtual_table_create")

    with pytest.raises(ValueError, match="must have a non-empty string `file`"):
        create.invoke({"name": "docs", "rows": [{"title": "missing"}], "runtime": _runtime({"messages": []})})


async def test_missing_file_is_materialized_as_a_row_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("virtual_table._worker", lambda *_: SimpleNamespace())
    middleware = VirtualTableMiddleware(
        backend=_backend(),
        initial_tables={"docs": [{"file": "/docs/missing.txt", "team": "a"}]},
    )
    middleware._model = FakeListChatModel(responses=["unused"])
    enrich = _tool(middleware, "virtual_table_enrich")

    assert enrich.coroutine is not None
    result = await enrich.coroutine(
        name="docs",
        enrichment_name="classification",
        worker_prompt="Classify sentiment.",
        output_schema={"type": "object", "properties": {"sentiment": {"type": "string"}}, "required": ["sentiment"]},
        worker_model=None,
        input_columns=["file", "team"],
        row_ids=None,
        concurrency=1,
        overwrite=False,
        runtime=_runtime({"messages": [], "_virtual_tables": middleware._initial_tables}),
    )

    row = result.update["_virtual_tables"]["docs"][0]
    assert row["classification_status"] == "error"
    assert "file_not_found" in row["classification_error"]


def test_middleware_adds_table_instructions() -> None:
    middleware = VirtualTableMiddleware(backend=_backend())
    model = FakeListChatModel(responses=["unused"])
    request = ModelRequest(model=model, messages=[], system_message=SystemMessage("Host instructions."))

    def handler(updated: ModelRequest[Any]) -> ModelResponse[Any]:
        assert updated.model is model
        assert "Host instructions." in updated.system_message.text
        assert "virtual_table_enrich" in updated.system_message.text
        assert "initial_tables` already exist" in updated.system_message.text
        assert "virtual_table_create" in updated.system_message.text
        assert "SELECT * FROM <table> LIMIT 3" in updated.system_message.text
        assert "Define the\n  row worker" in updated.system_message.text
        return ModelResponse(result=[AIMessage("done")])

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == "done"
    assert middleware._model is model
    assert {tool.name for tool in middleware.tools} == {"virtual_table_create", "virtual_table_query", "virtual_table_enrich"}


def test_initial_tables_are_middleware_owned_state() -> None:
    middleware = VirtualTableMiddleware(backend=_backend(), initial_tables={"docs": [{"file": "/docs/hello.txt"}]})
    agent = create_agent(FakeListChatModel(responses=["done"]), middleware=[middleware])
    input_schema = agent.get_input_schema().model_json_schema()
    assert "_virtual_tables" not in input_schema["$defs"]["InputSchema"]["properties"]
    assert "_virtual_tables" in agent.get_output_schema().model_json_schema()["properties"]

    update = middleware.before_agent({"messages": []}, None)
    assert update == {"_virtual_tables": {"docs": [{"file": "/docs/hello.txt", "_row_id": 1}]}}
    assert middleware.before_agent({"messages": [], **update}, None) is None
