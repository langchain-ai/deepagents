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

from virtual_table import DeepAgentOperator, VirtualTableMiddleware, _worker


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


def _middleware(*, operator: Any | None = None) -> VirtualTableMiddleware:
    return VirtualTableMiddleware(operator=operator or AsyncMock())


def _tool(middleware: VirtualTableMiddleware, name: str):
    return next(tool for tool in middleware.tools if tool.name == name)


def test_create_and_query() -> None:
    middleware = _middleware()
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
    tables = result.update["virtual_tables"]
    assert [row["_row_id"] for row in tables["docs"]] == [1, 2, 3]
    query = _tool(middleware, "virtual_table_query")
    output = query.invoke(
        {
            "name": "docs",
            "sql": "SELECT team, SUM(score) AS total FROM docs GROUP BY team ORDER BY team",
            "parameters": [],
            "runtime": _runtime({"messages": [], "virtual_tables": tables}),
        }
    )
    assert json.loads(output) == {"rows": [{"team": "a", "total": 5}, {"team": "b", "total": 4}], "count": 2}


def test_query_rejects_writes_and_unapproved_functions() -> None:
    middleware = _middleware()
    state = middleware.before_agent({"messages": [], "virtual_tables": {"docs": [{"file": "/docs/hello.txt"}]}}, None)
    query = _tool(middleware, "virtual_table_query")

    with pytest.raises(ValueError, match="Only SELECT"):
        query.invoke({"name": "docs", "sql": "DELETE FROM docs", "parameters": [], "runtime": _runtime(state)})
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        query.invoke({"name": "docs", "sql": "SELECT random() FROM docs", "parameters": [], "runtime": _runtime(state)})


async def test_enrich_materializes_filtered_derived_table(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class Worker:
        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            assert payload["files"] == {"/shared/context.txt": {"content": "context", "encoding": "utf-8"}}
            message = payload["messages"][0]
            row = json.loads(message.content.split("<row_data>\n", 1)[1].split("\n</row_data>", 1)[0])
            calls.append({"row": row, "config": config})
            return {"structured_response": {"sentiment": "positive", "length": len(row["file"])}}

    monkeypatch.setattr("virtual_table._worker", lambda *_: Worker())
    backend = _backend()
    operator = DeepAgentOperator(model=FakeListChatModel(responses=["unused"]), backend=backend, concurrency=2)
    middleware = _middleware(operator=operator)
    state = {
        "messages": [],
        "files": {"/shared/context.txt": {"content": "context", "encoding": "utf-8"}},
        **middleware.before_agent(
            {
                "messages": [],
                "virtual_tables": {
                    "docs": [
                        {"file": "/docs/great.txt", "team": "a", "score": 2},
                        {"file": "/docs/good.txt", "team": "b", "score": 1},
                    ]
                },
            },
            None,
        ),
    }
    enrich = _tool(middleware, "virtual_table_enrich")

    result = await enrich.coroutine(
        source_table="docs",
        output_table="classified_docs",
        instructions="Classify sentiment.",
        output_schema={
            "type": "object",
            "properties": {"sentiment": {"type": "string"}, "length": {"type": "integer"}},
            "required": ["sentiment", "length"],
        },
        input_columns=["file", "team"],
        where="score > ?",
        parameters=[1],
        runtime=_runtime(state),
    )

    assert isinstance(result, Command)
    assert result.update["virtual_tables"]["docs"] == state["virtual_tables"]["docs"]
    rows = result.update["virtual_tables"]["classified_docs"]
    assert rows == [
        {
            "_row_id": 1,
            "file": "/docs/great.txt",
            "team": "a",
            "sentiment": "positive",
            "length": len("/docs/great.txt"),
            "_enrichment_status": "succeeded",
            "_enrichment_error": None,
        }
    ]
    payload = json.loads(result.update["messages"][0].content)
    assert payload["table"] == "classified_docs"
    assert set(payload["schema"]) == {
        "_row_id",
        "file",
        "team",
        "sentiment",
        "length",
        "_enrichment_status",
        "_enrichment_error",
    }
    assert calls[0]["row"] == {"_row_id": 1, "file": "/docs/great.txt", "team": "a"}


def test_worker_is_a_normal_deep_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def create_agent_stub(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("virtual_table.create_deep_agent", create_agent_stub)
    backend = _backend()
    _worker(
        FakeListChatModel(responses=["unused"]),
        "Extract facts.",
        {"type": "object", "properties": {"fact": {"type": "string"}}},
        backend,
    )

    assert captured["backend"] is backend
    assert "filesystem tools" in captured["system_prompt"]
    assert "tools" not in captured
    assert "middleware" not in captured


def test_create_requires_a_file_reference() -> None:
    middleware = _middleware()
    create = _tool(middleware, "virtual_table_create")

    with pytest.raises(ValueError, match="must have a non-empty string `file`"):
        create.invoke({"name": "docs", "rows": [{"title": "missing"}], "runtime": _runtime({"messages": []})})


async def test_operator_failure_is_materialized_as_a_row_error() -> None:
    operator = AsyncMock()
    operator.enrich.return_value = [{"values": {}, "error": "file_not_found"}]
    middleware = _middleware(operator=operator)
    enrich = _tool(middleware, "virtual_table_enrich")

    result = await enrich.coroutine(
        source_table="docs",
        output_table="classified_docs",
        instructions="Classify sentiment.",
        output_schema={"type": "object", "properties": {"sentiment": {"type": "string"}}, "required": ["sentiment"]},
        input_columns=["file", "team"],
        where=None,
        parameters=[],
        runtime=_runtime(
            middleware.before_agent(
                {"messages": [], "virtual_tables": {"docs": [{"file": "/docs/missing.txt", "team": "a"}]}},
                None,
            )
        ),
    )

    row = result.update["virtual_tables"]["classified_docs"][0]
    assert row["sentiment"] is None
    assert row["_enrichment_status"] == "error"
    assert row["_enrichment_error"] == "file_not_found"


def test_middleware_adds_table_instructions() -> None:
    middleware = _middleware()
    model = FakeListChatModel(responses=["unused"])
    request = ModelRequest(model=model, messages=[], system_message=SystemMessage("Host instructions."))

    def handler(updated: ModelRequest[Any]) -> ModelResponse[Any]:
        assert updated.model is model
        assert "Host instructions." in updated.system_message.text
        assert "virtual_table_enrich" in updated.system_message.text
        assert "Do not read every row's file yourself" in updated.system_message.text
        assert "sample one or two files" in updated.system_message.text
        assert "virtual_tables` already exist" in updated.system_message.text
        assert "virtual_table_create" in updated.system_message.text
        assert "SELECT * FROM <table> LIMIT 3" in updated.system_message.text
        assert "Select a\n  source table" in updated.system_message.text
        return ModelResponse(result=[AIMessage("done")])

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == "done"
    assert {tool.name for tool in middleware.tools} == {"virtual_table_create", "virtual_table_query", "virtual_table_enrich"}


def test_tables_are_supplied_and_normalized_in_invocation_state() -> None:
    middleware = _middleware()
    agent = create_agent(FakeListChatModel(responses=["done"]), middleware=[middleware])
    input_schema = agent.get_input_schema().model_json_schema()
    assert "virtual_tables" in input_schema["$defs"]["InputSchema"]["properties"]
    assert "virtual_tables" in agent.get_output_schema().model_json_schema()["properties"]

    update = middleware.before_agent({"messages": [], "virtual_tables": {"docs": [{"file": "/docs/hello.txt"}]}}, None)
    assert update == {"virtual_tables": {"docs": [{"file": "/docs/hello.txt", "_row_id": 1}]}}
