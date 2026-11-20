"""Exercise launch recovery through actual SDK HTTP transports and Agent state."""

import asyncio
import json
import socket
import threading
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

from deepagents import create_deep_agent
from deepagents.middleware import async_subagents
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.fixture
async def launch_api(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {"visible": True, "run": None, "created": 0}
    owned = []
    original_sync, original_async = async_subagents.get_sync_client, async_subagents.get_client

    def sync_client(**kwargs: Any) -> SyncLangGraphClient:
        client = original_sync(**kwargs)
        owned.append(client.http.client)
        return client

    def async_client(**kwargs: Any) -> LangGraphClient:
        client = original_async(**kwargs)
        owned.append(client.http.client)
        return client

    monkeypatch.setattr(async_subagents, "get_sync_client", sync_client)
    monkeypatch.setattr(async_subagents, "get_client", async_client)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def reply(self, value: object) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_POST(self) -> None:  # BaseHTTPRequestHandler's method name.
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/threads":
                self.reply({"thread_id": "owned-thread"})
                return
            state["created"] += 1
            state["run"] = {"thread_id": "owned-thread", "run_id": "accepted-run", "status": "running", "metadata": body["metadata"]}
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

        def do_GET(self) -> None:  # BaseHTTPRequestHandler's method name.
            path = urlsplit(self.path).path
            self.reply(([state["run"]] if state["visible"] else []) if path.endswith("/runs") else state["run"])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        for client in owned:
            if hasattr(client, "aclose"):
                await client.aclose()
            else:
                client.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        runner.join(timeout=2)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("recovery", ["immediate", "later"])
async def test_accepted_launch_remains_recoverable_after_lost_http_response(launch_api: tuple, *, asynchronous: bool, recovery: str) -> None:
    url, remote = launch_api
    remote["visible"] = recovery == "immediate"
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "start_async_task", "args": {"description": "One job", "subagent_type": "worker"}, "id": "start", "type": "tool_call"}
            ],
        ),
        AIMessage(content="Parent finished"),
        AIMessage(content="", tool_calls=[{"name": "check_async_task", "args": {"task_id": "owned-thread"}, "id": "check", "type": "tool_call"}]),
        AIMessage(content="Task recovered"),
    ]
    graph = create_deep_agent(
        GenericFakeChatModel(messages=iter(responses)),
        subagents=[{"name": "worker", "description": "HTTP worker", "url": url, "graph_id": "worker"}],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "parent"}}
    input_state = {"messages": [HumanMessage("Delegate")]}
    result = await graph.ainvoke(input_state, config) if asynchronous else graph.invoke(input_state, config)
    saved = await graph.aget_state(config) if asynchronous else graph.get_state(config)
    if recovery == "later":
        assert saved.values["_unconfirmed_async_launches"]["owned-thread"]["thread_id"] == "owned-thread"
        assert not result.get("async_tasks")
    else:
        assert result["async_tasks"]["owned-thread"]["run_id"] == "accepted-run"
    remote["visible"] = True
    input_state = {"messages": [HumanMessage("Check the original task")]}
    result = await graph.ainvoke(input_state, config) if asynchronous else graph.invoke(input_state, config)
    saved = await graph.aget_state(config) if asynchronous else graph.get_state(config)
    assert result["async_tasks"]["owned-thread"]["run_id"] == "accepted-run"
    assert not saved.values.get("_unconfirmed_async_launches")
    assert remote["created"] == 1
