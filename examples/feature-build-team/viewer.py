"""A live view of the org: who's running, who was steered, and who reported back.

Reads everything from the local LangGraph server, so it needs no LangSmith UI.

    uv run uvicorn viewer:app --port 3000
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langgraph_sdk import get_client
from pydantic import BaseModel

client = get_client(url=os.environ.get("LANGGRAPH_URL", "http://127.0.0.1:2024"))
session: dict[str, Any] = {"thread_id": None, "pending": [], "nudged": set()}

NUDGE = (
    "{name} finished a newer run that you haven't seen. "
    "Call `check_async_task` for task `{task_id}` and continue from its result."
)


async def _nudge_missed_reports() -> None:
    """Wake the idle lead when a team lead's finished run never reached it."""
    while True:
        await asyncio.sleep(3)
        with contextlib.suppress(Exception):
            await _nudge_once()


async def _nudge_once() -> None:
    lead = session["thread_id"]
    if lead is None or (await client.threads.get(lead))["status"] != "idle":
        return
    values = (await client.threads.get_state(lead))["values"] or {}
    seen = {
        ((m.get("additional_kwargs") or {}).get("lc_data") or {}).get("run_id")
        for m in values.get("messages", [])
    }
    for task in values.get("async_tasks", {}).values():
        if (await client.threads.get(task["thread_id"]))["status"] != "idle":
            continue
        latest = (await client.runs.list(task["thread_id"], limit=1) or [None])[0]
        missed = (
            latest
            and latest["status"] == "success"
            and latest["run_id"] not in {task["run_id"], *seen}
        )
        if not missed or (task["task_id"], latest["run_id"]) in session["nudged"]:
            continue
        session["nudged"].add((task["task_id"], latest["run_id"]))
        text = NUDGE.format(name=task["agent_name"], task_id=task["task_id"])
        message = {
            "role": "user",
            "content": text,
            "additional_kwargs": {"lc_source": "viewer"},
        }
        await client.runs.create(
            lead, "lead", input={"messages": [message]}, multitask_strategy="reject"
        )
        return


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    nudger = asyncio.create_task(_nudge_missed_reports())
    yield
    nudger.cancel()


app = FastAPI(lifespan=lifespan)

REPORT = re.compile(r"\b(DONE|WORKING):\s*(.+)", re.DOTALL)


class Message(BaseModel):
    text: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "viewer.html")


@app.post("/api/send")
async def send(message: Message) -> dict[str, str]:
    if session["thread_id"] is None:
        session["thread_id"] = (await client.threads.create())["thread_id"]
    thread_id = session["thread_id"]
    session["pending"].append(message.text)
    await client.runs.create(
        thread_id,
        "lead",
        input={"messages": [{"role": "user", "content": message.text}]},
        multitask_strategy="enqueue",
    )
    return {"thread_id": thread_id}


@app.post("/api/open/{thread_id}")
async def open_run(thread_id: str) -> dict[str, str]:
    session.update(thread_id=thread_id, pending=[], nudged=set())
    return {"thread_id": thread_id}


@app.post("/api/new")
async def new() -> dict[str, None]:
    session.update(thread_id=None, pending=[], nudged=set())
    return {"thread_id": None}


@app.get("/api/state")
async def state() -> dict[str, Any]:
    if session["thread_id"] is None:
        return {"nodes": [], "chat": []}
    nodes: list[dict[str, Any]] = []
    queue = [(session["thread_id"], None)]
    chat: list[dict[str, str]] = []
    while queue:
        thread_id, parent = queue.pop(0)
        node, messages = await _node(thread_id, parent)
        nodes.append(node)
        if parent is None:
            chat = _chat(messages)
        children = await client.threads.search(
            metadata={"parent_thread_id": thread_id}, limit=20
        )
        queue.extend(
            (child["thread_id"], thread_id)
            for child in sorted(children, key=lambda t: t["created_at"])
        )
    lead, teams = nodes[0], [n for n in nodes if n["parent"] == nodes[0]["id"]]
    if lead["phase"] != "working":
        finished = teams and all(team["phase"] == "done" for team in teams)
        lead["phase"] = "done" if finished else "idle"
    return {"nodes": nodes, "chat": _with_pending(chat)}


async def _node(
    thread_id: str, parent: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thread = await client.threads.get(thread_id)
    values = (await client.threads.get_state(thread_id))["values"] or {}
    messages = values.get("messages", [])
    tasks = values.get("async_tasks", {})
    name = (thread["metadata"] or {}).get("subagent_name") or "lead"
    said = next(
        (_text(m) for m in reversed(messages) if m["type"] == "ai" and _text(m)), ""
    )
    node = {
        "id": thread_id,
        "parent": parent,
        "name": name,
        "phase": _phase(thread["status"], said),
        "activity": _activity(messages),
        "said": said[:280],
        "events": _events(thread_id, name, messages, tasks),
    }
    return node, messages


def _phase(status: str, said: str) -> str:
    if status == "busy":
        return "working"
    match = REPORT.search(said)
    return "done" if match and match.group(1) == "DONE" else "idle"


def _activity(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        for call in reversed(message.get("tool_calls") or []):
            args = call["args"]
            target = (
                args.get("file_path")
                or args.get("path")
                or args.get("command")
                or args.get("subagent_type")
            )
            return f"{call['name']} {target}" if target else call["name"]
    return ""


def _events(
    thread_id: str, name: str, messages: list[dict[str, Any]], tasks: dict[str, Any]
) -> list[dict[str, str]]:
    events = []
    for index, message in enumerate(messages):
        for event in _message_events(name, message, tasks):
            event["id"] = f"{thread_id}:{index}:{len(events)}"
            event["text"] = event["text"].strip()[:400]
            events.append(event)
    return events


def _message_events(
    name: str, message: dict[str, Any], tasks: dict[str, Any]
) -> list[dict[str, str]]:
    events = []
    for call in message.get("tool_calls") or []:
        args = call["args"]
        if call["name"] == "start_async_task":
            to = args.get("subagent_type", "")
            events.append(
                {
                    "kind": "spawn",
                    "from": name,
                    "to": to,
                    "text": args.get("description", ""),
                }
            )
        elif call["name"] == "check_async_task":
            sender = tasks.get(args.get("task_id", ""), {}).get("agent_name", "")
            events.append(
                {"kind": "report", "from": sender, "to": name, "text": "checked"}
            )
        elif call["name"] == "update_async_task":
            to = tasks.get(args.get("task_id", ""), {}).get("agent_name", "")
            events.append(
                {
                    "kind": "steer",
                    "from": name,
                    "to": to,
                    "text": args.get("message", ""),
                }
            )
    extra = message.get("additional_kwargs") or {}
    if extra.get("lc_source") == "async_subagent":
        data = extra.get("lc_data") or {}
        match = REPORT.search(_text(message))
        text = (
            f"{match.group(1)}: {match.group(2)}" if match else data.get("status", "")
        )
        events.append(
            {
                "kind": "report",
                "from": data.get("agent_name", ""),
                "to": name,
                "text": text,
            }
        )
    return events


def _chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat = []
    for message in messages:
        if (message.get("additional_kwargs") or {}).get("lc_source"):
            continue
        text = _text(message)
        if message["type"] in {"human", "ai"} and text:
            chat.append(
                {"role": "user" if message["type"] == "human" else "lead", "text": text}
            )
    return chat


def _with_pending(chat: list[dict[str, str]]) -> list[dict[str, str]]:
    sent = [m["text"] for m in chat if m["role"] == "user"]
    session["pending"] = [text for text in session["pending"] if text not in sent]
    for message in chat:
        if message["role"] == "user":
            message["status"] = "sent"
    queued = [
        {"role": "user", "text": t, "status": "queued"} for t in session["pending"]
    ]
    return chat + queued


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "") for b in content or [] if isinstance(b, dict)
    ).strip()
