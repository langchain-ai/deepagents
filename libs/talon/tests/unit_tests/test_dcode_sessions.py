"""Talon reaches only verified dcode sockets, and only with queued, framed prompts."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.dcode_sessions import DcodeSessions, discover
from deepagents_talon.interfaces import AgentRequest, ToolApprovalRequest
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.tool_approvals import APPROVAL_OPERATOR
from tests.unit_tests.test_research_subagents import ToolModel, _call

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    """Return a private directory short enough for a Unix socket path on macOS."""
    path = Path(tempfile.mkdtemp(prefix="tdc-", dir="/tmp"))  # AF_UNIX path limit
    path.chmod(0o700)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeDcode:
    """Accept one event per connection and answer the way dcode's listener does."""

    def __init__(self, *, ack: dict[str, object] | None = None) -> None:
        self.events: list[dict[str, object]] = []
        self._ack = ack
        self._server: asyncio.AbstractServer | None = None

    async def start(self, path: Path) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=str(path))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        event = json.loads(await reader.readline())
        self.events.append(event)
        ack = self._ack or {"ok": True, "correlation_id": event.get("correlation_id")}
        writer.write((json.dumps(ack) + "\n").encode())
        await writer.drain()
        writer.close()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


async def _bridge(sock_dir: Path, fake: FakeDcode) -> DcodeSessions:
    await fake.start(sock_dir / f"events-{os.getpid()}.sock")
    return DcodeSessions({}, dirs=[sock_dir])


async def test_send_queues_a_framed_prompt_and_nothing_else(sock_dir: Path) -> None:
    fake = FakeDcode()
    bridge = await _bridge(sock_dir, fake)
    _, send = bridge.tools
    token = APPROVAL_OPERATOR.set(True)
    try:
        receipt = await send.ainvoke({"session": str(os.getpid()), "message": "/quit"})
    finally:
        APPROVAL_OPERATOR.reset(token)
        await fake.stop()

    assert receipt["status"] == "delivered"
    [event] = fake.events
    assert set(event) == {"kind", "payload", "correlation_id"}
    assert event["kind"] == "prompt"
    assert event["correlation_id"] == receipt["request_id"]
    assert str(event["payload"]).startswith('<talon-relayed-message from="talon">')


async def test_rejected_or_uncorrelated_ack_is_reported(sock_dir: Path) -> None:
    fake = FakeDcode(ack={"ok": True, "correlation_id": "someone-else"})
    bridge = await _bridge(sock_dir, fake)
    _, send = bridge.tools
    token = APPROVAL_OPERATOR.set(True)
    try:
        result = await send.ainvoke({"session": str(os.getpid()), "message": "hello"})
    finally:
        APPROVAL_OPERATOR.reset(token)
        await fake.stop()

    assert "did not accept the message" in result


def test_discovery_skips_symlinks_dead_pids_and_shared_dirs(sock_dir: Path) -> None:
    live = sock_dir / f"events-{os.getpid()}.sock"
    elsewhere = sock_dir / "elsewhere.sock"
    for path in (elsewhere, sock_dir / "events-999999999.sock"):
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(path))
        server.close()
    live.symlink_to(elsewhere)
    assert discover([sock_dir]) == []

    live.unlink()
    elsewhere.rename(live)
    assert [session.pid for session in discover([sock_dir])] == [os.getpid()]
    sock_dir.chmod(0o755)
    assert discover([sock_dir]) == []


async def test_tools_deny_nonoperators_subagents_and_unlisted_targets(sock_dir: Path) -> None:
    bridge = DcodeSessions({}, dirs=[sock_dir])
    listing, send = bridge.tools

    assert "paired operator" in listing.invoke({})
    token = APPROVAL_OPERATOR.set(True)
    try:
        assert listing.invoke({}) == []
        assert "list_dcode_sessions" in await send.ainvoke(
            {"session": "../events-1.sock", "message": "hi"}
        )
        nested = _IN_SUBAGENT.set(True)
        try:
            assert "unavailable to subagents" in await send.ainvoke(
                {"session": str(os.getpid()), "message": "hi"}
            )
        finally:
            _IN_SUBAGENT.reset(nested)
    finally:
        APPROVAL_OPERATOR.reset(token)


@pytest.mark.parametrize("approval", [None, True, False])
async def test_send_honors_approval_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    approval: bool | None,
) -> None:
    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_call("message_dcode_session", session="1", message="hi")],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        env={"DEEPAGENTS_TALON_DCODE_SESSIONS": "1"},
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        include_web_tools=False,
        max_retries=1,
    )
    resolved: list[str] = []

    def resolve(_self: DcodeSessions, session: str) -> str:
        resolved.append(session)
        return "Unknown session."

    monkeypatch.setattr(DcodeSessions, "_resolve", resolve)
    if approval is not None:
        snapshot = runtime.approval_store.ensure()
        runtime.approval_store.update({"message_dcode_session": approval}, snapshot.revision)
    seen: list[ToolApprovalRequest] = []

    async def reject(request: ToolApprovalRequest) -> str:
        seen.append(request)
        return "reject"

    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest(
                "chat",
                "work",
                metadata={"tool_approval_operator": True},
                approval_handler=reject,
            )
        )
    finally:
        await runtime.stop()

    assert [r.action_requests[0]["name"] for r in seen] == (
        [] if approval is False else ["message_dcode_session"]
    )
    assert resolved == (["1"] if approval is False else [])


def test_dcode_tools_are_off_by_default(tmp_path: Path) -> None:
    runtime = DeepAgentRuntime(
        model="test:parent",
        env={},
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        include_web_tools=False,
    )

    assert runtime.dcode_sessions is None
    names = {getattr(item, "name", "") for item in runtime._build_tools()}
    assert not names & {"list_dcode_sessions", "message_dcode_session"}
