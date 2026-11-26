from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.messages import AIMessage

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.file_sending import (
    build_send_file_tool,
    display_filename,
    file_scope,
    send_file_to_channel,
)
from deepagents_talon.interfaces import AgentRequest, ChannelMedia, SendResult
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel

if TYPE_CHECKING:
    from pathlib import Path

Sent = tuple[bytes, str, str | None]


def _backend(tmp_path: Path) -> FilesystemBackend:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "q3.pdf").write_bytes(b"%PDF-report")
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def _recorder(sent: list[Sent], *, success: bool = True):
    async def handle(data: bytes, filename: str, caption: str | None) -> SendResult:
        sent.append((data, filename, caption))
        return SendResult(success=success, error="token=secret upstream failure")

    return handle


async def test_sends_backend_file_with_name_and_caption(tmp_path: Path) -> None:
    sent: list[Sent] = []
    send_file = build_send_file_tool(_backend(tmp_path), max_bytes=1024)

    with file_scope(_recorder(sent)):
        result = await send_file.ainvoke({"path": "/reports/q3.pdf", "caption": "Q3"})

    assert result == "Sent q3.pdf."
    assert sent == [(b"%PDF-report", "q3.pdf", "Q3")]


async def test_reports_the_name_the_recipient_sees(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    (tmp_path / "reports" / "Q3 report (final).txt").write_bytes(b"q3")
    sent: list[Sent] = []
    send_file = build_send_file_tool(backend, max_bytes=1024)

    with file_scope(_recorder(sent)):
        result = await send_file.ainvoke({"path": "/reports/Q3 report (final).txt"})

    assert result == "Sent Q3 report (final).txt."
    assert sent == [(b"q3", "Q3 report (final).txt", None)]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("../../etc/passwd", ".._.._etc_passwd"),
        ("a\\b.txt", "a_b.txt"),
        ("..", "file"),
        (".", "file"),
        ("   ", "file"),
        ("line\nbreak.txt", "line_break.txt"),
        ("invoice\u202efdp.exe", "invoice_fdp.exe"),
    ],
)
def test_display_filename_neutralizes_unsafe_names(name: str, expected: str) -> None:
    assert display_filename(name) == expected


def test_display_filename_truncates_long_names_but_keeps_the_extension() -> None:
    name = display_filename("é" * 300 + ".csv")

    assert len(name.encode()) <= 200
    assert name.endswith("é.csv")


@pytest.mark.parametrize(
    ("path", "max_bytes", "expected"),
    [
        ("/reports/missing.pdf", 1024, "Could not read /reports/missing.pdf"),
        ("/reports/q3.pdf", 5, "too large to send"),
    ],
)
async def test_unreadable_or_oversized_files_are_not_sent(
    tmp_path: Path, path: str, max_bytes: int, expected: str
) -> None:
    sent: list[Sent] = []
    send_file = build_send_file_tool(_backend(tmp_path), max_bytes=max_bytes)

    with file_scope(_recorder(sent)):
        result = await send_file.ainvoke({"path": path})

    assert expected in result
    assert sent == []


async def test_channel_failure_is_reported_without_transport_detail(tmp_path: Path) -> None:
    send_file = build_send_file_tool(_backend(tmp_path), max_bytes=1024)

    with file_scope(_recorder([], success=False)):
        result = await send_file.ainvoke({"path": "/reports/q3.pdf"})

    assert result == "Could not send q3.pdf to the chat."


async def test_without_a_chat_nothing_is_sent(tmp_path: Path) -> None:
    send_file = build_send_file_tool(_backend(tmp_path), max_bytes=1024)

    assert "nothing was sent" in await send_file.ainvoke({"path": "/reports/q3.pdf"})


async def test_subagents_cannot_send_files(tmp_path: Path) -> None:
    sent: list[Sent] = []
    send_file = build_send_file_tool(_backend(tmp_path), max_bytes=1024)
    token = _IN_SUBAGENT.set(True)
    try:
        with file_scope(_recorder(sent)):
            result = await send_file.ainvoke({"path": "/reports/q3.pdf"})
    finally:
        _IN_SUBAGENT.reset(token)

    assert "subagent" in result
    assert sent == []


class _ReadingChannel(RecordingChannel):
    """Records each file's bytes at send time, since the staged copy is then removed."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.contents: list[bytes] = []

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        self.contents.append(media.path.read_bytes())  # test double
        await super().send_media(conversation_id, media)
        return SendResult(success=not self.fail)


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("q3.pdf", "document"),
        ("chart.png", "image"),
        ("clip.mp4", "video"),
        ("notes", "document"),
        ("Q3 report (final).txt", "document"),
        ("résumé ✓.pdf", "document"),
        (".env", "document"),
        ("chart.svg", "document"),
        ("scan.tiff", "document"),
        ("anim.gif", "document"),
        ("clip.mov", "document"),
        ("photo.webp", "image"),
    ],
)
async def test_staged_file_keeps_its_name_and_type_then_is_removed(
    tmp_path: Path, filename: str, media_type: str
) -> None:
    channel = _ReadingChannel()
    root = tmp_path / "outbound"

    result = await send_file_to_channel(
        channel, "chat", (b"bytes", filename, "caption"), staging_root=root
    )

    assert result.success
    [(conversation_id, media)] = channel.media
    assert conversation_id == "chat"
    assert (media.path.name, media.media_type, media.caption) == (filename, media_type, "caption")
    assert media.path.is_relative_to(root)
    assert channel.contents == [b"bytes"]
    assert list(root.iterdir()) == []
    assert root.stat().st_mode & 0o777 == 0o700


async def test_staging_is_removed_when_the_send_fails(tmp_path: Path) -> None:
    channel = _ReadingChannel(fail=True)
    root = tmp_path / "outbound"

    result = await send_file_to_channel(
        channel, "chat", (b"bytes", "../../etc/passwd", None), staging_root=root
    )

    assert not result.success
    assert channel.media[0][1].path.parent.parent == root
    assert list(root.iterdir()) == []


async def test_runtime_binds_the_turns_file_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = {}

    class FileGraph:
        async def ainvoke(self, _payload: object, config: dict) -> dict:
            del config
            reply = await registered["send_file"].ainvoke({"path": "/reports/q3.pdf"})
            return {"messages": [AIMessage(content=reply)]}

    def create_graph(**kwargs: object) -> FileGraph:
        registered.update({item.name: item for item in kwargs["tools"]})
        return FileGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", create_graph)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        backend=_backend(tmp_path),
        env={},
    )
    await runtime.start()
    sent: list[Sent] = []
    try:
        result = await runtime.invoke(
            AgentRequest(conversation_id="chat", text="send it", file_handler=_recorder(sent))
        )
    finally:
        await runtime.stop()

    assert result.text == "Sent q3.pdf."
    assert sent == [(b"%PDF-report", "q3.pdf", None)]
