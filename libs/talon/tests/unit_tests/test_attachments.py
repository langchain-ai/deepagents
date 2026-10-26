from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileUploadResponse

from deepagents_talon.attachments import stage_documents
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost, _prepare_inbound_message
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def document(tmp_path: Path) -> tuple[TalonConfig, Path]:
    config = TalonConfig.from_env({}, base_home=tmp_path / "home")
    path = config.inbound_media_dir / "document.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"private document bytes")
    return config, path


class _StagingAgent(BlockingAgent):
    def __init__(self, runtime: DeepAgentRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.invoked = asyncio.Event()

    async def invoke(self, request: AgentRequest) -> AgentResult:
        result = await super().invoke(request)
        self.invoked.set()
        return result

    async def prepare_inbound_media(
        self, message: ChannelMessage, *, config: TalonConfig
    ) -> ChannelMessage:
        return await self.runtime.prepare_inbound_media(message, config=config)


@pytest.mark.parametrize("provider", ["slack", "telegram", "whatsapp"])
async def test_host_exposes_readable_sandbox_documents_without_inline_contents(
    tmp_path: Path, document: tuple[TalonConfig, Path], provider: str
) -> None:
    config, path = document
    backend = FilesystemBackend(root_dir=tmp_path / "sandbox", virtual_mode=True)
    runtime = DeepAgentRuntime(model="test", backend=backend, sandbox_working_dir="/work", env={})
    agent = _StagingAgent(runtime)
    channel = RecordingChannel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(
            channel,
            ChannelMessage(
                conversation_id="chat",
                text="read this",
                metadata={
                    "provider": provider,
                    "media_type": "document",
                    "media_paths": [str(path)],
                    "media_path": str(path),
                    "media_urls": [str(path)],
                    "media_mime_types": ["application/pdf"],
                },
            ),
        )
        async with asyncio.timeout(2):
            await agent.invoked.wait()
    finally:
        await host.stop()

    request = agent.requests[0]
    destination = request.metadata["media_paths"][0]
    assert destination.startswith("/work/.talon/attachments/")
    assert destination.endswith(".pdf")
    assert destination in request.text
    assert str(path) not in request.text
    assert "private document bytes" not in request.text
    downloaded = await backend.adownload_files([destination])
    assert downloaded[0].error is None
    assert downloaded[0].content == b"private document bytes"
    assert "media_urls" not in request.metadata
    assert "media_path" not in request.metadata


async def test_local_documents_are_read_on_demand(document: tuple[TalonConfig, Path]) -> None:
    config, path = document
    backend = FilesystemBackend(root_dir=config.home, virtual_mode=False)
    runtime = DeepAgentRuntime(model="test", backend=backend, env={})
    message = ChannelMessage(
        conversation_id="chat",
        text="",
        metadata={
            "media_type": "document",
            "media_paths": [str(path)],
        },
    )

    prepared = _prepare_inbound_message(await runtime.prepare_inbound_media(message, config=config))

    assert str(path) in prepared.text
    assert "private document bytes" not in prepared.text
    downloaded = await backend.adownload_files([str(path)])
    assert downloaded[0].error is None
    assert downloaded[0].content == b"private document bytes"


@pytest.mark.parametrize("failure", ["outside", "symlink", "oversized", "missing", "upload"])
async def test_failed_documents_are_not_advertised_as_accessible(
    tmp_path: Path, document: tuple[TalonConfig, Path], failure: str, monkeypatch
) -> None:
    config, path = document
    backend = FilesystemBackend(root_dir=tmp_path / "sandbox", virtual_mode=True)
    if failure in {"outside", "symlink"}:
        outside = tmp_path / "secret.txt"
        await asyncio.to_thread(outside.write_bytes, b"secret")
        if failure == "symlink":
            link = path.with_suffix(".txt")
            await asyncio.to_thread(link.symlink_to, outside)
            path = link
        else:
            path = outside
    elif failure == "missing":
        path = path.with_suffix(".missing")
    elif failure == "oversized":
        config = TalonConfig.from_env(
            {"DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "1"}, base_home=tmp_path / "home"
        )
    else:

        async def reject(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
            return [FileUploadResponse(path=files[0][0], error="permission_denied")]

        monkeypatch.setattr(backend, "aupload_files", reject)
    message = ChannelMessage(
        conversation_id="chat",
        text="read it",
        metadata={
            "provider": "telegram",
            "media_type": "document",
            "media_paths": [str(path)],
            "media_path": str(path),
            "media_urls": [str(path)],
        },
    )

    prepared = _prepare_inbound_message(
        await stage_documents(
            message,
            backend=backend,
            working_dir="/work",
            config=config,
        )
    )

    assert "unavailable" in prepared.text
    assert str(path) not in prepared.text
    assert prepared.metadata["media_paths"] == []
    assert "media_path" not in prepared.metadata
    assert "media_urls" not in prepared.metadata
