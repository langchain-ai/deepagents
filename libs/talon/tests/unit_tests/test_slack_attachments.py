from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from deepagents_talon.channels import slack as slack_module
from deepagents_talon.channels.slack import _SlackFile
from deepagents_talon.speech import _voice_path, transcribe_voice_message
from tests.channels.test_slack import _channel, _dm, _FakeResponse

if TYPE_CHECKING:
    from deepagents_talon.interfaces import ChannelMessage


def _mock_downloads(monkeypatch: pytest.MonkeyPatch, responses: list[object]) -> Mock:
    opener = Mock()
    opener.open.side_effect = responses
    monkeypatch.setattr(slack_module, "_build_opener", lambda: opener)
    return opener


@pytest.mark.parametrize(
    ("filename", "mimetype", "expected_mime"),
    [
        ("notes.txt", "text/plain", "text/plain"),
        ("notes.txt", None, "text/plain"),
        ("unknown.bin", None, "application/octet-stream"),
    ],
)
async def test_two_documents_are_downloaded_with_aligned_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mimetype: str | None,
    expected_mime: str,
) -> None:
    bodies = [b"a" * 600, b"b" * 600]
    opener = _mock_downloads(monkeypatch, [_FakeResponse(body) for body in bodies])
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    files = (
        _SlackFile("F1", "https://files.slack.com/a", filename, mimetype, 600),
        _SlackFile("F2", "https://files.slack.com/b", "report.pdf", "application/pdf", 600),
    )

    await gateway.handle_message(replace(_dm("read these"), files=files))

    metadata = messages[0].metadata
    paths = [Path(path) for path in metadata["media_paths"]]
    assert len(set(paths)) == 2
    assert all(path.parent == tmp_path / "inbound" for path in paths)
    assert [path.read_bytes() for path in paths] == bodies
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    assert metadata["media_path"] == str(paths[0])
    assert metadata["media_mime_types"] == [expected_mime, "application/pdf"]
    assert metadata["media_type"] == "document"
    assert metadata["has_media"] is True
    assert "media_error" not in metadata
    assert all(
        call.args[0].get_header("Authorization") == "Bearer xoxb-test"
        for call in opener.open.call_args_list
    )
    await channel.stop()


@pytest.mark.parametrize("native_type", ["image", "audio", "video"])
async def test_mixed_documents_preserve_first_native_media_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_type: str
) -> None:
    bodies = [b"document", b"native", b"later image"]
    _mock_downloads(monkeypatch, [_FakeResponse(body) for body in bodies])
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    files = (
        _SlackFile("F1", "https://files.slack.com/a", "notes.txt", "text/plain", 8),
        _SlackFile("F2", "https://files.slack.com/b", "native.bin", f"{native_type}/test", 6),
        _SlackFile("F3", "https://files.slack.com/c", "photo.png", "image/png", 11),
    )

    await gateway.handle_message(replace(_dm(), files=files))

    metadata = messages[0].metadata
    assert metadata["has_media"] is True
    assert metadata["media_type"] == native_type
    assert metadata["mime_type"] == f"{native_type}/test"
    assert metadata["media_mime_types"] == ["text/plain", f"{native_type}/test", "image/png"]
    downloaded = await asyncio.gather(
        *(asyncio.to_thread(Path(path).read_bytes) for path in metadata["media_paths"])
    )
    assert downloaded == bodies
    assert metadata["media_path"] == metadata["media_paths"][1]

    async def transcribe(message: ChannelMessage) -> str:
        path = _voice_path(message)
        assert path is not None
        return (await asyncio.to_thread(path.read_bytes)).decode()

    transcriber = Mock(transcribe=AsyncMock(side_effect=transcribe))
    transcribed = await transcribe_voice_message(transcriber, messages[0])
    if native_type in {"audio", "video"}:
        assert transcribed.text.endswith("native")
        assert transcribed.metadata["voice_transcribed"] is True
    else:
        transcriber.transcribe.assert_not_awaited()
    await channel.stop()


@pytest.mark.parametrize("native_type", ["image", "audio", "video"])
async def test_failed_native_media_does_not_send_document_to_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_type: str
) -> None:
    _mock_downloads(monkeypatch, [_FakeResponse(b"document"), OSError("download failed")])
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    files = (
        _SlackFile("F1", "https://files.slack.com/a", "report.pdf", "application/pdf", 8),
        _SlackFile("F2", "https://files.slack.com/b", "native.bin", f"{native_type}/test", 6),
    )

    await gateway.handle_message(replace(_dm(), files=files))

    metadata = messages[0].metadata
    assert metadata["media_type"] == "document"
    assert metadata["mime_type"] == "application/pdf"
    assert metadata["media_path"] == metadata["media_paths"][0]
    transcriber = Mock(transcribe=AsyncMock())
    await transcribe_voice_message(transcriber, messages[0])
    transcriber.transcribe.assert_not_awaited()
    await channel.stop()


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize("failure", ["declared_size", "body_size", "incomplete", "network", "host"])
async def test_per_file_failure_preserves_other_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int, failure: str
) -> None:
    files = [
        _SlackFile(f"F{index}", f"https://files.slack.com/{index}", "doc.txt", "text/plain", 2)
        for index in range(3)
    ]
    responses: list[object] = [_FakeResponse(b"ok") for _ in files]
    expected_error = "too large"
    if failure == "declared_size":
        files[failure_index] = replace(files[failure_index], size=1001)
        responses.pop(failure_index)
    elif failure == "body_size":
        responses[failure_index] = _FakeResponse(b"x" * 1001)
    elif failure == "incomplete":
        responses[failure_index] = _FakeResponse(b"short", {"content-length": "10"})
        expected_error = "incomplete"
    elif failure == "network":
        responses[failure_index] = OSError("connection reset")
        expected_error = "connection reset"
    else:
        files[failure_index] = replace(files[failure_index], url="https://evil.test/file")
        responses.pop(failure_index)
        expected_error = "unexpected host"
    opener = _mock_downloads(monkeypatch, responses)
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()

    await gateway.handle_message(replace(_dm(), files=tuple(files)))

    metadata = messages[0].metadata
    paths = [Path(path) for path in metadata["media_paths"]]
    assert len(paths) == 2
    assert [path.read_bytes() for path in paths] == [b"ok", b"ok"]
    assert metadata["has_media"] is True
    assert metadata["media_mime_types"] == ["text/plain", "text/plain"]
    assert expected_error in metadata["media_error"]
    assert sorted((tmp_path / "inbound").iterdir()) == sorted(paths)
    assert all(
        call.args[0].full_url.startswith("https://files.slack.com/")
        for call in opener.open.call_args_list
    )
    await channel.stop()


async def test_unadmitted_attachments_are_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener = _mock_downloads(monkeypatch, [])
    channel, gateway, messages, _ = _channel(tmp_path)
    await channel.start()
    files = tuple(
        _SlackFile(f"F{index}", f"https://files.slack.com/{index}", "doc.txt", "text/plain", 2)
        for index in range(2)
    )

    await gateway.handle_message(replace(_dm(sender="stranger"), files=files))

    assert messages == []
    opener.open.assert_not_called()
    assert list((tmp_path / "inbound").iterdir()) == []
    await channel.stop()
