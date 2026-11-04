from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from deepagents_talon.channels import imessage as module
from deepagents_talon.channels.base import ChannelExposure, ExposureMode
from deepagents_talon.channels.imessage import IMessageChannel, IMessageChannelConfig
from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMedia

if TYPE_CHECKING:
    from pathlib import Path


class FakeProcess:
    def __init__(self, output: bytes | None = b"", *, returncode: int | None = 0) -> None:
        self.stdout = asyncio.StreamReader(limit=module._MAX_OUTPUT_BYTES)
        if output is not None:
            self.stdout.feed_data(output)
            self.stdout.feed_eof()
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.ignore_terminate = False
        self.reaped = False
        self.exited = asyncio.Event()
        if returncode is not None:
            self.exited.set()

    async def wait(self):
        await self.exited.wait()
        self.reaped = True
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15
            self.stdout.feed_eof()
            self.exited.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.exited.set()


def _config(tmp_path: Path) -> IMessageChannelConfig:
    return IMessageChannelConfig(
        outbound_media_dir=tmp_path,
        exposure=ChannelExposure(operator_ids=frozenset({"operator@example.com"})),
    )


def _event(**changes: object) -> bytes:
    return (
        json.dumps(
            {
                "id": 3,
                "chat_id": 42,
                "sender": "operator@example.com",
                "is_from_me": False,
                "is_group": False,
                "chat_guid": "iMessage;-;operator@example.com",
                "text": "hello",
                **changes,
            }
        ).encode()
        + b"\n"
    )


def _subprocess(monkeypatch, process):
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/local/bin/imsg")
    return spawn


def test_config_requires_operator_and_uses_shared_media_policy(tmp_path):
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    with pytest.raises(ValueError, match="IMESSAGE_OPERATOR_ID"):
        IMessageChannelConfig.from_talon_config(config)
    configured = IMessageChannelConfig.from_talon_config(
        replace(
            config,
            env={
                "DEEPAGENTS_TALON_IMESSAGE_OPERATOR_ID": "operator@example.com,+15550001111",
                "DEEPAGENTS_TALON_IMESSAGE_CLI_PATH": "/opt/homebrew/bin/imsg",
                "DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR": str(tmp_path),
                "DEEPAGENTS_TALON_MAX_MEDIA_BYTES": "123",
            },
        )
    )
    assert configured.cli_path == "/opt/homebrew/bin/imsg"
    assert configured.exposure.operator_ids == {"operator@example.com", "+15550001111"}
    assert configured.outbound_media_dir == tmp_path
    assert configured.max_media_bytes == 123


@pytest.mark.parametrize("mode", ["allowlist", "open"])
def test_config_exposure_policy(tmp_path, mode):
    env = {"AGENT_ASSISTANT_ID": "test", "DEEPAGENTS_TALON_IMESSAGE_EXPOSURE": mode}
    config = TalonConfig.from_env(env, base_home=tmp_path)
    if mode == "open":
        with pytest.raises(ValueError, match="OPEN_ACK"):
            IMessageChannelConfig.from_talon_config(config)
        env["DEEPAGENTS_TALON_IMESSAGE_OPEN_ACK"] = "allow-arbitrary-senders"
    env["DEEPAGENTS_TALON_IMESSAGE_ALLOWLIST_CHATS"] = "42,43"
    config = TalonConfig.from_env(env, base_home=tmp_path)
    assert IMessageChannelConfig.from_talon_config(config).exposure.mode == mode


async def test_start_requires_macos_installed_cli_and_handles_spawn_failure(tmp_path, monkeypatch):
    channel = IMessageChannel(_config(tmp_path))
    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires macOS"):
        await channel.start()
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install steipete/imsg"):
        await channel.start()
    spawn = _subprocess(monkeypatch, FakeProcess())
    spawn.side_effect = OSError("private content")
    with pytest.raises(RuntimeError, match="check CLI_PATH") as error:
        await channel.start()
    assert "private content" not in str(error.value)
    assert not (await channel.status()).connected
    await channel.stop()


async def test_watch_dispatches_text_without_attachment_paths_and_disconnects(
    tmp_path, monkeypatch
):
    process = FakeProcess(_event(attachments=[{"filename": "/private/secret"}]))
    spawn = _subprocess(monkeypatch, process)
    channel = IMessageChannel(_config(tmp_path))
    handler = AsyncMock()
    channel.set_message_handler(handler)
    await channel.start()
    assert (await channel.status()).connected
    await channel._watch
    message = handler.call_args.args[0]
    assert (message.conversation_id, message.message_id, message.sender_id, message.text) == (
        "42",
        "3",
        "operator@example.com",
        "hello",
    )
    assert message.metadata == {"provider": "imessage", "chat_type": "private", "from_self": False}
    assert spawn.call_args.args == ("imsg", "watch", "--json")
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    assert not (await channel.status()).connected
    assert process.reaped
    await channel.stop()


@pytest.mark.parametrize(
    "changes",
    [
        {"is_from_me": True},
        {"is_from_me": None},
        {"is_from_me": 0},
        {"id": True},
        {"id": "3"},
        {"id": 0},
        {"chat_id": -1},
        {"chat_id": False},
        {"chat_id": "42"},
        {"chat_guid": "SMS;-;+15550001111"},
        {"chat_guid": "RCS;-;+15550001111"},
        {"chat_guid": None},
        {"chat_guid": "iMessage;-;"},
        {"sender": None},
        {"sender": " "},
        {"text": None},
        {"text": ""},
        {"is_group": None},
        {"type": "edited"},
        {"type": "unsent"},
        {"is_reaction": True},
        {"poll": {"event": "imessage.poll.voted"}},
    ],
)
async def test_watch_rejects_echoes_invalid_origins_and_mutations(tmp_path, monkeypatch, changes):
    _subprocess(monkeypatch, FakeProcess(_event(**changes)))
    exposure = ChannelExposure(mode=ExposureMode.OPEN)
    channel = IMessageChannel(replace(_config(tmp_path), exposure=exposure))
    handler = AsyncMock()
    channel.set_message_handler(handler)
    await channel.start()
    await channel._watch
    handler.assert_not_awaited()
    await channel.stop()


async def test_watch_rejects_missing_fields_and_malformed_json(tmp_path, monkeypatch):
    event = json.loads(_event())
    lines = [b"not json\n", b"[]\n", b"\xff\n"]
    lines.extend(
        json.dumps({key: value for key, value in event.items() if key != field}).encode() + b"\n"
        for field in ("id", "chat_id", "is_from_me", "is_group", "chat_guid", "sender")
    )
    _subprocess(monkeypatch, FakeProcess(b"".join(lines)))
    channel = IMessageChannel(_config(tmp_path))
    handler = AsyncMock()
    channel.set_message_handler(handler)
    await channel.start()
    await channel._watch
    handler.assert_not_awaited()
    await channel.stop()


@pytest.mark.parametrize("mode", [ExposureMode.SELF, ExposureMode.ALLOWLIST, ExposureMode.OPEN])
async def test_shared_admission_and_group_metadata(tmp_path, monkeypatch, mode):
    event = _event(sender="other@example.com", is_group=True, chat_guid="iMessage;+;chat123")
    _subprocess(monkeypatch, FakeProcess(event))
    exposure = ChannelExposure(mode=mode, conversations=frozenset({"42"}))
    channel = IMessageChannel(replace(_config(tmp_path), exposure=exposure))
    handler = AsyncMock()
    channel.set_message_handler(handler)
    await channel.start()
    await channel._watch
    if mode == ExposureMode.SELF:
        handler.assert_not_awaited()
    else:
        assert handler.call_args.args[0].metadata["chat_type"] == "group"
    await channel.stop()


@pytest.mark.parametrize("run_task", [False, True])
async def test_stop_reaps_even_immediate_or_uncooperative_watch(tmp_path, monkeypatch, run_task):
    process = FakeProcess(None, returncode=None)
    process.ignore_terminate = run_task
    spawn = _subprocess(monkeypatch, process)
    monkeypatch.setattr(module, "_STOP_TIMEOUT_SECONDS", 0.01)
    channel = IMessageChannel(_config(tmp_path))
    await channel.start()
    await channel.start()
    assert spawn.await_count == 1
    if run_task:
        await asyncio.sleep(0)
    await channel.stop()
    await channel.stop()
    assert process.terminated
    assert process.reaped
    assert process.killed == run_task
    assert not (await channel.status()).connected


async def test_oversized_watch_lines_are_skipped_without_disconnecting(tmp_path, monkeypatch):
    process = FakeProcess(None, returncode=None)
    _subprocess(monkeypatch, process)
    channel = IMessageChannel(_config(tmp_path))
    received = asyncio.Event()
    handler = AsyncMock(side_effect=lambda _: received.set())
    channel.set_message_handler(handler)
    await channel.start()
    oversized = b"x" * (module._MAX_OUTPUT_BYTES + 1)
    process.stdout.feed_data(oversized)
    for _ in range(5):
        await asyncio.sleep(0)
    process.stdout.feed_data(_event(text="smuggled tail") + oversized + b"\n" + _event())
    await asyncio.wait_for(received.wait(), timeout=5)
    assert [call.args[0].text for call in handler.call_args_list] == ["hello"]
    assert (await channel.status()).connected
    await channel.stop()
    assert process.reaped


async def test_send_shell_free_argv_chunks_and_unsupported_actions(tmp_path, monkeypatch):
    spawn = _subprocess(monkeypatch, FakeProcess())
    spawn.side_effect = [FakeProcess(b'{"status":"sent"}') for _ in range(3)]
    channel = IMessageChannel(_config(tmp_path))
    text = '$(touch /tmp/should-not-exist); "quoted"'
    assert (await channel.send_message("42", text)).success
    assert spawn.call_args.args == ("imsg", "send", "--chat-id", "42", "--text", text, "--json")
    assert "shell" not in spawn.call_args.kwargs
    assert (await channel.send_message("42", "a" * 4097)).success
    assert spawn.await_count == 3
    assert len(spawn.call_args_list[1].args[5]) == 4096
    assert len(spawn.call_args_list[2].args[5]) == 1
    assert not (await channel.edit_message("42", "3", "new")).success
    await channel.send_typing("42")


@pytest.mark.parametrize(
    "conversation",
    [
        "0",
        "-1",
        "--to",
        "a@example.com",
        "1;pwd",
        "1 2",
        "\u0661",
        "",
        "9" * 5000,
    ],
)
async def test_send_rejects_non_chat_selectors(tmp_path, monkeypatch, conversation):
    spawn = _subprocess(monkeypatch, FakeProcess())
    channel = IMessageChannel(_config(tmp_path))
    assert not (await channel.send_message(conversation, "hello")).success
    media = ChannelMedia(path=tmp_path, media_type="image")
    assert not (await channel.send_media(conversation, media)).success
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    ("output", "returncode"),
    [
        (b"private stderr", 1),
        (b"not json", 0),
        (b"{}", 0),
        (b'{"status":"error"}', 0),
    ],
)
async def test_send_failure_is_not_retryable_or_content_leaking(
    tmp_path, monkeypatch, output, returncode
):
    spawn = _subprocess(monkeypatch, FakeProcess(output, returncode=returncode))
    result = await IMessageChannel(_config(tmp_path)).send_message("42", "hello")
    assert not result.success
    assert not result.retryable
    assert "private" not in result.error
    assert spawn.await_count == 1


@pytest.mark.parametrize("overflow", [False, True])
async def test_send_timeout_and_output_limit_reap_without_retry(tmp_path, monkeypatch, overflow):
    output = b"x" * (module._MAX_OUTPUT_BYTES + 1) if overflow else None
    process = FakeProcess(output, returncode=None)
    spawn = _subprocess(monkeypatch, process)
    monkeypatch.setattr(module, "_SEND_TIMEOUT_SECONDS", 0.01)
    result = await IMessageChannel(_config(tmp_path)).send_message("42", "hello")
    assert not result.success
    assert not result.retryable
    assert process.terminated
    assert process.reaped
    assert spawn.await_count == 1


async def test_send_cancellation_reaps_child(tmp_path, monkeypatch):
    process = FakeProcess(None, returncode=None)
    _subprocess(monkeypatch, process)
    task = asyncio.create_task(IMessageChannel(_config(tmp_path)).send_message("42", "hello"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated
    assert process.reaped


async def test_send_media_validates_root_size_and_caption(tmp_path, monkeypatch):
    spawn = _subprocess(monkeypatch, FakeProcess(b'{"status":"sent"}'))
    root = tmp_path / "trusted"
    root.mkdir()
    image = root / "image.png"
    image.write_bytes(b"1234")
    channel = IMessageChannel(replace(_config(root), max_media_bytes=4))
    media = ChannelMedia(path=image, media_type="image", caption="caption")
    assert (await channel.send_media("42", media)).success
    assert spawn.call_args.args == (
        "imsg",
        "send",
        "--chat-id",
        "42",
        "--file",
        str(image.resolve()),
        "--text",
        "caption",
        "--json",
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"1234")
    link = root / "link.png"
    link.symlink_to(outside)
    for path in (outside, link, root / "missing.png"):
        assert not (await channel.send_media("42", replace(media, path=path))).success
    image.write_bytes(b"12345")
    assert not (await channel.send_media("42", media)).success
    assert spawn.await_count == 1


async def test_real_child_output_and_cleanup():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys, time; print('ready', flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=5) == b"ready\n"
    finally:
        await module._reap(process)
    assert process.returncode is not None


async def test_real_child_output_overflow_is_bounded():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 1000000)",
        stdout=asyncio.subprocess.PIPE,
        limit=module._MAX_OUTPUT_BYTES,
    )
    try:
        with pytest.raises(ValueError, match="exceeded limit"):
            await asyncio.wait_for(module._read_output(process), timeout=5)
    finally:
        await module._reap(process)
    assert process.returncode is not None
