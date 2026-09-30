from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from deepagents_talon.host import _BACKGROUND_FOLLOW_UP, TalonHost
from deepagents_talon.interfaces import ChannelMessage
from tests.conftest import RecordingChannel
from tests.test_host import RoutedAgent, _config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("transcript", ["transcribed voice", None])
async def test_background_follow_up_does_not_transcribe_previous_voice(
    tmp_path: Path, transcript: str | None
) -> None:
    channel = RecordingChannel()
    agent = RoutedAgent()
    transcriber = AsyncMock()
    transcriber.transcribe.return_value = transcript
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[channel],
        voice_transcriber=transcriber,
    )
    voice = ChannelMessage(
        conversation_id="chat",
        text="voice caption",
        metadata={"media_type": "voice", "voice_path": "voice.ogg"},
    )
    await host.start()
    try:
        await host.receive_message(channel, voice)
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        expected = f"voice caption\n\n{transcript}" if transcript else "voice caption"
        assert agent.requests[-1].text == expected

        agent.background.pending.add("test:chat")
        await host._dispatch_background_results()
        await asyncio.wait_for(host._tasks["test:chat"], 2)

        assert agent.requests[-1].text == _BACKGROUND_FOLLOW_UP
        assert agent.requests[-1].metadata["background_delivery"] is True
        transcriber.transcribe.assert_awaited_once_with(voice)

        agent.background.pending.clear()
        await host.receive_message(channel, voice)
        await asyncio.wait_for(host._tasks["test:chat"], 2)
        assert agent.requests[-1].text == expected
        assert transcriber.transcribe.await_count == 2
    finally:
        await host.stop()
