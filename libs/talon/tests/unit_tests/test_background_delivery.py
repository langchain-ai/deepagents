from __future__ import annotations

import asyncio
import json

from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent


async def test_background_delivery_waits_for_parent_and_discards_reset_results(tmp_path):
    channel = RecordingChannel()
    host = TalonHost(
        config=TalonConfig(assistant_id="test", home=tmp_path),
        agent=BlockingAgent(),
        channels=[channel],
    )
    record = {
        "id": "local-task",
        "owner": "chat",
        "status": "success",
        "result": "done",
        "origin": json.dumps(
            {"channel": "test", "conversation_root": "chat", "conversation_id": "chat"}
        ),
    }
    active = asyncio.create_task(asyncio.Event().wait())
    host._tasks["chat"] = active
    try:
        assert not await host._deliver_background_result(record, {"test": channel})
        assert channel.sent == []
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)
        assert await host._deliver_background_result(record, {"test": channel})
        assert channel.sent == [("chat", "Local task local-task (success):\ndone")]
        host._conversation_resets["chat"] = 1
        assert await host._deliver_background_result(record, {"test": channel})
        assert len(channel.sent) == 1
    finally:
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)
