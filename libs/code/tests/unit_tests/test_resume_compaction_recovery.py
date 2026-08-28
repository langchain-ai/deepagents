"""Unit coverage for compacting resumed threads with unfinished graph work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.tui.widgets.messages import ErrorMessage


def _app_with_remote() -> tuple[Any, Any]:
    remote = cast("Any", RemoteAgent(url="http://localhost:8123", graph_name="agent"))
    app = cast("Any", DeepAgentsApp(agent=remote, thread_id="thread-1"))
    app._context_tokens = 500_000
    app._push_screen_wait = AsyncMock(return_value=True)  # ty: ignore[invalid-assignment]
    app._handle_offload = AsyncMock()
    app._mount_message = AsyncMock(return_value=True)
    remote.aensure_thread = AsyncMock()
    remote.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={}, next=("tools",), tasks=(object(),), interrupts=()
        )
    )
    remote.aabandon_pending_work = AsyncMock()
    return app, remote
