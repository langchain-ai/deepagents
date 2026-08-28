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


async def test_confirmed_recovery_discards_pending_work_before_compaction() -> None:
    app, remote = _app_with_remote()

    await app._maybe_compact_after_resume()

    screen = app._push_screen_wait.await_args.args[0]
    assert screen._pending_work is True
    remote.aabandon_pending_work.assert_awaited_once()
    app._handle_offload.assert_awaited_once_with(reserved=True)


async def test_declined_recovery_preserves_pending_work_and_context() -> None:
    app, remote = _app_with_remote()
    app._push_screen_wait.return_value = False

    await app._maybe_compact_after_resume()

    remote.aabandon_pending_work.assert_not_awaited()
    app._handle_offload.assert_not_awaited()


async def test_idle_thread_uses_normal_compaction_path() -> None:
    app, remote = _app_with_remote()
    remote.aget_state.return_value = SimpleNamespace(
        values={}, next=(), tasks=(), interrupts=()
    )

    await app._maybe_compact_after_resume()

    screen = app._push_screen_wait.await_args.args[0]
    assert screen._pending_work is False
    remote.aabandon_pending_work.assert_not_awaited()
    app._handle_offload.assert_awaited_once()


async def test_failed_recovery_does_not_compact() -> None:
    app, remote = _app_with_remote()
    remote.aabandon_pending_work.side_effect = RuntimeError("still pending")

    await app._cancel_pending_work_and_compact()

    app._handle_offload.assert_not_awaited()
    mounted = app._mount_message.await_args.args[0]
    assert isinstance(mounted, ErrorMessage)
    assert "not compacted" in str(mounted._content)
