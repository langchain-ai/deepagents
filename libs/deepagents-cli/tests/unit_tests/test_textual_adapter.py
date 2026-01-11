"""Tests for textual adapter behavior."""

import asyncio

from deepagents_cli.textual_adapter import TextualUIAdapter, execute_task_textual


class _DummyAgent:
    def __init__(self) -> None:
        self.state_updates = []

    async def astream(self, *args, **kwargs):
        raise asyncio.CancelledError
        if False:  # pragma: no cover - keeps this an async generator
            yield None

    async def aupdate_state(self, config, update):
        self.state_updates.append((config, update))


class _DummySessionState:
    def __init__(self) -> None:
        self.thread_id = "thread-id"
        self.auto_approve = False


def test_early_interrupt_resets_status():
    """Early interrupts reset status to Ready after handling."""
    statuses: list[str] = []

    async def mount_message(_msg):
        return None

    async def request_approval(_action, _assistant_id):
        raise AssertionError("request_approval should not be called")

    adapter = TextualUIAdapter(
        mount_message=mount_message,
        update_status=lambda msg: statuses.append(msg),
        request_approval=request_approval,
    )

    agent = _DummyAgent()
    session_state = _DummySessionState()

    async def _run():
        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id=None,
            session_state=session_state,
            adapter=adapter,
        )

    asyncio.run(_run())

    assert statuses[-1] == "Ready"
    assert "Interrupted" in statuses
