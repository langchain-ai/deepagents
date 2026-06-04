"""Runtime host that coordinates Talon components in one event loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections import defaultdict
from collections.abc import Callable
from types import FrameType
from typing import TYPE_CHECKING, cast

from deepagents_talon.interfaces import (
    AgentRequest,
    AgentRuntime,
    ChannelAdapter,
    ChannelMessage,
    CronScheduler,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.config import TalonConfig

SignalHandler = Callable[[int, FrameType | None], object] | int | None

logger = logging.getLogger(__name__)

_STOP_COMMAND = "/stop"


class TalonHost:
    """Long-running process host for one Talon assistant.

    Args:
        config: Runtime configuration for this assistant.
        agent: Agent runtime invoked for channel and scheduler work.
        channels: Channel adapters managed by this host.
        scheduler: Optional cron scheduler managed by this host.
    """

    def __init__(
        self,
        *,
        config: TalonConfig,
        agent: AgentRuntime,
        channels: Sequence[ChannelAdapter] = (),
        scheduler: CronScheduler | None = None,
    ) -> None:
        """Initialize the host without starting managed components."""
        self.config = config
        self.agent = agent
        self.channels = tuple(channels)
        self.scheduler = scheduler
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopped = asyncio.Event()
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the host has started and not yet stopped."""
        return self._running

    async def start(self) -> None:
        """Start the agent runtime, scheduler, and channels."""
        if self._running:
            return

        self.config.ensure_home()
        await self.agent.start()

        if self.scheduler is not None:
            await self.scheduler.start()

        for channel in self.channels:
            channel.set_message_handler(
                lambda message, current=channel: self.receive_message(current, message),
            )
            await channel.start()

        self._stopped.clear()
        self._running = True
        logger.info("Talon host started for assistant %s", self.config.assistant_id)

    async def stop(self) -> None:
        """Stop managed components and cancel in-flight agent work."""
        if not self._running:
            self._stopped.set()
            return

        self._running = False
        await self._cancel_all()

        for channel in reversed(self.channels):
            await channel.stop()

        if self.scheduler is not None:
            await self.scheduler.stop()

        await self.agent.stop()
        self._stopped.set()
        logger.info("Talon host stopped for assistant %s", self.config.assistant_id)

    async def run_until_stopped(self) -> None:
        """Start the host and keep it alive until shutdown is requested."""
        await self.start()
        cleanup = self._install_signal_handlers()
        try:
            await self._stopped.wait()
        finally:
            cleanup()
            await self.stop()

    def request_shutdown(self) -> None:
        """Request graceful host shutdown."""
        self._stopped.set()

    async def receive_message(self, channel: ChannelAdapter, message: ChannelMessage) -> None:
        """Handle one inbound channel message.

        Args:
            channel: Channel that delivered the message.
            message: Inbound message to process.
        """
        if message.text.strip() == _STOP_COMMAND:
            await self._cancel_conversation(channel, message.conversation_id)
            return

        task = asyncio.create_task(
            self._run_agent_turn(channel, message),
            name=f"talon:{message.conversation_id}",
        )
        await task

    async def _run_agent_turn(self, channel: ChannelAdapter, message: ChannelMessage) -> None:
        lock = self._locks[message.conversation_id]
        async with lock:
            task = asyncio.current_task()
            if task is not None:
                self._tasks[message.conversation_id] = task

            try:
                result = await self.agent.invoke(
                    AgentRequest(
                        conversation_id=message.conversation_id,
                        text=message.text,
                        metadata={
                            "sender_id": message.sender_id,
                            "message_id": message.message_id,
                            **message.metadata,
                        },
                    ),
                )
                if result.text:
                    await channel.send_message(message.conversation_id, result.text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Unhandled agent error in conversation %s",
                    message.conversation_id,
                )
                raise
            finally:
                if self._tasks.get(message.conversation_id) is task:
                    del self._tasks[message.conversation_id]

    async def _cancel_conversation(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
    ) -> None:
        task = self._tasks.get(conversation_id)
        if task is None or task.done():
            await channel.send_message(conversation_id, "No in-flight run to stop.")
            return

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await channel.send_message(conversation_id, "Stopped current run.")

    async def _cancel_all(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _install_signal_handlers(self) -> Callable[[], None]:
        loop = asyncio.get_running_loop()
        previous_handlers: list[tuple[signal.Signals, SignalHandler]] = []

        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                previous_handlers.append((signum, cast("SignalHandler", signal.getsignal(signum))))
                loop.add_signal_handler(signum, self.request_shutdown)

        def cleanup() -> None:
            for signum, previous in previous_handlers:
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.remove_signal_handler(signum)
                signal.signal(signum, previous)

        return cleanup
