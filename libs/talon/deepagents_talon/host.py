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
    AgentResult,
    AgentRuntime,
    ChannelAdapter,
    ChannelMessage,
    CronScheduler,
)
from deepagents_talon.observability import langsmith_trace_context
from deepagents_talon.speech import transcribe_voice_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.cron.jobs import CronJob
    from deepagents_talon.speech import VoiceTranscriber

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
        voice_transcriber: VoiceTranscriber | None = None,
    ) -> None:
        """Initialize the host without starting managed components."""
        self.config = config
        self.agent = agent
        self.channels = tuple(channels)
        self.scheduler = scheduler
        self.voice_transcriber = voice_transcriber
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

        for channel in self.channels:
            channel.set_message_handler(
                lambda message, current=channel: self.receive_message(current, message),
            )
            await channel.start()

        if self.scheduler is not None:
            await self.scheduler.start()

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
        message = await transcribe_voice_message(self.voice_transcriber, message)
        if message.text.strip() == _STOP_COMMAND:
            await self._cancel_conversation(channel, message.conversation_id)
            return

        task = asyncio.create_task(
            self._run_agent_turn(channel, message),
            name=f"talon:{message.conversation_id}",
        )
        await task

    async def _run_agent_turn(self, channel: ChannelAdapter, message: ChannelMessage) -> None:
        result = await self._invoke_agent(
            conversation_id=message.conversation_id,
            text=message.text,
            metadata={
                "sender_id": message.sender_id,
                "message_id": message.message_id,
                **message.metadata,
            },
        )
        if result.text:
            await channel.send_message(message.conversation_id, result.text)

    async def run_scheduled_job(self, job: CronJob) -> str:
        """Invoke the agent for one scheduled job.

        Args:
            job: Claimed cron job to run.

        Returns:
            Agent text output for scheduler delivery handling.
        """
        result = await self._invoke_agent(
            conversation_id=job.origin.conversation_id,
            text=job.prompt,
            metadata={
                "cron_job_id": job.id,
                "cron_job_name": job.name,
                "cron_origin_message_id": job.origin.message_id,
                "trigger": "cron",
            },
        )
        return result.text

    async def deliver_scheduled_result(
        self,
        channel: ChannelAdapter,
        job: CronJob,
        text: str,
    ) -> None:
        """Deliver a scheduled job result to its origin conversation.

        Args:
            channel: Channel that should deliver the result.
            job: Cron job that produced the result.
            text: Message text to send.
        """
        await channel.send_message(job.origin.conversation_id, text)

    async def _invoke_agent(
        self,
        *,
        conversation_id: str,
        text: str,
        metadata: dict[str, object],
    ) -> AgentResult:
        lock = self._locks[conversation_id]
        async with lock:
            task = asyncio.current_task()
            if task is not None:
                self._tasks[conversation_id] = task

            try:
                with langsmith_trace_context(
                    self.config.env,
                    assistant_id=self.config.assistant_id,
                    conversation_id=conversation_id,
                    metadata=metadata,
                ):
                    return await self.agent.invoke(
                        AgentRequest(
                            conversation_id=conversation_id,
                            text=text,
                            metadata=metadata,
                        ),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Unhandled agent error in conversation %s",
                    conversation_id,
                )
                raise
            finally:
                if self._tasks.get(conversation_id) is task:
                    del self._tasks[conversation_id]

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
