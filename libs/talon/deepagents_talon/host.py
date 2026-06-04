"""Runtime host that coordinates Talon components in one event loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections import defaultdict
from collections.abc import Awaitable, Callable
from types import FrameType
from typing import TYPE_CHECKING, cast

from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    AgentRuntime,
    ChannelAdapter,
    ChannelMedia,
    ChannelMessage,
    CronScheduler,
)
from deepagents_talon.media import (
    MarkdownMediaRef,
    build_inbound_text,
    build_model_content,
    extract_markdown_media,
    outbound_channel_media,
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
        self._conversation_tasks: defaultdict[str, set[asyncio.Task[None]]] = defaultdict(set)
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
        if message.text.strip().lower() == _STOP_COMMAND:
            await self._cancel_conversation(channel, message.conversation_id)
            return

        task = asyncio.create_task(
            self._run_agent_turn(channel, message),
            name=f"talon:{message.conversation_id}",
        )
        self._track_conversation_task(message.conversation_id, task)

    async def _run_agent_turn(self, channel: ChannelAdapter, message: ChannelMessage) -> None:
        message = await transcribe_voice_message(self.voice_transcriber, message)
        message = _prepare_inbound_message(message)
        provider = await _channel_provider(channel)
        metadata: dict[str, object] = {
            "channel": provider,
            "sender_id": message.sender_id,
            "message_id": message.message_id,
            **message.metadata,
        }
        origin_conversation_id = message.metadata.get("chat_id_from")
        if isinstance(origin_conversation_id, str) and origin_conversation_id:
            metadata["origin_conversation_id"] = origin_conversation_id
        content = build_model_content(message.text, dict(message.metadata))
        if content != message.text:
            metadata["model_content"] = content

        await _send_typing(channel, message.conversation_id)
        result = await self._invoke_agent(
            conversation_id=message.conversation_id,
            text=message.text,
            metadata=metadata,
        )
        await self._deliver_agent_result(channel, message.conversation_id, result)

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
                "channel": job.origin.channel,
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
        tasks = {
            task
            for task in {
                *self._conversation_tasks.get(conversation_id, set()),
                self._tasks.get(conversation_id),
            }
            if task is not None and not task.done()
        }
        if not tasks:
            await channel.send_message(conversation_id, "No in-flight run to stop.")
            return

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await channel.send_message(conversation_id, "Stopped current run.")

    async def _cancel_all(self) -> None:
        tasks = {
            task
            for task in [
                *self._tasks.values(),
                *(task for tasks in self._conversation_tasks.values() for task in tasks),
            ]
            if not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._conversation_tasks.clear()

    async def _deliver_agent_result(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
        result: AgentResult,
    ) -> None:
        cleaned, refs = extract_markdown_media(result.text)
        if not refs:
            if result.text:
                await channel.send_message(conversation_id, result.text)
            return

        media, failed = _outbound_media_from_refs(refs, cleaned)
        text = _with_failed_attachment_text(cleaned, failed)
        sent_media, send_failed = await _send_channel_media(
            channel,
            conversation_id,
            media,
            fallback_caption=text,
        )
        if text and not sent_media:
            await channel.send_message(conversation_id, text)
        elif send_failed and sent_media:
            await channel.send_message(
                conversation_id,
                f"_(Could not attach: {', '.join(send_failed)}.)_",
            )

    def _track_conversation_task(
        self,
        conversation_id: str,
        task: asyncio.Task[None],
    ) -> None:
        self._conversation_tasks[conversation_id].add(task)
        task.add_done_callback(
            lambda done, current=conversation_id: self._complete_conversation_task(current, done),
        )

    def _complete_conversation_task(
        self,
        conversation_id: str,
        task: asyncio.Task[None],
    ) -> None:
        tasks = self._conversation_tasks.get(conversation_id)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                del self._conversation_tasks[conversation_id]
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unhandled channel task error in conversation %s",
                conversation_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

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


def _prepare_inbound_message(message: ChannelMessage) -> ChannelMessage:
    text = build_inbound_text(message.text, dict(message.metadata))
    if text == message.text:
        return message
    return ChannelMessage(
        conversation_id=message.conversation_id,
        text=text,
        sender_id=message.sender_id,
        message_id=message.message_id,
        metadata={**message.metadata, "media_text_augmented": True},
    )


def _outbound_media_from_refs(
    refs: list[MarkdownMediaRef],
    cleaned_text: str,
) -> tuple[list[ChannelMedia], list[str]]:
    media: list[ChannelMedia] = []
    failed: list[str] = []
    for index, ref in enumerate(refs):
        caption = cleaned_text if index == 0 and cleaned_text else getattr(ref, "alt", "") or None
        try:
            media.append(outbound_channel_media(ref, caption=caption))
        except ValueError:
            path = getattr(ref, "path", None)
            failed.append(getattr(ref, "alt", "") or getattr(path, "name", "attachment"))
    return media, failed


def _with_failed_attachment_text(text: str, failed: list[str]) -> str:
    if not failed:
        return text
    return f"{text.rstrip()}\n\n_(Could not attach: {', '.join(failed)}.)_".strip()


async def _send_channel_media(
    channel: ChannelAdapter,
    conversation_id: str,
    media: list[ChannelMedia],
    *,
    fallback_caption: str,
) -> tuple[bool, list[str]]:
    sent = False
    failed: list[str] = []
    for index, item in enumerate(media):
        payload = _media_with_fallback_caption(item, fallback_caption, is_first=index == 0)
        try:
            await channel.send_media(conversation_id, payload)
            sent = True
        except Exception:  # noqa: BLE001  # adapters raise transport-specific failures.
            logger.warning("Could not send outbound media: %s", payload.path, exc_info=True)
            failed.append(payload.caption or payload.path.name)
    return sent, failed


def _media_with_fallback_caption(
    media: ChannelMedia,
    fallback: str,
    *,
    is_first: bool,
) -> ChannelMedia:
    if not is_first or media.caption is not None or not fallback:
        return media
    return ChannelMedia(path=media.path, media_type=media.media_type, caption=fallback)


async def _send_typing(channel: ChannelAdapter, conversation_id: str) -> None:
    send_typing = getattr(channel, "send_typing", None)
    if not callable(send_typing):
        return
    try:
        result = send_typing(conversation_id)
        if isinstance(result, Awaitable):
            await result
    except Exception:  # noqa: BLE001  # typing indicators are best-effort adapter calls.
        logger.debug("Could not send typing indicator", exc_info=True)


async def _channel_provider(channel: ChannelAdapter) -> str | None:
    """Return the channel provider for origin metadata, if available."""
    try:
        return (await channel.status()).provider
    except Exception:  # noqa: BLE001
        logger.warning("Could not resolve channel provider for agent metadata", exc_info=True)
        return None
