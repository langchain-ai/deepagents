"""Runtime host that coordinates Talon components in one event loop.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, cast

from deepagents_talon.authorization import (
    AuthorizationBinding,
    AuthorizationCompleted,
    AuthorizationEvent,
    AuthorizationFailed,
    AuthorizationURL,
    CallbackURLRequested,
    DeviceCode,
)
from deepagents_talon.channels.base import outbound_media_root_from_env, send_with_retry
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    AgentRuntime,
    BackgroundTaskRuntime,
    ChannelAdapter,
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    CronScheduler,
    MCPReloadableRuntime,
    ReactionChannelAdapter,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from deepagents_talon.mcp_auth import extract_oauth_callback_url
from deepagents_talon.media import (
    MarkdownMediaRef,
    build_inbound_text,
    build_model_content,
    extract_markdown_media,
    outbound_channel_media,
)
from deepagents_talon.observability import langsmith_trace_context, log_event, stable_log_ref
from deepagents_talon.speech import transcribe_voice_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.cron.jobs import CronJob
    from deepagents_talon.speech import VoiceTranscriber

SignalHandler = Callable[[int, FrameType | None], object] | int | None

logger = logging.getLogger(__name__)

_STOP_COMMAND = "/stop"
_NEW_COMMAND = "/new"
_MCP_RELOAD_COMMAND = "/mcp-reload"
_NEW_CONVERSATION_MESSAGE = "Started a fresh conversation."
_MCP_RELOAD_SUCCESS_MESSAGE = "Reloaded MCP configuration."
_MCP_RELOAD_FAILURE_MESSAGE = "Could not reload MCP configuration. Check Talon logs."
_MCP_RELOAD_UNAVAILABLE_MESSAGE = "MCP configuration reload is unavailable."
_APPROVE_REPLIES = frozenset({"approve", "approved", "yes", "y"})
_DENY_REPLIES = frozenset({"deny", "denied", "reject", "rejected", "no", "n"})
_RESET_THREAD_SEPARATOR = ":talon-reset:"
_CRON_THREAD_SUFFIX = ":talon-cron"
_APPROVAL_LOG_RAW_IDS_ENV = "DEEPAGENTS_TALON_APPROVAL_LOG_RAW_IDS"
_TYPING_REFRESH_SECONDS = 4.0
_CANCEL_TIMEOUT_SECONDS = 30.0
_CANCEL_TIMEOUT_MESSAGE = (
    "Could not stop the current run within 30 seconds. Your new message was not started. "
    "Restart Talon to recover."
)
_EMOJI_VARIATION_SELECTOR = "\ufe0f"
_EMOJI_SKIN_TONES = frozenset(
    {
        "\U0001f3fb",
        "\U0001f3fc",
        "\U0001f3fd",
        "\U0001f3fe",
        "\U0001f3ff",
    }
)


class _CancelOutcome(StrEnum):
    NONE = "none"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class _Turn:
    conversation_root: str
    conversation_id: str
    provider: str | None
    generation: int
    recovery_degraded: bool


@dataclass(slots=True)
class _CronControl:
    lock: asyncio.Lock
    users: int = 0


@dataclass(slots=True)
class _PendingToolApproval:
    future: asyncio.Future[ToolApprovalDecision]
    provider: str
    channel_conversation_id: str
    agent_conversation_id: str
    prompt_message_id: str | None
    sender_id: str | None


@dataclass(slots=True)
class _PendingAuthorization:
    future: asyncio.Future[str]
    binding: AuthorizationBinding
    provider: str
    channel_conversation_id: str
    agent_conversation_id: str
    sender_id: str


@dataclass(frozen=True, slots=True)
class _AuthorizationFlow:
    binding: AuthorizationBinding
    provider: str
    channel_conversation_id: str
    sender_id: str


@dataclass(frozen=True, slots=True)
class _ReactionAudit:
    reaction: ChannelReaction
    provider: str
    decision: ToolApprovalDecision | None
    match_status: str
    resolution: str


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
        self._cron_controls: dict[str, _CronControl] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._conversation_tasks: defaultdict[str, set[asyncio.Task[None]]] = defaultdict(set)
        self._generations: defaultdict[str, int] = defaultdict(int)
        self._blocked: set[str] = set()
        self._conversation_resets = _load_conversation_resets(config.conversation_state_path)
        self._pending_tool_approvals: dict[str, _PendingToolApproval] = {}
        self._pending_authorizations: dict[str, _PendingAuthorization] = {}
        self._authorization_flows: dict[str, _AuthorizationFlow] = {}
        self._terminal_authorizations: set[str] = set()
        self._stopped = asyncio.Event()
        self._running = False
        self._background_delivery: asyncio.Task[None] | None = None

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
            if isinstance(channel, ReactionChannelAdapter):
                channel.set_reaction_handler(
                    lambda reaction, current=channel: self.receive_reaction(current, reaction),
                )
            await channel.start()

        if self.scheduler is not None:
            await self.scheduler.start()

        self._stopped.clear()
        self._running = True
        if isinstance(self.agent, BackgroundTaskRuntime) and self.agent.local_tasks is not None:
            self._background_delivery = asyncio.create_task(
                self._deliver_background_results(), name="talon-background-results"
            )
        logger.info("Talon host started for assistant %s", self.config.assistant_id)

    async def stop(self) -> None:
        """Stop managed components and cancel in-flight agent work."""
        if not self._running:
            self._stopped.set()
            return

        self._running = False
        if self._background_delivery is not None:
            self._background_delivery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background_delivery
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
        provider = await _channel_provider(channel)
        command = _command_name(message.text)
        channel_conversation_id = message.conversation_id
        conversation_root = self._conversation_root(
            provider or type(channel).__name__,
            channel_conversation_id,
        )
        async with self._locks[conversation_root]:
            agent_conversation_id = self._agent_conversation_id(conversation_root)

            if command == _NEW_COMMAND:
                await self._start_new_conversation(
                    channel,
                    channel_conversation_id,
                    conversation_root=conversation_root,
                )
                return

            if command == _STOP_COMMAND:
                await self._cancel_conversation(
                    channel,
                    agent_conversation_id,
                    reply_conversation_id=channel_conversation_id,
                )
                return

            if command == _MCP_RELOAD_COMMAND:
                await self._reload_mcp_configuration(channel, channel_conversation_id)
                return

            pending = self._pending_tool_approvals.get(agent_conversation_id)
            if pending is not None:
                authorized = pending.sender_id is None or message.sender_id == pending.sender_id
                if not authorized or _parse_tool_approval_reply(message.text) is not None:
                    await self._handle_tool_approval_reply(channel, message, pending)
                    return

            if await self._intercept_authorization_message(
                channel,
                message,
                provider=_channel_key(channel, provider),
                agent_conversation_id=agent_conversation_id,
            ):
                return

            await self._replace_agent_turn(
                channel,
                message,
                conversation_root,
                agent_conversation_id,
                provider,
            )

    async def _reload_mcp_configuration(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
    ) -> None:
        if not isinstance(self.agent, MCPReloadableRuntime):
            message = _MCP_RELOAD_UNAVAILABLE_MESSAGE
        else:
            try:
                await self.agent.reload_mcp_configuration()
            except Exception:  # noqa: BLE001  # Do not disclose config or transport errors.
                logger.warning("MCP configuration reload failed", exc_info=True)
                message = _MCP_RELOAD_FAILURE_MESSAGE
            else:
                message = _MCP_RELOAD_SUCCESS_MESSAGE
        await send_with_retry(lambda: channel.send_message(conversation_id, message))

    async def receive_reaction(self, channel: ChannelAdapter, reaction: ChannelReaction) -> None:
        """Handle one inbound channel reaction.

        Args:
            channel: Channel that delivered the reaction.
            reaction: Inbound reaction to process.
        """
        provider = await _channel_provider(channel)
        provider_key = _channel_key(channel, provider)
        conversation_root = self._conversation_root(provider_key, reaction.conversation_id)
        agent_conversation_id = self._agent_conversation_id(conversation_root)
        pending = self._pending_tool_approvals.get(agent_conversation_id)
        if pending is None:
            _log_tool_approval_reaction(
                self.config.env,
                _ReactionAudit(
                    reaction=reaction,
                    provider=provider_key,
                    decision=_parse_tool_approval_reaction(reaction.emoji),
                    match_status="ignored",
                    resolution="no_pending_approval",
                ),
            )
            return
        self._handle_tool_approval_reaction(
            reaction,
            pending,
            provider=provider_key,
            env=self.config.env,
        )

    async def _replace_agent_turn(
        self,
        channel: ChannelAdapter,
        message: ChannelMessage,
        conversation_root: str,
        conversation_id: str,
        provider: str | None,
    ) -> None:
        if conversation_id in self._blocked:
            await send_with_retry(
                lambda: channel.send_message(message.conversation_id, _CANCEL_TIMEOUT_MESSAGE)
            )
            return
        active = self._tasks.get(conversation_id)
        recovery_degraded = False
        if active is not None and not active.done():
            outcome = await self._cancel_active(conversation_id, active, recover=True)
            if outcome is _CancelOutcome.TIMEOUT:
                await send_with_retry(
                    lambda: channel.send_message(message.conversation_id, _CANCEL_TIMEOUT_MESSAGE)
                )
                return
            recovery_degraded = outcome is _CancelOutcome.DEGRADED
        generation = self._generations[conversation_id] + 1
        self._generations[conversation_id] = generation
        task = asyncio.create_task(
            self._run_agent_turn(
                channel,
                message,
                _Turn(
                    conversation_root,
                    conversation_id,
                    provider,
                    generation,
                    recovery_degraded,
                ),
            ),
            name=f"talon:{conversation_id}",
        )
        self._tasks[conversation_id] = task
        self._track_conversation_task(conversation_id, task)

    async def _run_agent_turn(
        self,
        channel: ChannelAdapter,
        message: ChannelMessage,
        turn: _Turn,
    ) -> None:
        agent_conversation_id = turn.conversation_id
        message = await transcribe_voice_message(self.voice_transcriber, message)
        message = _prepare_inbound_message(message)
        metadata: dict[str, object] = {
            "channel": turn.provider,
            "sender_id": message.sender_id,
            "message_id": message.message_id,
            **message.metadata,
            "talon_origin": {
                "channel": _channel_key(channel, turn.provider),
                "conversation_id": message.conversation_id,
                "conversation_root": turn.conversation_root,
            },
        }
        if turn.recovery_degraded:
            metadata["interruption_recovery"] = "failed"
        origin_conversation_id = _origin_conversation_id(message)
        if origin_conversation_id != agent_conversation_id:
            metadata["origin_conversation_id"] = origin_conversation_id
        content = build_model_content(message.text, dict(message.metadata))
        if content != message.text:
            metadata["model_content"] = content

        typing_task = asyncio.create_task(
            _typing_refresh_loop(channel, message.conversation_id),
        )
        suppress_result = False
        try:
            result = await self._invoke_agent(
                conversation_id=agent_conversation_id,
                text=message.text,
                metadata=metadata,
                approval_handler=lambda approval: self._request_tool_approval(
                    channel,
                    approval,
                    provider=_channel_key(channel, turn.provider),
                    reply_conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                ),
                authorization_handler=lambda event: self._handle_authorization_event(
                    channel,
                    event,
                    provider=_channel_key(channel, turn.provider),
                    reply_conversation_id=message.conversation_id,
                    agent_conversation_id=agent_conversation_id,
                    sender_id=message.sender_id,
                ),
            )
            suppress_result = agent_conversation_id in self._terminal_authorizations
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task
            self._clear_authorization(agent_conversation_id)
        async with self._locks[turn.conversation_root]:
            if (
                self._agent_conversation_id(turn.conversation_root) == agent_conversation_id
                and self._generations[agent_conversation_id] == turn.generation
                and not suppress_result
            ):
                await self._deliver_agent_result(channel, message.conversation_id, result)

    async def _deliver_background_results(self) -> None:
        if not isinstance(self.agent, BackgroundTaskRuntime) or self.agent.local_tasks is None:
            return
        supervisor = self.agent.local_tasks
        channels = {
            _channel_key(channel, await _channel_provider(channel)): channel
            for channel in self.channels
        }
        while self._running:
            try:
                for record in supervisor.pending_results():
                    if await self._deliver_background_result(record, channels):
                        supervisor.acknowledge(record["id"], revision=record["revision"])
            except Exception:  # noqa: BLE001  # delivery failures must not stop the ticker
                logger.warning("Could not deliver local subagent result; will retry")
            await asyncio.sleep(1)

    async def _deliver_background_result(
        self, record: dict[str, str], channels: Mapping[str, ChannelAdapter]
    ) -> bool:
        origin = json.loads(record["origin"])
        channel = channels.get(origin.get("channel"))
        root, reply = origin.get("conversation_root"), origin.get("conversation_id")
        if channel is None or not isinstance(root, str) or not isinstance(reply, str):
            return True
        async with self._locks[root]:
            if (
                isinstance(self.agent, BackgroundTaskRuntime)
                and self.agent.local_tasks is not None
                and not self.agent.local_tasks.result_is_current(record["id"], record["revision"])
            ):
                return False
            if self._agent_conversation_id(root) != record["owner"]:
                return True
            active = self._tasks.get(record["owner"])
            if record["owner"] in self._blocked or (active is not None and not active.done()):
                return False
            text = f"Local task {record['id']} ({record['status']}):\n{record['result']}"
            result = await send_with_retry(lambda: channel.send_message(reply, text))
            return result.success

    async def run_scheduled_job(self, job: CronJob) -> str:
        """Invoke the agent for one scheduled job.

        Args:
            job: Claimed cron job to run.

        Returns:
            Agent text output for scheduler delivery handling.
        """
        conversation_id = f"{job.id}{_CRON_THREAD_SUFFIX}"
        control = self._cron_controls.setdefault(job.id, _CronControl(asyncio.Lock()))
        control.users += 1
        try:
            async with control.lock:
                result = await self._invoke_agent(
                    conversation_id=conversation_id,
                    text=job.prompt,
                    metadata={
                        "channel": job.origin.channel,
                        "cron_job_id": job.id,
                        "cron_job_name": job.name,
                        "origin_conversation_id": job.origin.conversation_id,
                        "cron_origin_message_id": job.origin.message_id,
                        "trigger": "cron",
                    },
                )
                return result.text
        finally:
            control.users -= 1
            if control.users == 0 and self._cron_controls.get(job.id) is control:
                del self._cron_controls[job.id]

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
        await send_with_retry(lambda: channel.send_message(job.origin.conversation_id, text))

    async def _invoke_agent(
        self,
        *,
        conversation_id: str,
        text: str,
        metadata: dict[str, object],
        approval_handler: Callable[[ToolApprovalRequest], Awaitable[ToolApprovalDecision]]
        | None = None,
        authorization_handler: Callable[[AuthorizationEvent], Awaitable[str | None]] | None = None,
    ) -> AgentResult:
        try:
            with langsmith_trace_context(
                self.config.env,
                assistant_id=self.config.assistant_id,
                conversation_id=conversation_id,
                metadata={key: value for key, value in metadata.items() if key != "talon_origin"},
            ):
                return await self.agent.invoke(
                    AgentRequest(
                        conversation_id=conversation_id,
                        text=text,
                        metadata=metadata,
                        approval_handler=approval_handler,
                        authorization_handler=authorization_handler,
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

    async def _start_new_conversation(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
        *,
        conversation_root: str,
    ) -> None:
        current_conversation_id = self._agent_conversation_id(conversation_root)
        outcome = await self._cancel_conversation_tasks(current_conversation_id)
        if outcome is _CancelOutcome.TIMEOUT:
            await send_with_retry(
                lambda: channel.send_message(conversation_id, _CANCEL_TIMEOUT_MESSAGE)
            )
            return
        next_resets = {
            **self._conversation_resets,
            conversation_root: self._conversation_resets.get(conversation_root, 0) + 1,
        }
        _save_conversation_resets(self.config.conversation_state_path, next_resets)
        self._conversation_resets = next_resets
        await send_with_retry(
            lambda: channel.send_message(conversation_id, _NEW_CONVERSATION_MESSAGE)
        )

    async def _cancel_conversation(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
        *,
        reply_conversation_id: str | None = None,
    ) -> None:
        target_conversation_id = reply_conversation_id or conversation_id
        outcome = await self._cancel_conversation_tasks(conversation_id)
        if outcome is _CancelOutcome.NONE:
            message = "No in-flight run to stop."
        elif outcome is _CancelOutcome.TIMEOUT:
            message = _CANCEL_TIMEOUT_MESSAGE
        elif outcome is _CancelOutcome.DEGRADED:
            message = "Stopped current run."
        else:
            message = "Stopped current run."
        await send_with_retry(lambda: channel.send_message(target_conversation_id, message))

    async def _cancel_conversation_tasks(self, conversation_id: str) -> _CancelOutcome:
        task = self._tasks.get(conversation_id)
        if task is None or task.done():
            return _CancelOutcome.NONE
        return await self._cancel_active(conversation_id, task, recover=True)

    async def _cancel_active(
        self,
        conversation_id: str,
        task: asyncio.Task[None],
        *,
        recover: bool,
    ) -> _CancelOutcome:
        self._generations[conversation_id] += 1
        task.cancel()
        deadline = asyncio.get_running_loop().time() + _CANCEL_TIMEOUT_SECONDS
        try:
            done, _ = await asyncio.wait({task}, timeout=_CANCEL_TIMEOUT_SECONDS)
            if not done:
                return self._mark_cancellation_timeout(conversation_id)
            with contextlib.suppress(asyncio.CancelledError):
                task.result()
            if recover:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return self._mark_cancellation_timeout(conversation_id)
                await asyncio.wait_for(
                    self.agent.recover_interrupted(conversation_id),
                    timeout=remaining,
                )
        except TimeoutError:
            return self._mark_cancellation_timeout(conversation_id)
        except Exception:
            logger.exception(
                "Failed to recover interrupted conversation %s",
                stable_log_ref(conversation_id),
            )
            return _CancelOutcome.DEGRADED
        log_event(
            logger,
            "agent.interrupted",
            conversation_ref=stable_log_ref(conversation_id),
        )
        return _CancelOutcome.SUCCESS

    def _mark_cancellation_timeout(
        self,
        conversation_id: str,
    ) -> _CancelOutcome:
        self._blocked.add(conversation_id)
        log_event(
            logger,
            "agent.interrupt_timeout",
            conversation_ref=stable_log_ref(conversation_id),
        )
        return _CancelOutcome.TIMEOUT

    def _agent_conversation_id(self, conversation_id: str) -> str:
        reset = self._conversation_resets.get(conversation_id, 0)
        if reset == 0:
            return conversation_id
        return f"{conversation_id}{_RESET_THREAD_SEPARATOR}{reset}"

    def _conversation_root(
        self,
        channel_key: str,
        conversation_id: str,
    ) -> str:
        if len(self.channels) <= 1:
            return conversation_id
        return _conversation_key(channel_key, conversation_id)

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
        for pending in self._pending_tool_approvals.values():
            if not pending.future.done():
                pending.future.cancel()
        self._pending_tool_approvals.clear()
        for pending in self._pending_authorizations.values():
            if not pending.future.done():
                pending.future.cancel()
        self._pending_authorizations.clear()
        self._authorization_flows.clear()
        self._terminal_authorizations.clear()

    async def _handle_authorization_event(  # noqa: PLR0913  # binds all channel identities.
        self,
        channel: ChannelAdapter,
        event: AuthorizationEvent,
        *,
        provider: str,
        reply_conversation_id: str,
        agent_conversation_id: str,
        sender_id: str | None,
    ) -> str | None:
        if sender_id is None:
            msg = "Channel authorization requires an identified operator"
            raise RuntimeError(msg)
        if isinstance(event, AuthorizationURL):
            await self._begin_authorization(
                channel,
                event,
                provider=provider,
                reply_conversation_id=reply_conversation_id,
                agent_conversation_id=agent_conversation_id,
                sender_id=sender_id,
            )
            return None
        if isinstance(event, CallbackURLRequested):
            return await self._await_authorization_callback(event, agent_conversation_id)
        if isinstance(event, DeviceCode):
            self._register_authorization_flow(
                agent_conversation_id,
                event.binding,
                provider=provider,
                channel_conversation_id=reply_conversation_id,
                sender_id=sender_id,
            )
            await send_with_retry(
                lambda: channel.send_message(
                    reply_conversation_id,
                    "\n".join(
                        (
                            f"Authorization required for MCP server `{event.binding.server_name}`.",
                            f"Open: {event.verification_uri}",
                            f"Enter code: `{event.user_code}`",
                        )
                    ),
                )
            )
            return None
        if isinstance(event, AuthorizationCompleted):
            self._finish_authorization_flow(agent_conversation_id, event.binding)
            result = await send_with_retry(
                lambda: channel.send_message(
                    reply_conversation_id,
                    f"MCP server `{event.binding.server_name}` is authorized.",
                )
            )
            if event.terminal and result.success:
                self._terminal_authorizations.add(agent_conversation_id)
            return None
        if isinstance(event, AuthorizationFailed):
            self._finish_authorization_flow(agent_conversation_id, event.binding)
            await send_with_retry(
                lambda: channel.send_message(
                    reply_conversation_id,
                    f"Authorization for MCP server `{event.binding.server_name}` failed.",
                )
            )
            return None
        msg = "Unsupported authorization event"
        raise TypeError(msg)

    async def _begin_authorization(  # noqa: PLR0913  # persists all channel identities.
        self,
        channel: ChannelAdapter,
        event: AuthorizationURL,
        *,
        provider: str,
        reply_conversation_id: str,
        agent_conversation_id: str,
        sender_id: str,
    ) -> None:
        if event.binding.expires_at <= asyncio.get_running_loop().time():
            msg = "MCP authorization request expired"
            raise TimeoutError(msg)
        existing = self._pending_authorizations.get(agent_conversation_id)
        if existing is not None or agent_conversation_id in self._authorization_flows:
            msg = "Another MCP authorization request is already pending"
            raise RuntimeError(msg)
        future = asyncio.get_running_loop().create_future()
        pending = _PendingAuthorization(
            future=future,
            binding=event.binding,
            provider=provider,
            channel_conversation_id=reply_conversation_id,
            agent_conversation_id=agent_conversation_id,
            sender_id=sender_id,
        )
        self._pending_authorizations[agent_conversation_id] = pending
        self._authorization_flows[agent_conversation_id] = _AuthorizationFlow(
            binding=event.binding,
            provider=provider,
            channel_conversation_id=reply_conversation_id,
            sender_id=sender_id,
        )
        result = await send_with_retry(
            lambda: channel.send_message(
                reply_conversation_id,
                "\n".join(
                    (
                        f"Authorization required for MCP server `{event.binding.server_name}`.",
                        "Open this link and approve access:",
                        event.url,
                        "Then paste the full callback URL here.",
                    )
                ),
            )
        )
        if not result.success:
            if self._pending_authorizations.get(agent_conversation_id) is pending:
                del self._pending_authorizations[agent_conversation_id]
            self._authorization_flows.pop(agent_conversation_id, None)
            msg = "Could not deliver MCP authorization request"
            raise RuntimeError(msg)

    async def _await_authorization_callback(
        self,
        event: CallbackURLRequested,
        agent_conversation_id: str,
    ) -> str:
        pending = self._pending_authorizations.get(agent_conversation_id)
        if pending is None or pending.binding != event.binding:
            msg = "MCP authorization request does not match the active invocation"
            raise RuntimeError(msg)
        try:
            remaining = event.binding.expires_at - asyncio.get_running_loop().time()
            if remaining <= 0:
                if pending.future.done():
                    return pending.future.result()
                msg = "MCP authorization request expired"
                raise TimeoutError(msg)
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout=remaining)
        finally:
            if self._pending_authorizations.get(agent_conversation_id) is pending:
                del self._pending_authorizations[agent_conversation_id]
            if not pending.future.done():
                pending.future.cancel()

    async def _intercept_authorization_message(
        self,
        channel: ChannelAdapter,
        message: ChannelMessage,
        *,
        provider: str,
        agent_conversation_id: str,
    ) -> bool:
        callback_url = _callback_url(message.text)
        pending = self._pending_authorizations.get(agent_conversation_id)
        if pending is None:
            flow = self._authorization_flows.get(agent_conversation_id)
            if flow is not None:
                await self._remind_device_authorization(
                    channel,
                    message,
                    provider=provider,
                    flow=flow,
                )
            elif callback_url is None:
                return False
            else:
                await send_with_retry(
                    lambda: channel.send_message(
                        message.conversation_id,
                        "No matching MCP authorization request is pending.",
                    )
                )
            return True
        if (
            provider != pending.provider
            or message.conversation_id != pending.channel_conversation_id
            or message.sender_id != pending.sender_id
        ):
            await send_with_retry(
                lambda: channel.send_message(
                    message.conversation_id,
                    "Only the operator who started this authorization can complete it.",
                )
            )
            return True
        if asyncio.get_running_loop().time() >= pending.binding.expires_at:
            if not pending.future.done():
                msg = "MCP authorization request expired"
                pending.future.set_exception(TimeoutError(msg))
            return True
        if callback_url is None:
            await send_with_retry(
                lambda: channel.send_message(
                    message.conversation_id,
                    "Paste the full callback URL to finish MCP authorization, or send `/stop`.",
                )
            )
            return True
        if not pending.future.done():
            pending.future.set_result(callback_url)
        return True

    async def _remind_device_authorization(
        self,
        channel: ChannelAdapter,
        message: ChannelMessage,
        *,
        provider: str,
        flow: _AuthorizationFlow,
    ) -> None:
        if (
            provider != flow.provider
            or message.conversation_id != flow.channel_conversation_id
            or message.sender_id != flow.sender_id
        ):
            text = "Only the operator who started this authorization can complete it."
        elif asyncio.get_running_loop().time() >= flow.binding.expires_at:
            text = f"Authorization for MCP server `{flow.binding.server_name}` has expired."
        else:
            text = (
                f"Complete authorization for MCP server `{flow.binding.server_name}` "
                "in your browser, or send `/stop`."
            )
        await send_with_retry(lambda: channel.send_message(message.conversation_id, text))

    def _register_authorization_flow(
        self,
        agent_conversation_id: str,
        binding: AuthorizationBinding,
        *,
        provider: str,
        channel_conversation_id: str,
        sender_id: str,
    ) -> None:
        if binding.expires_at <= asyncio.get_running_loop().time():
            msg = "MCP authorization request expired"
            raise TimeoutError(msg)
        if agent_conversation_id in self._authorization_flows:
            msg = "Another MCP authorization request is already pending"
            raise RuntimeError(msg)
        self._authorization_flows[agent_conversation_id] = _AuthorizationFlow(
            binding=binding,
            provider=provider,
            channel_conversation_id=channel_conversation_id,
            sender_id=sender_id,
        )

    def _finish_authorization_flow(
        self,
        agent_conversation_id: str,
        binding: AuthorizationBinding,
    ) -> None:
        flow = self._authorization_flows.get(agent_conversation_id)
        if flow is None or flow.binding != binding:
            msg = "MCP authorization status does not match the active invocation"
            raise RuntimeError(msg)
        del self._authorization_flows[agent_conversation_id]

    def _clear_authorization(self, agent_conversation_id: str) -> None:
        pending = self._pending_authorizations.pop(agent_conversation_id, None)
        if pending is not None and not pending.future.done():
            pending.future.cancel()
        self._authorization_flows.pop(agent_conversation_id, None)
        self._terminal_authorizations.discard(agent_conversation_id)

    async def _request_tool_approval(
        self,
        channel: ChannelAdapter,
        approval: ToolApprovalRequest,
        *,
        provider: str,
        reply_conversation_id: str,
        sender_id: str | None,
    ) -> ToolApprovalDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolApprovalDecision] = loop.create_future()
        pending = _PendingToolApproval(
            future=future,
            provider=provider,
            channel_conversation_id=reply_conversation_id,
            agent_conversation_id=approval.conversation_id,
            prompt_message_id=None,
            sender_id=sender_id,
        )
        self._pending_tool_approvals[approval.conversation_id] = pending
        try:
            result = await send_with_retry(
                lambda: channel.send_message(
                    reply_conversation_id,
                    _format_tool_approval_prompt(approval),
                )
            )
            pending.prompt_message_id = result.message_id
            return await future
        finally:
            if self._pending_tool_approvals.get(approval.conversation_id) is pending:
                del self._pending_tool_approvals[approval.conversation_id]

    async def _handle_tool_approval_reply(
        self,
        channel: ChannelAdapter,
        message: ChannelMessage,
        pending: _PendingToolApproval,
    ) -> None:
        if pending.sender_id is not None and message.sender_id != pending.sender_id:
            await send_with_retry(
                lambda: channel.send_message(
                    message.conversation_id,
                    "Only the operator who started this run can approve or deny it.",
                )
            )
            return

        decision = _parse_tool_approval_reply(message.text)
        if decision is None:
            await send_with_retry(
                lambda: channel.send_message(
                    message.conversation_id,
                    "Reply `approve` to run the tool call or `deny` to skip it.",
                )
            )
            return

        if not pending.future.done():
            pending.future.set_result(decision)

    def _handle_tool_approval_reaction(
        self,
        reaction: ChannelReaction,
        pending: _PendingToolApproval,
        *,
        provider: str,
        env: Mapping[str, str],
    ) -> None:
        decision = _parse_tool_approval_reaction(reaction.emoji)
        resolution = _reaction_mismatch_resolution(
            reaction,
            pending,
            provider=provider,
            decision=decision,
        )
        if resolution is not None:
            _log_tool_approval_reaction(
                env,
                _ReactionAudit(
                    reaction=reaction,
                    provider=provider,
                    decision=decision,
                    match_status="ignored",
                    resolution=resolution,
                ),
            )
            return
        _log_tool_approval_reaction(
            env,
            _ReactionAudit(
                reaction=reaction,
                provider=provider,
                decision=decision,
                match_status="matched",
                resolution="operator_reaction",
            ),
        )
        if not pending.future.done():
            pending.future.set_result(cast("ToolApprovalDecision", decision))

    async def _deliver_agent_result(
        self,
        channel: ChannelAdapter,
        conversation_id: str,
        result: AgentResult,
    ) -> None:
        cleaned, refs = extract_markdown_media(result.text)
        if not refs:
            if result.text:
                await send_with_retry(lambda: channel.send_message(conversation_id, result.text))
            return

        media, failed = _outbound_media_from_refs(
            refs,
            cleaned,
            root=outbound_media_root_from_env(self.config.env),
        )
        text = _with_failed_attachment_text(cleaned, failed)
        sent_media, send_failed = await _send_channel_media(
            channel,
            conversation_id,
            media,
            fallback_caption=text,
        )
        if text and not sent_media:
            await send_with_retry(lambda: channel.send_message(conversation_id, text))
        elif send_failed and sent_media:
            await send_with_retry(
                lambda: channel.send_message(
                    conversation_id,
                    f"_(Could not attach: {', '.join(send_failed)}.)_",
                )
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


def _callback_url(text: str) -> str | None:
    return extract_oauth_callback_url(text)


def _command_name(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None
    first = parts[0].lower()
    if not first.startswith("/"):
        return None
    return first.split("@", maxsplit=1)[0]


def _origin_conversation_id(message: ChannelMessage) -> str:
    origin = message.metadata.get("chat_id_from")
    if isinstance(origin, str) and origin:
        return origin
    return message.conversation_id


def _outbound_media_from_refs(
    refs: list[MarkdownMediaRef],
    cleaned_text: str,
    *,
    root: Path,
) -> tuple[list[ChannelMedia], list[str]]:
    media: list[ChannelMedia] = []
    failed: list[str] = []
    for index, ref in enumerate(refs):
        caption = cleaned_text if index == 0 and cleaned_text else getattr(ref, "alt", "") or None
        try:
            media.append(outbound_channel_media(ref, caption=caption, root=root))
        except ValueError:
            path = getattr(ref, "path", None)
            failed.append(getattr(ref, "alt", "") or getattr(path, "name", "attachment"))
    return media, failed


def _conversation_key(provider: str, conversation_id: str) -> str:
    return f"{provider}:{conversation_id}"


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
        result = await send_with_retry(lambda p=payload: channel.send_media(conversation_id, p))
        if result.success:
            sent = True
        else:
            logger.warning(
                "Could not send outbound media: %s (%s)",
                payload.path,
                result.error,
            )
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
    try:
        await channel.send_typing(conversation_id)
    except Exception:  # noqa: BLE001  # typing indicators are best-effort adapter calls.
        logger.debug("Could not send typing indicator", exc_info=True)


async def _typing_refresh_loop(channel: ChannelAdapter, conversation_id: str) -> None:
    """Repeat the typing indicator for as long as an agent turn is in flight."""
    while True:
        await _send_typing(channel, conversation_id)
        await asyncio.sleep(_TYPING_REFRESH_SECONDS)


async def _channel_provider(channel: ChannelAdapter) -> str | None:
    """Return the channel provider for origin metadata, if available."""
    try:
        return (await channel.status()).provider
    except Exception:  # noqa: BLE001
        logger.warning("Could not resolve channel provider for agent metadata", exc_info=True)
        return None


def _channel_key(channel: ChannelAdapter, provider: str | None) -> str:
    return provider or type(channel).__name__


def _format_tool_approval_prompt(approval: ToolApprovalRequest) -> str:
    lines = ["Tool approval required."]
    for index, action in enumerate(approval.action_requests, start=1):
        name = action.get("name")
        tool_name = name if isinstance(name, str) and name else "unknown"
        lines.append(f"{index}. `{tool_name}`")
        args = action.get("args")
        if isinstance(args, dict) and args:
            lines.append(f"Args: `{_json_preview(args)}`")
        elif args not in (None, {}, []):
            lines.append(f"Args: `{args}`")
    lines.append("Reply `👍` / `approve` to run or `👎` / `deny` to skip.")
    return "\n".join(lines)


def _json_preview(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _load_conversation_resets(path: Path) -> dict[str, int]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        msg = f"failed to load conversation state from {path}"
        raise RuntimeError(msg) from exc
    if not isinstance(state, dict) or any(
        not isinstance(key, str)
        or not isinstance(reset, int)
        or isinstance(reset, bool)
        or reset < 1
        for key, reset in state.items()
    ):
        msg = f"invalid conversation state in {path}"
        raise RuntimeError(msg)
    return state


def _save_conversation_resets(path: Path, resets: Mapping[str, int]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(resets, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary_path.unlink()
        raise


def _parse_tool_approval_reply(text: str) -> ToolApprovalDecision | None:
    normalized = text.strip().lower().strip(".! ")
    if not normalized:
        return None
    first = normalized.split(maxsplit=1)[0]
    reaction_decision = _parse_tool_approval_reaction(first)
    if reaction_decision is not None:
        return reaction_decision
    if first in _APPROVE_REPLIES:
        return "approve"
    if first in _DENY_REPLIES:
        return "reject"
    return None


def _parse_tool_approval_reaction(emoji: str) -> ToolApprovalDecision | None:
    normalized = _normalize_reaction_emoji(emoji)
    if normalized == "\U0001f44d":
        return "approve"
    if normalized == "\U0001f44e":
        return "reject"
    return None


def _normalize_reaction_emoji(emoji: str) -> str:
    return "".join(
        char
        for char in emoji.strip()
        if char != _EMOJI_VARIATION_SELECTOR and char not in _EMOJI_SKIN_TONES
    )


def _reaction_mismatch_resolution(
    reaction: ChannelReaction,
    pending: _PendingToolApproval,
    *,
    provider: str,
    decision: ToolApprovalDecision | None,
) -> str | None:
    checks = (
        (decision is None, "unsupported_emoji"),
        (pending.prompt_message_id is None, "missing_prompt_message_id"),
        (provider != pending.provider, "provider_mismatch"),
        (reaction.conversation_id != pending.channel_conversation_id, "conversation_mismatch"),
        (reaction.message_id != pending.prompt_message_id, "message_mismatch"),
        (pending.sender_id is not None and reaction.sender_id is None, "sender_missing"),
        (
            pending.sender_id is not None and reaction.sender_id != pending.sender_id,
            "sender_mismatch",
        ),
    )
    for failed, resolution in checks:
        if failed:
            return resolution
    return None


def _log_tool_approval_reaction(
    env: Mapping[str, str],
    audit: _ReactionAudit,
) -> None:
    fields: dict[str, object] = {
        "provider": audit.provider,
        "channel_conversation_ref": stable_log_ref(audit.reaction.conversation_id),
        "prompt_message_ref": stable_log_ref(audit.reaction.message_id),
        "emoji": audit.reaction.emoji,
        "decision": audit.decision,
        "match_status": audit.match_status,
        "resolution": audit.resolution,
    }
    if audit.reaction.sender_id is not None:
        fields["reacting_sender_ref"] = stable_log_ref(audit.reaction.sender_id)
    if _approval_log_raw_ids(env):
        fields.update(
            {
                "raw_channel_conversation_id": audit.reaction.conversation_id,
                "raw_prompt_message_id": audit.reaction.message_id,
            }
        )
        if audit.reaction.sender_id is not None:
            fields["raw_reacting_sender_id"] = audit.reaction.sender_id
    log_event(logger, "tool_approval.reaction", **fields)


def _approval_log_raw_ids(env: Mapping[str, str]) -> bool:
    return env.get(_APPROVAL_LOG_RAW_IDS_ENV, "").lower() == "true"
