"""Single-owner coordinator for client-side Hooks v2 state.

`HooksManager` is the only place in the client that builds or holds a
`HooksRuntime`. The Textual app, the Textual stream adapter, and the headless
runner each hold a manager and call intention-revealing lifecycle methods on
it; none of them inspect the runtime, the hook service, or their availability.

A manager whose configuration failed to load stays usable and answers every
call with a neutral result, so consumers never need an availability check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents_code.hooks.client_lifecycle import (
    ClientHookContext,
    ClientHookService,
    ClientHookStopError,
)
from deepagents_code.hooks.models.domain import HookEvent
from deepagents_code.hooks.permissions import (
    PermissionHookOutcome,
    PermissionPlan,
)
from deepagents_code.hooks.presenter import HookPresenter
from deepagents_code.hooks.trust import WorkspaceTrust

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from langchain_core.messages import BaseMessage

    from deepagents_code._cli_context import CLIContext
    from deepagents_code.approval_mode import ApprovalMode
    from deepagents_code.hooks.models.domain import (
        CompactTrigger,
        DcodeNotificationKind,
        SessionEndCause,
        SessionStartCause,
        ToolCallData,
    )
    from deepagents_code.hooks.presenter import (
        HookNoticeCallback,
        HookStatusCallback,
    )
    from deepagents_code.hooks.runtime import HooksRuntime
    from deepagents_code.hooks.transcript import TranscriptRecorder
    from deepagents_code.plugins.models import PluginInstance


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HookSessionIdentity:
    """Live client identity projected into every hook invocation."""

    thread_id: str
    approval_mode: ApprovalMode
    prompt_id: str | UUID | None = None


SessionIdentityProvider = Callable[[], HookSessionIdentity]
"""Reads current session identity at invocation time, never cached."""


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """Result of a lifecycle hook that may halt the caller."""

    ok: bool = True
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PromptOutcome:
    """Result of `UserPromptSubmit`, including its prompt rewrites."""

    ok: bool = True
    stop_reason: str | None = None
    context: tuple[str, ...] = ()
    suppress_original_prompt: bool = False


@dataclass(frozen=True, slots=True)
class _InertTranscriptRuntime:
    """Transcript sink used when hooks are unavailable."""

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        *,
        agent_id: str | None = None,
    ) -> None:
        """Discard messages; no transcript is materialized for hooks to read."""


@dataclass(slots=True)
class HooksManager:
    """Owns the Hooks v2 runtime, presenter, hook service, and transcripts.

    The presenter is the manager's, not the runtime's: one instance is created
    once and handed to every runtime the manager loads, so a reload or a late
    UI attachment never leaves two presenters competing for the same output.
    """

    identity: SessionIdentityProvider
    presenter: HookPresenter = field(default_factory=HookPresenter)
    _runtime: HooksRuntime | None = None
    trust: WorkspaceTrust = field(default_factory=WorkspaceTrust)
    """Policy re-resolved on every reload; the manager is its only interpreter."""

    _service: ClientHookService | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Derive the hook service from whatever runtime was supplied."""
        self._service = self._build_service()

    @classmethod
    def create(
        cls,
        *,
        cwd: Path,
        identity: SessionIdentityProvider,
        notice: HookNoticeCallback | None = None,
        status: HookStatusCallback | None = None,
        trust: WorkspaceTrust | None = None,
    ) -> HooksManager:
        """Load hook configuration and return a ready manager.

        Hooks v2 is in progress, so it stays off unless
        `DEEPAGENTS_CODE_EXPERIMENTAL` is truthy; without it the manager is
        inert and no hook (client- or server-owned) fires.

        Never raises: a failed load yields an inert manager whose lifecycle
        methods are all no-ops.

        Args:
            cwd: Session working directory used to resolve hook configuration.
            identity: Reads current thread, approval mode, and prompt id.
            notice: Sink for user-visible notices. When omitted, output is only
                logged until `attach_output` binds a sink.
            status: Sink for transient hook-owned status text.
            trust: Project-hook trust policy. Defaults to trusting nothing
                beyond what the persisted trust store already records.

        Returns:
            A manager owning the loaded runtime, or an inert one on failure.
        """
        policy = trust if trust is not None else WorkspaceTrust.none()
        presenter = HookPresenter(notice=notice, status=status)
        runtime = _load_runtime(cwd, trust=policy, presenter=presenter)
        _present_load_diagnostics(runtime)
        return cls(identity, presenter, runtime, policy)

    @classmethod
    def adopting(
        cls,
        runtime: HooksRuntime | None,
        *,
        identity: SessionIdentityProvider,
        notice: HookNoticeCallback | None = None,
        status: HookStatusCallback | None = None,
    ) -> HooksManager:
        """Wrap an already-loaded runtime.

        A supplied runtime brought its own presenter, so that one is adopted
        rather than displaced; the given sinks are bound onto it so the runtime
        and the manager keep sharing a single instance.

        Args:
            runtime: Preloaded runtime, or `None` when loading failed.
            identity: Reads current thread, approval mode, and prompt id.
            notice: Sink for user-visible notices.
            status: Sink for transient hook-owned status text.

        Returns:
            A manager owning `runtime`.
        """
        if runtime is None:
            return cls(identity, HookPresenter(notice=notice, status=status))
        if notice is not None or status is not None:
            runtime.presenter.attach(notice=notice, status=status)
        return cls(identity, runtime.presenter, runtime)

    @classmethod
    def inert(cls) -> HooksManager:
        """Return a manager that runs nothing, for state built before setup.

        Returns:
            A manager with neutral identity and no runtime.
        """
        from deepagents_code.approval_mode import ApprovalMode

        return cls(lambda: HookSessionIdentity("", ApprovalMode.MANUAL))

    def attach_output(
        self,
        *,
        notice: HookNoticeCallback | None,
        status: HookStatusCallback | None = None,
    ) -> None:
        """Route hook notices, diagnostics, and progress to a client's UI.

        For callers handed a manager that was loaded before their UI existed.
        Load diagnostics are re-presented so anything the earlier load could
        only log now reaches the user.

        Args:
            notice: Sink for user-visible notices.
            status: Sink for transient hook-owned status text.
        """
        self.presenter.attach(notice=notice, status=status)
        _present_load_diagnostics(self._runtime)

    @property
    def enabled(self) -> bool:
        """Whether hook configuration loaded successfully for this session."""
        return self._service is not None

    def has_handlers(self, event: HookEvent) -> bool:
        """Return whether any handler is configured for `event`.

        Args:
            event: Lifecycle event to inspect.

        Returns:
            `False` whenever hooks are unavailable.
        """
        service = self._service
        return service is not None and service.has_handlers(event)

    async def reload(
        self, *, cwd: Path, plugins: tuple[PluginInstance, ...] | None = None
    ) -> None:
        """Rebuild the runtime after configuration or working-directory changes.

        Workspace trust is re-resolved for `cwd`, so moving from a trusted
        project into an untrusted one drops project hooks instead of carrying
        the previous grant forward. The presenter survives the reload, so the
        client's output sinks stay bound.

        Pending `SessionStart` context is dropped with the old runtime, matching
        the lifecycle boundary that triggers a reload.

        Args:
            cwd: New session working directory.
            plugins: Already-discovered plugins, or `None` to discover them.
        """
        import asyncio

        self._runtime = await asyncio.to_thread(
            _load_runtime,
            cwd,
            trust=self.trust,
            presenter=self.presenter,
            plugins=plugins,
        )
        self._service = self._build_service()
        _present_load_diagnostics(self._runtime)

    async def on_session_start(
        self,
        cause: SessionStartCause,
        *,
        model: str | None = None,
    ) -> HookOutcome:
        """Run `SessionStart` and accumulate any context it returns.

        Context is retained rather than returned; `take_pending_context`
        consumes it when the next model turn is assembled.

        Args:
            cause: Lifecycle boundary that started the session.
            model: Active model identifier, when known.

        Returns:
            `ok=False` only when a handler explicitly stopped processing.
        """
        service = self._service
        if service is None or not service.has_handlers(HookEvent.SESSION_START):
            return HookOutcome()
        try:
            decision = await service.session_start(
                self._context(),
                cause,
                model=model,
            )
        except Exception:
            logger.warning("SessionStart hook invocation failed", exc_info=True)
            return HookOutcome()
        if decision.continue_processing:
            return HookOutcome()
        return HookOutcome(
            ok=False,
            stop_reason=decision.stop_reason or "Session start was stopped by a hook.",
        )

    async def on_session_end(
        self,
        cause: SessionEndCause,
        *,
        thread_id: str | None = None,
    ) -> None:
        """Run `SessionEnd` for the outgoing thread.

        Args:
            cause: Reason the session ended.
            thread_id: Outgoing thread, when it differs from current identity.
        """
        service = self._service
        if service is None or not service.has_handlers(HookEvent.SESSION_END):
            return
        try:
            await service.session_end(self._context(thread_id=thread_id), cause)
        except Exception:
            logger.warning("SessionEnd hook invocation failed", exc_info=True)

    async def on_user_prompt(self, prompt: str) -> PromptOutcome:
        """Run `UserPromptSubmit` before a user turn reaches the model.

        Args:
            prompt: Original user prompt.

        Returns:
            The handlers' verdict plus any injected context or suppression.
        """
        service = self._service
        if service is None or not service.has_handlers(HookEvent.USER_PROMPT_SUBMIT):
            return PromptOutcome()
        try:
            decision = await service.user_prompt_submit(self._context(), prompt)
        except Exception:
            logger.warning("UserPromptSubmit hook invocation failed", exc_info=True)
            return PromptOutcome()
        if not decision.continue_processing:
            return PromptOutcome(
                ok=False,
                stop_reason=(
                    decision.stop_reason or "User prompt submission stopped by hook"
                ),
            )
        return PromptOutcome(
            context=tuple(decision.context),
            suppress_original_prompt=decision.suppress_original_prompt,
        )

    async def on_pre_compact(
        self,
        trigger: CompactTrigger,
        *,
        custom_instructions: str = "",
    ) -> HookOutcome:
        """Run `PreCompact` before conversation compaction.

        Args:
            trigger: Manual or automatic compaction source.
            custom_instructions: Optional compaction instructions.

        Returns:
            `ok=False` only when a handler explicitly stopped processing.
        """
        service = self._service
        if service is None or not service.has_handlers(HookEvent.PRE_COMPACT):
            return HookOutcome()
        try:
            decision = await service.pre_compact(
                self._context(),
                trigger,
                custom_instructions=custom_instructions,
            )
        except Exception:
            logger.warning("PreCompact hook invocation failed", exc_info=True)
            return HookOutcome()
        if decision.continue_processing:
            return HookOutcome()
        return HookOutcome(
            ok=False,
            stop_reason=decision.stop_reason or "Compaction stopped by hook",
        )

    async def on_permission_request(
        self,
        calls: Sequence[ToolCallData | None],
    ) -> PermissionPlan:
        """Run `PermissionRequest` for each pending tool call.

        Args:
            calls: Tool actions awaiting approval, in request order. A `None`
                entry is left to human review without invoking any handler.

        Returns:
            One outcome per call, in the same order. Unresolved entries carry
            a `None` decision and fall through to human review.
        """
        service = self._service
        if service is None or not service.has_handlers(HookEvent.PERMISSION_REQUEST):
            return PermissionPlan(tuple(PermissionHookOutcome(None) for _ in calls))
        context = self._context()
        outcomes: list[PermissionHookOutcome] = []
        for call in calls:
            if call is None:
                outcomes.append(PermissionHookOutcome(None))
                continue
            try:
                outcome = await service.resolve_permission(context, call)
            except Exception:
                logger.warning(
                    "PermissionRequest hook invocation failed",
                    exc_info=True,
                )
                outcomes.append(PermissionHookOutcome(None))
                continue
            outcomes.append(outcome)
        return PermissionPlan(tuple(outcomes))

    async def notify(
        self,
        kind: DcodeNotificationKind,
        message: str,
        *,
        title: str | None = None,
    ) -> None:
        """Run `Notification` for one supported dcode notification.

        Args:
            kind: Supported dcode notification kind.
            message: User-facing notification text.
            title: Optional notification title.

        Raises:
            ClientHookStopError: If a handler stopped lifecycle processing.
        """
        service = self._service
        if service is None:
            return
        try:
            await service.notification(self._context(), kind, message, title=title)
        except ClientHookStopError:
            raise
        except Exception:
            logger.warning("Notification hook invocation failed", exc_info=True)

    def take_pending_context(self, *, thread_id: str | None = None) -> tuple[str, ...]:
        """Consume `SessionStart` context accumulated for the next model turn.

        Args:
            thread_id: Thread to drain, defaulting to current identity.

        Returns:
            Ordered context strings, removed from the service.
        """
        service = self._service
        if service is None:
            return ()
        return service.take_session_context(thread_id or self.identity().thread_id)

    def recorder(self, thread_id: str) -> TranscriptRecorder:
        """Return a transcript recorder for one stream.

        Args:
            thread_id: Thread whose transcript the stream contributes to.

        Returns:
            A recorder that discards messages when hooks are unavailable.
        """
        from deepagents_code.hooks.transcript import TranscriptRecorder

        runtime = self._runtime
        return TranscriptRecorder(
            runtime if runtime is not None else _InertTranscriptRuntime(),
            thread_id,
        )

    def record_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        thread_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Project checkpoint messages into the client transcript.

        Args:
            messages: LangChain messages to project.
            thread_id: Thread to record under, defaulting to current identity.
            agent_id: Optional subagent scope.
        """
        runtime = self._runtime
        if runtime is None or not messages:
            return
        runtime.append_messages(
            thread_id or self.identity().thread_id,
            messages,
            agent_id=agent_id,
        )

    def apply_graph_context(self, context: CLIContext) -> CLIContext:
        """Stamp snapshot identity and server event gates onto graph context.

        Args:
            context: Mutable per-run graph context.

        Returns:
            The same context, updated in place.
        """
        from deepagents_code.hooks.context import apply_hooks_context

        prompt_id = self.identity().prompt_id
        return apply_hooks_context(
            context,
            self._runtime,
            prompt_id=str(prompt_id) if prompt_id is not None else None,
        )

    async def fulfill_interrupt(self, payload: object) -> dict[str, object]:
        """Execute one server-owned hook interrupt on the client runtime.

        Args:
            payload: Raw LangGraph interrupt value.

        Returns:
            Resume value for `Command(resume=...)`.

        Raises:
            RuntimeError: If hooks are unavailable, or the payload is not a
                parseable hook invocation.
        """
        from deepagents_code.hooks.client import fulfill_hook_interrupt

        runtime = self._runtime
        if runtime is None:
            msg = "Received hook invocation interrupt without a HooksRuntime"
            raise RuntimeError(msg)
        resume = await fulfill_hook_interrupt(runtime, payload)
        if resume is None:
            msg = "Failed to parse hook interrupt"
            raise RuntimeError(msg)
        return resume

    async def fulfill_pending_interrupts(
        self,
        pending: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        """Execute a batch of server-owned hook interrupts.

        Args:
            pending: LangGraph interrupt id to raw interrupt payload.

        Returns:
            Resume values keyed by interrupt id.

        Raises:
            RuntimeError: If hooks are unavailable, or a payload is not a
                parseable hook invocation.
        """
        from deepagents_code.hooks.client import fulfill_pending_hook_interrupts

        runtime = self._runtime
        if runtime is None:
            msg = "Received hook invocation interrupt without a HooksRuntime"
            raise RuntimeError(msg)
        return await fulfill_pending_hook_interrupts(runtime, pending)

    def _build_service(self) -> ClientHookService | None:
        runtime = self._runtime
        if runtime is None:
            return None
        return ClientHookService(runtime)

    def _context(self, *, thread_id: str | None = None) -> ClientHookContext:
        identity = self.identity()
        return ClientHookContext.create(
            thread_id=thread_id or identity.thread_id,
            approval_mode=identity.approval_mode,
            prompt_id=identity.prompt_id,
        )


def _present_load_diagnostics(runtime: HooksRuntime | None) -> None:
    """Surface configuration diagnostics collected while loading the snapshot.

    Args:
        runtime: Freshly loaded runtime, or `None` when loading failed.
    """
    if runtime is None:
        return
    runtime.presenter.present_diagnostics(runtime.snapshot.diagnostics)


def _load_runtime(
    cwd: Path,
    *,
    trust: WorkspaceTrust,
    presenter: HookPresenter,
    plugins: tuple[PluginInstance, ...] | None = None,
) -> HooksRuntime | None:
    """Resolve workspace trust for `cwd` and load a runtime under it.

    Trust is resolved here rather than by the caller so that a reload after a
    working-directory change re-reads the trust store for the new directory.

    Args:
        cwd: Session working directory.
        trust: Policy deciding whether project hooks may load.
        presenter: The manager's presenter, shared with the new runtime.
        plugins: Already-discovered plugins, or `None` to discover them.

    Returns:
        The loaded runtime, `None` when configuration could not be loaded, and
        `None` whenever `DEEPAGENTS_CODE_EXPERIMENTAL` is not truthy.
    """
    from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy
    from deepagents_code.hooks.runtime import HooksRuntime
    from deepagents_code.plugins.adapters.hooks import discover_plugin_hook_sources
    from deepagents_code.project_utils import ProjectContext

    if not is_env_truthy(EXPERIMENTAL):
        return None
    try:
        project_context = ProjectContext.from_user_cwd(cwd)
        plugin_sources, plugin_diagnostics = discover_plugin_hook_sources(
            project_dir=project_context.project_root or project_context.user_cwd,
            plugins=plugins,
        )
        return HooksRuntime.create(
            cwd=cwd,
            workspace_trusted=trust.allows(cwd),
            presenter=presenter,
            plugin_sources=plugin_sources,
            plugin_diagnostics=plugin_diagnostics,
        )
    except Exception:
        logger.exception("Failed to load hook configuration; hooks disabled")
        return None
