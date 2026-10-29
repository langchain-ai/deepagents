"""Talon adapter for the shared command-hook engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage

from deepagents_code.hooks.interrupt import build_hook_resume_value, parse_hook_interrupt_payload
from deepagents_code.hooks.models.transport import HookInvocationResponse
from deepagents_code.hooks.runtime import HooksRuntime

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from deepagents_code.hooks.models.transport import HookInvocationRequest

_SUPPORTED_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"}


@dataclass
class TalonHooks:
    """Own one turn's hook snapshot, transcripts, and completed invocations."""

    runtime: HooksRuntime
    context: dict[str, object]
    _responses: dict[UUID, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, home: Path, cwd: Path, thread_id: str) -> TalonHooks | None:
        """Load operator hooks, excluding workspace and plugin configuration.

        Args:
            home: Talon configuration directory containing `hooks.json`.
            cwd: Host working directory for hook commands.
            thread_id: Conversation owning this turn.

        Returns:
            An isolated turn adapter, or `None` when no hooks file exists.
        """
        if not (home / "hooks.json").is_file():
            return None
        runtime = HooksRuntime.create(
            cwd=cwd,
            config_dir=home,
            transcript_root=home / "transcripts",
            workspace_trusted=False,
        )
        runtime.presenter.present_diagnostics(runtime.snapshot.diagnostics)
        return cls(
            runtime=runtime,
            context={
                "thread_id": thread_id,
                "prompt_id": str(uuid4()),
                "approval_mode": "manual",
                "hooks_snapshot_id": runtime.snapshot_id,
                "hooks_server_events": sorted(
                    _SUPPORTED_EVENTS.intersection(runtime.configured_server_events())
                ),
            },
        )

    def record(self, state: object) -> None:
        """Record committed messages before fulfilling hooks.

        Args:
            state: Graph result, including interrupted results.
        """
        if isinstance(state, dict):
            self.runtime.append_messages(
                str(self.context["thread_id"]),
                [
                    message
                    for message in state.get("messages", ())
                    if isinstance(message, BaseMessage)
                ],
            )

    async def resume(
        self, interrupts: Sequence[object]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Fulfill hook interrupts and leave approvals to Talon.

        Args:
            interrupts: Interrupts returned by the graph.

        Returns:
            Remaining interrupts and completed hook resume values.

        Raises:
            ValueError: If a hook request has an invalid identity or snapshot.
        """
        remaining: list[object] = []
        responses: dict[str, object] = {}
        seen: set[str] = set()
        for interrupt in interrupts:
            interrupt_id = getattr(interrupt, "id", None)
            if not isinstance(interrupt_id, str) or not interrupt_id or interrupt_id in seen:
                msg = "Interrupts must have unique resumable ids"
                raise ValueError(msg)
            seen.add(interrupt_id)
            request = parse_hook_interrupt_payload(getattr(interrupt, "value", None))
            if request is None:
                remaining.append(interrupt)
                continue
            responses[interrupt_id] = await self._fulfill(request)
        return tuple(remaining), responses

    async def _fulfill(self, request: HookInvocationRequest) -> dict[str, object]:
        if (
            request.snapshot_id != self.runtime.snapshot_id
            or request.invocation.context.thread_id != self.context["thread_id"]
            or str(request.invocation.context.prompt_id) != self.context["prompt_id"]
        ):
            msg = "Hook request does not belong to this turn"
            raise ValueError(msg)
        if request.invocation_id not in self._responses:
            decision = await self.runtime.invoke(request.invocation)
            self.runtime.presenter.present_decision(decision)
            self._responses[request.invocation_id] = build_hook_resume_value(
                HookInvocationResponse(
                    protocol_version=1,
                    invocation_id=request.invocation_id,
                    snapshot_id=request.snapshot_id,
                    decision=decision,
                )
            )
        return self._responses[request.invocation_id]
