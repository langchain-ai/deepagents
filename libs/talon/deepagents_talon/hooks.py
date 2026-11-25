"""Talon adapter for the shared command-hook engine."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage

from deepagents_code.hooks.client import HookFulfillmentLedger
from deepagents_code.hooks.engine import HookEngine
from deepagents_code.hooks.interrupt import build_hook_resume_value, parse_hook_interrupt_payload
from deepagents_code.hooks.loading import FileHooksSource, load_hooks_config
from deepagents_code.hooks.migration import is_legacy_hooks_document, migrate_legacy_hooks
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import AgentIdentity, HookDiagnostic
from deepagents_code.hooks.models.transport import HookInvocationResponse
from deepagents_code.hooks.presenter import HookPresenter
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.server_middleware import hook_permission_behavior
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_talon.hook_transcripts import TRANSCRIPT_LOCK, ConversationTranscripts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain.tools.tool_node import ToolCallRequest

    from deepagents_code.hooks.models.transport import HookInvocationRequest
    from deepagents_code.json_types import JsonValue

_SUPPORTED_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"}
ACTIVE_HOOKS: ContextVar[TalonHooks | None] = ContextVar("talon_hooks", default=None)


def _load_snapshot(home: Path) -> HooksSnapshot | None:
    path = home.resolve() / "hooks.json"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            msg = "Hooks configuration must be a regular file"
            raise ValueError(msg)
        try:
            document = json.loads(stream.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return HooksSnapshot.from_config(
                HooksConfig(hooks={}),
                diagnostics=(
                    HookDiagnostic(
                        code="config_read_failed",
                        severity="warning",
                        message=f"Failed to read hooks config at {path}: {exc}",
                        field=str(path),
                    ),
                ),
            )
    config, diagnostics = _parse_hooks_document(document, path)
    config = config if config is not None else HooksConfig(hooks={})
    source = FileHooksSource(location=str(path))
    return HooksSnapshot.from_config(
        config,
        groups={
            event: tuple((source, group) for group in groups)
            for event, groups in config.hooks.items()
        },
        diagnostics=diagnostics,
    )


def _parse_hooks_document(
    data: JsonValue, path: Path
) -> tuple[HooksConfig | None, tuple[HookDiagnostic, ...]]:
    """Validate decoded file configuration, including legacy migration.

    Args:
        data: Decoded file document.
        path: Source location for diagnostics.

    Returns:
        Validated configuration and diagnostics.
    """
    if is_legacy_hooks_document(data):
        hooks = data.get("hooks", []) if isinstance(data, dict) else []
        if not isinstance(hooks, list):
            diagnostics = (
                HookDiagnostic(
                    code="invalid_config",
                    severity="warning",
                    message=f"Legacy hooks list missing at {path}",
                    field=str(path),
                ),
            )
            return None, diagnostics
        legacy_entries: list[dict[str, object]] = [
            {str(key): value for key, value in item.items()}
            for item in hooks
            if isinstance(item, Mapping)
        ]
        migrated = migrate_legacy_hooks(legacy_entries)
        migration_message = (
            f"Migrated semantically equivalent legacy hooks from {path}; "
            "unsupported legacy events remain unmapped"
            if migrated.hooks
            else (
                f"Legacy hooks at {path} contained no events that are safe to migrate to Hooks v2"
            )
        )
        diagnostics = (
            HookDiagnostic(
                code="legacy_deprecated",
                severity="warning",
                message=(
                    f"Legacy hooks configuration at {path} is deprecated and will "
                    f"stop being supported on September 1, 2026"
                ),
                field=str(path),
            ),
            HookDiagnostic(
                code="legacy_migrated" if migrated.hooks else "legacy_unmapped",
                severity="warning",
                message=migration_message,
                field=str(path),
            ),
        )
        return migrated, diagnostics

    loaded = load_hooks_config(
        project_root=path.parent,
        workspace_trusted=False,
        paths=(),
        documents=((FileHooksSource(location=str(path)), data),),
    )
    return loaded.config, loaded.diagnostics


def hook_approval_required(request: ToolCallRequest) -> bool:
    """Require normal approval unless a hook already denied the call."""
    return hook_permission_behavior(request.state, request.tool_call["id"] or "") != "deny"


class TalonHookContext(TypedDict):
    """Trusted per-invocation context consumed by the shared hook middleware."""

    thread_id: str
    prompt_id: str
    approval_mode: str
    hooks_snapshot_id: str
    hooks_server_events: list[str]


@dataclass
class TalonHooks:
    """Own one turn's hook snapshot, transcripts, and completed invocations."""

    runtime: HooksRuntime
    context: TalonHookContext
    agent: AgentIdentity | None = field(default=None, kw_only=True)
    _responses: dict[UUID, dict[str, object]] = field(default_factory=dict)

    def child(self, name: str) -> TalonHooks:
        """Create an isolated subagent session using this turn's hook snapshot.

        Args:
            name: Configured subagent name.

        Returns:
            Adapter with independent invocation identities and transcripts.
        """
        identity = str(uuid4())
        return TalonHooks(
            runtime=self.runtime,
            context={**self.context, "thread_id": identity, "prompt_id": str(uuid4())},
            agent=AgentIdentity(id=identity, name=name),
        )

    @classmethod
    def load(cls, home: Path, cwd: Path, thread_id: str) -> TalonHooks | None:
        """Load trusted assistant hooks once per turn, excluding workspace hooks."""
        snapshot = _load_snapshot(home)
        if snapshot is None:
            return None
        with TRANSCRIPT_LOCK:
            transcripts = ConversationTranscripts(home, thread_id, reopen=True)
            runtime = HooksRuntime(
                snapshot=snapshot,
                transcripts=transcripts,
                engine=HookEngine(snapshot),
                cwd=cwd.resolve(),
                workspace_trusted=False,
                project_hooks_loaded=False,
                project_hooks_fingerprint=None,
                presenter=HookPresenter(),
                fulfillments=HookFulfillmentLedger(),
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
        """Record committed messages before fulfilling hooks."""
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
        """Fulfill hook interrupts, validating identities and leaving approvals to Talon."""
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
            invocation = request.invocation
            if self.agent is not None:
                invocation = invocation.model_copy(
                    update={"context": invocation.context.model_copy(update={"agent": self.agent})}
                )
            decision = await self.runtime.invoke(invocation)
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
