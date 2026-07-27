"""Typed helpers for Hooks v2 differential wire fixtures."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast
from uuid import UUID

from langchain_core.messages import ToolMessage

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.capabilities import get_event_spec
from deepagents_code.hooks.models.adapters import HOOK_DECISION_ADAPTER
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    CompactTrigger,
    DcodeNotification,
    DcodeNotificationKind,
    HookContext,
    HookDiagnostic,
    HookEvent,
    HookInvocation,
    NotificationEvent,
    PermissionRequestEvent,
    PostToolUseEvent,
    PreCompactEvent,
    PreToolUseEvent,
    SessionEndCause,
    SessionEndEvent,
    SessionStartCause,
    SessionStartEvent,
    StopEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    ToolCallData,
    UserPromptSubmitEvent,
)
from deepagents_code.hooks.runner import HandlerResult, run_command_handler
from deepagents_code.hooks.snapshot import HookHandler

if TYPE_CHECKING:
    from deepagents_code.hooks.models.domain import HookDecision, HookDomainEvent
    from deepagents_code.json_types import JsonObject

FIXTURES_DIR: Final[Path] = Path(__file__).resolve().parent / "fixtures" / "wire"
INPUTS_DIR: Final[Path] = FIXTURES_DIR / "inputs"
OUTPUTS_DIR: Final[Path] = FIXTURES_DIR / "outputs"
REGISTRY_POLICIES_PATH: Final[Path] = FIXTURES_DIR / "registry_policies.json"
REDUCTION_CASES_PATH: Final[Path] = OUTPUTS_DIR / "reduction_cases.json"

FIXED_PROMPT_ID: Final[UUID] = UUID("00000000-0000-4000-8000-000000000001")
FIXED_CWD: Final[Path] = Path("/workspace")
FIXED_TRANSCRIPT_PATH: Final[Path] = Path("/tmp/thread.jsonl")
FIXED_AGENT_TRANSCRIPT_PATH: Final[Path] = Path("/tmp/agent.jsonl")
FIXED_HANDLER_ID: Final[str] = "fixture:0:0"

# Keys whose values are inherently environment- or generation-dependent.
# Both the projected payload and the fixture pass through normalize_wire_payload
# so committed fixtures may store concrete sample values while comparison uses
# stable placeholders.
_VOLATILE_WIRE_KEYS: Final[dict[str, str]] = {
    "prompt_id": "<prompt_id>",
    "cwd": "<cwd>",
    "transcript_path": "<transcript_path>",
    "agent_transcript_path": "<agent_transcript_path>",
}

_FIXTURE_HANDLER_TIMEOUT_SECONDS: Final[float] = 20.0


class RegistryPolicyFixture(TypedDict):
    """Pinned capability-matrix row for one hook event."""

    owner: str
    matcher_field: str | None
    default_timeout_seconds: float
    exit_code_policy: str
    plain_output_policy: str
    aggregation_policy: str


class ReductionCaseFixture(TypedDict):
    """One handler-exit scenario and its reduced domain decision snapshot."""

    id: str
    event: str
    exit_code: int
    stdout: str
    stderr: str
    expected: JsonObject


def load_json_object(path: Path) -> JsonObject:
    """Load a JSON object fixture from disk.

    Args:
        path: Absolute or relative path to a JSON object file.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Fixture must be a JSON object: {path}"
        raise TypeError(msg)
    return cast("JsonObject", raw)


def load_registry_policies() -> dict[str, RegistryPolicyFixture]:
    """Load the pinned registry policy matrix."""
    raw = load_json_object(REGISTRY_POLICIES_PATH)
    return cast("dict[str, RegistryPolicyFixture]", raw)


def load_reduction_cases() -> list[ReductionCaseFixture]:
    """Load handler-output reduction cases."""
    raw: object = json.loads(REDUCTION_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        msg = f"Reduction cases fixture must be a JSON array: {REDUCTION_CASES_PATH}"
        raise TypeError(msg)
    return cast("list[ReductionCaseFixture]", raw)


def load_wire_input_fixture(event: HookEvent) -> JsonObject:
    """Load the committed wire-input fixture for `event`."""
    return load_json_object(INPUTS_DIR / f"{event.value}.json")


def normalize_wire_payload(payload: JsonObject) -> JsonObject:
    """Stabilize non-deterministic wire fields for exact fixture comparison.

    Rewrites known volatile keys (`prompt_id`, `cwd`, `transcript_path`,
    `agent_transcript_path`) to documented placeholders. All other keys and
    nested values are left unchanged so added, removed, or renamed fields fail
    the exact comparison.

    Args:
        payload: Serialized hook wire input (external field names).

    Returns:
        A deep copy with volatile keys replaced by placeholders.
    """
    normalized = cast("JsonObject", copy.deepcopy(payload))
    for key, placeholder in _VOLATILE_WIRE_KEYS.items():
        if key in normalized:
            normalized[key] = placeholder
    return normalized


def assert_exact_mapping(
    actual: JsonObject,
    expected: JsonObject,
    *,
    label: str,
) -> None:
    """Assert two JSON objects match with exact key sets.

    Args:
        actual: Observed mapping.
        expected: Fixture mapping.
        label: Context included in assertion messages.

    Raises:
        AssertionError: On key drift or value mismatch.
    """
    actual_keys = set(actual)
    expected_keys = set(expected)
    added = sorted(actual_keys - expected_keys)
    removed = sorted(expected_keys - actual_keys)
    assert actual_keys == expected_keys, (
        f"{label} key drift: added={added!r} removed={removed!r}"
    )
    assert actual == expected, f"{label} value mismatch"


def fixture_context(*, agent: AgentIdentity | None = None) -> HookContext:
    """Build the shared deterministic invocation context for wire fixtures."""
    return HookContext(
        thread_id="thread-1",
        cwd=FIXED_CWD,
        prompt_id=FIXED_PROMPT_ID,
        approval_mode=ApprovalMode.MANUAL,
        effort="high",
        agent=agent,
    )


def representative_invocation(event: HookEvent) -> HookInvocation:
    """Return a deterministic domain invocation for `event`.

    Args:
        event: Lifecycle event to project onto the wire.

    Returns:
        A domain invocation suitable for golden wire-input comparison.
    """
    agent = AgentIdentity(id="agent-1", name="researcher")
    domain = _representative_event(event, agent=agent)
    identity = (
        agent if isinstance(domain, (SubagentStartEvent, SubagentStopEvent)) else None
    )
    return HookInvocation(context=fixture_context(agent=identity), event=domain)


def agent_transcript_path_for(event: HookEvent) -> Path | None:
    """Return the agent transcript path required by SubagentStop projection."""
    if event is HookEvent.SUBAGENT_STOP:
        return FIXED_AGENT_TRANSCRIPT_PATH
    return None


async def handler_result_from_exit(
    *,
    event: HookEvent,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    handler_id: str = FIXED_HANDLER_ID,
) -> HandlerResult:
    """Run a real command handler that reproduces one exit/stdout/stderr shape.

    Exit-code and output interpretation is owned by `run_command_handler`, so
    these fixtures execute a real process rather than restating that mapping.

    Args:
        event: Event the synthetic handler belongs to.
        exit_code: Exit status the handler should return.
        stdout: Text the handler should write to stdout.
        stderr: Text the handler should write to stderr.
        handler_id: Stable handler id recorded in diagnostics.

    Returns:
        The handler result the reducer consumes.
    """
    script = (
        "import sys;"
        f"sys.stdin.buffer.read();"
        f"sys.stdout.write({stdout!r});"
        f"sys.stderr.write({stderr!r});"
        f"sys.exit({exit_code})"
    )
    handler = HookHandler(
        id=handler_id,
        event=event,
        command="",
        timeout=None,
        status_message=None,
        matcher=None,
        matcher_text=None,
        argv=(sys.executable, "-c", script),
    )
    return await run_command_handler(
        handler,
        b"{}",
        cwd=Path.cwd(),
        default_timeout=_FIXTURE_HANDLER_TIMEOUT_SECONDS,
    )


def decision_snapshot(decision: HookDecision) -> JsonObject:
    """Serialize a domain decision for fixture comparison."""
    dumped = HOOK_DECISION_ADAPTER.dump_python(decision, mode="json")
    return cast("JsonObject", dumped)


def registry_policy_row(event: HookEvent) -> RegistryPolicyFixture:
    """Build the policy-matrix row for `event` from the live capability registry."""
    spec = get_event_spec(event)
    return {
        "owner": spec.owner.value,
        "matcher_field": spec.matcher_field,
        "default_timeout_seconds": spec.default_timeout_seconds,
        "exit_code_policy": spec.exit_code_policy.value,
        "plain_output_policy": spec.plain_output_policy.value,
        "aggregation_policy": spec.aggregation_policy.value,
    }


def _representative_event(
    event: HookEvent,
    *,
    agent: AgentIdentity,
) -> HookDomainEvent:
    match event:
        case HookEvent.SESSION_START:
            return SessionStartEvent(
                event=event,
                cause=SessionStartCause.STARTUP,
                model="provider:model",
            )
        case HookEvent.USER_PROMPT_SUBMIT:
            return UserPromptSubmitEvent(event=event, prompt="Review this change")
        case HookEvent.SESSION_END:
            return SessionEndEvent(event=event, cause=SessionEndCause.OTHER)
        case HookEvent.PERMISSION_REQUEST:
            return PermissionRequestEvent(
                event=event,
                call=ToolCallData(id="call-1", name="Bash", args={"command": "pwd"}),
            )
        case HookEvent.NOTIFICATION:
            return NotificationEvent(
                event=event,
                notification=DcodeNotification(
                    type=DcodeNotificationKind.PERMISSION_REQUIRED,
                    message="Approval required",
                    title="Permission",
                ),
            )
        case HookEvent.PRE_TOOL_USE:
            return PreToolUseEvent(
                event=event,
                call=ToolCallData(
                    id="call-1",
                    name="Write",
                    args={"file_path": "notes.txt", "content": "hello"},
                ),
            )
        case HookEvent.POST_TOOL_USE:
            return PostToolUseEvent(
                event=event,
                call=ToolCallData(id="call-2", name="Bash", args={"command": "pwd"}),
                result=ToolMessage(
                    content="/workspace",
                    tool_call_id="call-2",
                    name="Bash",
                ),
                duration_ms=12,
            )
        case HookEvent.PRE_COMPACT:
            return PreCompactEvent(
                event=event,
                trigger=CompactTrigger.MANUAL,
                custom_instructions="Keep the plan",
            )
        case HookEvent.STOP:
            return StopEvent(
                event=event,
                continuation_count=0,
                last_assistant_message="Done",
            )
        case HookEvent.SUBAGENT_START:
            return SubagentStartEvent(event=event, agent=agent)
        case HookEvent.SUBAGENT_STOP:
            return SubagentStopEvent(
                event=event,
                agent=agent,
                continuation_count=0,
                last_assistant_message="Found it",
            )
        case _:
            msg = f"Unsupported hook event: {event}"
            raise ValueError(msg)
