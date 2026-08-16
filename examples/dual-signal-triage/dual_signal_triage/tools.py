"""Mock SOC remediation tools for the dual-signal triage example.

Containment is gated: callers should consult `gate.decide_*` before invoking
`contain_host`, and the agent is wired with `interrupt_on` for HITL.
"""

from __future__ import annotations

from typing import Any

from dual_signal_triage.gate import decide_for_event_id, decide_from_event, index_events

_AUDIT: list[dict[str, Any]] = []


def audit_log() -> list[dict[str, Any]]:
    return list(_AUDIT)


def clear_audit() -> None:
    _AUDIT.clear()


def _record(action: str, **payload: Any) -> dict[str, Any]:
    entry = {"action": action, **payload}
    _AUDIT.append(entry)
    return entry


def contain_host(event_id: str, host: str, *, force: bool = False) -> dict[str, Any]:
    """Simulate host containment. Denied for ML-only highs unless force (tests only)."""
    decision = decide_for_event_id(event_id)
    if not decision.allow_contain and not force:
        return _record(
            "contain_denied",
            event_id=event_id,
            host=host,
            reason=decision.reason,
            disposition=decision.disposition,
        )
    return _record(
        "contain_simulated",
        event_id=event_id,
        host=host,
        disposition=decision.disposition,
        require_interrupt=decision.require_interrupt,
    )


def escalate_alert(event_id: str, note: str = "") -> dict[str, Any]:
    decision = decide_for_event_id(event_id)
    return _record(
        "escalated",
        event_id=event_id,
        note=note or decision.reason,
        disposition=decision.disposition,
    )


def accept_alert(event_id: str, note: str = "") -> dict[str, Any]:
    decision = decide_for_event_id(event_id)
    return _record(
        "accepted",
        event_id=event_id,
        note=note or decision.reason,
        disposition=decision.disposition,
    )


def corroborate_alert(event_id: str) -> dict[str, Any]:
    """Request signature / second-signal corroboration (ML-only high path)."""
    decision = decide_for_event_id(event_id)
    return _record(
        "corroborate_requested",
        event_id=event_id,
        disposition=decision.disposition,
        reason=decision.reason,
    )


def list_open_alerts() -> list[dict[str, Any]]:
    """Return the labeled corpus events for the agent to triage."""
    return list(index_events().values())


def recommend_action(event: dict[str, Any]) -> dict[str, Any]:
    decision = decide_from_event(event)
    return {
        "event_id": decision.event_id,
        "disposition": decision.disposition,
        "gate_action": decision.gate_action,
        "allow_contain": decision.allow_contain,
        "require_interrupt": decision.require_interrupt,
        "reason": decision.reason,
    }
