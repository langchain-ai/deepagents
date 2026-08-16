"""Dual-signal triage gate: ML probability ≠ signature true positive.

Pure-Python policy used by the Deep Agents example. Offline-testable without
an LLM — the agent harness must not auto-contain on ML-only highs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Disposition = Literal[
    "fix_now",
    "escalate",
    "accept",
    "suppress_fp",
    "triage_t1",
    "triage_t2",
]
GateAction = Literal["allow_contain_interrupt", "deny_contain", "escalate", "accept"]

HIGH_ML_THRESHOLD = 0.8
FORBIDDEN_AUTO_ACTIONS = frozenset({"auto_contain", "contain", "block", "fix_now"})

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "labeled_events.json"


@dataclass(frozen=True)
class GateDecision:
    event_id: str
    signal_class: str
    disposition: Disposition
    gate_action: GateAction
    allow_contain: bool
    require_interrupt: bool
    reason: str


def load_corpus(path: Path | None = None) -> dict[str, Any]:
    corpus_path = path or DEFAULT_CORPUS
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if data.get("never_equate_ml_to_signature") is not True:
        raise ValueError("corpus must set never_equate_ml_to_signature: true")
    return data


def index_events(corpus: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    data = corpus if corpus is not None else load_corpus()
    return {event["event_id"]: event for event in data["events"]}


def decide_from_event(event: dict[str, Any]) -> GateDecision:
    """Map a labeled (or live) alert event to a remediation gate decision."""
    event_id = str(event.get("event_id") or event.get("alert_id") or "unknown")
    signal = str(event.get("signal_class") or _infer_signal_class(event))
    disposition = str(event.get("label_disposition") or _infer_disposition(event, signal))
    ml_score = event.get("ml_score")
    if ml_score is None:
        ml_score = event.get("snortml_score")

    # Hard rule: ML-only must never auto-contain / fix_now.
    if signal == "ml_only":
        if disposition in FORBIDDEN_AUTO_ACTIONS or disposition == "fix_now":
            disposition = "escalate"
        high = isinstance(ml_score, (int, float)) and float(ml_score) >= HIGH_ML_THRESHOLD
        if high:
            return GateDecision(
                event_id=event_id,
                signal_class=signal,
                disposition="escalate",
                gate_action="deny_contain",
                allow_contain=False,
                require_interrupt=True,
                reason="ml_only_high_requires_corroboration",
            )
        return GateDecision(
            event_id=event_id,
            signal_class=signal,
            disposition="accept" if disposition not in {"accept", "suppress_fp"} else disposition,  # type: ignore[arg-type]
            gate_action="accept",
            allow_contain=False,
            require_interrupt=False,
            reason="ml_only_low_is_noise",
        )

    if disposition == "fix_now" and signal in {"signature_only", "signature_plus_ml"}:
        return GateDecision(
            event_id=event_id,
            signal_class=signal,
            disposition="fix_now",
            gate_action="allow_contain_interrupt",
            allow_contain=True,
            require_interrupt=True,
            reason="signature_tp_contain_requires_hitl",
        )

    if disposition in {"triage_t2", "escalate"}:
        return GateDecision(
            event_id=event_id,
            signal_class=signal,
            disposition=disposition,  # type: ignore[arg-type]
            gate_action="escalate",
            allow_contain=False,
            require_interrupt=True,
            reason="notable_or_escalate_path",
        )

    if disposition == "triage_t1":
        return GateDecision(
            event_id=event_id,
            signal_class=signal,
            disposition="triage_t1",
            gate_action="accept",
            allow_contain=False,
            require_interrupt=False,
            reason="low_urgency_t1",
        )

    return GateDecision(
        event_id=event_id,
        signal_class=signal,
        disposition="accept" if disposition == "accept" else disposition,  # type: ignore[arg-type]
        gate_action="accept",
        allow_contain=False,
        require_interrupt=False,
        reason="low_urgency_accept",
    )


def decide_for_event_id(event_id: str, corpus: dict[str, Any] | None = None) -> GateDecision:
    events = index_events(corpus)
    if event_id not in events:
        raise KeyError(f"unknown event_id: {event_id}")
    return decide_from_event(events[event_id])


def validate_corpus(corpus: dict[str, Any] | None = None) -> list[str]:
    """Return validation errors (empty = OK). Rejects ML-only high labeled fix_now/auto_contain."""
    data = corpus if corpus is not None else load_corpus()
    errors: list[str] = []
    if data.get("never_equate_ml_to_signature") is not True:
        errors.append("never_equate_ml_to_signature must be true")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
        return errors

    required_cases = {
        "signature_only_high",
        "snortml_gid411_high_ml_only",
        "signature_plus_ml_corroboration",
        "snortml_low",
        "splunk_notable_high_risk",
        "splunk_notable_low",
    }
    seen: set[str] = set()
    for idx, event in enumerate(events):
        case = event.get("case")
        seen.add(case)
        signal = event.get("signal_class")
        disposition = event.get("label_disposition")
        if signal == "ml_only" and disposition in {"fix_now", "auto_contain", "contain"}:
            errors.append(
                f"events[{idx}] ({event.get('event_id')}): ml_only cannot be labeled {disposition!r}"
            )
        if case == "snortml_gid411_high_ml_only" and disposition != "escalate":
            errors.append(
                f"events[{idx}]: snortml_gid411_high_ml_only must label escalate, got {disposition!r}"
            )
    missing = required_cases - seen
    if missing:
        errors.append(f"missing required cases: {sorted(missing)}")
    return errors


def _infer_signal_class(event: dict[str, Any]) -> str:
    gid = event.get("gid")
    sid = event.get("sid")
    ml = event.get("ml_score", event.get("snortml_score"))
    if event.get("_adapter") == "splunk_notable" or event.get("search_name"):
        return "splunk_notable"
    if gid == 411 and not sid:
        return "ml_only"
    if sid and isinstance(ml, (int, float)) and float(ml) >= HIGH_ML_THRESHOLD:
        return "signature_plus_ml"
    if sid:
        return "signature_only"
    return "ml_only"


def _infer_disposition(event: dict[str, Any], signal: str) -> Disposition:
    ml = event.get("ml_score", event.get("snortml_score"))
    if signal == "ml_only":
        if isinstance(ml, (int, float)) and float(ml) >= HIGH_ML_THRESHOLD:
            return "escalate"
        return "accept"
    if signal in {"signature_only", "signature_plus_ml"}:
        return "fix_now"
    urgency = str(event.get("urgency") or "").lower()
    risk = event.get("risk_score")
    if urgency == "high" or (isinstance(risk, (int, float)) and float(risk) >= 50):
        return "triage_t2"
    return "triage_t1"
