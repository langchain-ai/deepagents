"""Dual-signal SnortML + signature + Splunk-notable triage for Deep Agents."""

from dual_signal_triage.gate import (
    GateDecision,
    decide_for_event_id,
    decide_from_event,
    load_corpus,
    validate_corpus,
)

__all__ = [
    "GateDecision",
    "decide_for_event_id",
    "decide_from_event",
    "load_corpus",
    "validate_corpus",
]
