"""Offline tests for the dual-signal triage gate (no LLM required)."""

from __future__ import annotations

import copy

from dual_signal_triage.gate import (
    decide_for_event_id,
    load_corpus,
    validate_corpus,
)
from dual_signal_triage.tools import clear_audit, contain_host, escalate_alert


def test_corpus_validates():
    assert validate_corpus() == []


def test_never_equate_flag():
    corpus = load_corpus()
    assert corpus["never_equate_ml_to_signature"] is True


def test_ml_only_high_denies_contain():
    decision = decide_for_event_id("ml-sqli-002")
    assert decision.disposition == "escalate"
    assert decision.allow_contain is False
    assert decision.gate_action == "deny_contain"


def test_signature_high_allows_contain_with_interrupt():
    decision = decide_for_event_id("sig-sqli-001")
    assert decision.disposition == "fix_now"
    assert decision.allow_contain is True
    assert decision.require_interrupt is True


def test_corroborated_allows_contain_with_interrupt():
    decision = decide_for_event_id("corr-003")
    assert decision.allow_contain is True
    assert decision.require_interrupt is True


def test_ml_low_accepts():
    decision = decide_for_event_id("ml-low-004")
    assert decision.disposition == "accept"
    assert decision.allow_contain is False


def test_contain_tool_blocks_ml_only():
    clear_audit()
    result = contain_host("ml-sqli-002", "10.0.0.5")
    assert result["action"] == "contain_denied"
    result_ok = contain_host("sig-sqli-001", "10.0.0.5")
    assert result_ok["action"] == "contain_simulated"


def test_escalate_tool():
    clear_audit()
    result = escalate_alert("ml-sqli-002", note="needs signature")
    assert result["action"] == "escalated"


def test_bad_corpus_rejected():
    bad = copy.deepcopy(load_corpus())
    for event in bad["events"]:
        if event["case"] == "snortml_gid411_high_ml_only":
            event["label_disposition"] = "fix_now"
            break
    errors = validate_corpus(bad)
    assert any("ml_only" in e for e in errors)
