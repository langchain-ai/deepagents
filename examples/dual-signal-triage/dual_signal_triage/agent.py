"""Deep Agents wiring for dual-signal SOC triage with HITL on contain."""

from __future__ import annotations

from typing import Any

from dual_signal_triage import tools as triage_tools

SYSTEM_PROMPT = """You are a SOC triage agent for dual-signal alerts (Snort signature,
SnortML GID 411, and Splunk notables).

Hard rule: an ML probability is NEVER equivalent to a signature true positive.
- ML-only high (GID 411, no SID): escalate / corroborate — NEVER call contain_host.
- Signature high or signature+ML: may propose contain_host, which pauses for human approval.
- Low ML / low Splunk notables: accept or T1 triage.

Always call recommend_action(event) before proposing remediation.
Prefer escalate_alert or corroborate_alert over containment when unsure.
"""


def _langchain_tools():
    """Build LangChain tools (lazy import so offline CLI needs no deepagents)."""
    from langchain_core.tools import tool

    @tool
    def list_open_alerts() -> list[dict[str, Any]]:
        """List labeled open alerts awaiting triage."""
        return triage_tools.list_open_alerts()

    @tool
    def recommend_action(event_id: str) -> dict[str, Any]:
        """Return the policy gate decision for an alert event_id."""
        events = {e["event_id"]: e for e in triage_tools.list_open_alerts()}
        if event_id not in events:
            return {"error": f"unknown event_id: {event_id}"}
        return triage_tools.recommend_action(events[event_id])

    @tool
    def contain_host(event_id: str, host: str) -> dict[str, Any]:
        """Simulate host containment. Blocked for ML-only highs; HITL interrupt required."""
        return triage_tools.contain_host(event_id, host)

    @tool
    def escalate_alert(event_id: str, note: str = "") -> dict[str, Any]:
        """Escalate an alert for human / T3 review."""
        return triage_tools.escalate_alert(event_id, note=note)

    @tool
    def accept_alert(event_id: str, note: str = "") -> dict[str, Any]:
        """Accept / close low-urgency noise."""
        return triage_tools.accept_alert(event_id, note=note)

    @tool
    def corroborate_alert(event_id: str) -> dict[str, Any]:
        """Request second-signal corroboration before containment."""
        return triage_tools.corroborate_alert(event_id)

    return [
        list_open_alerts,
        recommend_action,
        contain_host,
        escalate_alert,
        accept_alert,
        corroborate_alert,
    ]


def create_triage_agent(*, model: str = "openai:gpt-4.1-mini"):
    """Build a Deep Agent with HITL interrupt on contain_host."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=_langchain_tools(),
        system_prompt=SYSTEM_PROMPT,
        interrupt_on={"contain_host": True},
    )


def main() -> None:
    """Print offline gate recommendations for the shipped corpus (no LLM)."""
    for event in triage_tools.list_open_alerts():
        decision = triage_tools.recommend_action(event)
        print(
            f"{decision['event_id']}: disposition={decision['disposition']} "
            f"gate={decision['gate_action']} allow_contain={decision['allow_contain']} "
            f"({decision['reason']})"
        )


if __name__ == "__main__":
    main()
