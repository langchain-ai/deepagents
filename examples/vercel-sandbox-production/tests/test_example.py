"""Tests for the example's reusable helpers.

Covers the two pieces of non-trivial logic — the network-policy builder and the
per-thread backend factory — without touching the network or creating a real
sandbox.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

# The example modules live at the package root (flat layout), next to this
# tests/ dir. Add the parent so `import agent` / `import network_policy` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
import network_policy  # noqa: E402


def test_network_policy_denies_private_subnets_and_plain_allows() -> None:
    policy = network_policy.build_network_policy()

    assert "169.254.0.0/16" in policy.subnets.deny  # cloud metadata endpoint
    assert "10.0.0.0/8" in policy.subnets.deny
    # PyPI + LLM hosts are plain-allowed (no auth transform).
    assert policy.allow["pypi.org"] == []
    assert policy.allow["api.anthropic.com"] == []


def test_network_policy_brokers_credentials_at_the_firewall() -> None:
    policy = network_policy.build_network_policy(
        extra_allow_hosts=("data.example.gov",),
        brokered_hosts={"api.myapp.com": "secret-token"},
    )

    assert "data.example.gov" in policy.allow
    rule = policy.allow["api.myapp.com"][0]
    assert rule.transform[0].headers == {"Authorization": "Bearer secret-token"}


def _runtime(thread_id: str | None) -> types.SimpleNamespace:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    return types.SimpleNamespace(config=config)


def test_get_backend_reuses_one_sandbox_per_thread(monkeypatch) -> None:
    created: list[object] = []
    monkeypatch.setattr(agent, "_WARM", {})
    monkeypatch.setattr(agent, "_create_sandbox", lambda: created.append(1) or object())

    first = agent.get_backend(_runtime("t-A"))
    again = agent.get_backend(_runtime("t-A"))
    other = agent.get_backend(_runtime("t-B"))

    assert first is again  # same thread reuses
    assert other is not first  # different thread gets its own
    assert len(created) == 2


def test_get_backend_recreates_after_lifetime_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_WARM", {})
    monkeypatch.setattr(agent, "_create_sandbox", lambda: object())

    first = agent.get_backend(_runtime("t-A"))
    # Age the warm entry past the pre-emptive recreate window.
    agent._WARM["t-A"].created_at = time.monotonic() - (agent.PREEMPTIVE_RECREATE_S + 1)
    second = agent.get_backend(_runtime("t-A"))

    assert second is not first


def test_get_backend_falls_back_to_a_stable_default_key(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_WARM", {})
    monkeypatch.setattr(agent, "_create_sandbox", lambda: object())

    first = agent.get_backend(_runtime(None))
    again = agent.get_backend(_runtime(None))

    assert first is again
