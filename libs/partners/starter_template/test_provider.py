"""Smoke tests for the provider starter scaffold (Wave 4 P1)."""

from __future__ import annotations

from provider import StarterProvider


def test_capabilities_shape():
    caps = StarterProvider().capabilities()
    assert caps["name"] == "starter"
    assert "supports_streaming" in caps
    assert "supports_tools" in caps
    assert "max_context_tokens" in caps


def test_invoke_returns_dict():
    out = StarterProvider().invoke("hello", temperature=0.0)
    assert out["provider"] == "starter"
    assert out["prompt"] == "hello"
    assert out["kwargs"] == {"temperature": 0.0}


def test_health_ok():
    h = StarterProvider().health()
    assert h["ok"] is True
