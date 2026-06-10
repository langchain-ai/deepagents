"""Tests for langgraph `DeltaChannel` feature detection."""

from __future__ import annotations

import importlib
import sys
import warnings

from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import add_messages

import deepagents._delta_channel as dc


def test_supported_returns_delta_channel() -> None:
    """When langgraph supports it, the reducer is a `DeltaChannel`."""
    assert dc.delta_channel_supported() is True
    assert isinstance(dc.messages_channel_reducer(), DeltaChannel)


def test_no_warning_when_supported() -> None:
    """No warning is emitted on a supported langgraph install."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dc.warn_if_delta_channel_unsupported()


def test_unsupported_falls_back_and_warns(monkeypatch) -> None:
    """Simulate an old langgraph (no `DeltaChannel`): fall back + warn."""
    real_delta = sys.modules.pop("langgraph.channels.delta", None)
    monkeypatch.setitem(sys.modules, "langgraph.channels.delta", None)
    try:
        reloaded = importlib.reload(dc)
        assert reloaded.delta_channel_supported() is False
        assert reloaded.messages_channel_reducer() is add_messages
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            reloaded.warn_if_delta_channel_unsupported()
        assert len(captured) == 1
        assert issubclass(captured[0].category, UserWarning)
        assert "DeltaChannel" in str(captured[0].message)
    finally:
        if real_delta is not None:
            sys.modules["langgraph.channels.delta"] = real_delta
        importlib.reload(dc)
