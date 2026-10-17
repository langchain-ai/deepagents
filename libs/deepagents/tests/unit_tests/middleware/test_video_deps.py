"""Unit tests for `video_dependencies_available`, the optional-extra gate.

These tests monkeypatch `importlib.util.find_spec` so they run without the
`[video]` extra actually being installed — unlike `test_video.py`, which
`importorskip`s the real PyAV decoder.
"""

import importlib.util

import pytest

from deepagents.middleware._video import video_dependencies_available


def test_video_dependencies_available_true_when_both_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both specs discoverable -> the extra is reported available."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert video_dependencies_available() is True


def test_video_dependencies_available_false_when_av_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing `av` short-circuits to unavailable (exercises the `and`)."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "av" else object())
    assert video_dependencies_available() is False


def test_video_dependencies_available_false_when_pillow_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing `PIL.Image` reports unavailable even when `av` is present."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "PIL.Image" else object())
    assert video_dependencies_available() is False


def test_video_dependencies_available_false_on_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raised probe error is caught and treated as unavailable (except branch)."""

    def _raise(_name: str) -> object:
        msg = "corrupt spec"
        raise ValueError(msg)

    monkeypatch.setattr(importlib.util, "find_spec", _raise)
    assert video_dependencies_available() is False
