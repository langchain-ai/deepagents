"""Tests for the shared environment-variable flag helper."""

from __future__ import annotations

import pytest

from deepagents.profiles._env import _env_flag

_VAR = "DEEPAGENTS_TEST_ENV_FLAG"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on ", "\tTrue\n"])
def test_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_VAR, value)
    assert _env_flag(_VAR) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe", "2"])
def test_falsey_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_VAR, value)
    assert _env_flag(_VAR) is False


def test_unset_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_VAR, raising=False)
    assert _env_flag(_VAR) is False
