"""Tests for configurable conversation cost warnings."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from deepagents_code.app import DeepAgentsApp, _load_cost_warning_threshold_usd
from deepagents_code.config_manifest import OptionKind, get_option, resolve_scalar


def test_cost_warning_manifest_option() -> None:
    """The threshold is discoverable and resolves config-file numbers."""
    option = get_option("warnings.cost_threshold_usd")

    assert option is not None
    assert option.kind is OptionKind.FLOAT
    assert option.default is None
    assert option.toml_keys == ("warnings", "cost_threshold_usd")
    assert resolve_scalar(option, toml_data={}) == (None, "default")
    assert resolve_scalar(
        option,
        toml_data={"warnings": {"cost_threshold_usd": 2.5}},
    ) == (2.5, "config.toml")


@pytest.mark.parametrize("value", [-0.01, math.inf, -math.inf, math.nan])
def test_cost_warning_loader_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    """Negative and non-finite thresholds disable the warning."""
    monkeypatch.setattr(
        "deepagents_code.config_manifest.load_config_toml",
        lambda: {"warnings": {"cost_threshold_usd": value}},
    )

    assert _load_cost_warning_threshold_usd() is None


def test_cost_warning_loader_accepts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero threshold warns on the first positive estimated cost."""
    monkeypatch.setattr(
        "deepagents_code.config_manifest.load_config_toml",
        lambda: {"warnings": {"cost_threshold_usd": 0}},
    )

    assert _load_cost_warning_threshold_usd() == pytest.approx(0.0)


def test_cost_warning_is_strict_and_shown_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only cost above the threshold warns, including provisional spend."""
    monkeypatch.setattr(
        "deepagents_code.app._load_cost_warning_threshold_usd",
        lambda: 1.0,
    )
    app = DeepAgentsApp()

    with patch.object(app, "notify") as notify:
        app._set_session_cost(1.0)
        notify.assert_not_called()

        app._add_provisional_cost(0.01)
        notify.assert_called_once_with(
            (
                "Estimated conversation cost is $1.01, above your $1.00 "
                "warning threshold. Use /cost for details."
            ),
            title="Cost warning",
            severity="warning",
            timeout=8,
            markup=False,
        )

        app._set_session_cost(1.01)
        app._add_provisional_cost(0.5)
        assert notify.call_count == 1


def test_cost_warning_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No notification is shown when the threshold is not configured."""
    monkeypatch.setattr(
        "deepagents_code.app._load_cost_warning_threshold_usd",
        lambda: None,
    )
    app = DeepAgentsApp()

    with patch.object(app, "notify") as notify:
        app._set_session_cost(100.0)

    notify.assert_not_called()


def test_cost_warning_resets_for_loaded_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resetting usage lets a restored over-threshold conversation warn."""
    monkeypatch.setattr(
        "deepagents_code.app._load_cost_warning_threshold_usd",
        lambda: 1.0,
    )
    app = DeepAgentsApp()

    with patch.object(app, "notify") as notify:
        app._set_session_cost(1.25)
        app._reset_thread_usage(1.5)

    assert notify.call_count == 2
