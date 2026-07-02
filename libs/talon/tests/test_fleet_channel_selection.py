from __future__ import annotations

import pytest

from deepagents_talon.fleet_channel_selection import (
    ChannelCandidate,
    ChannelSelectionError,
    resolve_fleet_channel_selection,
)
from deepagents_talon.fleet_manifest import ChannelSelection


def test_explicit_channel_selection_is_ready() -> None:
    result = resolve_fleet_channel_selection({}, explicit_channel="telegram")

    assert result.ready
    assert result.selected_channel == ChannelSelection(
        provider="telegram",
        source="explicit",
        status="ready",
    )


def test_single_enabled_env_channel_is_inferred() -> None:
    result = resolve_fleet_channel_selection(
        {"DEEPAGENTS_TALON_WHATSAPP_ENABLED": "true"},
    )

    assert result.selected_channel == ChannelSelection(
        provider="whatsapp",
        source="inferred",
        status="ready",
        metadata={"selection_source": "environment"},
    )


def test_single_cli_channel_flag_is_inferred() -> None:
    result = resolve_fleet_channel_selection({}, channel_flags={"telegram": True})

    assert result.selected_channel == ChannelSelection(
        provider="telegram",
        source="inferred",
        status="ready",
        metadata={"selection_source": "cli"},
    )


def test_ambiguous_env_selection_is_non_ready_without_prompt() -> None:
    result = resolve_fleet_channel_selection(
        {
            "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
            "DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1",
        },
    )

    assert not result.ready
    assert result.status == "ambiguous"
    assert result.selected_channel is None
    assert "Multiple Talon channels" in result.message


def test_absent_selection_is_non_ready_without_prompt() -> None:
    result = resolve_fleet_channel_selection({})

    assert not result.ready
    assert result.status == "missing_config"
    assert result.selected_channel is None
    assert "No Talon channel" in result.message


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(ChannelSelectionError, match="Unsupported Talon channel"):
        resolve_fleet_channel_selection({}, explicit_channel="email")


def test_interactive_prompt_resolves_ambiguous_selection() -> None:
    seen: list[tuple[str, tuple[ChannelCandidate, ...]]] = []

    def prompt(status, candidates):
        seen.append((status, tuple(candidates)))
        return "telegram"

    result = resolve_fleet_channel_selection(
        {
            "DEEPAGENTS_TALON_TELEGRAM_ENABLED": "1",
            "DEEPAGENTS_TALON_WHATSAPP_ENABLED": "1",
        },
        interactive=True,
        prompt=prompt,
    )

    assert result.selected_channel == ChannelSelection(
        provider="telegram",
        source="prompted",
        status="ready",
        metadata={
            "prompt_status": "ambiguous",
            "providers": "telegram,whatsapp",
            "selection_sources": "environment,environment",
        },
    )
    assert seen == [
        (
            "ambiguous",
            (
                ChannelCandidate(provider="telegram", source="environment"),
                ChannelCandidate(provider="whatsapp", source="environment"),
            ),
        )
    ]


def test_interactive_prompt_resolves_missing_selection() -> None:
    result = resolve_fleet_channel_selection(
        {},
        interactive=True,
        prompt=lambda _status, _candidates: "whatsapp",
    )

    assert result.selected_channel == ChannelSelection(
        provider="whatsapp",
        source="prompted",
        status="ready",
        metadata={"prompt_status": "missing_config"},
    )
