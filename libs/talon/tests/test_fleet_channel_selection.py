from __future__ import annotations

import pytest

from deepagents_talon.fleet_channel_selection import (
    ChannelSelectionError,
    resolve_fleet_channel_selection,
)
from deepagents_talon.fleet_manifest import ChannelSelection


def test_explicit_channel_selection_is_ready() -> None:
    result = resolve_fleet_channel_selection(explicit_channel="telegram")

    assert result.ready
    assert result.selected_channel == ChannelSelection(
        provider="telegram",
        source="explicit",
        status="ready",
    )


def test_missing_channel_selection_is_non_ready() -> None:
    result = resolve_fleet_channel_selection()

    assert not result.ready
    assert result.status == "missing_config"
    assert result.selected_channel is None
    assert "--channel is required" in result.message


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(ChannelSelectionError, match="Unsupported Talon channel"):
        resolve_fleet_channel_selection(explicit_channel="email")
