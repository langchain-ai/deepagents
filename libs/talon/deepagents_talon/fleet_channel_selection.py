"""Channel selection helpers for Fleet-backed Talon runs.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from deepagents_talon.fleet_manifest import ChannelSelection

if TYPE_CHECKING:
    from collections.abc import Mapping

ChannelProvider = Literal["telegram", "whatsapp"]
ChannelSelectionSource = Literal["explicit"]
ChannelSelectionStatus = Literal["ready", "missing_config"]

SUPPORTED_CHANNEL_PROVIDERS: tuple[ChannelProvider, ...] = ("telegram", "whatsapp")


class ChannelSelectionError(ValueError):
    """Raised when a channel selection request is invalid."""


@dataclass(frozen=True, slots=True)
class ChannelSelectionResolution:
    """Resolved channel selection state.

    Args:
        selected_channel: Ready manifest channel selection, when one was resolved.
        status: Resolver status.
        message: Human-readable explanation for non-ready results.
    """

    selected_channel: ChannelSelection | None
    status: ChannelSelectionStatus
    message: str = ""

    @property
    def ready(self) -> bool:
        """Return whether the resolver selected a channel."""
        return self.selected_channel is not None and self.status == "ready"


def resolve_fleet_channel_selection(
    *,
    explicit_channel: str | None = None,
) -> ChannelSelectionResolution:
    """Resolve the single channel provider for a Fleet-backed Talon run.

    Args:
        explicit_channel: Provider supplied by `--channel` or an embedding host.

    Returns:
        Channel selection resolution. Missing explicit input returns a non-ready
        resolution instead of guessing from the runtime environment.

    Raises:
        ChannelSelectionError: If a supplied provider is unsupported.
    """
    if explicit_channel:
        provider = _validate_provider(explicit_channel)
        return _ready(provider, source="explicit")

    return ChannelSelectionResolution(
        selected_channel=None,
        status="missing_config",
        message="--channel is required for Fleet-backed Talon runs.",
    )


def _ready(
    provider: ChannelProvider,
    *,
    source: ChannelSelectionSource,
    metadata: Mapping[str, str] | None = None,
) -> ChannelSelectionResolution:
    return ChannelSelectionResolution(
        selected_channel=ChannelSelection(
            provider=provider,
            source=source,
            status="ready",
            metadata={} if metadata is None else dict(metadata),
        ),
        status="ready",
    )


def _validate_provider(value: str) -> ChannelProvider:
    provider = value.strip().lower()
    if provider in SUPPORTED_CHANNEL_PROVIDERS:
        return cast("ChannelProvider", provider)
    msg = (
        f"Unsupported Talon channel {value!r}; expected one of "
        f"{', '.join(SUPPORTED_CHANNEL_PROVIDERS)}"
    )
    raise ChannelSelectionError(msg)
