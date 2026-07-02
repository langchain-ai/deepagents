"""Channel selection helpers for Fleet-backed Talon runs.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from deepagents_talon.fleet_manifest import ChannelSelection

ChannelProvider = Literal["telegram", "whatsapp"]
ChannelSelectionSource = Literal["explicit", "inferred", "prompted"]
ChannelSelectionStatus = Literal["ready", "missing_config", "ambiguous"]

SUPPORTED_CHANNEL_PROVIDERS: tuple[ChannelProvider, ...] = ("telegram", "whatsapp")

_CHANNEL_ENV = {
    "telegram": "DEEPAGENTS_TALON_TELEGRAM_ENABLED",
    "whatsapp": "DEEPAGENTS_TALON_WHATSAPP_ENABLED",
}


class ChannelSelectionError(ValueError):
    """Raised when a channel selection request is invalid."""


@dataclass(frozen=True, slots=True)
class ChannelCandidate:
    """Candidate channel discovered from CLI flags or environment.

    Args:
        provider: Supported Talon channel provider.
        source: Where the candidate came from.
    """

    provider: ChannelProvider
    source: str


@dataclass(frozen=True, slots=True)
class ChannelSelectionResolution:
    """Resolved channel selection state.

    Args:
        selected_channel: Ready manifest channel selection, when one was resolved.
        status: Resolver status.
        candidates: Candidate channels considered by the resolver.
        message: Human-readable explanation for non-ready results.
    """

    selected_channel: ChannelSelection | None
    status: ChannelSelectionStatus
    candidates: tuple[ChannelCandidate, ...] = ()
    message: str = ""

    @property
    def ready(self) -> bool:
        """Return whether the resolver selected a channel."""
        return self.selected_channel is not None and self.status == "ready"


PromptChannel = Callable[[ChannelSelectionStatus, Sequence[ChannelCandidate]], str]


def resolve_fleet_channel_selection(
    env: Mapping[str, str],
    *,
    explicit_channel: str | None = None,
    channel_flags: Mapping[str, bool] | None = None,
    interactive: bool = False,
    prompt: PromptChannel | None = None,
) -> ChannelSelectionResolution:
    """Resolve the single channel provider for a Fleet-backed Talon run.

    Args:
        env: Talon runtime environment mapping.
        explicit_channel: Provider supplied by `--channel` or an embedding host.
        channel_flags: CLI channel flags keyed by provider.
        interactive: Whether prompting the operator is allowed.
        prompt: Optional prompt callback for interactive selection.

    Returns:
        Channel selection resolution. Non-interactive missing or ambiguous inputs
        return a non-ready resolution instead of blocking for input.

    Raises:
        ChannelSelectionError: If a supplied or prompted provider is unsupported.
    """
    if explicit_channel:
        provider = _validate_provider(explicit_channel)
        return _ready(provider, source="explicit")

    candidates = _channel_candidates(
        env,
        channel_flags={} if channel_flags is None else channel_flags,
    )
    providers = _unique_providers(candidates)
    if len(providers) == 1:
        return _ready(
            providers[0],
            source="inferred",
            metadata={"selection_source": candidates[0].source},
            candidates=candidates,
        )

    status: ChannelSelectionStatus = "missing_config" if not providers else "ambiguous"
    if interactive and prompt is not None:
        provider = _validate_provider(prompt(status, candidates))
        metadata = {"prompt_status": status}
        if candidates:
            metadata["providers"] = ",".join(candidate.provider for candidate in candidates)
            metadata["selection_sources"] = ",".join(candidate.source for candidate in candidates)
        return _ready(provider, source="prompted", metadata=metadata, candidates=candidates)

    message = (
        "No Talon channel is configured; pass --channel or enable exactly one channel."
        if status == "missing_config"
        else "Multiple Talon channels are configured; pass --channel to choose one."
    )
    return ChannelSelectionResolution(
        selected_channel=None,
        status=status,
        candidates=candidates,
        message=message,
    )


def _channel_candidates(
    env: Mapping[str, str],
    *,
    channel_flags: Mapping[str, bool],
) -> tuple[ChannelCandidate, ...]:
    candidates: list[ChannelCandidate] = []
    for provider in SUPPORTED_CHANNEL_PROVIDERS:
        if channel_flags.get(provider, False):
            candidates.append(ChannelCandidate(provider=provider, source="cli"))
        elif _env_enabled(env, _CHANNEL_ENV[provider]):
            candidates.append(ChannelCandidate(provider=provider, source="environment"))

    return tuple(candidates)


def _ready(
    provider: ChannelProvider,
    *,
    source: ChannelSelectionSource,
    metadata: Mapping[str, str] | None = None,
    candidates: tuple[ChannelCandidate, ...] = (),
) -> ChannelSelectionResolution:
    return ChannelSelectionResolution(
        selected_channel=ChannelSelection(
            provider=provider,
            source=source,
            status="ready",
            metadata={} if metadata is None else dict(metadata),
        ),
        status="ready",
        candidates=candidates,
    )


def _unique_providers(candidates: Sequence[ChannelCandidate]) -> tuple[ChannelProvider, ...]:
    providers: list[ChannelProvider] = []
    for candidate in candidates:
        if candidate.provider not in providers:
            providers.append(candidate.provider)
    return tuple(providers)


def _validate_provider(value: str) -> ChannelProvider:
    provider = value.strip().lower()
    if provider in SUPPORTED_CHANNEL_PROVIDERS:
        return cast("ChannelProvider", provider)
    msg = (
        f"Unsupported Talon channel {value!r}; expected one of "
        f"{', '.join(SUPPORTED_CHANNEL_PROVIDERS)}"
    )
    raise ChannelSelectionError(msg)


def _env_enabled(env: Mapping[str, str], key: str) -> bool:
    return env.get(key, "").lower() in {"1", "true", "yes"}
