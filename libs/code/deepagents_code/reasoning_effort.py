"""Provider-specific reasoning effort support for `/effort`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable

from deepagents_code.model_config import ModelSpec

logger = logging.getLogger(__name__)

EffortLabel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]
"""Closed vocabulary of effort labels across all supported providers.

Typing the per-provider tuples with this alias catches typos in the vocabulary
at check time. It does not express the deeper invariant that a label must be
supported by a *specific* model — that is enforced at runtime by
`supported_efforts_for_model`.
"""

ReasoningProvider: TypeAlias = Literal[
    "anthropic", "fireworks", "google_genai", "openai", "openai_codex"
]
"""Provider identifiers that support model-specific reasoning effort controls.

Values must stay byte-identical to the provider strings from `ModelSpec.parse`
used throughout `model_config.py` (e.g. `CODEX_PROVIDER`).
"""


class ReasoningProviderConfig(NamedTuple):
    """Provider-specific reasoning effort behavior."""

    supported_efforts: Callable[[str], tuple[EffortLabel, ...]]
    """Return supported effort labels for a lowercased model name."""

    default_effort: Callable[[str], EffortLabel | None]
    """Return the provider default effort for a lowercased model name, if known."""

    model_params: Callable[[str], dict[str, Any]]
    """Translate an effort label into provider-specific model params."""

    current_effort: Callable[[dict[str, Any]], str | None]
    """Read the configured effort label from provider-specific model params."""


OPENAI_EFFORTS: tuple[EffortLabel, ...] = ("none", "low", "medium", "high", "xhigh")
"""OpenAI GPT-5 effort labels for `reasoning.effort`.

See https://platform.openai.com/docs/guides/reasoning.
"""

ANTHROPIC_EFFORTS: tuple[EffortLabel, ...] = ("low", "medium", "high", "xhigh", "max")
"""Anthropic `output_config.effort` labels for Opus 4.7+ and Sonnet 5.

See https://platform.claude.com/docs/en/build-with-claude/effort.
"""

ANTHROPIC_EFFORTS_NO_XHIGH: tuple[EffortLabel, ...] = ("low", "medium", "high", "max")
"""Anthropic effort labels for Opus 4.6 and Sonnet 4.6.

These models predate `xhigh`; Sonnet 4.5 rejects `effort` entirely.
See https://platform.claude.com/docs/en/build-with-claude/effort.
"""

ANTHROPIC_EFFORTS_NO_MAX: tuple[EffortLabel, ...] = ("low", "medium", "high")
"""Anthropic effort labels for Opus 4.5.

Opus 4.5 predates both `max` (Opus 4.6+) and `xhigh` (Opus 4.7+).
See https://platform.claude.com/docs/en/build-with-claude/effort.
"""

GOOGLE_EFFORTS: tuple[EffortLabel, ...] = ("low", "medium", "high")
"""Gemini 3 effort labels for `thinking_level`.

Gemini 3.1 Pro and 3.5 Flash accept low/medium/high only; `minimal` is
Flash-Lite / original-Pro territory, neither of which is offered here. See
https://ai.google.dev/gemini-api/docs/thinking.
"""

FIREWORKS_REASONING_EFFORTS: tuple[EffortLabel, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
"""Fireworks `reasoning_effort` labels for DeepSeek V4 Pro.

See https://docs.fireworks.ai/guides/reasoning.
"""

FIREWORKS_KIMI_EFFORTS: tuple[EffortLabel, ...] = ("low", "medium", "high")
"""Fireworks `reasoning_effort` labels for Kimi K2 models.

See https://docs.fireworks.ai/guides/reasoning.
"""

FIREWORKS_GLM_EFFORTS: tuple[EffortLabel, ...] = ("none", "high", "max")
"""Fireworks `reasoning_effort` labels for GLM 5 models.

See https://docs.fireworks.ai/guides/reasoning.
"""

_REASONING_KEYS: frozenset[str] = frozenset(
    {"effort", "reasoning", "thinking", "thinking_level"}
)
"""Runtime config keys that may already carry provider reasoning settings."""


def _openai_supported_efforts(_model: str) -> tuple[EffortLabel, ...]:
    """Return OpenAI reasoning effort levels."""
    return OPENAI_EFFORTS


def _openai_default_effort(model: str) -> EffortLabel | None:
    """Return the OpenAI default reasoning effort when known."""
    return "medium" if model.startswith("gpt-5.5") else None


def _openai_model_params(effort: str) -> dict[str, Any]:
    """Return OpenAI reasoning params for an effort label."""
    if effort == "none":
        return {"reasoning": {"effort": "none"}}
    return {"reasoning": {"effort": effort, "summary": "auto"}}


def _openai_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the OpenAI reasoning effort from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    reasoning = model_params.get("reasoning")
    if isinstance(reasoning, dict):
        value = reasoning.get("effort")
        return value if isinstance(value, str) else None
    return None


def _anthropic_supported_efforts(model: str) -> tuple[EffortLabel, ...]:
    """Return the effort levels an Anthropic model accepts.

    Args:
        model: Lowercased Anthropic model name (e.g. `claude-opus-4-8`).

    Returns:
        Supported effort labels, or an empty tuple when the model does not
        accept `effort` (e.g. Sonnet 4.5).
    """
    if model.startswith("claude-opus-"):
        if "opus-4-5" in model:
            # Opus 4.5 predates both `max` (4.6+) and `xhigh` (4.7+).
            return ANTHROPIC_EFFORTS_NO_MAX
        # Opus 4.6 predates `xhigh`; 4.7+ accept the full range.
        return ANTHROPIC_EFFORTS_NO_XHIGH if "opus-4-6" in model else ANTHROPIC_EFFORTS
    if model.startswith("claude-sonnet-"):
        if "sonnet-4-5" in model:
            return ()
        # Sonnet 4.6 predates `xhigh`; Sonnet 5 accepts the full range.
        return (
            ANTHROPIC_EFFORTS_NO_XHIGH if "sonnet-4-6" in model else ANTHROPIC_EFFORTS
        )
    return ()


def _anthropic_default_effort(model: str) -> EffortLabel | None:
    """Return the Anthropic default reasoning effort when known."""
    return "high" if _anthropic_supported_efforts(model) else None


def _anthropic_model_params(effort: str) -> dict[str, Any]:
    """Return Anthropic reasoning params for an effort label."""
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "effort": effort,
    }


def _anthropic_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the Anthropic reasoning effort from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    value = model_params.get("effort")
    return value if isinstance(value, str) else None


def _google_supported_efforts(_model: str) -> tuple[EffortLabel, ...]:
    """Return Gemini thinking levels."""
    return GOOGLE_EFFORTS


def _google_default_effort(model: str) -> EffortLabel | None:
    """Return the Gemini default thinking level when known."""
    if model.startswith("gemini-3.5-flash"):
        return "medium"
    if model.startswith(("gemini-3.1-pro", "gemini-3-flash", "gemini-3-pro")):
        return "high"
    return None


def _google_model_params(effort: str) -> dict[str, Any]:
    """Return Gemini thinking params for an effort label."""
    return {"thinking_level": effort}


def _google_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the Gemini thinking level from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    value = model_params.get("thinking_level")
    return value if isinstance(value, str) else None


def _fireworks_supported_efforts(model: str) -> tuple[EffortLabel, ...]:
    """Return Fireworks reasoning effort levels for a model."""
    if "kimi-k2" in model:
        return FIREWORKS_KIMI_EFFORTS
    if "glm-5" in model:
        return FIREWORKS_GLM_EFFORTS
    if "deepseek-v4-pro" in model:
        return FIREWORKS_REASONING_EFFORTS
    return ()


def _fireworks_default_effort(model: str) -> EffortLabel | None:
    """Return the Fireworks default reasoning effort when known."""
    if "deepseek-v4-pro" in model:
        return "high"
    if "glm-5p2" in model:
        return "max"
    return None


def _fireworks_model_params(effort: str) -> dict[str, Any]:
    """Return Fireworks reasoning params for an effort label."""
    return {"model_kwargs": {"reasoning_effort": effort}}


def _fireworks_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the Fireworks reasoning effort from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    kwargs = model_params.get("model_kwargs")
    if isinstance(kwargs, dict):
        value = kwargs.get("reasoning_effort")
        return value if isinstance(value, str) else None
    return None


_OPENAI_CONFIG = ReasoningProviderConfig(
    supported_efforts=_openai_supported_efforts,
    default_effort=_openai_default_effort,
    model_params=_openai_model_params,
    current_effort=_openai_current_effort,
)
"""Shared config for OpenAI-compatible GPT-5 reasoning providers.

`openai` and `openai_codex` use different provider names so model selection can
route to the right client, but `/effort` maps both to the same reasoning params.
"""

_PROVIDER_CONFIGS: dict[ReasoningProvider, ReasoningProviderConfig] = {
    "openai": _OPENAI_CONFIG,
    "openai_codex": _OPENAI_CONFIG,
    "anthropic": ReasoningProviderConfig(
        supported_efforts=_anthropic_supported_efforts,
        default_effort=_anthropic_default_effort,
        model_params=_anthropic_model_params,
        current_effort=_anthropic_current_effort,
    ),
    "google_genai": ReasoningProviderConfig(
        supported_efforts=_google_supported_efforts,
        default_effort=_google_default_effort,
        model_params=_google_model_params,
        current_effort=_google_current_effort,
    ),
    "fireworks": ReasoningProviderConfig(
        supported_efforts=_fireworks_supported_efforts,
        default_effort=_fireworks_default_effort,
        model_params=_fireworks_model_params,
        current_effort=_fireworks_current_effort,
    ),
}
"""Provider-specific reasoning effort behavior keyed by `ModelSpec` provider."""


def _classify_reasoning_provider(provider: str, model: str) -> ReasoningProvider | None:
    """Classify provider/model parts into a reasoning-capable provider.

    Returns:
        The registry key for supported reasoning models, or `None` otherwise.
    """
    model_lower = model.lower()
    if provider == "openai" and model_lower.startswith("gpt-5"):
        return "openai"
    if provider == "openai_codex" and model_lower.startswith("gpt-5"):
        return "openai_codex"
    if provider == "anthropic" and model_lower.startswith(
        ("claude-opus-", "claude-sonnet-")
    ):
        return "anthropic"
    if provider == "google_genai" and model_lower.startswith("gemini-3"):
        return "google_genai"
    if provider == "fireworks" and model_lower.startswith("accounts/fireworks/models/"):
        return "fireworks"
    return None


def _reasoning_config(model_spec: str) -> tuple[ReasoningProviderConfig, str] | None:
    """Return provider config and lowercased model when reasoning is supported."""
    parsed = ModelSpec.try_parse(model_spec)
    if parsed is None:
        return None
    provider = _classify_reasoning_provider(parsed.provider, parsed.model)
    if provider is None:
        return None
    return _PROVIDER_CONFIGS[provider], parsed.model.lower()


def supported_efforts_for_model(model_spec: str | None) -> tuple[str, ...]:
    """Return reasoning efforts supported by `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.

    Returns:
        Supported effort labels, or an empty tuple when the model is unsupported.
    """
    if not model_spec:
        return ()
    context = _reasoning_config(model_spec)
    if context is None:
        return ()
    config, model = context
    efforts = config.supported_efforts(model)
    if not efforts:
        # A recognized reasoning provider that yields no configurable efforts
        # usually means the model-version heuristics need updating for a newer
        # release — surface it at debug level rather than silently reporting
        # "not configurable".
        logger.debug("No configurable reasoning efforts for %s", model_spec)
    return efforts


def default_effort_for_model(model_spec: str | None) -> EffortLabel | None:
    """Return the documented default reasoning effort when known.

    Args:
        model_spec: `provider:model` spec for the active model.

    Returns:
        The provider default effort label, or `None` when the default is unknown
            or cannot be represented by the supported effort labels.
    """
    if not model_spec:
        return None
    context = _reasoning_config(model_spec)
    if context is None:
        return None
    config, model = context
    return config.default_effort(model)


def model_params_for_effort(model_spec: str, effort: str) -> dict[str, Any] | None:
    """Translate an effort label into provider-specific model params.

    Args:
        model_spec: `provider:model` spec for the active model.
        effort: Effort label accepted by `supported_efforts_for_model`.

    Returns:
        Model params to merge into the per-session override, or `None` when the
        model/effort pair is unsupported.
    """
    context = _reasoning_config(model_spec)
    if context is None:
        return None
    config, model = context
    if effort not in config.supported_efforts(model):
        return None
    return config.model_params(effort)


def current_effort_from_model_params(
    model_spec: str | None, model_params: dict[str, Any] | None
) -> str | None:
    """Read the configured effort from model params when present.

    Args:
        model_spec: `provider:model` spec for the active model.
        model_params: Per-session model params.

    Returns:
        The configured effort, or `None` when no recognized effort override is set.
    """
    if not model_spec or not model_params:
        return None
    context = _reasoning_config(model_spec)
    if context is None:
        return None
    config, _ = context
    return config.current_effort(model_params)


def merge_effort_model_params(
    existing: dict[str, Any] | None, effort_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge effort params into existing per-session model params.

    Args:
        existing: Current per-session model params.
        effort_params: Params returned by `model_params_for_effort`.

    Returns:
        A new merged dictionary preserving unrelated nested `model_kwargs`.
    """
    merged = dict(existing) if existing else {}
    for key, value in effort_params.items():
        if key == "model_kwargs" and isinstance(value, dict):
            current = merged.get("model_kwargs")
            base = dict(current) if isinstance(current, dict) else {}
            base.update(value)
            merged["model_kwargs"] = base
        else:
            merged[key] = value
    return merged


def without_effort_model_params(
    existing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Remove known effort params while preserving unrelated model params.

    Args:
        existing: Current per-session model params.

    Returns:
        A cleaned dictionary, or `None` when no params remain.
    """
    if not existing:
        return None
    # Exclude `model_kwargs` from the comprehension and rebuild it below.
    # Leaving it here would retain a stale `reasoning_effort` when the cleaned
    # nested dict ends up empty — the empty-check would then skip the overwrite
    # and the original (still-populated) copy would survive.
    cleaned = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in existing.items()
        if key not in _REASONING_KEYS and key != "model_kwargs"
    }
    kwargs = existing.get("model_kwargs")
    if isinstance(kwargs, dict):
        model_kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
        if model_kwargs:
            cleaned["model_kwargs"] = model_kwargs
    elif kwargs is not None:
        cleaned["model_kwargs"] = kwargs
    return cleaned or None
