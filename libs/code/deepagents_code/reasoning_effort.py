"""Provider-specific reasoning effort support for `/effort`."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, get_args

if TYPE_CHECKING:
    from collections.abc import Callable

from deepagents_code.model_config import CODEX_PROVIDER, ModelSpec

logger = logging.getLogger(__name__)

EffortLabel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]
"""Closed vocabulary of effort labels across all supported providers.

Typing the per-provider tuples with this alias catches typos in the vocabulary
at check time. It does not express the deeper invariant that a label must be
supported by a *specific* model — that is enforced at runtime by
`supported_efforts_for_model`.

This vocabulary is also hand-duplicated as display text in the `/effort`
`argument_hint` (`command_registry.py`) and in `COMMANDS.md`; those are not
type-checked against this alias, so update them in lockstep when it changes.
"""

ReasoningProvider: TypeAlias = Literal[
    "anthropic", "fireworks", "google_genai", "openai", "openai_codex", "xai"
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
"""OpenAI GPT-5 effort labels before GPT-5.6 for `reasoning.effort`.

See https://platform.openai.com/docs/guides/reasoning.
"""

OPENAI_56_EFFORTS: tuple[EffortLabel, ...] = (*OPENAI_EFFORTS, "max")
"""OpenAI GPT-5.6 effort labels for `reasoning.effort`."""

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
"""Gemini `thinking_level` labels.

Applied to every `gemini-3*` model (the gate in `_classify_reasoning_provider`),
including Gemini 3 Pro/Flash, 3.1 Pro, and 3.5 Flash — all accept
low/medium/high. `minimal` is Flash-Lite / original-Pro territory, neither of
which is offered here. See https://ai.google.dev/gemini-api/docs/thinking.
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

_FIREWORKS_MODEL_ID_PREFIXES: tuple[str, ...] = (
    "accounts/fireworks/models/",
    "accounts/fireworks/routers/",
)
"""Fully-qualified Fireworks model-id prefixes that carry reasoning support.

Fireworks exposes both individual models (`accounts/fireworks/models/...`) and
routers that dispatch across models (`accounts/fireworks/routers/...`); both
forms accept the `reasoning_effort` param, so `/effort` gates on either."""

XAI_EFFORTS: tuple[EffortLabel, ...] = ("low", "medium", "high")
"""xAI `reasoning_effort` labels for Grok 4.5.

See https://docs.x.ai/developers/model-capabilities/text/reasoning.
"""

_REASONING_KEYS: frozenset[str] = frozenset(
    {
        "effort",
        "extra_body",
        "output_config",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "thinking_level",
    }
)
"""Runtime config keys that may already carry provider reasoning settings."""


def _openai_supported_efforts(model: str) -> tuple[EffortLabel, ...]:
    """Return OpenAI reasoning effort levels."""
    return OPENAI_56_EFFORTS if _has_version(model, "gpt-5.6") else OPENAI_EFFORTS


def _openai_default_effort(model: str) -> EffortLabel | None:
    """Return the OpenAI default reasoning effort when known."""
    if _has_version(model, "gpt-5.5") or _has_version(model, "gpt-5.6"):
        return "medium"
    return None


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
        if value is not None and not isinstance(value, str):
            # Present but mistyped (e.g. a hand-edited config or bad
            # `--model-params` JSON). Discard it, but log the *type* — never the
            # value — so the drop is greppable instead of silently read as "no
            # effort set" while the malformed param still ships on the wire.
            logger.warning(
                "Ignoring non-str OpenAI reasoning.effort of type %s",
                type(value).__name__,
            )
        return value if isinstance(value, str) else None
    if reasoning is not None:
        logger.warning(
            "Ignoring OpenAI reasoning params of unexpected type %s",
            type(reasoning).__name__,
        )
    return None


def _has_version(model: str, token: str) -> bool:
    """Return whether `model` carries version `token` not followed by a digit.

    A plain substring test would match a longer version by accident — e.g.
    `"opus-4-1" in "claude-opus-4-16"` is true. Anchoring on a non-digit
    boundary keeps `opus-4-1` from matching a future `opus-4-16` while still
    matching a dated suffix like `opus-4-1-20250805`. `token` is always a
    hardcoded constant, but `re.escape` keeps the match literal regardless.
    """
    return re.search(rf"{re.escape(token)}(?!\d)", model) is not None


def _anthropic_supported_efforts(model: str) -> tuple[EffortLabel, ...]:
    """Return the effort levels an Anthropic model accepts.

    Args:
        model: Lowercased Anthropic model name (e.g. `claude-opus-4-8`).

    Returns:
        Supported effort labels, or an empty tuple when the model does not
        accept `effort` (e.g. Sonnet 4.5).
    """
    if model.startswith("claude-opus-"):
        if _has_version(model, "opus-4-0") or _has_version(model, "opus-4-1"):
            # Opus 4.0/4.1 predate reasoning effort entirely.
            return ()
        if _has_version(model, "opus-4-5"):
            # Opus 4.5 predates both `max` (4.6+) and `xhigh` (4.7+).
            return ANTHROPIC_EFFORTS_NO_MAX
        # Opus 4.6 predates `xhigh`; 4.7+ (and newer, unrecognized versions)
        # get the full range.
        return (
            ANTHROPIC_EFFORTS_NO_XHIGH
            if _has_version(model, "opus-4-6")
            else ANTHROPIC_EFFORTS
        )
    if model.startswith("claude-sonnet-"):
        if (
            _has_version(model, "sonnet-4-0")
            or _has_version(model, "sonnet-4-1")
            or _has_version(model, "sonnet-4-5")
        ):
            # Sonnet 4.0/4.1 predate effort; Sonnet 4.5 rejects it.
            return ()
        # Sonnet 4.6 predates `xhigh`; Sonnet 5 (and newer) get the full range.
        return (
            ANTHROPIC_EFFORTS_NO_XHIGH
            if _has_version(model, "sonnet-4-6")
            else ANTHROPIC_EFFORTS
        )
    return ()


def _anthropic_default_effort(model: str) -> EffortLabel | None:
    """Return the Anthropic default reasoning effort when known."""
    return "high" if _anthropic_supported_efforts(model) else None


def _anthropic_model_params(effort: str) -> dict[str, Any]:
    """Return Anthropic reasoning params for an effort label."""
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }


def _anthropic_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the Anthropic reasoning effort from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    output_config = model_params.get("output_config")
    if isinstance(output_config, dict):
        value = output_config.get("effort")
        if value is not None and not isinstance(value, str):
            logger.warning(
                "Ignoring non-str Anthropic output_config.effort of type %s",
                type(value).__name__,
            )
        return value if isinstance(value, str) else None
    if output_config is not None:
        logger.warning(
            "Ignoring Anthropic output_config params of unexpected type %s",
            type(output_config).__name__,
        )
    return None


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
    if value is not None and not isinstance(value, str):
        logger.warning(
            "Ignoring non-str Gemini thinking_level of type %s",
            type(value).__name__,
        )
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
        if value is not None and not isinstance(value, str):
            logger.warning(
                "Ignoring non-str Fireworks reasoning_effort of type %s",
                type(value).__name__,
            )
        return value if isinstance(value, str) else None
    # A non-dict `model_kwargs` is a legitimate shape here (it may hold
    # unrelated params, or be preserved verbatim by `without_effort_model_params`),
    # so treat it as "no effort configured" without warning.
    return None


def _is_xai_grok_45(model: str) -> bool:
    """Return whether `model` is Grok 4.5 or a documented alias."""
    return model in {"grok-4.5", "grok-4.5-latest", "grok-build-latest"}


def _xai_supported_efforts(model: str) -> tuple[EffortLabel, ...]:
    """Return xAI reasoning effort levels for a model."""
    return XAI_EFFORTS if _is_xai_grok_45(model) else ()


def _xai_default_effort(model: str) -> EffortLabel | None:
    """Return the xAI default reasoning effort when known."""
    return "high" if _is_xai_grok_45(model) else None


def _xai_model_params(effort: str) -> dict[str, Any]:
    """Return xAI reasoning params for an effort label."""
    return {"extra_body": {"reasoning_effort": effort}}


def _xai_current_effort(model_params: dict[str, Any]) -> str | None:
    """Read the xAI reasoning effort from model params.

    Returns:
        The configured effort label, or `None` when unset.
    """
    extra = model_params.get("extra_body")
    if isinstance(extra, dict):
        value = extra.get("reasoning_effort")
        if value is not None and not isinstance(value, str):
            logger.warning(
                "Ignoring non-str xAI extra_body.reasoning_effort of type %s",
                type(value).__name__,
            )
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
    "xai": ReasoningProviderConfig(
        supported_efforts=_xai_supported_efforts,
        default_effort=_xai_default_effort,
        model_params=_xai_model_params,
        current_effort=_xai_current_effort,
    ),
}
"""Provider-specific reasoning effort behavior keyed by `ModelSpec` provider."""

if set(_PROVIDER_CONFIGS) != set(get_args(ReasoningProvider)):  # pragma: no cover
    # `_classify_reasoning_provider` only ever returns members of the
    # `ReasoningProvider` vocabulary, and `_reasoning_config` indexes
    # `_PROVIDER_CONFIGS` with the result — so the two must stay in lockstep or
    # that lookup raises `KeyError` at runtime. Fail loudly at import instead.
    msg = "_PROVIDER_CONFIGS keys must match the ReasoningProvider vocabulary"
    raise RuntimeError(msg)


def _classify_reasoning_provider(provider: str, model: str) -> ReasoningProvider | None:
    """Classify provider/model parts into a reasoning-capable provider.

    Returns:
        The registry key for supported reasoning models, or `None` otherwise.
    """
    model_lower = model.lower()
    if provider == "openai" and model_lower.startswith("gpt-5"):
        return "openai"
    if provider == CODEX_PROVIDER and model_lower.startswith("gpt-5"):
        return "openai_codex"
    if provider == "anthropic" and model_lower.startswith(
        ("claude-opus-", "claude-sonnet-")
    ):
        return "anthropic"
    if provider == "google_genai" and model_lower.startswith("gemini-3"):
        return "google_genai"
    if provider == "fireworks" and model_lower.startswith(_FIREWORKS_MODEL_ID_PREFIXES):
        return "fireworks"
    if provider == "xai" and _is_xai_grok_45(model_lower):
        return "xai"
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

    Returns plain `str` labels rather than `EffortLabel`: this is the public
    boundary where the label vocabulary is intentionally dropped, since the
    values flow straight to the UI.

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
        # release. Log at info so the maintenance gap is visible at default
        # verbosity rather than silently reporting "not configurable".
        logger.info("No configurable reasoning efforts for %s", model_spec)
    return efforts


def default_effort_for_model(model_spec: str | None) -> str | None:
    """Return the documented default reasoning effort when known.

    Returns a plain `str` rather than `EffortLabel`: like
    `supported_efforts_for_model`, this is the public boundary where the label
    vocabulary is intentionally dropped, since every caller treats the value as
    display text.

    Args:
        model_spec: `provider:model` spec for the active model.

    Returns:
        The provider default effort label, or `None` when the default is unknown.
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
        A new merged dictionary preserving unrelated nested config objects.
    """
    merged = dict(existing) if existing else {}
    for key, value in effort_params.items():
        if key in {"extra_body", "model_kwargs", "output_config"} and isinstance(
            value, dict
        ):
            current = merged.get(key)
            base = dict(current) if isinstance(current, dict) else {}
            base.update(value)
            merged[key] = base
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
    # Exclude nested config objects from the comprehension and rebuild them below.
    # Leaving them here would retain stale nested effort keys when the cleaned
    # nested dict ends up empty — the empty-check would then skip the overwrite
    # and the original (still-populated) copy would survive.
    cleaned = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in existing.items()
        if key not in _REASONING_KEYS
        and key not in {"extra_body", "model_kwargs", "output_config"}
    }
    kwargs = existing.get("model_kwargs")
    if isinstance(kwargs, dict):
        model_kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
        if model_kwargs:
            cleaned["model_kwargs"] = model_kwargs
    elif kwargs is not None:
        cleaned["model_kwargs"] = kwargs
    output_config = existing.get("output_config")
    if isinstance(output_config, dict):
        output_config_params = {k: v for k, v in output_config.items() if k != "effort"}
        if output_config_params:
            cleaned["output_config"] = output_config_params
    elif output_config is not None:
        cleaned["output_config"] = output_config
    extra = existing.get("extra_body")
    if isinstance(extra, dict):
        extra_params = {k: v for k, v in extra.items() if k != "reasoning_effort"}
        if extra_params:
            cleaned["extra_body"] = extra_params
    elif extra is not None:
        cleaned["extra_body"] = extra
    return cleaned or None
