"""Provider-specific reasoning effort support for `/effort`."""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeAlias, assert_never

from deepagents_code.model_config import ModelSpec

logger = logging.getLogger(__name__)

EffortLabel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]
"""Closed vocabulary of effort labels across all supported providers.

Typing the per-provider tuples with this alias catches typos in the vocabulary
at check time. It does not express the deeper invariant that a label must be
supported by a *specific* model — that is enforced at runtime by
`supported_efforts_for_model`.
"""

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

ReasoningProvider: TypeAlias = Literal[
    "anthropic", "fireworks", "google_genai", "openai", "openai_codex"
]
"""Provider identifiers that support model-specific reasoning effort controls.

Values must stay byte-identical to the provider strings from `ModelSpec.parse`
used throughout `model_config.py` (e.g. `CODEX_PROVIDER`).
"""

_REASONING_KEYS: frozenset[str] = frozenset(
    {"effort", "reasoning", "thinking", "thinking_level"}
)
"""Runtime config keys that may already carry provider reasoning settings."""


def _anthropic_efforts(model: str) -> tuple[str, ...]:
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


def _reasoning_provider(model_spec: str) -> ReasoningProvider | None:
    """Classify `model_spec` into a reasoning-capable provider, or `None`.

    Args:
        model_spec: `provider:model` spec for the active model.

    Returns:
        The reasoning provider, or `None` when the model family does not support
            reasoning effort (or the spec is unparseable).
    """
    parsed = ModelSpec.try_parse(model_spec)
    if parsed is None:
        return None
    provider, model = parsed.provider, parsed.model
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


def supported_efforts_for_model(model_spec: str | None) -> tuple[str, ...]:
    """Return reasoning efforts supported by `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.

    Returns:
        Supported effort labels, or an empty tuple when the model is unsupported.
    """
    if not model_spec:
        return ()
    provider = _reasoning_provider(model_spec)
    if provider is None:
        return ()
    efforts: tuple[str, ...]
    match provider:
        case "openai" | "openai_codex":
            efforts = OPENAI_EFFORTS
        case "anthropic":
            parsed = ModelSpec.try_parse(model_spec)
            efforts = _anthropic_efforts(parsed.model.lower()) if parsed else ()
        case "google_genai":
            efforts = GOOGLE_EFFORTS
        case "fireworks":
            parsed = ModelSpec.try_parse(model_spec)
            model = parsed.model.lower() if parsed else ""
            if "kimi-k2" in model:
                efforts = FIREWORKS_KIMI_EFFORTS
            elif "glm-5" in model:
                efforts = FIREWORKS_GLM_EFFORTS
            elif "deepseek-v4-pro" in model:
                efforts = FIREWORKS_REASONING_EFFORTS
            else:
                efforts = ()
        case _:
            assert_never(provider)
    if not efforts:
        # A recognized reasoning provider that yields no configurable efforts
        # usually means the model-version heuristics need updating for a newer
        # release — surface it at debug level rather than silently reporting
        # "not configurable".
        logger.debug(
            "No configurable reasoning efforts for %s (provider=%s)",
            model_spec,
            provider,
        )
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
    provider = _reasoning_provider(model_spec)
    if provider is None:
        return None
    parsed = ModelSpec.try_parse(model_spec)
    model = parsed.model.lower() if parsed else ""
    match provider:
        case "openai" | "openai_codex":
            if model.startswith("gpt-5.5"):
                return "medium"
        case "anthropic":
            return "high" if _anthropic_efforts(model) else None
        case "google_genai":
            if model.startswith("gemini-3.5-flash"):
                return "medium"
            if model.startswith(("gemini-3.1-pro", "gemini-3-flash", "gemini-3-pro")):
                return "high"
        case "fireworks":
            if "deepseek-v4-pro" in model:
                return "high"
            if "glm-5p2" in model:
                return "max"
        case _:
            assert_never(provider)
    return None


def model_params_for_effort(model_spec: str, effort: str) -> dict[str, Any] | None:
    """Translate an effort label into provider-specific model params.

    Args:
        model_spec: `provider:model` spec for the active model.
        effort: Effort label accepted by `supported_efforts_for_model`.

    Returns:
        Model params to merge into the per-session override, or `None` when the
        model/effort pair is unsupported.
    """
    if effort not in supported_efforts_for_model(model_spec):
        return None
    provider = _reasoning_provider(model_spec)
    if provider is None:
        return None
    match provider:
        case "openai" | "openai_codex":
            if effort == "none":
                return {"reasoning": {"effort": "none"}}
            return {"reasoning": {"effort": effort, "summary": "auto"}}
        case "anthropic":
            return {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "effort": effort,
            }
        case "google_genai":
            return {"thinking_level": effort}
        case "fireworks":
            return {"model_kwargs": {"reasoning_effort": effort}}
        case _:
            assert_never(provider)


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
    provider = _reasoning_provider(model_spec)
    if provider is None:
        return None
    value: object = None
    match provider:
        case "openai" | "openai_codex":
            reasoning = model_params.get("reasoning")
            if isinstance(reasoning, dict):
                value = reasoning.get("effort")
        case "anthropic":
            value = model_params.get("effort")
        case "google_genai":
            value = model_params.get("thinking_level")
        case "fireworks":
            kwargs = model_params.get("model_kwargs")
            if isinstance(kwargs, dict):
                value = kwargs.get("reasoning_effort")
        case _:
            assert_never(provider)
    return value if isinstance(value, str) else None


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
