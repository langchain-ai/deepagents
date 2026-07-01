"""Provider-specific reasoning effort support for `/effort`."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from deepagents_code.model_config import ModelSpec

OPENAI_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
# `effort` maps to Anthropic's `output_config.effort`, whose accepted values
# vary by model version: `xhigh` exists only on Opus 4.7+ and Sonnet 5, `max`
# is unavailable before Opus 4.6 / Sonnet 4.6, and Sonnet 4.5 rejects `effort`.
ANTHROPIC_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_EFFORTS_NO_XHIGH: tuple[str, ...] = ("low", "medium", "high", "max")
# Gemini 3.1 Pro and 3.5 Flash accept low/medium/high only (`minimal` is
# Flash-Lite / original-Pro territory, neither of which is offered here).
GOOGLE_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
FIREWORKS_REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
FIREWORKS_KIMI_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
FIREWORKS_GLM_EFFORTS: tuple[str, ...] = ("none", "high", "max")

ReasoningProvider: TypeAlias = Literal[
    "anthropic", "fireworks", "google_genai", "openai", "openai_codex"
]

_REASONING_KEYS: frozenset[str] = frozenset(
    {"effort", "reasoning", "thinking", "thinking_level"}
)


def _anthropic_efforts(model: str) -> tuple[str, ...]:
    """Return the effort levels an Anthropic model accepts.

    Args:
        model: Lowercased Anthropic model name (e.g. `claude-opus-4-8`).

    Returns:
        Supported effort labels, or an empty tuple when the model does not
        accept `effort` (e.g. Sonnet 4.5).
    """
    if model.startswith("claude-opus-"):
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
    if provider in {"openai", "openai_codex"}:
        return OPENAI_EFFORTS
    if provider == "anthropic":
        parsed = ModelSpec.try_parse(model_spec)
        return _anthropic_efforts(parsed.model.lower()) if parsed is not None else ()
    if provider == "google_genai":
        return GOOGLE_EFFORTS
    if provider == "fireworks":
        parsed = ModelSpec.try_parse(model_spec)
        model = parsed.model.lower() if parsed is not None else ""
        if "kimi-k2" in model:
            return FIREWORKS_KIMI_EFFORTS
        if "glm-5" in model:
            return FIREWORKS_GLM_EFFORTS
        if "deepseek-v4-pro" in model:
            return FIREWORKS_REASONING_EFFORTS
    return ()


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
    if provider in {"openai", "openai_codex"}:
        if effort == "none":
            return {"reasoning": {"effort": "none"}}
        return {"reasoning": {"effort": effort, "summary": "auto"}}
    if provider == "anthropic":
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "effort": effort,
        }
    if provider == "google_genai":
        return {"thinking_level": effort}
    if provider == "fireworks":
        return {"model_kwargs": {"reasoning_effort": effort}}
    return None


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
    value: object = None
    if provider in {"openai", "openai_codex"}:
        reasoning = model_params.get("reasoning")
        if isinstance(reasoning, dict):
            value = reasoning.get("effort")
    elif provider == "anthropic":
        value = model_params.get("effort")
    elif provider == "google_genai":
        value = model_params.get("thinking_level")
    elif provider == "fireworks":
        kwargs = model_params.get("model_kwargs")
        if isinstance(kwargs, dict):
            value = kwargs.get("reasoning_effort")
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
    cleaned = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in existing.items()
        if key not in _REASONING_KEYS
    }
    kwargs = existing.get("model_kwargs")
    if isinstance(kwargs, dict):
        model_kwargs = dict(kwargs)
        model_kwargs.pop("reasoning_effort", None)
        if model_kwargs:
            cleaned["model_kwargs"] = model_kwargs
    return cleaned or None
