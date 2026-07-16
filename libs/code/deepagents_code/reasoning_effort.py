"""Reasoning effort support for `/effort`.

Supported levels come from LangChain model profiles. Provider-specific
translation of `reasoning_effort` into each provider's API is handled by the
corresponding LangChain chat model implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeAlias

from deepagents_code.model_config import get_model_profiles

logger = logging.getLogger(__name__)

EffortLabel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]
"""Closed vocabulary of effort labels across all supported providers.

Typing call sites with this alias catches typos in the vocabulary at check
time. It does not express the deeper invariant that a label must be
supported by a *specific* model — that is enforced at runtime by
`supported_efforts_for_model`.

This vocabulary is also hand-duplicated as display text in the `/effort`
`argument_hint` (`command_registry.py`) and in `COMMANDS.md`; those are not
type-checked against this alias, so update them in lockstep when it changes.
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
"""Runtime config keys that may carry reasoning settings.

Includes both the standard `reasoning_effort` key and provider-specific
variants, allowing `/effort clear` to remove either.
"""


def _model_profile(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Look up `model_spec`'s merged profile dict, if any.

    Returns:
        The profile dict, or `None` when `model_spec` is unset or unrecognized.
    """
    if not model_spec:
        return None
    entry = get_model_profiles(cli_override=cli_override).get(model_spec)
    if entry is None:
        return None
    return entry["profile"]


def supported_efforts_for_model(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> tuple[str, ...]:
    """Return reasoning efforts supported by `model_spec`.

    Returns plain `str` labels rather than `EffortLabel`: this is the public
    boundary where the label vocabulary is intentionally dropped, since the
    values flow straight to the UI.

    Args:
        model_spec: `provider:model` spec for the active model.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        Supported effort labels, or an empty tuple when the model is
            unsupported or unrecognized.
    """
    profile = _model_profile(model_spec, cli_override=cli_override)
    if profile is None:
        return ()
    levels = profile.get("reasoning_effort_levels")
    if not isinstance(levels, (list, tuple)):
        return ()
    return tuple(str(level) for level in levels)


def default_effort_for_model(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> str | None:
    """Return the documented default reasoning effort when known.

    Args:
        model_spec: `provider:model` spec for the active model.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        The provider default effort label, or `None` when the default is
            unknown or the model doesn't support reasoning effort.
    """
    profile = _model_profile(model_spec, cli_override=cli_override)
    if profile is None:
        return None
    default = profile.get("reasoning_effort_default")
    return default if isinstance(default, str) else None


def is_effort_supported_for_model(
    model_spec: str, effort: str, *, cli_override: dict[str, Any] | None = None
) -> bool:
    """Return whether `effort` is a supported level for `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.
        effort: Effort label to check.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        `True` if `effort` is one of `model_spec`'s supported levels.
    """
    return effort in supported_efforts_for_model(model_spec, cli_override=cli_override)


def _str_or_none(value: object, *, key: str) -> str | None:
    """Return `value` if it's a `str`, else log and return `None`.

    Logs and ignores mistyped values instead of raising, so malformed effort
    settings don't fail silently.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    logger.warning("Ignoring non-str %s of type %s", key, type(value).__name__)
    return None


def current_effort_from_model_params(
    model_spec: str | None, model_params: dict[str, Any] | None
) -> str | None:
    """Read the configured effort from model params when present.

        `/effort` now writes only the flat `reasoning_effort` sentinel, which is
        checked first. Provider-specific fields are recognized only as a fallback for
        raw `--model-params` values, so explicit user settings aren't overwritten by
        a saved preference.

    Args:
        model_spec: `provider:model` spec for the active model.
        model_params: Per-session model params.

    Returns:
        The configured effort, or `None` when no recognized effort override is set.
    """
    if not model_spec or not model_params:
        return None

    sentinel = _str_or_none(
        model_params.get("reasoning_effort"), key="reasoning_effort"
    )
    if sentinel is not None:
        return sentinel

    reasoning = model_params.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _str_or_none(reasoning.get("effort"), key="reasoning.effort")
        if effort is not None:
            return effort

    output_config = model_params.get("output_config")
    if isinstance(output_config, dict):
        effort = _str_or_none(output_config.get("effort"), key="output_config.effort")
        if effort is not None:
            return effort

    model_kwargs = model_params.get("model_kwargs")
    if isinstance(model_kwargs, dict):
        effort = _str_or_none(
            model_kwargs.get("reasoning_effort"), key="model_kwargs.reasoning_effort"
        )
        if effort is not None:
            return effort

    extra_body = model_params.get("extra_body")
    if isinstance(extra_body, dict):
        effort = _str_or_none(
            extra_body.get("reasoning_effort"), key="extra_body.reasoning_effort"
        )
        if effort is not None:
            return effort

    return _str_or_none(model_params.get("thinking_level"), key="thinking_level")


def merge_effort_model_params(
    existing: dict[str, Any] | None, effort_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge effort params into existing per-session model params.

    Args:
        existing: Current per-session model params.
        effort_params: Params to merge in (typically `{"reasoning_effort": ...}`).

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
