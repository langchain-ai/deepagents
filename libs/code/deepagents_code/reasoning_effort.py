"""Reasoning effort support for `/effort`.

Supported levels and defaults come from LangChain model profiles. Provider
integrations translate the standard `reasoning_effort` constructor parameter
into their native request shapes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from deepagents_code.model_config import CODEX_PROVIDER, ModelSpec, get_model_profiles

logger = logging.getLogger(__name__)

_LEGACY_ANTHROPIC_THINKING = {"type": "adaptive", "display": "summarized"}


def _model_profile(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> Mapping[str, Any] | None:
    """Return the reasoning-capable profile for `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        The merged model profile when `reasoning_output` is `True`, otherwise
        `None`.
    """
    if not model_spec:
        return None
    entry = get_model_profiles(cli_override=cli_override).get(model_spec)
    profile = cli_override if entry is None else entry.get("profile")
    if profile is None:
        return None
    if not isinstance(profile, Mapping):
        logger.warning(
            "Ignoring model profile for %s with unexpected type %s",
            model_spec,
            type(profile).__name__,
        )
        return None
    reasoning_output = profile.get("reasoning_output")
    if reasoning_output is not None and not isinstance(reasoning_output, bool):
        logger.warning(
            "Ignoring reasoning_output for %s with unexpected type %s",
            model_spec,
            type(reasoning_output).__name__,
        )
        return None
    if reasoning_output is not True:
        return None
    return profile


def supported_efforts_for_model(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> tuple[str, ...]:
    """Return the ordered reasoning effort levels supported by `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        Supported effort labels, or an empty tuple when effort is not
        configurable or the profile is malformed.
    """
    profile = _model_profile(model_spec, cli_override=cli_override)
    if profile is None or "reasoning_effort_levels" not in profile:
        return ()
    levels = profile["reasoning_effort_levels"]
    if not isinstance(levels, list):
        logger.warning(
            "Ignoring reasoning_effort_levels for %s with unexpected type %s",
            model_spec,
            type(levels).__name__,
        )
        return ()
    for level in levels:
        if not isinstance(level, str):
            logger.warning(
                "Ignoring reasoning_effort_levels for %s containing type %s",
                model_spec,
                type(level).__name__,
            )
            return ()
    return tuple(levels)


def default_effort_for_model(
    model_spec: str | None, *, cli_override: dict[str, Any] | None = None
) -> str | None:
    """Return the profile's reasoning effort default independently of its levels.

    Args:
        model_spec: `provider:model` spec for the active model.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        The default effort label, or `None` when absent or malformed.
    """
    profile = _model_profile(model_spec, cli_override=cli_override)
    if profile is None or "reasoning_effort_default" not in profile:
        return None
    default = profile["reasoning_effort_default"]
    if not isinstance(default, str):
        logger.warning(
            "Ignoring reasoning_effort_default for %s with unexpected type %s",
            model_spec,
            type(default).__name__,
        )
        return None
    return default


def is_effort_supported_for_model(
    model_spec: str, effort: str, *, cli_override: dict[str, Any] | None = None
) -> bool:
    """Return whether `effort` is a supported level for `model_spec`.

    Args:
        model_spec: `provider:model` spec for the active model.
        effort: Effort label to check.
        cli_override: Extra profile fields from `--profile-override`, if any.

    Returns:
        `True` when the active profile advertises `effort`.
    """
    return effort in supported_efforts_for_model(model_spec, cli_override=cli_override)


def _str_or_none(value: object, *, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    logger.warning("Ignoring non-str %s of type %s", key, type(value).__name__)
    return None


def _effort_value(model_params: Mapping[str, Any], key: str) -> tuple[bool, str | None]:
    if key not in model_params or model_params[key] is None:
        return False, None
    return True, _str_or_none(model_params[key], key=key)


def _nested_effort_value(
    model_params: Mapping[str, Any], container: str, key: str
) -> tuple[bool, str | None]:
    nested = model_params.get(container)
    if not isinstance(nested, Mapping) or key not in nested or nested[key] is None:
        return False, None
    return True, _str_or_none(nested[key], key=f"{container}.{key}")


def _first_effort_value(
    model_params: Mapping[str, Any], *paths: tuple[str, ...]
) -> str | None:
    for path in paths:
        result = (
            _effort_value(model_params, path[0])
            if len(path) == 1
            else _nested_effort_value(model_params, path[0], path[1])
        )
        present, value = result
        if present:
            return value
    return None


def _effort_paths(provider: str) -> tuple[tuple[str, ...], ...]:
    if provider in {"openai", CODEX_PROVIDER}:
        return (("reasoning", "effort"), ("reasoning_effort",))
    if provider == "anthropic":
        return (
            ("effort",),
            ("reasoning_effort",),
            ("output_config", "effort"),
        )
    if provider == "google_genai":
        return (
            ("thinking_level",),
            ("reasoning_effort",),
            ("thinking_config", "thinking_level"),
        )
    if provider == "fireworks":
        return (("reasoning_effort",), ("model_kwargs", "reasoning_effort"))
    if provider == "xai":
        return (("reasoning_effort",), ("extra_body", "reasoning_effort"))
    return (("reasoning_effort",),)


def _path_is_present(model_params: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    if len(path) == 1:
        return path[0] in model_params
    nested = model_params.get(path[0])
    return isinstance(nested, Mapping) and path[1] in nested


def has_explicit_effort_model_params(
    model_spec: str | None, model_params: dict[str, Any] | None
) -> bool:
    """Return whether canonical or native effort parameters are present.

    Args:
        model_spec: `provider:model` spec for the active model.
        model_params: Per-session model constructor parameters.

    Returns:
        `True` when an explicit effort setting should block persisted restoration.
    """
    if not model_spec or not model_params:
        return False
    parsed = ModelSpec.try_parse(model_spec)
    provider = parsed.provider if parsed is not None else ""
    return any(_path_is_present(model_params, path) for path in _effort_paths(provider))


def current_effort_from_model_params(
    model_spec: str | None, model_params: dict[str, Any] | None
) -> str | None:
    """Read canonical or native effort settings using integration precedence.

    This compatibility reader does not modify the supplied parameters. It only
    reports settings that may come from `--model-params`, `/model`, or a resumed
    thread.

    Args:
        model_spec: `provider:model` spec for the active model.
        model_params: Per-session model constructor parameters.

    Returns:
        The effective configured effort, or `None` when none is recognized.
    """
    if not model_spec or not model_params:
        return None
    parsed = ModelSpec.try_parse(model_spec)
    provider = parsed.provider if parsed is not None else ""

    paths = _effort_paths(provider)
    if provider in {"openai", CODEX_PROVIDER}:
        reasoning = model_params.get("reasoning")
        if isinstance(reasoning, Mapping) and "effort" in reasoning:
            return _str_or_none(reasoning["effort"], key="reasoning.effort")
    elif provider == "anthropic" and "effort" in model_params:
        effort = model_params["effort"]
        if effort is not None:
            return _str_or_none(effort, key="effort")
        return _first_effort_value(model_params, ("output_config", "effort"))
    elif provider == "google_genai" and "thinking_level" in model_params:
        effort = model_params["thinking_level"]
        if effort is not None:
            return _str_or_none(effort, key="thinking_level")
        return _first_effort_value(model_params, ("thinking_config", "thinking_level"))
    elif provider == "fireworks" and all(
        _path_is_present(model_params, path) for path in paths
    ):
        logger.warning("Ignoring conflicting Fireworks reasoning effort parameters")
        return None
    return _first_effort_value(model_params, *paths)


def _remove_nested_key(params: dict[str, Any], container: str, key: str) -> None:
    nested = params.get(container)
    if not isinstance(nested, Mapping):
        return
    remaining = dict(nested)
    remaining.pop(key, None)
    if remaining:
        params[container] = remaining
    else:
        params.pop(container, None)


def without_effort_model_params(
    model_spec: str, existing: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Remove canonical and native effort settings without changing siblings.

    Args:
        model_spec: `provider:model` spec for the active model.
        existing: Current per-session model constructor parameters.

    Returns:
        Cleaned parameters, or `None` when no parameters remain.
    """
    if not existing:
        return None
    cleaned = dict(existing)
    cleaned.pop("reasoning_effort", None)

    parsed = ModelSpec.try_parse(model_spec)
    provider = parsed.provider if parsed is not None else ""
    if provider in {"openai", CODEX_PROVIDER}:
        _remove_nested_key(cleaned, "reasoning", "effort")
    elif provider == "anthropic":
        cleaned.pop("effort", None)
        _remove_nested_key(cleaned, "output_config", "effort")
        if cleaned.get("thinking") == _LEGACY_ANTHROPIC_THINKING:
            cleaned.pop("thinking")
    elif provider == "google_genai":
        cleaned.pop("thinking_level", None)
        _remove_nested_key(cleaned, "thinking_config", "thinking_level")
    elif provider == "fireworks":
        _remove_nested_key(cleaned, "model_kwargs", "reasoning_effort")
    elif provider == "xai":
        _remove_nested_key(cleaned, "extra_body", "reasoning_effort")
    return cleaned or None


def with_effort_model_params(
    model_spec: str, existing: dict[str, Any] | None, effort: str
) -> dict[str, Any]:
    """Replace existing effort settings with the standard flat parameter.

    Args:
        model_spec: `provider:model` spec for the active model.
        existing: Current per-session model constructor parameters.
        effort: Profile-advertised effort label to apply.

    Returns:
        New model parameters containing `reasoning_effort` and all unrelated
        existing settings.
    """
    updated = without_effort_model_params(model_spec, existing) or {}
    updated["reasoning_effort"] = effort
    return updated
