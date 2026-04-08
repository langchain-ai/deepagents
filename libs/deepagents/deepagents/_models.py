"""Shared helpers for resolving and inspecting chat models."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from packaging.version import Version

OPENROUTER_MIN_VERSION = "0.2.0"
"""Minimum required version of `langchain-openrouter`.

Used by both the SDK (`resolve_model`) and the CLI (`config.py`) to enforce
a consistent version floor at runtime.
"""

_OPENROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""

_BASETEN_TIMEOUT_S = 90
"""Default timeout for Baseten chat models to avoid indefinite network stalls."""


def _baseten_auth_kwargs() -> dict[str, Any]:
    """Build Baseten auth kwargs, deferring to env var overrides."""
    api_key = os.environ.get("BASETEN_API_KEY")
    kwargs: dict[str, Any] = {"timeout": _BASETEN_TIMEOUT_S}
    if api_key:
        kwargs["baseten_api_key"] = api_key
    return kwargs


def _openrouter_attribution_kwargs() -> dict[str, Any]:
    """Build OpenRouter attribution kwargs, deferring to env var overrides.

    `ChatOpenRouter` reads `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE` via
    `from_env()` defaults. Explicit kwargs passed to the constructor take
    precedence over those env-var defaults, so we only inject our SDK defaults
    when the corresponding env var is **not** set — otherwise the user's env var
    would be overridden.

    Returns:
        Dictionary of attribution kwargs to spread into `init_chat_model`.
    """
    kwargs: dict[str, Any] = {}
    if not os.environ.get("OPENROUTER_APP_URL"):
        kwargs["app_url"] = _OPENROUTER_APP_URL
    if not os.environ.get("OPENROUTER_APP_TITLE"):
        kwargs["app_title"] = _OPENROUTER_APP_TITLE
    return kwargs


def check_openrouter_version() -> None:
    """Raise if the installed `langchain-openrouter` is below the minimum.

    If the package is not installed at all the check is skipped;
    `init_chat_model` will surface its own missing-dependency error downstream.

    Raises:
        ImportError: If the installed version is too old.
    """
    try:
        installed = pkg_version("langchain-openrouter")
    except PackageNotFoundError:
        return
    if Version(installed) < Version(OPENROUTER_MIN_VERSION):
        msg = (
            f"deepagents requires langchain-openrouter>={OPENROUTER_MIN_VERSION}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-openrouter>={OPENROUTER_MIN_VERSION}'"
        )
        raise ImportError(msg)


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model string to a `BaseChatModel`.

    If `model` is already a `BaseChatModel`, returns it unchanged.

    String models are resolved via `init_chat_model`. OpenAI models
    (prefixed with `openai:`) default to the Responses API.

    OpenRouter models include default app attribution headers unless overridden
    via `OPENROUTER_APP_URL` / `OPENROUTER_APP_TITLE` env vars.

    Args:
        model: Model string or pre-configured model instance.

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if isinstance(model, BaseChatModel):
        return model
    if model.startswith("openai:"):
        return init_chat_model(model, use_responses_api=True)
    if model.startswith("baseten:"):
        return init_chat_model(model, **_baseten_auth_kwargs())
    if model.startswith("openrouter:"):
        check_openrouter_version()
        return init_chat_model(model, **_openrouter_attribution_kwargs())
    return init_chat_model(model)


def get_model_identifier(model: BaseChatModel) -> str | None:
    """Extract the provider-native model identifier from a chat model.

    Providers do not agree on a single field name for the identifier. Some use
    `model_name`, while others use `model`. Reading the serialized model config
    lets us inspect both without relying on reflective attribute access.

    Args:
        model: Chat model instance to inspect.

    Returns:
        The configured model identifier, or `None` if it is unavailable.
    """
    config = model.model_dump()
    return _string_value(config, "model_name") or _string_value(config, "model")


def model_matches_spec(model: BaseChatModel, spec: str) -> bool:
    """Check whether a model instance already matches a string model spec.

    Matching is performed in two ways: first by exact string equality between
    `spec` and the model identifier, then by comparing only the model-name
    portion of a `provider:model` spec against the identifier. For example,
    `"openai:gpt-5"` matches a model with identifier `"gpt-5"`.

    Assumes the `provider:model` convention (single colon separator).

    Args:
        model: Chat model instance to inspect.
        spec: Model spec in `provider:model` format (e.g., `openai:gpt-5`).

    Returns:
        `True` if the model already matches the spec, otherwise `False`.
    """
    current = get_model_identifier(model)
    if current is None:
        return False
    if spec == current:
        return True

    _, separator, model_name = spec.partition(":")
    return bool(separator) and model_name == current


def _string_value(config: dict[str, Any], key: str) -> str | None:
    """Return a non-empty string value from a serialized model config."""
    value = config.get(key)
    if isinstance(value, str) and value:
        return value
    return None
