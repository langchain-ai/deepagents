"""Shared helpers for resolving and inspecting chat models."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from deepagents.profiles import _get_provider_profile


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model string to a `BaseChatModel`.

    If `model` is already a `BaseChatModel`, returns it unchanged.

    String models are resolved via `init_chat_model`, composed with any
    provider-specific initialization behavior registered in the
    `_ProviderProfile` registry. Built-in registrations supply
    the OpenAI Responses API default and OpenRouter app attribution headers;
    additional providers can register their own profiles without modifying
    this function.

    Args:
        model: Model string (e.g. `"openai:gpt-5.4"`) or pre-configured
            `BaseChatModel` subclass instance.

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if isinstance(model, BaseChatModel):
        return model

    profile = _get_provider_profile(model)

    # Execute any pre-initialization logic
    if profile.pre_init is not None:
        profile.pre_init(model)

    # Combine static and factory kwargs, with factory taking precedence
    kwargs: dict[str, Any] = {**profile.init_kwargs}
    if profile.init_kwargs_factory is not None:
        kwargs.update(profile.init_kwargs_factory())

    return init_chat_model(model, **kwargs)  # kwargs may be empty


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


def get_model_provider(model: BaseChatModel) -> str | None:
    """Extract the provider name from a chat model instance.

    Uses the model's `_get_ls_params` method. The base `BaseChatModel`
    implementation derives `ls_provider` from the class name, and all major
    providers override it with a hardcoded value (e.g. `"anthropic"`).

    Args:
        model: Chat model instance to inspect.

    Returns:
        The provider name, or `None` if unavailable.
    """
    try:
        ls_params = model._get_ls_params()
    except (AttributeError, TypeError, NotImplementedError):
        return None
    provider = ls_params.get("ls_provider")
    if isinstance(provider, str) and provider:
        return provider
    return None


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
