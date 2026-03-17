"""Shared helpers for resolving and inspecting chat models."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel


def resolve_model(model: str | BaseChatModel | _ConfigurableModel) -> BaseChatModel | _ConfigurableModel:
    """Resolve a model string to a `BaseChatModel`.

    If `model` is already a `BaseChatModel` or `_ConfigurableModel`, returns it unchanged.

    String models are resolved via `init_chat_model`. OpenAI models
    (prefixed with `openai:`) default to the Responses API.

    Args:
        model: Model string or pre-configured model instance.

    Returns:
        Resolved `BaseChatModel` or `_ConfigurableModel` instance.
    """
    if isinstance(model, (BaseChatModel, _ConfigurableModel)):
        return model
    if model.startswith("openai:"):
        return init_chat_model(model, use_responses_api=True)
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
