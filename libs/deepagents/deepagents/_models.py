"""Shared helpers for resolving and inspecting chat models."""
from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model string to a `BaseChatModel`.

    If `model` is already a `BaseChatModel`, returns it unchanged.
    String models are resolved via `init_chat_model`. OpenAI models
    (prefixed with `openai:`) default to the Responses API.

    Args:
        model: Model string or pre-configured model instance.

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if isinstance(model, BaseChatModel):
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


def get_model_provider(model: BaseChatModel) -> str | None:
    """Extract the provider name from a chat model instance.

    Reads the serialized model config for a ``model_provider`` field, which
    LangChain populates when a model is initialised via ``init_chat_model``
    with a ``provider:model`` spec. Falls back to inspecting the class name
    as a best-effort when the field is absent (e.g. for directly instantiated
    model objects).

    Examples::

        m = init_chat_model("openai:gpt-5")
        get_model_provider(m)   # "openai"

        m = init_chat_model("anthropic:claude-opus-4-6")
        get_model_provider(m)   # "anthropic"

    Args:
        model: Chat model instance to inspect.

    Returns:
        The provider name as a lowercase string, or ``None`` if it cannot be
        determined.
    """
    config = model.model_dump()

    # Prefer the explicit field set by init_chat_model.
    provider = _string_value(config, "model_provider")
    if provider:
        return provider.lower()

    # Fall back to the class name, which conventionally starts with the
    # provider name (e.g. ChatOpenAI → "openai", ChatAnthropic → "anthropic").
    class_name = type(model).__name__
    if class_name.startswith("Chat"):
        return class_name[len("Chat"):].lower() or None

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
