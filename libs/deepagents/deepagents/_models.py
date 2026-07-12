"""Shared helpers for resolving and inspecting chat models."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from deepagents.profiles.provider.provider_profiles import apply_provider_profile

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES = {
    "azure_openai": "azure",
    "mistralai": "mistral",
}
"""Known provider aliases between LangChain specs and LangSmith params.

LangChain specs and LangSmith params use different provider names for some
integrations. Canonicalize only known aliases before comparing providers.
"""

_BEDROCK_PROVIDERS = frozenset({"amazon_bedrock", "anthropic_bedrock", "aws", "bedrock", "bedrock_converse"})
"""Normalized provider names that identify AWS Bedrock chat models."""

_BEDROCK_MODEL_CLASSES = frozenset({"ChatAnthropicBedrock", "ChatBedrock", "ChatBedrockConverse", "ChatBedrockNovaSonic"})
"""`langchain-aws` chat model class names that identify AWS Bedrock models."""

_BEDROCK_REGIONAL_PREFIXES = ("apac.", "amer.", "au.", "eu.", "global.", "jp.", "sa.", "us.", "us-gov.")
"""Regional inference profile prefixes stripped from Bedrock model identifiers."""


def resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    """Resolve a model string to a `BaseChatModel`.

    If `model` is already a `BaseChatModel`, returns it unchanged.

    String models are resolved via `init_chat_model`, composed with any
    provider-specific initialization behavior registered in the
    `ProviderProfile` registry. Built-in registrations supply NVIDIA NIM and
    OpenRouter app attribution headers plus the OpenAI Responses API default;
    users can layer additional providers or overrides via
    `register_provider_profile`.

    Args:
        model: Model string (e.g. `"openai:gpt-5.4"`) or pre-configured
            `BaseChatModel` subclass instance.

    Returns:
        Resolved `BaseChatModel` instance.
    """
    if isinstance(model, BaseChatModel):
        return model

    resolved = init_chat_model(model, **apply_provider_profile(model))
    # OpenAI-compatible gateway clients reached via a custom base_url log LLM
    # spans with null ls_provider/ls_model_name, so cost never resolves. Tag the
    # resolved model's span metadata with the inspected provider (defaulting to
    # "openai" for the gateway path) and effective model id so LangSmith can look
    # up pricing.
    provider = get_model_provider(resolved) or "openai"
    model_id = get_model_identifier(resolved)
    metadata = {key: value for key, value in {"ls_provider": provider, "ls_model_name": model_id}.items() if value}
    if metadata:
        resolved = resolved.with_config(metadata=metadata)
    return resolved


def get_model_identifier(model: BaseChatModel) -> str | None:
    """Extract the provider-native model identifier from a chat model.

    Providers do not agree on a single field name for the identifier. Some use
    `model_name`, while others use `model`.

    Args:
        model: Chat model instance to inspect.

    Returns:
        The configured model identifier, or `None` if it is unavailable.
    """
    return _string_attr(model, "model_name") or _string_attr(model, "model")


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
    except (AttributeError, TypeError, NotImplementedError) as exc:
        # INFO rather than DEBUG: a missing or raising `_get_ls_params` causes
        # profile resolution to silently miss for that model. Custom
        # integrations need this to be visible at default log levels so users
        # can debug "my profile isn't applying" without enabling DEBUG.
        logger.info(
            "Could not extract provider from %s.%s via _get_ls_params: %s",
            type(model).__module__,
            type(model).__name__,
            exc,
        )
        return None
    if not isinstance(ls_params, Mapping):
        # A custom integration may return `None` (or another non-mapping)
        # instead of raising. Treat that as "provider unavailable" rather than
        # letting the subsequent `.get` raise `AttributeError`. Logged at INFO
        # for the same reason as the `except` branch above: the user-visible
        # outcome is identical (provider silently unavailable), so this path
        # must be just as discoverable at default log levels.
        logger.info(
            "Could not extract provider from %s.%s: _get_ls_params returned %s, not a mapping",
            type(model).__module__,
            type(model).__name__,
            type(ls_params).__name__,
        )
        return None
    provider = ls_params.get("ls_provider")
    if isinstance(provider, str) and provider:
        return provider
    return None


def is_bedrock_model(model: str | BaseChatModel) -> bool:
    """Check whether a model targets AWS Bedrock."""
    if isinstance(model, str):
        if _is_bedrock_nova_model_id(model):
            return True
        provider, separator, _ = model.partition(":")
        return bool(separator) and _normalize_provider(provider) in _BEDROCK_PROVIDERS

    provider = get_model_provider(model)
    if provider is not None and _normalize_provider(provider) in _BEDROCK_PROVIDERS:
        return True
    return type(model).__name__ in _BEDROCK_MODEL_CLASSES


def _is_bedrock_nova_model_id(model: str) -> bool:
    """Check for cache-capable Bedrock Nova model identifiers."""
    identifier = model
    for prefix in _BEDROCK_REGIONAL_PREFIXES:
        if identifier.startswith(prefix):
            identifier = identifier.removeprefix(prefix)
            break
    return identifier.startswith("amazon.nova-")


def model_matches_spec(model: BaseChatModel, spec: str) -> bool:
    """Check whether a model instance already matches a string model spec.

    Bare specs match by model identifier. Provider-prefixed specs match by both
    model identifier and provider when the current model exposes a provider via
    `_get_ls_params`; if the provider cannot be inspected, the check falls back
    to identifier-only matching for backwards compatibility with custom models.
    Provider comparison is normalized, so case, hyphen/underscore spelling, and
    known aliases do not read as a mismatch (see `_normalize_provider`).

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

    provider, separator, model_name = spec.partition(":")
    if not separator or model_name != current:
        return False

    current_provider = get_model_provider(model)
    if current_provider is None:
        # Provider could not be inspected, so the spec's provider cannot be
        # confirmed. Fall back to the identifier-only match. Logged at DEBUG so
        # that a consumer skipping a model swap on the strength of this match
        # (e.g. the runtime model override) is traceable when it surprises.
        logger.debug(
            "Matched spec %r on identifier alone; provider for %s.%s is uninspectable, so the spec's %r provider was not verified",
            spec,
            type(model).__module__,
            type(model).__name__,
            provider,
        )
        return True
    return _normalize_provider(provider) == _normalize_provider(current_provider)


def _normalize_provider(provider: str) -> str:
    """Canonicalize a provider name so equal providers compare equal.

    Specs use the `provider:model` spelling (lowercase, underscore-separated,
    e.g. `azure_openai`), while the `ls_provider` reported by `_get_ls_params`
    may differ in case, use hyphens (`openai-codex`), or use an entirely
    different name (`mistralai` vs `mistral`). Folding both sides through this
    function before comparison keeps those spellings from reading as a
    mismatch.
    """
    normalized = provider.lower().replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _string_attr(obj: object, attr: str) -> str | None:
    """Return a non-empty string attribute from `obj`, or `None`."""
    value = getattr(obj, attr, None)
    if isinstance(value, str) and value:
        return value
    return None
