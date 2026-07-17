"""Middleware for runtime model selection via LangGraph runtime context.

Allows switching the model per invocation by passing a `CLIContext` via
`context=` on `agent.astream()` / `agent.invoke()` without recompiling
the graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from deepagents._models import (  # noqa: PLC2701
    get_model_identifier,
    model_matches_spec,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel

    from deepagents_code.config import ModelResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResolvedModelRequest:
    """Model request plus the checkpoint metadata it should persist."""

    request: ModelRequest
    """Request to pass to the downstream model handler."""

    model_spec: str | None
    """Resolved `provider:model` spec to persist for resume, when known."""

    model_params: dict[str, Any] | None = None
    """Invocation params to persist, or `None` to clear checkpointed params."""

    model_params_known: bool = False
    """Whether `model_params` is known and should be written to the checkpoint."""


def _get_ls_provider(model: object) -> str | None:
    """Return the LangSmith provider name reported by a chat model.

    Returns:
        The `ls_provider` string when the model reports one, otherwise `None`
            (including when `_get_ls_params` is missing, raises, or yields a
            non-string provider).
    """
    try:
        ls_params = model._get_ls_params()  # ty: ignore[unresolved-attribute]
    except (AttributeError, TypeError, RuntimeError, NotImplementedError):
        logger.debug("_get_ls_params raised for %s", type(model).__name__)
        return None
    if isinstance(ls_params, dict):
        provider = ls_params.get("ls_provider")
        if isinstance(provider, str):
            return provider
    return None


def _is_anthropic_model(model: object) -> bool:
    """Check whether a resolved model reports `'anthropic'` as its provider.

    Uses `_get_ls_params` from `BaseChatModel` to read the provider name.

    Args:
        model: A model instance to inspect.

            Typed as `object` (rather than `BaseChatModel`) so the caller can
            pass any model without an import-time dependency on a specific
            provider package.

    Returns:
        `True` if the model's `ls_provider` is `'anthropic'`.
    """
    return _get_ls_provider(model) == "anthropic"


def _is_fireworks_model(model: object) -> bool:
    """Check whether a resolved model reports `'fireworks'` as its provider.

    Returns:
        `True` if the model's `ls_provider` is `'fireworks'`.
    """
    return _get_ls_provider(model) == "fireworks"


def _is_openai_model(model: object) -> bool:
    """Check whether a resolved model targets OpenAI's official API.

    `ChatOpenAI` reports `'openai'` even when configured with a custom base URL,
    so provider metadata alone cannot establish support for OpenAI-specific
    request fields. Unknown endpoints are treated conservatively as
    incompatible.

    Returns:
        `True` if the model reports `'openai'` and uses `api.openai.com` or an
            official regional subdomain.
    """
    if _get_ls_provider(model) != "openai":
        return False

    # Prefer the SDK client's resolved base URL: a default `ChatOpenAI()` leaves
    # `openai_api_base` unset, but its `root_client.base_url` still resolves to
    # the official `api.openai.com` default. Fall back to the constructor field
    # only when the client is unavailable.
    client = getattr(model, "root_client", None)
    base_url = getattr(client, "base_url", None)
    if base_url is None:
        base_url = getattr(model, "openai_api_base", None)
    if base_url is None:
        # The provider is 'openai' yet no endpoint is discoverable. A genuine
        # `ChatOpenAI` always exposes one, so this points to an unexpected model
        # shape (e.g. an attribute renamed by an upstream upgrade). Skip the
        # optimization instead of guessing, and leave a trace so the silent
        # regression is diagnosable.
        logger.debug("OpenAI model exposes no base URL; skipping prompt_cache_key")
        return False

    try:
        hostname = urlsplit(str(base_url)).hostname
    except ValueError:
        logger.debug("OpenAI base URL is unparseable; skipping prompt_cache_key")
        return False
    return hostname == "api.openai.com" or (
        hostname is not None and hostname.endswith(".api.openai.com")
    )


_ANTHROPIC_ONLY_SETTINGS: set[str] = {"cache_control"}
"""Keys injected by Anthropic-specific middleware (e.g.
`AnthropicPromptCachingMiddleware`) that are not accepted by other providers and
must be stripped on cross-provider swap."""

_FIREWORKS_SESSION_AFFINITY_HEADER = "x-session-affinity"
"""Fireworks prompt-cache affinity header populated from the active thread ID."""


def _has_header(headers: Mapping[object, object], target: str) -> bool:
    """Return whether a headers mapping already includes `target`.

    Comparison is case-insensitive; `target` must be supplied in lowercase.

    Returns:
        `True` if a string key case-insensitively equal to `target` is present.
    """
    return any(isinstance(key, str) and key.lower() == target for key in headers)


def _with_fireworks_session_settings(
    model_settings: dict[str, Any], thread_id: str
) -> dict[str, Any] | None:
    """Return model settings with Fireworks session settings added if needed.

    Existing settings are preserved and never overwritten. Missing
    `x-session-affinity` headers are populated directly so Fireworks can route
    the conversation to the prompt-cache session for the active thread.

    Returns:
        A new `model_settings` dict with the missing session settings added, or
            `None` when nothing needed adding or `extra_headers` is present but
            not a mapping (leaving the request untouched).
    """
    raw_headers = model_settings.get("extra_headers")
    if raw_headers is None:
        headers: dict[object, object] = {}
    elif isinstance(raw_headers, Mapping):
        headers = dict(raw_headers)
    else:
        logger.warning(
            "Cannot inject Fireworks session settings because extra_headers is %s",
            type(raw_headers).__name__,
        )
        return None

    updated: dict[str, Any] = {}
    has_session_affinity = _has_header(headers, _FIREWORKS_SESSION_AFFINITY_HEADER)
    if "prompt_cache_key" not in model_settings and not has_session_affinity:
        updated["prompt_cache_key"] = thread_id

    if not has_session_affinity:
        headers[_FIREWORKS_SESSION_AFFINITY_HEADER] = thread_id
        updated["extra_headers"] = headers

    if not updated:
        return None
    return {**model_settings, **updated}


def _with_openai_prompt_cache_key(
    model: object, model_settings: dict[str, Any], thread_id: str
) -> dict[str, Any] | None:
    """Return model settings with an OpenAI `prompt_cache_key` added if needed.

    Setting `prompt_cache_key` lets OpenAI route a conversation to a stable
    prompt-cache prefix across turns, giving more reliable cache hits than the
    automatic (keyless) matching; it is optional, and requests still cache
    without it. It is a supported top-level `ChatOpenAI` invocation setting
    forwarded to the OpenAI request payload, so passing it through
    `model_settings` reaches the wire unchanged. `prompt_cache_key` is an
    optional, additive request field: it sharpens prefix-cache routing on model
    families that support it (GPT-5.6 and later) and is otherwise inert, so the
    same key is sent to every OpenAI model without a version gate. This mirrors
    the Fireworks path (`_with_fireworks_session_settings`), which sets
    `prompt_cache_key` the same way and additionally sends an
    `x-session-affinity` header.

    A user-supplied `prompt_cache_key` is always preserved, whether it was
    configured on the model (`model_kwargs`) or supplied for this invocation
    (`model_settings`).

    Returns:
        A new `model_settings` dict with `prompt_cache_key` added, or `None` when
            a key is already present on the model or in the settings (nothing to
            add).
    """
    model_kwargs = getattr(model, "model_kwargs", None)
    if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
        # A non-mapping `model_kwargs` cannot carry a user-supplied key, so it is
        # treated as "no key present" and injection proceeds. Trace the anomaly
        # since a real `ChatOpenAI` always exposes a mapping here.
        logger.debug(
            "Ignoring non-mapping model_kwargs (%s) when checking for a "
            "user-supplied prompt_cache_key",
            type(model_kwargs).__name__,
        )
    if "prompt_cache_key" in model_settings or (
        isinstance(model_kwargs, Mapping) and "prompt_cache_key" in model_kwargs
    ):
        return None
    return {**model_settings, "prompt_cache_key": thread_id}


def _get_context(request: ModelRequest) -> CLIContextSchema | None:
    """Return runtime context when it matches the CLI context shape."""
    runtime = request.runtime
    if runtime is None:
        return None

    ctx = runtime.context
    if isinstance(ctx, CLIContextSchema):
        return ctx
    if isinstance(ctx, dict):
        raw_key = ctx.get("approval_mode_key")
        raw_thread_id = ctx.get("thread_id")
        return CLIContextSchema(
            model=ctx.get("model"),
            model_params=ctx.get("model_params") or {},
            profile_overrides=ctx.get("profile_overrides") or {},
            model_context_limit=ctx.get("model_context_limit"),
            auto_approve=bool(ctx.get("auto_approve", False)),
            approval_mode_key=raw_key if isinstance(raw_key, str) else None,
            thread_id=raw_thread_id if isinstance(raw_thread_id, str) else None,
        )
    return None


def _model_spec_from_model(model: BaseChatModel) -> str | None:
    """Return a resumable `provider:model` spec for a model object."""
    provider = _get_ls_provider(model)
    model_name = get_model_identifier(model)
    if provider and model_name:
        return f"{provider}:{model_name}"

    from deepagents_code.config import settings

    settings_provider = settings.model_provider or ""
    settings_model = settings.model_name or ""
    if settings_provider and settings_model:
        return f"{settings_provider}:{settings_model}"
    return None


def _model_spec_from_result(
    model_result: ModelResult | None, model: BaseChatModel
) -> str | None:
    """Return the resolved spec from `create_model`, falling back to model metadata."""
    if model_result is not None and model_result.provider and model_result.model_name:
        return f"{model_result.provider}:{model_result.model_name}"
    return _model_spec_from_model(model)


def _build_overrides(
    request: ModelRequest, ctx: CLIContextSchema, model_result: ModelResult | None
) -> ModelRequest:
    """Build the overridden request from a (possibly resolved) model result.

    Holds the post-construction logic shared by the sync and async override
    paths: applying the model swap, merging `model_params`, stripping
    Anthropic-only settings on a cross-provider swap, and patching the
    `### Model Identity` system-prompt section. The only thing that differs
    between the two callers is how `model_result` is produced (a direct
    `create_model` call vs. an `asyncio.to_thread` offload).

    Args:
        request: The incoming model request from the middleware chain.
        ctx: Runtime CLI context carrying the requested overrides.
        model_result: The resolved model result from `create_model`, or `None`
            when no model swap was requested.

    Returns:
        The original request when no overrides apply, otherwise a new request
            with overrides applied via `request.override()`.
    """
    overrides: dict[str, Any] = {}

    new_model = model_result.model if model_result is not None else None
    if new_model is not None:
        overrides["model"] = new_model

    # Param merge
    model_params = ctx.model_params
    if model_params:
        from deepagents_code.config import CLI_MAX_RETRIES_KEY

        provider_settings = {
            key: value
            for key, value in model_params.items()
            if key != CLI_MAX_RETRIES_KEY
        }
        if provider_settings:
            overrides["model_settings"] = {
                **request.model_settings,
                **provider_settings,
            }

    # Inject the provider's prompt-cache routing hint from the active thread.
    # Only one provider path applies per call; both share the fetch/guard/log
    # tail below. `overrides.get` is side-effect-free, so resolving `settings`
    # before the provider check is equivalent to doing it inside each branch.
    effective_model = new_model if new_model is not None else request.model
    if ctx.thread_id:
        settings = overrides.get("model_settings", request.model_settings)
        if _is_fireworks_model(effective_model):
            updated_settings = _with_fireworks_session_settings(settings, ctx.thread_id)
            injected = "Fireworks session settings"
        elif _is_openai_model(effective_model):
            updated_settings = _with_openai_prompt_cache_key(
                effective_model, settings, ctx.thread_id
            )
            injected = "OpenAI prompt_cache_key"
        else:
            updated_settings = None
            injected = ""
        if updated_settings is not None:
            overrides["model_settings"] = updated_settings
            # The thread ID is a sensitive session identifier, so it is kept out
            # of the log line; the line firing at all confirms injection ran.
            logger.debug("Injected %s", injected)

    if not overrides:
        return request

    # When switching away from Anthropic, strip provider-specific settings
    # that would cause errors on other providers (e.g. cache_control passed
    # to the OpenAI SDK raises TypeError).
    if new_model is not None and not _is_anthropic_model(new_model):
        settings = overrides.get("model_settings", request.model_settings)
        dropped = settings.keys() & _ANTHROPIC_ONLY_SETTINGS
        if dropped:
            logger.debug(
                "Stripped Anthropic-only settings %s for non-Anthropic model",
                dropped,
            )
            overrides["model_settings"] = {
                k: v for k, v in settings.items() if k not in dropped
            }

    # Patch the Model Identity section in the system prompt so the new model
    # sees its own name/provider/context-limit, not the original's.
    # We read metadata from model_result (not the app's settings singleton)
    # because the middleware runs in the server subprocess where settings
    # are never updated by /model.
    if model_result is not None and request.system_prompt:
        from deepagents_code.agent import (
            MODEL_IDENTITY_RE,
            build_model_identity_section,
        )

        prompt = request.system_prompt
        new_identity = build_model_identity_section(
            model_result.model_name,
            provider=model_result.provider,
            context_limit=model_result.context_limit,
            unsupported_modalities=model_result.unsupported_modalities,
        )
        patched = MODEL_IDENTITY_RE.sub(new_identity, prompt, count=1)
        if patched != prompt:
            overrides["system_prompt"] = patched
        elif "### Model Identity" in prompt:
            logger.warning(
                "System prompt contains '### Model Identity' but regex "
                "did not match; identity section was NOT updated for "
                "model '%s'. The regex may be out of sync with the "
                "prompt template.",
                model_result.model_name,
            )

    return request.override(**overrides)


def _model_creation_kwargs(
    ctx: CLIContextSchema,
    current_model: BaseChatModel,
) -> dict[str, dict[str, Any]]:
    """Build constructor kwargs needed for a runtime model switch.

    Invocation settings remain on `ModelRequest.model_settings`; only the
    internal CLI retry override must also reach `create_model` so the switched
    model carries the same authoritative retry budget as the startup model.

    Args:
        ctx: Runtime context carrying model and profile overrides.
        current_model: Active model whose current-process CLI override wins over
            restored checkpoint metadata.

    Returns:
        Keyword arguments for `create_model`.
    """
    from deepagents_code.config import (
        CLI_MAX_RETRIES_KEY,
        get_model_retry_override,
    )

    kwargs: dict[str, dict[str, Any]] = {}
    retry_override = get_model_retry_override(current_model)
    if retry_override is None:
        raw_override = ctx.model_params.get(CLI_MAX_RETRIES_KEY)
        if (
            isinstance(raw_override, int)
            and not isinstance(raw_override, bool)
            and raw_override >= 0
        ):
            retry_override = raw_override
    if retry_override is not None:
        kwargs["extra_kwargs"] = {CLI_MAX_RETRIES_KEY: retry_override}
    if ctx.profile_overrides:
        kwargs["profile_overrides"] = ctx.profile_overrides
    return kwargs


def _persisted_model_params(
    ctx: CLIContextSchema,
    model: BaseChatModel,
    *,
    include_context_params: bool = True,
) -> dict[str, Any] | None:
    """Return checkpoint params with the model's effective CLI override.

    Args:
        ctx: Runtime context carrying user and checkpoint params.
        model: Model that handled the completed request.
        include_context_params: Whether params targeting the requested model
            completed successfully and may be persisted.

    Returns:
        Params to persist, or `None` when no params or override are active.
    """
    from deepagents_code.config import (
        CLI_MAX_RETRIES_KEY,
        get_model_retry_override,
    )

    params = dict(ctx.model_params) if include_context_params else {}
    retry_override = get_model_retry_override(model)
    if retry_override is not None:
        params[CLI_MAX_RETRIES_KEY] = retry_override
    return params or None


def _apply_overrides(request: ModelRequest) -> _ResolvedModelRequest:
    """Apply model/param overrides and return checkpoint persistence metadata.

    Reads `'model'` and `'model_params'` from `runtime.context` and, when
    present, swaps the model and/or merges extra settings into the request.
    On a cross-provider swap away from Anthropic, Anthropic-only settings
    (e.g. `cache_control`) are stripped. The `### Model Identity` section
    in the system prompt is also patched to reflect the new model.

    Args:
        request: The incoming model request from the middleware chain.

    Returns:
        The request to send downstream plus the actual model spec and user-supplied
            model params that should be recorded for resume.
    """
    ctx = _get_context(request)
    if ctx is None:
        return _ResolvedModelRequest(request, _model_spec_from_model(request.model))

    model_result = None
    model = ctx.model
    if model and not model_matches_spec(request.model, model):
        from deepagents_code.config import create_model
        from deepagents_code.model_config import ModelConfigError

        logger.debug("Overriding model to %s", model)
        model_kwargs = _model_creation_kwargs(ctx, request.model)
        try:
            model_result = create_model(model, **model_kwargs)
        except ModelConfigError:
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            return _ResolvedModelRequest(
                request,
                _model_spec_from_model(request.model),
                _persisted_model_params(
                    ctx,
                    request.model,
                    include_context_params=False,
                ),
                model_params_known=True,
            )

    updated = _build_overrides(request, ctx, model_result)
    params = _persisted_model_params(ctx, updated.model)
    return _ResolvedModelRequest(
        updated,
        _model_spec_from_result(model_result, updated.model),
        params,
        model_params_known=True,
    )


async def _apply_overrides_async(request: ModelRequest) -> _ResolvedModelRequest:
    """Async variant of `_apply_overrides` that offloads model construction.

    Returns:
        The request to send downstream plus the actual model spec and user-supplied
            model params that should be recorded for resume.
    """
    ctx = _get_context(request)
    if ctx is None:
        return _ResolvedModelRequest(request, _model_spec_from_model(request.model))

    model_result = None
    model = ctx.model
    if model and not model_matches_spec(request.model, model):
        from deepagents_code.config import create_model
        from deepagents_code.model_config import ModelConfigError

        logger.debug("Overriding model to %s", model)
        model_kwargs = _model_creation_kwargs(ctx, request.model)
        try:
            model_result = await asyncio.to_thread(
                create_model,
                model,
                **model_kwargs,
            )
        except ModelConfigError:
            logger.exception(
                "Failed to resolve runtime model override '%s'; "
                "continuing with current model",
                model,
            )
            return _ResolvedModelRequest(
                request,
                _model_spec_from_model(request.model),
                _persisted_model_params(
                    ctx,
                    request.model,
                    include_context_params=False,
                ),
                model_params_known=True,
            )

    updated = _build_overrides(request, ctx, model_result)
    params = _persisted_model_params(ctx, updated.model)
    return _ResolvedModelRequest(
        updated,
        _model_spec_from_result(model_result, updated.model),
        params,
        model_params_known=True,
    )


def _checkpoint_command(resolved: _ResolvedModelRequest) -> Command[Any] | None:
    """Build the private resume-state update for a completed model call.

    Returns:
        Command with private checkpoint updates, or `None` when nothing is known.
    """
    update: dict[str, Any] = {}
    if resolved.model_spec:
        update["_model_spec"] = resolved.model_spec
    if resolved.model_params_known:
        update["_model_params"] = resolved.model_params
    if not update:
        return None
    return Command(update=update)


class ConfigurableModelMiddleware(AgentMiddleware):
    """Swap the model or per-call settings from `runtime.context`.

    Reads two optional keys from the runtime context dict:

    - `'model'` — a `provider:model` spec (e.g. `"openai:gpt-5"`).
        When present and different from the current model, the request is
        re-routed to the new model.
    - `'model_params'` — a dict of extra model settings (e.g.
        `{"temperature": 0}`) that are shallow-merged into the
        request's `model_settings`.

    This middleware is typically the outermost layer so it intercepts every
    model call before provider-specific middleware (like
    `AnthropicPromptCachingMiddleware`) runs.
    """

    def __init__(self, *, persist_model_state: bool = True) -> None:
        """Initialize the middleware.

        Args:
            persist_model_state: Whether completed calls should write private
                resume metadata. Subagent instances disable this because they do
                not own the parent thread's resume state.
        """
        self._persist_model_state = persist_model_state

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply runtime overrides and delegate to the next handler.

        Returns:
            The downstream response plus a private resume-state update when the
            completed call has model metadata to checkpoint.
        """
        resolved = _apply_overrides(request)
        response = handler(resolved.request)
        command = _checkpoint_command(resolved) if self._persist_model_state else None
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply runtime overrides and delegate to the next async handler.

        Returns:
            The downstream response plus a private resume-state update when the
            completed call has model metadata to checkpoint.
        """
        resolved = await _apply_overrides_async(request)
        response = await handler(resolved.request)
        command = _checkpoint_command(resolved) if self._persist_model_state else None
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)
