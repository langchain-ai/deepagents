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
    """Check whether a resolved model targets OpenAI's chat/responses API.

    `prompt_cache_key` is an optional, additive OpenAI request field, so it is
    attempted for every model whose LangSmith provider is `'openai'` regardless
    of base URL. `ChatOpenAI` reports `'openai'` for the official API, the
    LangSmith gateway, and other OpenAI-compatible endpoints alike; treating all
    of them as eligible is intentional so the cache-key optimization is not
    silently dropped behind a proxy. Endpoints that reject unknown request
    fields can opt out via the `models.openai_prompt_cache_key` config option.

    Returns:
        `True` if the model reports `'openai'` as its provider.
    """
    return _get_ls_provider(model) == "openai"


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

    Adds `thread_id` as a top-level `prompt_cache_key` when the model and the
    current invocation settings do not already carry one. Callers decide
    eligibility (provider check + `models.openai_prompt_cache_key` opt-out)
    before invoking this helper.

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


def _resolve_openai_prompt_cache_key_enabled() -> bool:
    """Resolve the `models.openai_prompt_cache_key` opt-out (default on).

    Called once when `ConfigurableModelMiddleware` is constructed. The read is
    kept off the blockbuster-guarded server loop by the caller: on the server
    path `create_cli_agent` runs inside `asyncio.to_thread` (see
    `server_graph._make_graph`), so the synchronous `config.toml` read happens
    on a worker thread.

    On an unexpected failure this defaults to enabled: breaking agent
    construction over a config hiccup is worse than injecting the key, and the
    ordinary failure modes (a missing or corrupt `config.toml`) are already
    absorbed by `load_config_toml`. The trade-off is real, not cosmetic — a user
    who opted out *because their endpoint 400s on unknown request fields* would
    then see that per-request failure rather than a benign extra key — so the
    fallback logs at `warning` (not `debug`) to leave a breadcrumb.

    `BlockingError` is deliberately excluded from the fail-open: it signals a
    real blocking-I/O-on-the-event-loop regression (construction moved back onto
    the guarded loop), and swallowing it would mask that bug *and* silently
    defeat the opt-out. It is re-raised so the violation surfaces loudly. It is
    matched by class name because `blockbuster` is not a runtime dependency of
    this package (it is supplied by the langgraph runtime), so it cannot be
    imported here for an `isinstance` check.

    Returns:
        `True` when injection is enabled (the default), `False` when the opt-out
            is set.
    """
    try:
        from deepagents_code.config import is_openai_prompt_cache_key_enabled

        return is_openai_prompt_cache_key_enabled()
    except Exception as exc:
        if any(cls.__name__ == "BlockingError" for cls in type(exc).__mro__):
            raise
        logger.warning(
            "Could not resolve models.openai_prompt_cache_key; defaulting to ON "
            "(an opt-out you set may not take effect)",
            exc_info=True,
        )
        return True


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
            approval_mode=(
                ctx.get("approval_mode")
                if isinstance(ctx.get("approval_mode"), str)
                else "manual"
            ),
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
    request: ModelRequest,
    ctx: CLIContextSchema,
    model_result: ModelResult | None,
    *,
    openai_prompt_cache_key: bool,
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
        openai_prompt_cache_key: Whether OpenAI `prompt_cache_key` injection is
            enabled (the resolved `models.openai_prompt_cache_key` opt-out).

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
        overrides["model_settings"] = {**request.model_settings, **model_params}

    # Inject the provider's prompt-cache routing hint from the active thread.
    # Only one provider path applies per call; both share the fetch/guard/log
    # tail below. `overrides.get` is side-effect-free, so resolving `settings`
    # before the provider check is equivalent to doing it inside each branch.
    effective_model = new_model if new_model is not None else request.model
    if ctx.thread_id:
        settings = overrides.get("model_settings", request.model_settings)
        if _is_fireworks_model(effective_model):
            # Fireworks has no opt-out gate. The classifier is provider-only
            # (like the OpenAI one), so this does not *verify* a fixed endpoint;
            # it rests on the assumption that `ChatFireworks` in practice targets
            # Fireworks' hosted API, where unknown-field rejection is not the
            # concern it is for the broadened, proxy-reachable OpenAI path below.
            updated_settings = _with_fireworks_session_settings(settings, ctx.thread_id)
            injected = "Fireworks session settings"
        elif _is_openai_model(effective_model):
            if openai_prompt_cache_key:
                updated_settings = _with_openai_prompt_cache_key(
                    effective_model, settings, ctx.thread_id
                )
                injected = "OpenAI prompt_cache_key"
            else:
                # Opt-out fired: leave the request untouched but log it so a user
                # verifying `models.openai_prompt_cache_key=false` sees a positive
                # signal rather than having to infer it from an absent log line.
                updated_settings = None
                injected = ""
                logger.debug("Skipped OpenAI prompt_cache_key (opt-out)")
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

    # Layer the swapped-in model's harness profile suffix onto the prompt.
    # `create_deep_agent` resolves harness profiles against the *default*
    # configurable model at construction time, so a runtime swap to a model
    # with its own suffix (e.g. GLM-5.2's `<execution>` block) would otherwise
    # never see it. Append it on top of any Model Identity patch above, using
    # the already-patched prompt when present so both edits coexist.
    if model_result is not None:
        base_prompt = overrides.get("system_prompt", request.system_prompt)
        if base_prompt:
            from deepagents.profiles.harness.harness_profiles import (
                _harness_profile_for_model,  # noqa: PLC2701  # Mirrors SDK profile lookup.
            )

            spec = _model_spec_from_result(model_result, model_result.model)
            suffix = _harness_profile_for_model(
                model_result.model, spec
            ).system_prompt_suffix
            if suffix and suffix not in base_prompt:
                overrides["system_prompt"] = f"{base_prompt}\n\n{suffix}"

    return request.override(**overrides)


def _apply_overrides(
    request: ModelRequest, *, openai_prompt_cache_key: bool
) -> _ResolvedModelRequest:
    """Apply model/param overrides and return checkpoint persistence metadata.

    Reads `'model'` and `'model_params'` from `runtime.context` and, when
    present, swaps the model and/or merges extra settings into the request.
    On a cross-provider swap away from Anthropic, Anthropic-only settings
    (e.g. `cache_control`) are stripped. The `### Model Identity` section
    in the system prompt is also patched to reflect the new model.

    Args:
        request: The incoming model request from the middleware chain.
        openai_prompt_cache_key: The resolved `models.openai_prompt_cache_key`
            opt-out, threaded through to `_build_overrides`.

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
        model_kwargs = (
            {"profile_overrides": ctx.profile_overrides}
            if ctx.profile_overrides
            else {}
        )
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
                model_params_known=True,
            )

    updated = _build_overrides(
        request, ctx, model_result, openai_prompt_cache_key=openai_prompt_cache_key
    )
    params = dict(ctx.model_params) if ctx.model_params else None
    return _ResolvedModelRequest(
        updated,
        _model_spec_from_result(model_result, updated.model),
        params,
        model_params_known=True,
    )


async def _apply_overrides_async(
    request: ModelRequest, *, openai_prompt_cache_key: bool
) -> _ResolvedModelRequest:
    """Async variant of `_apply_overrides` that offloads model construction.

    Args:
        request: The incoming model request from the middleware chain.
        openai_prompt_cache_key: The resolved `models.openai_prompt_cache_key`
            opt-out, threaded through to `_build_overrides`.

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
        model_kwargs = (
            {"profile_overrides": ctx.profile_overrides}
            if ctx.profile_overrides
            else {}
        )
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
                model_params_known=True,
            )

    updated = _build_overrides(
        request, ctx, model_result, openai_prompt_cache_key=openai_prompt_cache_key
    )
    params = dict(ctx.model_params) if ctx.model_params else None
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

    def __init__(
        self,
        *,
        persist_model_state: bool = True,
        openai_prompt_cache_key: bool | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            persist_model_state: Whether completed calls should write private
                resume metadata. Subagent instances disable this because they do
                not own the parent thread's resume state.
            openai_prompt_cache_key: Whether to inject a per-thread OpenAI
                `prompt_cache_key`. Left as `None` (the default) it is resolved
                once here from `models.openai_prompt_cache_key` and cached, so no
                per-call read happens. The one-time `config.toml` read assumes
                current callers construct the middleware off the
                blockbuster-guarded server loop (the server path offloads
                `create_cli_agent` via `asyncio.to_thread`); if that assumption
                is ever broken the read would trip `BlockingError`, which
                `_resolve_openai_prompt_cache_key_enabled` re-raises rather than
                masks. Pass an explicit bool to bypass the config read (mainly
                for tests).
        """
        self._persist_model_state = persist_model_state
        self._openai_prompt_cache_key = (
            _resolve_openai_prompt_cache_key_enabled()
            if openai_prompt_cache_key is None
            else openai_prompt_cache_key
        )

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
        resolved = _apply_overrides(
            request, openai_prompt_cache_key=self._openai_prompt_cache_key
        )
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
        resolved = await _apply_overrides_async(
            request, openai_prompt_cache_key=self._openai_prompt_cache_key
        )
        response = await handler(resolved.request)
        command = _checkpoint_command(resolved) if self._persist_model_state else None
        if command is None:
            return response
        return ExtendedModelResponse(model_response=response, command=command)
