"""The agent's current model, recorded in state under `model_selection`.

LangChain is adding this key to `AgentState` (langchain-ai/langchain#41696):
the model node records a serializable `ModelSpec` of the model each call went to,
and anything that chooses a model may write `current` ahead of the next call.
When the installed LangChain provides it, everything here is LangChain's. Until
then this module mirrors that API, and `create_deep_agent` adds
`_ModelSelectionMiddleware` to record the spec itself. Delete the fallback once
the LangChain minimum includes it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, Required, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ExtendedModelResponse, PrivateStateAttr, TracePolicy, omit_payload
from langchain.chat_models.base import _BUILTIN_PROVIDERS, init_chat_model
from langgraph.types import Command
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain.agents.middleware.model_selection import (  # ty: ignore[unresolved-import]
        MODEL_SELECTION_KEY,
        current_model_spec,
        init_model_from_spec,
        model_spec_from_model,
    )
    from langchain.agents.middleware.types import ModelSelection, ModelSpec  # ty: ignore[unresolved-import]

    UPSTREAM_MODEL_SELECTION = True
    """Whether LangChain records `model_selection` itself."""

    ModelSelectionState = AgentState

except ImportError:
    UPSTREAM_MODEL_SELECTION = False

    MODEL_SELECTION_KEY = "model_selection"
    """Agent state key holding the `ModelSelection`."""

    SAFE_INIT_KWARGS = ("temperature", "top_p", "top_k", "max_tokens", "reasoning_effort")
    """Model attributes copied into `ModelSpec.init_kwargs`; credentials, headers and endpoints never are."""

    _REBUILT_MODEL_CACHE_SIZE = 32

    class ModelSpec(TypedDict, total=False):
        """Serializable description of a chat model, safe to store in agent state."""

        provider: Required[str]
        """Provider key accepted by `init_chat_model`, else the model's LangSmith provider."""

        model: Required[str]
        """Model name, as reported to LangSmith (`ls_model_name`)."""

        max_input_tokens: int
        """Context window from the model's profile, when known."""

        init_kwargs: dict[str, str | int | float | bool]
        """Safe init args (sampling settings and output limits) used to rebuild the model."""

    class ModelSelection(TypedDict, total=False):
        """The model currently selected for the agent; a replaced `current` moves to `previous`."""

        current: ModelSpec
        """The model selected for the next call, or used by the last one."""

        previous: ModelSpec | None
        """The model `current` replaced, if any."""

    def _merge_model_selection(left: ModelSelection | None, right: ModelSelection | None) -> ModelSelection:
        """Reducer for `model_selection`: move a replaced `current` to `previous`."""
        if not right:
            return left or {}
        if not left:
            return cast("ModelSelection", dict(right))
        merged = cast("ModelSelection", {**left, **right})
        replaced = left.get("current")
        current = right.get("current")
        if "previous" not in right and current is not None and replaced not in (None, current):
            merged["previous"] = replaced
        return merged

    class ModelSelectionState(AgentState):
        """Agent state with the `model_selection` key LangChain is adding."""

        model_selection: NotRequired[Annotated[ModelSelection, PrivateStateAttr, _merge_model_selection]]
        """The model selected for the agent; see `ModelSelection`."""

    def _safe_init_kwargs(values: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
        return {key: value for key, value in values.items() if key in SAFE_INIT_KWARGS and isinstance(value, (str, int, float, bool))}

    def _init_provider(model: BaseChatModel, ls_provider: object) -> str | None:
        """Map a model's class back to its `init_chat_model` provider key."""
        cls = type(model)
        package = cls.__module__.split(".", maxsplit=1)[0]
        candidates = [
            provider
            for provider, (module, class_name, _) in _BUILTIN_PROVIDERS.items()
            if class_name == cls.__name__ and module.split(".", maxsplit=1)[0] == package
        ]
        if isinstance(ls_provider, str) and ls_provider and (ls_provider in candidates or not candidates):
            return ls_provider
        return candidates[0] if candidates else None

    def model_spec_from_model(model: BaseChatModel) -> ModelSpec | None:
        """Describe `model` as a serializable `ModelSpec`, or `None` without a model name."""
        try:
            params: Mapping[str, Any] = model._get_ls_params()  # the provider-neutral source of name and provider
        except Exception:  # noqa: BLE001  # a broken integration must not fail the model call
            return None
        name = params.get("ls_model_name")
        provider = _init_provider(model, params.get("ls_provider"))
        if not isinstance(name, str) or not name or provider is None:
            return None
        spec: ModelSpec = {"provider": provider, "model": name}
        profile = getattr(model, "profile", None)
        max_input_tokens = profile.get("max_input_tokens") if isinstance(profile, Mapping) else None
        if isinstance(max_input_tokens, int) and not isinstance(max_input_tokens, bool):
            spec["max_input_tokens"] = max_input_tokens
        init_kwargs = _safe_init_kwargs({key: getattr(model, key, None) for key in SAFE_INIT_KWARGS})
        if init_kwargs:
            spec["init_kwargs"] = init_kwargs
        return spec

    def current_model_spec(state: Mapping[str, Any]) -> ModelSpec | None:
        """Return `state["model_selection"]["current"]`, or `None` when nothing is recorded."""
        selection: ModelSelection | None = state.get(MODEL_SELECTION_KEY)
        return selection.get("current") if selection else None

    def init_model_from_spec(spec: ModelSpec) -> BaseChatModel:
        """Build (or reuse) a chat model for `spec`, for `init_chat_model`'s built-in providers only.

        Raises:
            ValueError: If the spec's provider is not a built-in provider.
        """
        provider = spec.get("provider")
        if provider not in _BUILTIN_PROVIDERS:
            msg = (
                f"Cannot rebuild a chat model for provider {provider!r}: only `init_chat_model`'s built-in providers can be rebuilt from agent state."
            )
            raise ValueError(msg)
        key = {"provider": provider, "model": spec["model"], "init_kwargs": _safe_init_kwargs(spec.get("init_kwargs") or {})}
        return _init_cached(json.dumps(key, sort_keys=True))

    @lru_cache(maxsize=_REBUILT_MODEL_CACHE_SIZE)
    def _init_cached(key: str) -> BaseChatModel:
        spec = json.loads(key)
        return init_chat_model(str(spec["model"]), model_provider=str(spec["provider"]), **spec["init_kwargs"])


def same_model(first: ModelSpec | None, second: ModelSpec | None) -> bool:
    """Whether two specs name the same provider and model."""
    return first is not None and second is not None and first["provider"] == second["provider"] and first["model"] == second["model"]


class _ModelSelectionMiddleware(AgentMiddleware):
    """Records the model each call went to under `model_selection`, as LangChain's model node will.

    Runs innermost, so it sees the model a router or fallback swapped in. When
    retries or fallbacks run several models, the last one wins.
    """

    state_schema = ModelSelectionState
    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    @staticmethod
    def _record(request: ModelRequest, response: ModelResponse) -> ModelResponse | ExtendedModelResponse:
        spec = model_spec_from_model(request.model)
        if spec is None or spec == current_model_spec(request.state):
            return response
        return ExtendedModelResponse(model_response=response, command=Command(update={MODEL_SELECTION_KEY: {"current": spec}}))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Record the model the call went to."""
        return self._record(request, handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Record the model the call went to (async)."""
        return self._record(request, await handler(request))


def model_selection_middleware() -> list[AgentMiddleware]:
    """Middleware that records `model_selection`, unless LangChain already does."""
    return [] if UPSTREAM_MODEL_SELECTION else [_ModelSelectionMiddleware()]
