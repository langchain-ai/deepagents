"""Per-chat model selection behind the `/model` command.

Talon is an experimental runtime and is subject to change or removal at any time.

A chat's choice travels to the runtime on each `AgentRequest`. The runtime turns
it into a chat model the first time any chat picks it, keeps that model for
later turns, and binds it for the turn; `ModelSelectionMiddleware` then swaps it
into every main-agent model call, so switching never recompiles the graph.

Swapping the model is not enough on its own. Deep Agents runs its summarizer
ahead of custom middleware, so it would size the context against the startup
model. `SelectedModelSummarization` takes over the summarizer's slot, the way
`deepagents-code` does for its `/model`, so a switch brings the selected model's
context budget and compaction thresholds with it.

Which models may be chosen is discovered rather than configured: the tool-calling
models that installed LangChain provider packages advertise, limited to
providers whose credentials are set in Talon's own environment. Chat input is
untrusted, so a requested model must match that catalog exactly before anything
is built from it.
"""

from __future__ import annotations

import contextvars
import re
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationState,
    create_summarization_middleware,
)
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.model_config import (
    get_available_models,
    get_credential_env_var,
    is_langsmith_gateway_host,
)
from deepagents_talon.background import _IN_SUBAGENT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.language_models import BaseChatModel

ACTIVE_MODEL: contextvars.ContextVar[BaseChatModel | None] = contextvars.ContextVar(
    "talon_active_model",
    default=None,
)
"""Model the current turn's chat selected, or `None` to keep the graph's default."""


class UnknownModelError(ValueError):
    """Requested model is not in the discovered catalog."""


_GATEWAY_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")


def gateway_connection(env: Mapping[str, str]) -> tuple[str, str] | None:
    """Return the authenticated unified gateway endpoint, if configured."""
    gateway = env.get("LANGSMITH_GATEWAY", "")
    if base_url := env.get("OPENAI_BASE_URL"):
        key = env.get("OPENAI_API_KEY", "")
        if not key and gateway and gateway.lower() not in {"false", "0", "no"}:
            key = env.get("LANGSMITH_GATEWAY_API_KEY", "")
    elif gateway and gateway.lower() not in {"false", "0", "no"}:
        base = (
            "https://gateway.smith.langchain.com"
            if gateway.lower() in {"true", "1", "yes"}
            else gateway.rstrip("/")
        )
        base_url = f"{base}/v1"
        key = env.get("LANGSMITH_GATEWAY_API_KEY") or env.get("LANGSMITH_API_KEY", "")
    else:
        return None
    try:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.port not in (None, 443) or parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or not is_langsmith_gateway_host(parsed.hostname)
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
        or not key.strip()
    ):
        return None
    return f"https://{parsed.hostname}/v1", key


def _gateway_models(env: Mapping[str, str]) -> list[str]:
    if (connection := gateway_connection(env)) is None:
        return []
    base_url, key = connection
    with httpx.Client(timeout=5.0, follow_redirects=False) as client:
        response = client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        data = response.json()
    if (
        not isinstance(data, dict)
        or data.get("object") != "list"
        or not isinstance(data.get("data"), list)
    ):
        msg = "Invalid gateway model catalog"
        raise ValueError(msg)
    return [
        identifier
        for item in data["data"]
        if isinstance(item, dict)
        and isinstance(identifier := item.get("id"), str)
        and _GATEWAY_MODEL_ID.fullmatch(identifier)
        and (
            "supported_endpoints" not in item
            or "/v1/chat/completions" in item["supported_endpoints"]
        )
    ]


def discover_models(env: Mapping[str, str]) -> dict[str, list[str]]:
    """List the models each credentialed provider offers.

    Native models come from `deepagents-code`; configured gateway models come
    from its unified endpoint. Credentials are checked against `env` alone: a
    key only in `dcode`'s store cannot build Talon's models.

    Args:
        env: Environment Talon's models are built from.

    Returns:
        Model names keyed by provider, for providers with credentials set.
    """
    catalog: dict[str, list[str]] = {}
    for provider, models in get_available_models().items():
        variable = get_credential_env_var(provider)
        if variable and env.get(variable, "").strip():
            catalog[provider] = list(models)
    for identifier in _gateway_models(env):
        if identifier not in catalog.setdefault("openai", []):
            catalog["openai"].append(identifier)
    return catalog


class ModelSelection:
    """Catalog of selectable models and a cache of the ones already built.

    Args:
        default: Model spec the graph was built with. Always selectable, and
            never rebuilt, because selecting it means making no override.
        build: Builds a chat model from a `provider:model` spec.
        discover: Returns the selectable models keyed by provider.
    """

    def __init__(
        self,
        default: str,
        *,
        build: Callable[[str], BaseChatModel],
        discover: Callable[[], Mapping[str, Sequence[str]]],
    ) -> None:
        """Initialize without discovering or building anything."""
        self.default = default
        self._build = build
        self._discover = discover
        self._models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()

    def catalog(self) -> dict[str, list[str]]:
        """Return the selectable models keyed by provider.

        Returns:
            The discovered catalog, with the default model included.
        """
        catalog = {provider: list(models) for provider, models in self._discover().items()}
        provider, _, name = self.default.partition(":")
        if name and name not in catalog.setdefault(provider, []):
            catalog[provider].append(name)
        return catalog

    def allows(self, spec: str) -> bool:
        """Report whether `spec` names a selectable model exactly.

        Args:
            spec: Requested `provider:model` spec.

        Returns:
            Whether the spec is the default or appears in the catalog.
        """
        if spec == self.default:
            return True
        provider, separator, name = spec.partition(":")
        return bool(separator) and name in self.catalog().get(provider, ())

    def resolve(self, spec: str) -> BaseChatModel | None:
        """Return the chat model for `spec`, building it on first use.

        Blocking: discovery reads provider profile data and building a model may
        import its provider package, so async callers run this in a thread.

        Args:
            spec: Requested `provider:model` spec.

        Returns:
            The built model, or `None` for the default model.

        Raises:
            UnknownModelError: If `spec` is not a selectable model.
        """
        if spec == self.default:
            return None
        if not self.allows(spec):
            msg = "Requested model is not available"
            raise UnknownModelError(msg)
        with self._lock:
            model = self._models.get(spec)
            if model is None:
                model = self._models[spec] = self._build(spec)
        return model


class ModelSelectionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Swap the turn's selected model into each main-agent model call.

    It is installed outermost, so everything inside it — prompt caching
    included — sees the selected model rather than the one the graph was built
    with.
    """

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Call the model the turn selected.

        Returns:
            The handler's response for the selected model.
        """
        return handler(_selected(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Call the model the turn selected.

        Returns:
            The handler's response for the selected model.
        """
        return await handler(_selected(request))


def _selected(request: ModelRequest[Any]) -> ModelRequest[Any]:
    model = ACTIVE_MODEL.get()
    return request if model is None else request.override(model=model)


def _summary_trim_limit(model: BaseChatModel) -> int:
    """Reserve 20% of the input window for the summary prompt and counting overhead."""
    profile = getattr(model, "profile", None)
    limit = profile.get("max_input_tokens") if isinstance(profile, dict) else None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return 4_000
    return max(1, limit * 4 // 5)


class SelectedModelSummarization(AgentMiddleware[Any, Any, Any]):
    """Summarize against the turn's selected model rather than the startup model.

    It reports the Deep Agents summarizer's name, so `create_deep_agent` puts it
    in that summarizer's slot, ahead of custom middleware. Each selected model
    gets its own summarizer, built the first time it is needed, whose trigger
    thresholds follow that model's profile. The request's model is swapped before
    delegating, because the summarizer checks the input budget against
    `request.model`.

    Delegated subagents keep the startup summarizer: Deep Agents hands this
    middleware to its general-purpose subagent too, and that subagent still calls
    the startup model.

    Args:
        startup: Returns the startup model the default summarizer sizes against.
            Called on first use, so building the graph builds no model.
        backend: Backend the summarizer offloads evicted history to.
    """

    state_schema = SummarizationState
    trace_policy = SummarizationMiddleware.trace_policy

    def __init__(self, startup: Callable[[], BaseChatModel], backend: BackendProtocol) -> None:
        """Keep what summarizers need; each is built the first time a turn needs it."""
        self._startup = startup
        self._backend = backend
        self._default: SummarizationMiddleware | None = None
        self._summarizers: dict[int, tuple[BaseChatModel, SummarizationMiddleware]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Take over the Deep Agents summarizer's slot by sharing its name."""
        return SummarizationMiddleware.serialized_name

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Summarize, if needed, against the selected model's context budget.

        Returns:
            The handler's response.
        """
        summarizer, request = self._for_turn(request)
        return summarizer.wrap_model_call(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Summarize, if needed, against the selected model's context budget.

        Returns:
            The handler's response.
        """
        summarizer, request = self._for_turn(request)
        return await summarizer.awrap_model_call(request, handler)

    def _for_turn(
        self, request: ModelRequest[Any]
    ) -> tuple[SummarizationMiddleware, ModelRequest[Any]]:
        model = ACTIVE_MODEL.get()
        if model is None or _IN_SUBAGENT.get():
            return self._startup_summarizer(), request
        with self._lock:
            cached = self._summarizers.get(id(model))
            if cached is None or cached[0] is not model:
                cached = (
                    model,
                    create_summarization_middleware(
                        model, self._backend, trim_tokens_to_summarize=_summary_trim_limit(model)
                    ),
                )
                self._summarizers[id(model)] = cached
        return cached[1], request.override(model=model)

    def _startup_summarizer(self) -> SummarizationMiddleware:
        with self._lock:
            if self._default is None:
                model = self._startup()
                self._default = create_summarization_middleware(
                    model, self._backend, trim_tokens_to_summarize=_summary_trim_limit(model)
                )
            return self._default
