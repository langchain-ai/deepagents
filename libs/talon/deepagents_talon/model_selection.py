"""Per-chat model selection behind the `/model` command.

Talon is an experimental runtime and is subject to change or removal at any time.

A chat's choice travels to the runtime on each `AgentRequest`. The runtime turns
it into a chat model the first time any chat picks it, keeps that model for
later turns, and binds it for the turn; `ModelSelectionMiddleware` then swaps it
into every main-agent model call, so switching never recompiles the graph.

Which models may be chosen is discovered rather than configured: the tool-calling
models that installed LangChain provider packages advertise, limited to
providers whose credentials are set in Talon's own environment. Chat input is
untrusted, so a requested model must match that catalog exactly before anything
is built from it.
"""

from __future__ import annotations

import contextvars
import threading
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.model_config import get_available_models, get_credential_env_var

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel

ACTIVE_MODEL: contextvars.ContextVar[BaseChatModel | None] = contextvars.ContextVar(
    "talon_active_model",
    default=None,
)
"""Model the current turn's chat selected, or `None` to keep the graph's default."""


class UnknownModelError(ValueError):
    """Requested model is not in the discovered catalog."""


def discover_models(env: Mapping[str, str]) -> dict[str, list[str]]:
    """List the models each credentialed provider offers.

    Discovery reuses `deepagents-code`'s model catalog, so Talon and `dcode` agree
    on what a provider offers. Credentials are checked against `env` alone: a key
    that only `dcode`'s credential store holds never reaches Talon's models, so
    counting it would list models that cannot be built.

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
