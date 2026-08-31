"""Fake chat model base shared by integration tests and tool enumeration.

Holds the tool-binding base that both the local integration-test fakes
(`_testing_models`) and the `dcode tools list` tool-enumeration path
(`tool_catalog._CatalogModel`) build on. It lives in a use-neutral module — not
under a `_testing_`-prefixed name — so a production import path never depends on
something that reads as test-only and might be pruned or excluded from the wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


_TOOL_BINDING_MODEL_PROFILE: dict[str, Any] = {
    "tool_calling": True,
    "max_input_tokens": 8000,
}
"""Minimal capability profile the agent runtime reads while compiling a model.

Only `tool_calling` is load-bearing — the agent negotiates tool support at
setup. `max_input_tokens` is part of the profile surface but inert here: these
models are compiled to bind tools and are never invoked, so no token budget ever
applies. Defined once so both the integration-test fakes and
`tool_catalog._CatalogModel` share a single source of truth.
"""


class _ToolBindingFakeModel(GenericFakeChatModel):
    """Base for fake chat models that must bind tools but are never invoked.

    The agent runtime calls `model.bind_tools(schemas)` and reads `model.profile`
    while compiling the graph, and a bare `GenericFakeChatModel` cannot be
    compiled into an agent graph: it inherits `BaseChatModel.bind_tools`, which
    raises `NotImplementedError`, and its `profile` is `None`, which breaks
    capability negotiation. This base supplies a no-op `bind_tools` passthrough
    and a minimal `profile`, leaving subclasses to add generation behavior
    (tests) or nothing at all (tool enumeration).
    """

    # Required by `GenericFakeChatModel`, but subclasses never consume it.
    messages: object = Field(default_factory=lambda: iter(()))
    profile: dict[str, Any] | None = Field(
        default_factory=lambda: dict(_TOOL_BINDING_MODEL_PROFILE)
    )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return self so the agent can bind tool schemas without a real model."""
        return self
