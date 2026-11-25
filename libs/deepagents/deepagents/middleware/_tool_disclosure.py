"""Disclose tools mid-conversation without changing a request's `tools`.

A tool that becomes available partway through a conversation can't just be
appended to `request.tools`: the tool list sits at the front of the provider's
cached prefix, so changing it invalidates the cache for the whole conversation.
Models whose profile sets `mid_conversation_tools` can instead be given the
tool inline, in a provider-native block placed in the conversation where the
tool appeared. That block keeps its index and bytes on every later call, so the
cached prefix survives.

Middleware builds a `ToolDisclosure` saying which tools to disclose and the
message each is anchored to, then passes it to `disclose_tools`, which places
the blocks inline when the model supports it and binds the tools in
`request.tools` otherwise.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents.middleware._utils import is_anthropic_chat_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import ContextT, ModelRequest
    from langchain_openai.chat_models.base import BaseChatOpenAI

logger = logging.getLogger(__name__)

DEFER_LOADING = "defer_loading"
"""Tool `extras` key (and provider field) that withholds a tool's schema until searched for."""

_ANTHROPIC_ROOT_COMBINATORS = ("oneOf", "anyOf", "allOf")
"""Root `input_schema` keys the Anthropic API rejects, failing the whole request."""

ToolBlock = dict[str, Any]
"""A provider-native content block that makes one tool callable from its position on."""


@dataclass
class ToolDisclosure:
    """The tools one model call discloses mid-conversation, and where each is anchored."""

    deferred: dict[str, BaseTool] = field(default_factory=dict)
    """Deferred tools already in `request.tools`, disclosed ahead of a search."""

    added: dict[str, BaseTool] = field(default_factory=dict)
    """Tools missing from `request.tools`."""

    anchors: dict[str, int] = field(default_factory=dict)
    """Index in `request.messages` of the message each tool is disclosed after."""

    def tools(self) -> dict[str, BaseTool]:
        """Return every disclosed tool by name."""
        return {**self.deferred, **self.added}

    def discard(self, name: str) -> None:
        """Withhold the tool `name`, so the model isn't shown it."""
        for disclosed in (self.deferred, self.added, self.anchors):
            disclosed.pop(name, None)


def disclose_tools(request: ModelRequest[ContextT], disclosure: ToolDisclosure) -> ModelRequest[ContextT]:
    """Return `request` disclosing `disclosure`'s tools.

    Tools whose schema the provider would reject are first discarded from
    `disclosure`, so it ends up equal to what the model is shown. The rest are
    placed inline when the model takes tools mid-conversation, and bound in
    `request.tools` otherwise, which changes the cached prefix.

    Args:
        request: The model request to disclose the tools to.
        disclosure: The tools to disclose and their anchors. Modified in place.

    Returns:
        The request to send.
    """
    discard_rejected_schemas(disclosure, request.model)
    if not disclosure.anchors:
        return request
    build = inline_block_builder(request.model)
    if build is None:
        return request.override(tools=bind_disclosures(request.tools, disclosure))
    return request.override(messages=insert_disclosures(request.messages, disclosure, build))


def tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """Return a bound tool's name, whether it is a `BaseTool` or a provider dict."""
    if isinstance(tool, BaseTool):
        return tool.name
    function = tool.get("function")
    return tool.get("name") or (function.get("name") if isinstance(function, dict) else None)


def insertion_point(messages: Sequence[AnyMessage], anchor: int) -> int:
    """Index after the anchor's tool-result batch and any follow-ups queued behind it.

    Anthropic needs the disclosure after a user turn and before an assistant turn,
    so it may not split a tool-result batch or sit in front of a user message;
    OpenAI needs it to keep its position. An empty reply is skipped too: Anthropic
    drops it, leaving no assistant turn there.
    """
    index = anchor + 1
    while index < len(messages) and (
        isinstance(message := messages[index], ToolMessage | HumanMessage)
        or (isinstance(message, AIMessage) and not message.content and not message.tool_calls)
    ):
        index += 1
    return index


def insert_disclosures(
    messages: Sequence[AnyMessage],
    disclosure: ToolDisclosure,
    build: Callable[[BaseTool], ToolBlock],
) -> list[AnyMessage]:
    """Insert one `SystemMessage` per insertion point, carrying the tools anchored there.

    Each message lands at the same index with the same bytes on every call while
    its anchor stays visible, so the provider's cached prefix survives.
    """
    tools = disclosure.tools()
    names_by_index: dict[int, list[str]] = {}
    for name in tools:
        names_by_index.setdefault(insertion_point(messages, disclosure.anchors[name]), []).append(name)
    result = list(messages)
    # Indexes were computed against the unmodified list, so insert back to front.
    for index in sorted(names_by_index, reverse=True):
        # Authored in `content`: the standard-content path drops provider-native blocks.
        result.insert(index, SystemMessage(content=[build(tools[name]) for name in sorted(names_by_index[index])]))
    return result


def bind_disclosures(request_tools: Sequence[BaseTool | dict[str, Any]], disclosure: ToolDisclosure) -> list[BaseTool | dict[str, Any]]:
    """Return `request_tools` with deferred disclosures undeferred and added ones appended."""
    tools: list[BaseTool | dict[str, Any]] = []
    for tool in request_tools:
        if isinstance(tool, BaseTool) and disclosure.deferred.get(tool.name) is tool:
            # A copy without `defer_loading`, so its schema is sent up front.
            extras = {key: value for key, value in (tool.extras or {}).items() if key != DEFER_LOADING}
            tools.append(tool.model_copy(update={"extras": extras}))
        else:
            tools.append(tool)
    return [*tools, *(disclosure.added[name] for name in sorted(disclosure.added))]


def discard_rejected_schemas(disclosure: ToolDisclosure, model: object) -> None:
    """Withhold each disclosed tool whose schema `model`'s provider rejects, failing the whole request.

    Anthropic rejects a root `oneOf`, `anyOf` or `allOf` in a tool's input schema,
    in `tools` and inline alike. `bind_tools` drops such a tool from `tools`
    (bar `allOf`), but an inline disclosure never passes through it, and
    withholding it on every path keeps the disclosure equal to what the model
    was shown.
    """
    if not is_anthropic_chat_model(unwrap_bound(model)):
        return
    # Already loaded: the model is a `ChatAnthropic`.
    from langchain_anthropic import convert_to_anthropic_tool  # noqa: PLC0415

    for name, tool in disclosure.tools().items():
        schema = convert_to_anthropic_tool(tool).get("input_schema")
        keys = [key for key in _ANTHROPIC_ROOT_COMBINATORS if isinstance(schema, dict) and key in schema]
        if keys:
            logger.warning(
                "Not disclosing tool '%s': its input_schema has a top-level %s, which the Anthropic API does not support", name, "/".join(keys)
            )
            disclosure.discard(name)


def inline_block_builder(model: object) -> Callable[[BaseTool], ToolBlock] | None:
    """Return how `model` is given a tool mid-conversation, or `None` if it can't be."""
    chat_model = unwrap_bound(model)
    if is_anthropic_chat_model(chat_model):
        build = _anthropic_tool_addition
    elif _is_openai_chat_model(chat_model):
        build = _openai_additional_tools
    else:
        return None
    return build if (chat_model.profile or {}).get("mid_conversation_tools") else None


def _is_openai_chat_model(model: object) -> TypeGuard[BaseChatOpenAI]:
    """Return whether `model` is a `langchain_openai` chat model.

    Reads `sys.modules` rather than importing `langchain_openai`: an instance of
    its classes implies the module is loaded, and importing it eagerly would add
    the `openai` SDK to every `deepagents` import.
    """
    base = sys.modules.get("langchain_openai.chat_models.base")
    return base is not None and isinstance(model, base.BaseChatOpenAI)


def unwrap_bound(model: object) -> object:
    """Return the chat model inside `model.bind(...)` or `model.with_config(...)`.

    Chained calls merge into one `RunnableBinding`, so one unwrap is enough.
    """
    return model.bound if isinstance(model, RunnableBinding) else model


def _anthropic_tool_addition(tool: BaseTool) -> ToolBlock:
    """Build the Anthropic `tool_addition` block carrying `tool`'s full definition."""
    # Deferred: only an Anthropic model needs this, and it loads the `anthropic` SDK.
    from langchain_anthropic import convert_to_anthropic_tool  # noqa: PLC0415

    definition: dict[str, Any] = dict(convert_to_anthropic_tool(tool))
    definition.pop(DEFER_LOADING, None)
    definition.pop("cache_control", None)
    return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}


def _openai_additional_tools(tool: BaseTool) -> ToolBlock:
    """Build the OpenAI Responses `additional_tools` item carrying `tool`'s function schema."""
    function: dict[str, Any] = {"type": "function", **convert_to_openai_tool(tool)["function"]}
    function.pop(DEFER_LOADING, None)
    return {"type": "additional_tools", "role": "developer", "tools": [function]}
