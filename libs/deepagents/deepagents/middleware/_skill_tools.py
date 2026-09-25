"""Disclose the tools a skill names once its `SKILL.md` has been read.

`SkillsMiddleware` owns the behavior; this module holds the pieces it is
built from: which reads count as skill loads, which tools they disclose, where
the disclosure goes in the conversation, and the provider-native blocks that
carry each tool's definition.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic, convert_to_anthropic_tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents.backends.utils import validate_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from langchain_openai import ChatOpenAI

    from deepagents.middleware.skills import SkillMetadata

logger = logging.getLogger(__name__)

INCLUDE_TOOLS_KEY = "include_tools"
"""`SKILL.md` frontmatter `metadata` key listing the skill's tools, space-separated."""

SKILL_TOOLS_DISCLOSED_KEY = "_skill_tools_disclosed"
"""Private state key recording the skill tools disclosed to the latest model call."""

_DEFER_LOADING = "defer_loading"
"""Tool `extras` key (and provider field) that withholds a tool's schema until searched for."""

_ANTHROPIC_INLINE_TOOL_MODELS = ("claude-opus-5", "claude-fable-5", "claude-mythos-5", "claude-opus-4-8")
"""Model ID prefixes that accept an inline `tool_definition` mid-conversation.

Every prefix must also pass `langchain_anthropic`'s mid-conversation system
message check; otherwise the converter hoists the disclosure into `system` and
strips its blocks with only a warning.
"""

_OPENAI_INLINE_TOOL_MODELS = ("gpt-6-", "gpt-5.6-")
"""Model ID prefixes whose Responses API accepts an `additional_tools` input item."""

_MAX_BINDING_DEPTH = 10
"""How many `RunnableBinding` layers to unwrap when looking for the chat model."""

ToolDisclosure = dict[str, Any]
"""A provider-native content block that makes one tool callable from its position on."""


def coerce_skill_tools(entries: Sequence[BaseTool | Callable[..., Any]]) -> dict[str, BaseTool]:
    """Convert `skill_tools` entries as `create_agent` converts tools, keyed by name.

    Raises:
        TypeError: If an entry is a provider-native tool dict.
        ValueError: If two entries share a name.
    """
    tools: list[BaseTool] = []
    for entry in entries:
        if isinstance(entry, dict):
            msg = "skill_tools entries must be BaseTool instances or callables; provider-native tool dicts are not supported"
            raise TypeError(msg)
        tools.append(entry if isinstance(entry, BaseTool) else create_tool(entry))
    names = [t.name for t in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"skill_tools contains duplicate tool name(s): {', '.join(duplicates)}"
        raise ValueError(msg)
    return {t.name: t for t in tools}


def included_tool_names(skill: SkillMetadata) -> list[str]:
    """Return the tool names a skill's frontmatter lists under `metadata.include_tools`."""
    return (skill.get("metadata") or {}).get(INCLUDE_TOOLS_KEY, "").split()


def skills_naming(skills: Sequence[SkillMetadata], tool_name: str) -> list[SkillMetadata]:
    """Return the skills whose `include_tools` names `tool_name`, in listing order."""
    return [skill for skill in skills if tool_name in included_tool_names(skill)]


def not_disclosed_error(tool_name: str, skills: Sequence[SkillMetadata]) -> str:
    """Build the error returned for a call to a skill tool its model call never saw."""
    if len(skills) == 1:
        return f"Error: {tool_name} is not available yet. Read the {skills[0]['name']} skill ({skills[0]['path']}) to make it available."
    listed = ", ".join(f"{skill['name']} ({skill['path']})" for skill in skills)
    return f"Error: {tool_name} is not available yet. Read one of these skills to make it available: {listed}."


@dataclass
class Disclosure:
    """The tools one model call discloses, and where each is anchored."""

    deferred: dict[str, BaseTool] = field(default_factory=dict)
    """Deferred tools from `request.tools`: disclosed early, never gated."""

    gated: dict[str, BaseTool] = field(default_factory=dict)
    """Skill tools: callable only while disclosed."""

    anchors: dict[str, int] = field(default_factory=dict)
    """Index of the earliest skill read naming each tool."""

    def tools(self) -> dict[str, BaseTool]:
        """Return every disclosed tool by name."""
        return {**self.deferred, **self.gated}


def resolve_disclosure(
    messages: Sequence[AnyMessage],
    skills: Sequence[SkillMetadata],
    request_tools: Sequence[BaseTool | dict[str, Any]],
    skill_tools: Mapping[str, BaseTool],
) -> Disclosure:
    """Resolve the tools named by the skills read in `messages` against both pools.

    Anything already in `request_tools` wins over a skill tool of the same name,
    so a tool the model can already see is never hidden or disclosed twice.
    """
    disclosure = Disclosure()
    if not any(included_tool_names(skill) for skill in skills):
        return disclosure
    present = {_tool_name(t): t for t in request_tools}
    for name, (index, skill_name) in _anchors(messages, skills).items():
        tool = present.get(name)
        if isinstance(tool, BaseTool) and (tool.extras or {}).get(_DEFER_LOADING) is True:
            disclosure.deferred[name] = tool
        elif name in present:
            continue
        elif name in skill_tools:
            disclosure.gated[name] = skill_tools[name]
        else:
            logger.debug("Skill '%s' names tool '%s', which is not available in this request", skill_name, name)
            continue
        disclosure.anchors[name] = index
    return disclosure


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """Return a bound tool's name, whether it is a `BaseTool` or a provider dict."""
    if isinstance(tool, BaseTool):
        return tool.name
    function = tool.get("function")
    return tool.get("name") or (function.get("name") if isinstance(function, dict) else None)


def _normalized_path(path: object) -> str | None:
    """Normalize `path` as `read_file` does, or return `None` if it is not a valid path."""
    if not isinstance(path, str):
        return None
    try:
        return validate_path(path)
    except ValueError:
        return None


def _skill_loads(messages: Sequence[AnyMessage], skills: Sequence[SkillMetadata]) -> list[tuple[int, SkillMetadata]]:
    """Return `(index, skill)` for every successful `read_file` result of a skill's `SKILL.md`.

    Any `offset` or `limit` counts, and so does a result whose content was later
    truncated or clipped, since only the call and the result's status are read.
    """
    skills_by_path = {path: skill for skill in skills if (path := _normalized_path(skill["path"])) is not None}
    read_paths = {
        tool_call["id"]: tool_call["args"].get("file_path")
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
        if tool_call["name"] == "read_file" and tool_call["id"]
    }
    loads: list[tuple[int, SkillMetadata]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        skill = skills_by_path.get(_normalized_path(read_paths.get(message.tool_call_id)) or "")
        if skill is not None:
            loads.append((index, skill))
    return loads


def _anchors(messages: Sequence[AnyMessage], skills: Sequence[SkillMetadata]) -> dict[str, tuple[int, str]]:
    """Map each named tool to the index of the earliest read of a skill naming it, and that skill."""
    anchors: dict[str, tuple[int, str]] = {}
    for index, skill in _skill_loads(messages, skills):
        for name in included_tool_names(skill):
            anchors.setdefault(name, (index, skill["name"]))
    return anchors


def _insertion_point(messages: Sequence[AnyMessage], anchor: int) -> int:
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
    disclosure: Disclosure,
    build: Callable[[BaseTool], ToolDisclosure],
) -> list[AnyMessage]:
    """Insert one `SystemMessage` per insertion point, carrying the tools anchored there.

    Each message lands at the same index with the same bytes on every call while
    its anchor stays visible, so the provider's cached prefix survives.
    """
    tools = disclosure.tools()
    names_by_index: dict[int, list[str]] = {}
    for name in tools:
        names_by_index.setdefault(_insertion_point(messages, disclosure.anchors[name]), []).append(name)
    result = list(messages)
    # Indexes were computed against the unmodified list, so insert back to front.
    for index in sorted(names_by_index, reverse=True):
        # Authored in `content`: the standard-content path drops provider-native blocks.
        result.insert(index, SystemMessage(content=[build(tools[name]) for name in sorted(names_by_index[index])]))
    return result


def bind_disclosures(request_tools: Sequence[BaseTool | dict[str, Any]], disclosure: Disclosure) -> list[BaseTool | dict[str, Any]]:
    """Return `request_tools` with deferred disclosures undeferred and gated ones appended."""
    tools = [_undeferred(t) if isinstance(t, BaseTool) and disclosure.deferred.get(t.name) is t else t for t in request_tools]
    return [*tools, *(disclosure.gated[name] for name in sorted(disclosure.gated))]


def _undeferred(tool: BaseTool) -> BaseTool:
    """Return a copy of `tool` without `defer_loading`, so its schema is sent up front."""
    extras = {key: value for key, value in (tool.extras or {}).items() if key != _DEFER_LOADING}
    return tool.model_copy(update={"extras": extras})


def disclosure_builder(model: object) -> Callable[[BaseTool], ToolDisclosure] | None:
    """Return how `model` is given a tool mid-conversation, or `None` if it can't be.

    The one place that decides support, so a model-profile capability can replace
    the allowlists later.
    """
    chat_model = _unwrap_bound(model)
    if isinstance(chat_model, ChatAnthropic) and chat_model.model.startswith(_ANTHROPIC_INLINE_TOOL_MODELS):
        return _anthropic_tool_addition
    if _is_inline_openai_model(chat_model):
        return _openai_additional_tools
    return None


def _is_inline_openai_model(model: object) -> bool:
    """Return whether `model` is a Responses API `ChatOpenAI` that accepts `additional_tools`."""
    chat_openai = _chat_openai_type()
    # Exact type, not subclasses: a subclass may lift system messages elsewhere
    # (e.g. into `instructions`) and reject a non-text block there.
    if chat_openai is None or type(model) is not chat_openai:
        return False
    return model.use_responses_api is True and model.model_name.startswith(_OPENAI_INLINE_TOOL_MODELS)


def _unwrap_bound(model: object) -> object:
    """Return the chat model beneath any `RunnableBinding` layers."""
    current = model
    for _ in range(_MAX_BINDING_DEPTH):
        bound = getattr(current, "bound", None)
        if bound is None or bound is current:
            break
        current = bound
    return current


@functools.cache
def _chat_openai_type() -> type[ChatOpenAI] | None:
    """Return `ChatOpenAI`, or `None` when `langchain-openai` isn't installed."""
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415  # optional dependency
    except ImportError:
        return None
    return ChatOpenAI


def _anthropic_tool_addition(tool: BaseTool) -> ToolDisclosure:
    """Build the Anthropic `tool_addition` block carrying `tool`'s full definition."""
    definition: dict[str, Any] = dict(convert_to_anthropic_tool(tool))
    definition.pop(_DEFER_LOADING, None)
    definition.pop("cache_control", None)
    return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}


def _openai_additional_tools(tool: BaseTool) -> ToolDisclosure:
    """Build the OpenAI Responses `additional_tools` item carrying `tool`'s function schema."""
    function: dict[str, Any] = {"type": "function", **convert_to_openai_tool(tool)["function"]}
    function.pop(_DEFER_LOADING, None)
    return {"type": "additional_tools", "role": "developer", "tools": [function]}
