"""Disclose the tools a skill names once its `SKILL.md` has been read.

`SkillsMiddleware` owns the behavior; this module holds the pieces it is
built from: which reads of a `SKILL.md` count, how the include names a skill
lists resolve to tools and which of those are disclosed, where the disclosure
goes in the conversation, and the provider-native blocks that carry each tool's
definition.

An include name is one entry in a skill's `metadata.include_tools`: a tool's
exact name, or a name a resolver maps to tools.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langchain_anthropic import ChatAnthropic, convert_to_anthropic_tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents.backends.utils import validate_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from langchain_openai import ChatOpenAI
    from langgraph.runtime import Runtime

    from deepagents.middleware.skills import SkillMetadata, SkillToolResolver

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
except ImportError:
    _CHAT_OPENAI_TYPE: type[ChatOpenAI] | None = None
else:
    _CHAT_OPENAI_TYPE = _ChatOpenAI

logger = logging.getLogger(__name__)

_INCLUDE_TOOLS_KEY = "include_tools"
"""`SKILL.md` frontmatter `metadata` key listing the skill's include names, space-separated."""

_SKILL_TOOLS_DISCLOSED_KEY = "_skill_tools_disclosed"
"""Private state key mapping each skill tool disclosed to the latest model call to the include name that produced it."""

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

_ANTHROPIC_ROOT_COMBINATORS = ("oneOf", "anyOf", "allOf")
"""Root `input_schema` keys the Anthropic API rejects, failing the whole request."""

_ToolDisclosure = dict[str, Any]
"""A provider-native content block that makes one tool callable from its position on."""


def _normalize_skill_tools(tools: Sequence[BaseTool | Callable[..., Any]] | SkillToolResolver | None) -> SkillToolResolver:
    """Return the resolver `SkillsMiddleware`'s `tools` stands for.

    A sequence is converted as `create_agent` converts tools, then resolved by
    exact name.

    Raises:
        TypeError: If `tools` is a bare tool or neither a sequence nor
            callable, or if an entry is a provider-native tool dict.
        ValueError: If two entries share a name.
    """
    if tools is not None and not isinstance(tools, Sequence):
        if isinstance(tools, BaseTool) or not callable(tools):
            msg = f"tools must be a list of tools or a resolver function, got {type(tools).__name__}; wrap a single tool in a list"
            raise TypeError(msg)
        return tools
    converted: list[BaseTool] = []
    # ty keeps a callable-and-sequence intersection that no value inhabits.
    for entry in cast("Sequence[BaseTool | Callable[..., Any]]", tools or ()):
        if isinstance(entry, dict):
            msg = "tools entries must be BaseTool instances or callables; provider-native tool dicts are not supported"
            raise TypeError(msg)
        converted.append(entry if isinstance(entry, BaseTool) else create_tool(entry))
    names = [t.name for t in converted]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"tools contains duplicate tool name(s): {', '.join(duplicates)}"
        raise ValueError(msg)
    by_name = {t.name: t for t in converted}

    def resolve_by_exact_name(name: str, runtime: Runtime[Any]) -> list[BaseTool]:  # noqa: ARG001  # resolver signature
        return [by_name[name]] if name in by_name else []

    return resolve_by_exact_name


def _call_resolver(resolver: SkillToolResolver, name: str, runtime: Runtime[Any]) -> list[BaseTool]:
    """Resolve `name` on a sync hook.

    Raises:
        TypeError: If the resolver is async, or returns something other than tools.
    """
    result = resolver(name, runtime)
    if inspect.isawaitable(result):
        # Close it, so discarding it unawaited raises no "never awaited" warning.
        close = getattr(result, "close", None)
        if callable(close):
            close()
        msg = f"skill tool resolver returned an awaitable for {name!r}; an async resolver needs the agent's async entry point (e.g. `ainvoke`)"
        raise TypeError(msg)
    return _checked(name, result)


async def _acall_resolver(resolver: SkillToolResolver, name: str, runtime: Runtime[Any]) -> list[BaseTool]:
    """Resolve `name` on an async hook, awaiting the resolver if it's async.

    Raises:
        TypeError: If the resolver returns something other than tools.
    """
    result = resolver(name, runtime)
    if inspect.isawaitable(result):
        result = await result
    return _checked(name, result)


def _checked(name: str, result: object) -> list[BaseTool]:
    """Validate a resolver's tools for `name`, keeping the first of each tool name.

    Raises:
        TypeError: If `result` isn't a sequence of `BaseTool`s.
    """
    if not isinstance(result, Sequence) or isinstance(result, str):
        msg = f"skill tool resolver must return a sequence of BaseTool for {name!r}, got {type(result).__name__}"
        raise TypeError(msg)
    tools: dict[str, BaseTool] = {}
    for item in result:
        if not isinstance(item, BaseTool):
            msg = f"skill tool resolver returned a {type(item).__name__} for {name!r}; expected BaseTool instances"
            raise TypeError(msg)
        tools.setdefault(item.name, item)
    return list(tools.values())


def _disclosed_record(state: Mapping[str, object]) -> dict[str, str]:
    """Return the recorded `{tool name: include name}` map, or `{}` if it's missing or malformed.

    A record of any other shape, such as one checkpointed by an earlier build,
    reads as empty rather than failing the gate.
    """
    record = state.get(_SKILL_TOOLS_DISCLOSED_KEY)
    if not isinstance(record, dict):
        return {}
    entries = {k: v for k, v in record.items() if isinstance(k, str) and isinstance(v, str)}
    return entries if len(entries) == len(record) else {}


def _include_names(skill: SkillMetadata) -> list[str]:
    """Return the include names a skill's frontmatter lists under `metadata.include_tools`."""
    return (skill.get("metadata") or {}).get(_INCLUDE_TOOLS_KEY, "").split()


@dataclass
class _Disclosure:
    """The tools one model call discloses, and where each is anchored."""

    deferred: dict[str, BaseTool] = field(default_factory=dict)
    """Deferred tools from `request.tools`: disclosed early, never gated."""

    gated: dict[str, BaseTool] = field(default_factory=dict)
    """Skill tools: callable only while disclosed."""

    anchors: dict[str, int] = field(default_factory=dict)
    """Index of the earliest skill read producing each tool."""

    produced_by: dict[str, str] = field(default_factory=dict)
    """The include name that produced each gated tool at its anchor, for tool-time lookup."""

    def tools(self) -> dict[str, BaseTool]:
        """Return every disclosed tool by name."""
        return {**self.deferred, **self.gated}

    def record(self) -> dict[str, str]:
        """Return the include name that produced each gated tool, sorted by tool name, for the tool-time gate."""
        return {name: self.produced_by[name] for name in sorted(self.gated)}

    def discard(self, name: str) -> None:
        """Withhold the tool `name`, so the model isn't shown it and the gate doesn't admit it."""
        for disclosed in (self.deferred, self.gated, self.anchors, self.produced_by):
            disclosed.pop(name, None)


_SkillRead = tuple[int, "SkillMetadata"]
"""A successful read of a skill's `SKILL.md`: the tool result's message index, and the skill."""


def _unclaimed_include_names(reads: Sequence[_SkillRead], request_tools: Sequence[BaseTool | dict[str, Any]]) -> list[str]:
    """Return each distinct include name the read skills list that no request tool claims, in read order."""
    present = {_tool_name(t) for t in request_tools}
    return list(dict.fromkeys(name for _, skill in reads for name in _include_names(skill) if name not in present))


def _plan_disclosure(
    reads: Sequence[_SkillRead],
    request_tools: Sequence[BaseTool | dict[str, Any]],
    resolved: Mapping[str, Sequence[BaseTool]],
) -> _Disclosure:
    """Classify what each read skill's include names produce, anchoring each tool at the earliest read producing it.

    An include name that matches a request tool's name produces that tool. Any
    other produces what the resolver returned for it (`resolved`).
    """
    disclosure = _Disclosure()
    present = {_tool_name(t): t for t in request_tools}
    unresolved: set[str] = set()
    for index, skill in reads:
        for include_name in _include_names(skill):
            produced = [present[include_name]] if include_name in present else resolved.get(include_name, [])
            if not produced and include_name not in unresolved:
                unresolved.add(include_name)
                logger.debug("Skill '%s' names tool '%s', which is not available in this request", skill["name"], include_name)
            for tool in produced:
                _classify(disclosure, tool, present, index, include_name)
    return disclosure


def _classify(
    disclosure: _Disclosure,
    tool: BaseTool | dict[str, Any],
    present: Mapping[str | None, BaseTool | dict[str, Any]],
    index: int,
    include_name: str,
) -> None:
    """Add `tool`, produced by `include_name` at read `index`, unless an earlier read already did.

    If the request already has a tool with this name, that tool wins, because
    it's the one that runs. A deferred one is disclosed early. Any other adds
    nothing, since the model already sees it.

    Tools are matched by name, not by object, because outer middleware such as
    tool search can replace the request's tools with copies.
    """
    if not isinstance(tool, BaseTool) or tool.name in disclosure.anchors:
        return
    entry = present.get(tool.name)
    if entry is None:
        disclosure.gated[tool.name] = tool
        disclosure.produced_by[tool.name] = include_name
    elif isinstance(entry, BaseTool) and (entry.extras or {}).get(_DEFER_LOADING) is True:
        disclosure.deferred[tool.name] = entry
    else:
        return
    disclosure.anchors[tool.name] = index


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


def _find_skill_reads(messages: Sequence[AnyMessage], skills: Sequence[SkillMetadata]) -> list[_SkillRead]:
    """Return `(index, skill)` for every successful `read_file` result of a `SKILL.md` naming tools.

    Any `offset` or `limit` counts, and so does a result whose content was later
    truncated or clipped, since only the call and the result's status are read.
    """
    skills_by_path = {path: skill for skill in skills if _include_names(skill) and (path := _normalized_path(skill["path"])) is not None}
    if not skills_by_path:
        return []
    read_paths = {
        tool_call["id"]: tool_call["args"].get("file_path")
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
        if tool_call["name"] == "read_file" and tool_call["id"]
    }
    reads: list[_SkillRead] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        skill = skills_by_path.get(_normalized_path(read_paths.get(message.tool_call_id)) or "")
        if skill is not None:
            reads.append((index, skill))
    return reads


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


def _insert_disclosures(
    messages: Sequence[AnyMessage],
    disclosure: _Disclosure,
    build: Callable[[BaseTool], _ToolDisclosure],
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


def _bind_disclosures(request_tools: Sequence[BaseTool | dict[str, Any]], disclosure: _Disclosure) -> list[BaseTool | dict[str, Any]]:
    """Return `request_tools` with deferred disclosures undeferred and gated ones appended."""
    tools: list[BaseTool | dict[str, Any]] = []
    for tool in request_tools:
        if isinstance(tool, BaseTool) and disclosure.deferred.get(tool.name) is tool:
            # A copy without `defer_loading`, so its schema is sent up front.
            extras = {key: value for key, value in (tool.extras or {}).items() if key != _DEFER_LOADING}
            tools.append(tool.model_copy(update={"extras": extras}))
        else:
            tools.append(tool)
    return [*tools, *(disclosure.gated[name] for name in sorted(disclosure.gated))]


def _discard_rejected_schemas(disclosure: _Disclosure, model: object) -> None:
    """Withhold each disclosed tool whose schema `model`'s provider rejects, failing the whole request.

    Anthropic rejects a root `oneOf`, `anyOf` or `allOf` in a tool's input schema,
    in `tools` and inline alike. `bind_tools` drops such a tool from `tools`
    (bar `allOf`), but an inline disclosure never passes through it, and
    withholding it on every path keeps the record equal to what the model was
    shown.
    """
    if not isinstance(_unwrap_bound(model), ChatAnthropic):
        return
    for name, tool in disclosure.tools().items():
        schema = convert_to_anthropic_tool(tool).get("input_schema")
        keys = [key for key in _ANTHROPIC_ROOT_COMBINATORS if isinstance(schema, dict) and key in schema]
        if keys:
            logger.warning(
                "Not disclosing tool '%s': its input_schema has a top-level %s, which the Anthropic API does not support", name, "/".join(keys)
            )
            disclosure.discard(name)


def _inline_block_builder(model: object) -> Callable[[BaseTool], _ToolDisclosure] | None:
    """Return how `model` is given a tool mid-conversation, or `None` if it can't be.

    The one place that decides support, so a model-profile capability can replace
    the allowlists later.
    """
    chat_model = _unwrap_bound(model)
    if isinstance(chat_model, ChatAnthropic) and chat_model.model.startswith(_ANTHROPIC_INLINE_TOOL_MODELS):
        return _anthropic_tool_addition
    # Exact type, not subclasses: a subclass may lift system messages elsewhere
    # (e.g. into `instructions`) and reject a non-text block there.
    if (
        _CHAT_OPENAI_TYPE is not None
        and type(chat_model) is _CHAT_OPENAI_TYPE
        and chat_model.use_responses_api is True
        and chat_model.model_name.startswith(_OPENAI_INLINE_TOOL_MODELS)
    ):
        return _openai_additional_tools
    return None


def _unwrap_bound(model: object) -> object:
    """Return the chat model inside `model.bind(...)` or `model.with_config(...)`.

    Chained calls merge into one `RunnableBinding`, so one unwrap is enough.
    """
    return model.bound if isinstance(model, RunnableBinding) else model


def _anthropic_tool_addition(tool: BaseTool) -> _ToolDisclosure:
    """Build the Anthropic `tool_addition` block carrying `tool`'s full definition."""
    definition: dict[str, Any] = dict(convert_to_anthropic_tool(tool))
    definition.pop(_DEFER_LOADING, None)
    definition.pop("cache_control", None)
    return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}


def _openai_additional_tools(tool: BaseTool) -> _ToolDisclosure:
    """Build the OpenAI Responses `additional_tools` item carrying `tool`'s function schema."""
    function: dict[str, Any] = {"type": "function", **convert_to_openai_tool(tool)["function"]}
    function.pop(_DEFER_LOADING, None)
    return {"type": "additional_tools", "role": "developer", "tools": [function]}
