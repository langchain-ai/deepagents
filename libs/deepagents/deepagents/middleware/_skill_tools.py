"""Disclose the tools a skill names once its `SKILL.md` has been read.

`SkillsMiddleware` owns the behavior; this module holds the pieces it is
built from: which reads of a `SKILL.md` count, and how the include names a skill
lists resolve to tools and which of those are disclosed. Placing the disclosure
in the request is `_tool_disclosure`'s job.

An include name is one entry in a skill's `metadata.include_tools`: a tool's
exact name, or a name a resolver maps to tools.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool as create_tool

from deepagents.backends.utils import validate_path
from deepagents.middleware._tool_disclosure import DEFER_LOADING, ToolDisclosure, tool_name

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from langgraph.runtime import Runtime

    from deepagents.middleware.skills import SkillMetadata, SkillToolResolver

logger = logging.getLogger(__name__)

_INCLUDE_TOOLS_KEY = "include_tools"
"""`SKILL.md` frontmatter `metadata` key listing the skill's include names, space-separated."""

_SKILL_TOOLS_DISCLOSED_KEY = "_skill_tools_disclosed"
"""Private state key mapping each skill tool disclosed to the latest model call to the include name that produced it."""


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


@dataclass
class _Disclosure(ToolDisclosure):
    """The tools one model call discloses, and where each is anchored.

    Deferred tools from `request.tools` are disclosed early and never gated.
    Added ones are skill tools: callable only while disclosed.
    """

    produced_by: dict[str, str] = field(default_factory=dict)
    """The include name that produced each gated tool at its anchor, for tool-time lookup."""

    def record(self) -> dict[str, str]:
        """Return the include name that produced each gated tool, sorted by tool name, for the tool-time gate."""
        return {name: self.produced_by[name] for name in sorted(self.added)}

    def discard(self, name: str) -> None:
        """Withhold the tool `name`, so the model isn't shown it and the gate doesn't admit it."""
        super().discard(name)
        self.produced_by.pop(name, None)


_SkillRead = tuple[int, "SkillMetadata"]
"""A read of a skill's `SKILL.md`: the index of the tool result or pinned skill message, and the skill."""


def _unclaimed_include_names(reads: Sequence[_SkillRead], request_tools: Sequence[BaseTool | dict[str, Any]]) -> list[str]:
    """Return each distinct include name the read skills list that no request tool claims, in read order."""
    present = {tool_name(t) for t in request_tools}
    return list(
        dict.fromkeys(name for _, skill in reads for name in (skill.get("metadata") or {}).get(_INCLUDE_TOOLS_KEY, "").split() if name not in present)
    )


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
    present = {tool_name(t): t for t in request_tools}
    unresolved: set[str] = set()
    for index, skill in reads:
        for include_name in (skill.get("metadata") or {}).get(_INCLUDE_TOOLS_KEY, "").split():
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
        disclosure.added[tool.name] = tool
        disclosure.produced_by[tool.name] = include_name
    elif isinstance(entry, BaseTool) and (entry.extras or {}).get(DEFER_LOADING) is True:
        disclosure.deferred[tool.name] = entry
    else:
        return
    disclosure.anchors[tool.name] = index


def _normalized_path(path: object) -> str | None:
    """Normalize `path` as `read_file` does, or return `None` if it is not a valid path."""
    if not isinstance(path, str):
        return None
    try:
        return validate_path(path)
    except ValueError:
        return None


def _find_skill_reads(messages: Sequence[AnyMessage], skills: Sequence[SkillMetadata]) -> list[_SkillRead]:
    """Return `(index, skill)` for every read of a `SKILL.md` naming tools.

    A read is a successful `read_file` result or a message pinning the
    skill. Any `offset` or `limit` counts, and so does a result whose content was
    later truncated or clipped, since only the call and the result's status are
    read.
    """
    skills_by_path = {
        path: skill
        for skill in skills
        if (skill.get("metadata") or {}).get(_INCLUDE_TOOLS_KEY, "").split() and (path := _normalized_path(skill["path"])) is not None
    }
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
        # A tool result's path is on the `read_file` call it answers; a pinned skill message carries its own.
        if isinstance(message, ToolMessage) and message.status != "error":
            path = read_paths.get(message.tool_call_id)
        elif isinstance(message, HumanMessage) and message.additional_kwargs.get("lc_source") == "pinned_skill":
            marker = message.additional_kwargs.get("skill")
            path = marker.get("path") if isinstance(marker, dict) else None
        else:
            continue
        skill = skills_by_path.get(_normalized_path(path) or "")
        if skill is not None:
            reads.append((index, skill))
    return reads
