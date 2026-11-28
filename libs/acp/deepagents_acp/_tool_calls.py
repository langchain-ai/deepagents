"""How tool calls and plans are shown to ACP clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from acp.schema import (
    AgentPlanUpdate,
    ContentToolCallContent,
    FileEditToolCallContent,
    PlanEntry,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)
from deepagents.backends.utils import validate_path

from deepagents_acp.utils import format_execute_result

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from acp.schema import ToolKind
    from langchain_core.messages import AIMessage, ToolMessage

_ChunkKey: TypeAlias = "tuple[tuple[str, ...], int]"
_Chunks: TypeAlias = "dict[_ChunkKey, dict[str, str]]"
"""Streamed tool-call fragments by `(namespace, index)`, each with its `id`, `name`, and `args`.

An entry is dropped once its call starts.
"""

_Updates: TypeAlias = "Iterator[ToolCallStart | ToolCallProgress | AgentPlanUpdate]"
"""Updates from the trackers below, which the live stream and session replay share.

Do not collect them into a list: each tracker changes `calls` between yields, so a
send that fails leaves the calls after it unrecorded.
"""

_TOOL_KINDS: dict[str, ToolKind] = {
    "read_file": "read",
    "edit_file": "edit",
    "write_file": "edit",
    "delete": "delete",
    "ls": "search",
    "glob": "search",
    "grep": "search",
    "execute": "execute",
    "write_todos": "think",
    "fetch_url": "fetch",
    "web_search": "fetch",
    "http_request": "fetch",
}
"""ACP tool kinds for built-in Deep Agents tools; anything else is `other`."""


@dataclass(frozen=True, slots=True)
class ToolCallDisplay:
    """How to show one tool call instead of the default; see `tool_call_display`."""

    title: str | None = None
    """Replaces the default title, which is the tool's name for most tools."""
    meta: Mapping[str, Any] | None = None
    """Sent as the call's ACP `_meta`."""


@dataclass
class _ToolCall:
    """A tool call the client has been told about during one turn."""

    name: str
    args: dict[str, Any]
    finished: bool = False
    linked: bool = False
    """Whether a permission request already used this call's id."""
    scope: tuple[str, ...] = ()
    """The agent that made the call: `()` for the main agent, else the subagent's namespace root."""


def _tool_title(name: str, args: Mapping[str, Any]) -> str:
    """Return the client-facing title for a tool call, naming the path it really uses."""
    path = _tool_path(name, args)
    titles = {"read_file": "Read", "edit_file": "Edit", "write_file": "Write", "delete": "Delete"}
    if name in titles and path:
        return f"{titles[name]} `{path}`"
    if name == "execute":
        command = args.get("command")
        return command if isinstance(command, str) and command else "Execute command"
    return name


def _tool_path(name: str, args: Mapping[str, Any]) -> str | None:
    """Return the file or directory a built-in tool touches, if any.

    Normalized exactly as the filesystem tools do, so a relative path shows as
    the path the tool uses (rooted at `/`), never as one under the session cwd.
    Clients approve edits by these locations.
    """
    key = "file_path" if name in {"read_file", "edit_file", "write_file", "delete"} else "path"
    path = args.get(key) if _TOOL_KINDS.get(name) in {"read", "edit", "delete", "search"} else None
    if not isinstance(path, str) or not path:
        return None
    try:
        return validate_path(path)
    except ValueError:
        return None


def _tool_call_result(tool_id: str, call: _ToolCall, message: ToolMessage) -> ToolCallProgress:
    """Build a tool call's final update from its result message."""
    failed = message.status == "error"
    text = message.text
    if call.name == "execute":
        text = format_execute_result(command=str(call.args.get("command", "")), result=text)
    # A successful edit keeps the diff it started with instead of the tool's summary.
    keep_diff = call.name in {"edit_file", "write_file"} and not failed
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=tool_id,
        status="failed" if failed else "completed",
        content=None
        if keep_diff
        else [
            ContentToolCallContent(type="content", content=TextContentBlock(type="text", text=text))
        ],
    )


def _create_tool_call_start(
    tool_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
) -> ToolCallStart:
    """Create the `tool_call` update: title, kind, locations, and edit diffs."""
    location = _tool_path(tool_name, tool_args)
    content: (
        list[ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent] | None
    ) = None
    if location and tool_name == "edit_file":
        old, new = tool_args.get("old_string"), tool_args.get("new_string")
        if isinstance(old, str) and isinstance(new, str):
            content = [
                FileEditToolCallContent(type="diff", path=location, old_text=old, new_text=new)
            ]
    elif location and tool_name == "write_file":
        text = tool_args.get("content")
        if isinstance(text, str):
            content = [FileEditToolCallContent(type="diff", path=location, new_text=text)]
    return ToolCallStart(
        session_update="tool_call",
        tool_call_id=tool_id,
        title=_tool_title(tool_name, tool_args),
        name=tool_name,
        kind=_TOOL_KINDS.get(tool_name, "other"),
        status="pending",
        content=content,
        locations=[ToolCallLocation(path=location)] if location else None,
        raw_input=tool_args,
    )


def _plan_update(todos: list[dict[str, Any]]) -> AgentPlanUpdate:
    """Build the plan update for a `write_todos` list; an unknown status shows as pending."""
    entries = []
    for todo in todos:
        status = todo.get("status", "pending")
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        entries.append(PlanEntry(content=todo.get("content", ""), status=status, priority="medium"))
    return AgentPlanUpdate(session_update="plan", entries=entries)


def _start_tool_call(calls: dict[str, _ToolCall], tool_id: str, call: _ToolCall) -> _Updates:
    """Record a tool call and yield the updates that tell the client it started."""
    calls[tool_id] = call
    yield _create_tool_call_start(tool_id, call.name, call.args)
    if call.name == "write_todos":
        yield _plan_update(call.args.get("todos", []))


def _start_message_tool_calls(
    calls: dict[str, _ToolCall], message: AIMessage, scope: tuple[str, ...] = ()
) -> _Updates:
    """Start the tool calls of a finished model message that are not started yet."""
    for tool_call in message.tool_calls:
        tool_id = tool_call.get("id")
        if tool_id and tool_id not in calls:
            args = tool_call.get("args")
            call = _ToolCall(
                name=tool_call["name"], args=args if isinstance(args, dict) else {}, scope=scope
            )
            yield from _start_tool_call(calls, tool_id, call)


def _add_chunks(chunks: _Chunks, message_chunk: Any, namespace: tuple[str, ...]) -> list[_ChunkKey]:
    """Add a streamed message chunk's tool-call fragments; return the keys they extended."""
    fragments = getattr(message_chunk, "tool_call_chunks", None)
    if isinstance(message_chunk, str) or not fragments:
        return []
    touched = []
    for fragment in fragments:
        key = (namespace, fragment.get("index") or 0)
        chunk_id, name = fragment.get("id"), fragment.get("name")
        acc = chunks.get(key)
        if chunk_id and name and (acc is None or acc["id"] != chunk_id):
            acc = chunks[key] = {"id": chunk_id, "name": name, "args": ""}
        if acc is not None and fragment.get("args"):
            acc["args"] += fragment["args"]
            touched.append(key)
    return touched


def _start_streamed_tool_calls(
    chunks: _Chunks,
    calls: dict[str, _ToolCall],
    message_chunk: Any,  # a streamed message chunk, or plain text
    namespace: tuple[str, ...] = (),
) -> _Updates:
    """Start streamed tool calls as soon as their arguments parse.

    Calls whose arguments never parse while streaming (a tool with no
    arguments streams `""`) start from the finished model message instead.
    """
    for key in _add_chunks(chunks, message_chunk, namespace):
        acc = chunks.get(key)
        if acc is None:
            continue
        if acc["id"] in calls:
            del chunks[key]
            continue
        # Do not parse on every fragment: that reparses a long argument from the
        # start each time. Only a closing brace can complete the object.
        if not acc["args"].rstrip().endswith("}"):
            continue
        try:
            args = json.loads(acc["args"])
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            del chunks[key]
            call = _ToolCall(name=acc["name"], args=args, scope=namespace[:1])
            yield from _start_tool_call(calls, acc["id"], call)


def _finish_tool_call(calls: dict[str, _ToolCall], message: ToolMessage) -> _Updates:
    """Yield the one final update for a tool call when its result arrives."""
    tool_id = message.tool_call_id
    if not tool_id:
        return
    call = calls.get(tool_id)
    if call is None:
        call = _ToolCall(name=message.name or "tool", args={})
        yield from _start_tool_call(calls, tool_id, call)
    if call.finished:
        return
    call.finished = True
    yield _tool_call_result(tool_id, call, message)


def _fail_open_calls(calls: dict[str, _ToolCall]) -> _Updates:
    """Yield a `failed` update for each call that never got a result."""
    for tool_id, call in calls.items():
        if not call.finished:
            call.finished = True
            yield ToolCallProgress(
                session_update="tool_call_update", tool_call_id=tool_id, status="failed"
            )
