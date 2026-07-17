#!/usr/bin/env python3
"""Inspect conversations in the local Deep Agents Code session store."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess  # noqa: S404  # Used only to probe resolved dcode Python launchers.
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.messages import MessageLikeRepresentation

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"


def _has_runtime() -> bool:
    try:
        return (
            importlib.util.find_spec("langgraph.checkpoint.serde.jsonplus") is not None
        )
    except ModuleNotFoundError:
        return False


def _ensure_runtime() -> None:
    if _has_runtime():
        return
    if os.environ.get("DEEPAGENTS_THREAD_INSPECTOR_REEXEC") == "1":
        msg = (
            "The selected Python runtime does not contain Deep Agents Code "
            "dependencies."
        )
        raise SystemExit(msg)

    candidates: list[Path] = []
    for command in ("dcode", "deepagents-code"):
        executable = shutil.which(command)
        if not executable:
            continue
        launcher = Path(executable).resolve()
        try:
            first_line = launcher.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, UnicodeDecodeError, IndexError):
            first_line = ""
        if first_line.startswith("#!"):
            candidates.append(Path(first_line[2:].strip()))
        candidates.extend((launcher.parent / "python", launcher.parent / "python3"))

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.absolute())
        if candidate_key in seen or not candidate.is_file():
            continue
        seen.add(candidate_key)
        check = subprocess.run(  # noqa: S603  # Runs a resolved dcode interpreter.
            [
                str(candidate),
                "-c",
                "from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        if check.returncode == 0:
            env = os.environ.copy()
            env["DEEPAGENTS_THREAD_INSPECTOR_REEXEC"] = "1"
            os.execve(  # noqa: S606  # Replaces this process with that interpreter.
                str(candidate),
                [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
                env,
            )

    msg = "Could not import Deep Agents Code dependencies or locate dcode on PATH."
    raise SystemExit(msg)


def _default_db_path() -> Path:
    explicit = os.environ.get("DEEPAGENTS_SESSIONS_DB")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".deepagents" / ".state" / "sessions.db"


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        msg = f"Sessions database not found: {resolved}"
        raise SystemExit(msg)
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing = {"checkpoints", "writes"} - tables
    if missing:
        conn.close()
        names = ", ".join(sorted(missing))
        msg = f"Not a supported sessions database; missing tables: {names}"
        raise SystemExit(msg)
    return conn


def _resolve_thread_id(conn: sqlite3.Connection, value: str) -> str:
    exact = conn.execute(
        "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '' LIMIT 1",
        (value,),
    ).fetchone()
    if exact:
        return value
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints "
        "WHERE checkpoint_ns = '' AND thread_id LIKE ? ESCAPE '\\' "
        "ORDER BY thread_id LIMIT 11",
        (escaped + "%",),
    ).fetchall()
    matches = [str(row[0]) for row in rows]
    if not matches:
        msg = f"Thread not found: {value}"
        raise SystemExit(msg)
    if len(matches) > 1:
        rendered = "\n".join(f"  {match}" for match in matches[:10])
        msg = f"Thread prefix is ambiguous:\n{rendered}"
        raise SystemExit(msg)
    return matches[0]


def _decode_metadata(value: object) -> dict[str, object]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        return {}
    return {str(key): item for key, item in decoded.items()}


def _thread_summary(
    conn: sqlite3.Connection,
    thread_id: str,
    message_count: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate = conn.execute(
        "SELECT COUNT(*) AS checkpoint_count, "
        "MIN(json_extract(metadata, '$.updated_at')) AS created_at, "
        "MAX(json_extract(metadata, '$.updated_at')) AS updated_at, "
        "MAX(checkpoint_id) AS latest_checkpoint_id "
        "FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ''",
        (thread_id,),
    ).fetchone()
    latest = conn.execute(
        "SELECT metadata FROM checkpoints "
        "WHERE thread_id = ? AND checkpoint_ns = '' "
        "ORDER BY checkpoint_id DESC LIMIT 1",
        (thread_id,),
    ).fetchone()
    metadata = _decode_metadata(latest[0] if latest else None)
    writes_count = conn.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = ? AND checkpoint_ns = ''",
        (thread_id,),
    ).fetchone()[0]
    summary: dict[str, object] = {
        "thread_id": thread_id,
        "agent_name": metadata.get("agent_name"),
        "created_at": aggregate["created_at"],
        "updated_at": aggregate["updated_at"],
        "latest_checkpoint_id": aggregate["latest_checkpoint_id"],
        "checkpoint_count": aggregate["checkpoint_count"],
        "writes_count": writes_count,
        "git_branch": metadata.get("git_branch"),
        "git_commit_sha": metadata.get("git_commit_sha"),
        "cwd": metadata.get("cwd"),
        "repository_name": metadata.get("repository_name"),
        "repository_url": metadata.get("repository_url"),
    }
    if message_count is not None:
        summary["message_count"] = message_count
    return summary, metadata


def _list_threads(conn: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    rows = conn.execute(
        "SELECT thread_id, "
        "MAX(json_extract(metadata, '$.updated_at')) AS updated_at, "
        "MIN(json_extract(metadata, '$.updated_at')) AS created_at, "
        "MAX(json_extract(metadata, '$.agent_name')) AS agent_name, "
        "MAX(json_extract(metadata, '$.git_branch')) AS git_branch, "
        "MAX(json_extract(metadata, '$.cwd')) AS cwd, "
        "COUNT(*) AS checkpoint_count "
        "FROM checkpoints WHERE checkpoint_ns = '' "
        "GROUP BY thread_id ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def _reconstruct_messages(
    conn: sqlite3.Connection, thread_id: str
) -> list[MessageLikeRepresentation]:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    rows = conn.execute(
        "SELECT checkpoint_id, task_id, idx, type, value FROM writes "
        "WHERE thread_id = ? AND checkpoint_ns = '' AND channel = 'messages' "
        "ORDER BY checkpoint_id ASC, task_id ASC, idx ASC",
        (thread_id,),
    ).fetchall()
    serde = JsonPlusSerializer()
    messages: list[MessageLikeRepresentation] = []
    for row in rows:
        type_name = row["type"]
        value = row["value"]
        if not type_name or value is None:
            continue
        delta = serde.loads_typed((type_name, value))
        if isinstance(delta, Overwrite):
            # Message-channel overwrites contain a list; ignore malformed values.
            messages = (
                list(cast("list[MessageLikeRepresentation]", delta.value))
                if isinstance(delta.value, list)
                else []
            )
        else:
            # `add_messages` normalizes both inputs to a list despite its broad alias.
            messages = cast(
                "list[MessageLikeRepresentation]", add_messages(messages, delta)
            )
    return messages


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            phase = block.get("phase")
            if block_type in {"reasoning", "thinking"} or phase == "analysis":
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            nested = block.get("content")
            if isinstance(nested, str):
                parts.append(nested)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


def _bounded_value(value: object, limit: int) -> tuple[object, bool]:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        rendered = str(value)
    if len(rendered) <= limit:
        return value, False
    return rendered[:limit] + "…", True


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "unknown")
    type_name = getattr(message, "type", type(message).__name__)
    return {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }.get(type_name, str(type_name))


def _message_record(index: int, message: object, max_content: int) -> dict[str, object]:
    if isinstance(message, dict):
        content = message.get("content")
        name = message.get("name")
        message_id = message.get("id")
        raw_tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")
        status = message.get("status")
    else:
        content = getattr(message, "content", None)
        name = getattr(message, "name", None)
        message_id = getattr(message, "id", None)
        raw_tool_calls = getattr(message, "tool_calls", None)
        tool_call_id = getattr(message, "tool_call_id", None)
        status = getattr(message, "status", None)
    tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []

    text = _content_text(content)
    bounded_text, content_truncated = _truncate(text, max_content)
    record: dict[str, object] = {
        "index": index,
        "role": _message_role(message),
        "name": name,
        "id": message_id,
        "content": bounded_text,
        "content_chars": len(text),
        "content_truncated": content_truncated,
    }
    if tool_calls:
        rendered_calls: list[dict[str, object]] = []
        for call in tool_calls:
            if isinstance(call, dict):
                args, args_truncated = _bounded_value(call.get("args"), max_content)
                rendered_calls.append(
                    {
                        "name": call.get("name"),
                        "id": call.get("id"),
                        "args": args,
                        "args_truncated": args_truncated,
                    }
                )
            else:
                rendered_calls.append({"value": str(call)})
        record["tool_calls"] = rendered_calls
    if tool_call_id:
        record["tool_call_id"] = tool_call_id
    if status:
        record["status"] = status
    return record


def _is_user_message(message: object) -> bool:
    role = _message_role(message)
    if role not in {"user", "human"}:
        return False
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    return not _content_text(content).startswith("[SYSTEM]")


def _turns(
    messages: Sequence[object], max_content: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    starts = [
        index for index, message in enumerate(messages) if _is_user_message(message)
    ]
    if not starts:
        return [], [
            _message_record(index, message, max_content)
            for index, message in enumerate(messages)
        ]
    preamble = [
        _message_record(index, message, max_content)
        for index, message in enumerate(messages[: starts[0]])
    ]
    turns: list[dict[str, object]] = []
    for turn_index, start in enumerate(starts):
        end = starts[turn_index + 1] if turn_index + 1 < len(starts) else len(messages)
        turns.append(
            {
                "number": turn_index + 1,
                "start_message_index": start,
                "end_message_index": end - 1,
                "messages": [
                    _message_record(index, messages[index], max_content)
                    for index in range(start, end)
                ],
            }
        )
    return turns, preamble


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Deep Agents Code thread state without modifying the database."
        )
    )
    parser.add_argument("thread_id", nargs="?", help="Full thread ID or unique prefix")
    parser.add_argument(
        "--db", type=Path, default=_default_db_path(), help="Path to sessions.db"
    )
    parser.add_argument(
        "--mode",
        choices=("summary", "latest-turn", "transcript"),
        default="latest-turn",
    )
    parser.add_argument(
        "--list", type=int, metavar="N", dest="list_limit", help="List recent threads"
    )
    parser.add_argument(
        "--max-content",
        type=int,
        default=4000,
        help="Maximum characters retained per message or tool-call argument",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include latest checkpoint metadata",
    )
    return parser


def main() -> None:
    """Parse arguments, inspect the session store, and write JSON to stdout.

    Raises:
        SystemExit: If arguments or the local session store are invalid.
    """
    args = _build_parser().parse_args()
    if args.max_content < 1:
        msg = "--max-content must be positive"
        raise SystemExit(msg)
    if args.list_limit is not None and args.list_limit < 1:
        msg = "--list must be positive"
        raise SystemExit(msg)
    if args.list_limit is None and not args.thread_id:
        msg = "Provide a thread ID or use --list N"
        raise SystemExit(msg)
    if args.list_limit is not None and args.thread_id:
        msg = "Use either a thread ID or --list N, not both"
        raise SystemExit(msg)

    _ensure_runtime()
    warnings.filterwarnings(
        "ignore",
        message=(
            "Core Pydantic V1 functionality isn't compatible with Python 3.14 "
            "or greater.*"
        ),
    )
    conn = _connect_read_only(args.db)
    try:
        if args.list_limit is not None:
            result: dict[str, object] = {
                "database": str(args.db.expanduser().resolve()),
                "threads": _list_threads(conn, args.list_limit),
            }
        else:
            thread_id = _resolve_thread_id(conn, args.thread_id)
            messages = _reconstruct_messages(conn, thread_id)
            summary, metadata = _thread_summary(conn, thread_id, len(messages))
            result = {
                "database": str(args.db.expanduser().resolve()),
                "thread": summary,
            }
            if args.include_metadata:
                result["latest_metadata"] = metadata
            if args.mode != "summary":
                turns, preamble = _turns(messages, args.max_content)
                result["turn_count"] = len(turns)
                if args.mode == "latest-turn":
                    latest_turn = turns[-1] if turns else None
                    result["latest_turn"] = latest_turn
                    if latest_turn is not None:
                        latest_turn["stored_turn_number"] = metadata.get("turn_number")
                        latest_turn["turn_id"] = metadata.get("turn_id")
                else:
                    result["preamble"] = preamble
                    result["turns"] = turns
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
