from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_talon.archive import SQLiteConversationArchive
from deepagents_talon.archive_saver import ConversationSaver
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    import aiosqlite
    from langgraph.checkpoint.memory import InMemorySaver


def make_saver(
    connection: aiosqlite.Connection,
    backend: type[InMemorySaver | AsyncSqliteSaver] = AsyncSqliteSaver,
) -> ConversationSaver:
    return ConversationSaver(
        backend(connection) if backend is AsyncSqliteSaver else backend(),
        archive=SQLiteConversationArchive(connection),
    )


def make_runtime(saver: ConversationSaver, directory: Path) -> DeepAgentRuntime:
    return DeepAgentRuntime(
        model="test:model",
        checkpointer=saver,
        assistant_dir=directory,
        include_web_tools=False,
        skills=(),
        memory=(),
    )
