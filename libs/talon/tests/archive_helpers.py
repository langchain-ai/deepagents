from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_talon.archive_saver import ConversationSaver
from deepagents_talon.config import TalonConfig
from deepagents_talon.history_backends import _sqlite_store, open_history
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.store_archive import StoreConversationArchive

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from langgraph.checkpoint.memory import InMemorySaver


def open_archive(path: str | Path) -> AbstractAsyncContextManager[StoreConversationArchive]:
    path = Path(path).resolve()
    return open_history(
        TalonConfig(
            assistant_id="test",
            home=path.parent,
            env={"DEEPAGENTS_TALON_HISTORY_URI": path.as_uri()},
        )
    )


@asynccontextmanager
async def make_saver(
    path: str | Path,
    backend: type[InMemorySaver | AsyncSqliteSaver] = AsyncSqliteSaver,
) -> AsyncIterator[ConversationSaver]:
    async with AsyncExitStack() as stack:
        checkpointer = (
            await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(path)))
            if backend is AsyncSqliteSaver
            else backend()
        )
        archive = await stack.enter_async_context(open_archive(path))
        yield ConversationSaver(checkpointer, archive=archive)


def make_runtime(saver: ConversationSaver, directory: Path) -> DeepAgentRuntime:
    return DeepAgentRuntime(
        model="test:model",
        checkpointer=saver,
        assistant_dir=directory,
        include_web_tools=False,
        skills=(),
        memory=(),
    )


@asynccontextmanager
async def open_vector_archive(path, *, store=None, vector_search=True, search_visibility="unknown"):
    async with (
        _sqlite_store(Path(path).as_uri()) as metadata,
        StoreConversationArchive(
            metadata,
            namespace=("talon", "test"),
            vector_store=store,
            vector_search=vector_search,
            search_visibility=search_visibility,
        ).open() as archive,
    ):
        yield archive
