"""Conversation archives on any durable LangGraph key/value Store.

Warning:
    Experimental API; subject to change with the Talon runtime.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections import deque
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from deepagents_talon.history_messages import message_revisions
from deepagents_talon.store_records import Record, StoreRecords, Write, digest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from langchain_core.messages import BaseMessage
    from langgraph.store.base import BaseStore

    from deepagents_talon.history import ArchiveEntry, ArchiveScope, ConversationSummary

_MAX_PAGE_SIZE = 20


def number(record: Record, key: str) -> int:
    """Read a numeric ordering field from a versioned archive record."""
    return cast("int", record.get(key, 0))


def scope_key(scope: ArchiveScope) -> str:
    """Encode a trusted chat scope as a backend-independent lookup key."""
    return "scope:" + digest(scope["talon_history_channel"], scope["talon_history_chat"])


def _bounds(after: int, limit: int) -> None:
    if not 1 <= limit <= _MAX_PAGE_SIZE or after < 0:
        msg = "limit must be between 1 and 20 and after must be non-negative"
        raise ValueError(msg)


def _chunks(messages: Sequence[BaseMessage], timestamp: str) -> Iterator[Record]:
    for message in message_revisions(messages, timestamp):
        for part, text in message.chunks():
            yield {
                "message_id": message.message_id,
                "revision": message.revision,
                "part": part,
                "role": message.role,
                "text": text,
                "search_text": message.text if part == 0 else "",
            }


class StoreConversationArchive:
    """Store transcripts, summaries, and recovery state without database-specific SQL.

    Use one active writer per namespace, a metadata Store with read-after-write
    consistency and no embeddings or TTL, with caller-owned connections.
    Cursor links and a bounded redo journal avoid assumptions about search order
    or atomic batches. Keyword search scans the chat's transcript chunks.

    Args:
        store: Initialized, caller-owned metadata Store without vector indexing.
        namespace: Stable identity separating assistants sharing a database.
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        namespace: tuple[str, ...],
    ) -> None:
        """Keep ownership of the Store connection with the caller."""
        self.records = StoreRecords(store, namespace)
        self._ready = False
        self._closed = False
        self._setup_lock = asyncio.Lock()

    @asynccontextmanager
    async def open(self) -> AsyncIterator[StoreConversationArchive]:
        """Initialize the archive while leaving its caller-owned Store open."""
        try:
            await self.setup()
            yield self
        finally:
            await self.aclose()

    async def setup(self) -> None:
        """Recover interrupted writes before enabling retrieval."""
        async with self._setup_lock:
            if self._closed:
                msg = "Conversation archive is closed; create a new instance to reopen it"
                raise RuntimeError(msg)
            if self._ready:
                return
            async with self.records.access():
                root = await self.records.root()
                await self.records.commit([("root", root)])
            self._ready = True

    async def aclose(self) -> None:
        """Close the archive without closing its caller-owned Store."""
        self._closed = True

    async def session(self, session_id: str) -> Record | None:
        """Read the authoritative session registration under the records lock."""
        lookup = await self.records.get("session:" + digest(session_id))
        return await self.records.get(str(number(lookup, "cursor"))) if lookup else None

    async def _register(self, scope: ArchiveScope, session_id: str, timestamp: str) -> Record:
        session = await self.session(session_id)
        if session is not None:
            if session["scope"] != scope or session.get("deleting"):
                msg = "Session belongs to another scope or is being deleted"
                raise ValueError(msg)
            return session
        root = await self.records.root()
        scoped = await self.records.get(scope_key(scope)) or {}
        cursor = number(root, "last") + 1
        session = {
            "kind": "session",
            "cursor": cursor,
            "session_id": session_id,
            "scope": dict(scope),
            "previous_global": number(root, "sessions"),
            "previous_scope": number(scoped, "sessions"),
            "head": 0,
            "started_at": timestamp,
            "updated_at": timestamp,
            "message_count": 0,
            "preview": "",
        }
        await self.records.commit(
            [
                (str(cursor), session),
                ("session:" + digest(session_id), {"cursor": cursor}),
                (scope_key(scope), {**scoped, "sessions": cursor}),
                ("root", {**root, "last": cursor, "sessions": cursor}),
            ]
        )
        return session

    async def append(
        self,
        scope: ArchiveScope,
        session_id: str,
        timestamp: str,
        messages: Sequence[BaseMessage],
    ) -> None:
        """Append idempotent message revisions within a trusted session scope.

        Args:
            scope: Host-supplied channel and chat identity.
            session_id: Checkpointer thread whose ownership must match the scope.
            timestamp: Checkpoint timestamp used for fallback message identities.
            messages: Committed messages; an empty list only registers ownership.
        """
        await self.setup()
        async with self.records.access():
            await self._register(scope, session_id, timestamp)
            for chunk in _chunks(messages, timestamp):
                await self._append_chunk(scope, session_id, timestamp, chunk)

    async def _append_chunk(
        self, scope: ArchiveScope, session_id: str, timestamp: str, chunk: Record
    ) -> None:
        identifier, revision = str(chunk["message_id"]), str(chunk["revision"])
        key = "dedup:" + digest(session_id, identifier, revision, str(chunk["part"]))
        if await self.records.get(key) is not None:
            return
        session = cast("Record", await self.session(session_id))
        root, scoped = await self.records.root(), await self.records.get(scope_key(scope)) or {}
        cursor = number(root, "last") + 1
        entry = {
            "cursor": cursor,
            "session_id": session_id,
            "timestamp": timestamp,
            **{field: chunk[field] for field in ("role", "message_id", "part", "text")},
        }
        message_key = "message:" + digest(session_id, identifier)
        fresh = await self.records.get(message_key) is None
        record = {
            "kind": "chunk",
            "entry": entry,
            "owner": session["cursor"],
            "previous_scope": number(scoped, "head"),
            "previous_session": number(session, "head"),
            "dedup": key,
            "message": message_key,
            "search_text": chunk["search_text"],
        }
        updated = self._summary_update(session, cursor, timestamp, str(chunk["text"]), fresh=fresh)
        await self.records.commit(
            [
                (str(cursor), record),
                (str(session["cursor"]), updated),
                (key, {"cursor": cursor}),
                (message_key, {"owner": session["cursor"]}),
                (scope_key(scope), {**scoped, "head": cursor}),
                ("root", {**root, "last": cursor}),
            ]
        )

    @staticmethod
    def _summary_update(
        session: Record, cursor: int, timestamp: str, text: str, *, fresh: bool
    ) -> Record:
        return {
            **session,
            "head": cursor,
            "started_at": min(str(session["started_at"]), timestamp)
            if number(session, "head")
            else timestamp,
            "updated_at": max(str(session["updated_at"]), timestamp)
            if number(session, "head")
            else timestamp,
            "message_count": number(session, "message_count") + int(fresh),
            "preview": session["preview"] if number(session, "head") else text[:300],
        }

    async def visible(self, record: Record, scope: ArchiveScope) -> ArchiveEntry | None:
        """Authorize a transcript against its live session ownership."""
        if record.get("kind") != "chunk":
            return None
        owner = await self.records.get(str(number(record, "owner")))
        entry = cast("ArchiveEntry", record["entry"])
        if (
            owner is None
            or owner.get("kind") != "session"
            or owner.get("deleting")
            or owner["scope"] != scope
            or owner["session_id"] != entry["session_id"]
        ):
            return None
        return entry

    async def _text_entries(
        self,
        scope: ArchiveScope,
        *,
        query: str,
        session_id: str,
        after: int,
        limit: int,
    ) -> list[ArchiveEntry]:
        async with self.records.access():
            if session_id:
                session = await self.session(session_id)
                if session is None or session["scope"] != scope:
                    return []
                cursor, link = number(session, "head"), "previous_session"
            else:
                scoped = await self.records.get(scope_key(scope)) or {}
                cursor, link = number(scoped, "head"), "previous_scope"
            hits: deque[ArchiveEntry] = deque(maxlen=limit)
            async for identifier, record in self.records.chain(cursor, link):
                if (session_id and identifier <= after) or (
                    not session_id and after and identifier >= after
                ):
                    continue
                entry = await self.visible(record, scope)
                if (
                    entry is not None
                    and (not session_id or entry["session_id"] == session_id)
                    and _matches(str(record["search_text"]), query)
                ):
                    hits.append(entry)
                    if not session_id and len(hits) == limit:
                        break
            return list(reversed(hits)) if session_id else list(hits)

    async def entries(
        self,
        scope: ArchiveScope,
        *,
        query: str = "",
        session_id: str = "",
        after: int = 0,
        limit: int = 5,
    ) -> list[ArchiveEntry]:
        """Read or search bounded, scoped transcripts.

        Args:
            scope: Trusted channel and chat identity.
            query: Literal words to match against complete message revisions.
            session_id: Read this session chronologically when supplied.
            after: Previous result cursor.
            limit: Maximum result count, from 1 to 20.
        """
        _bounds(after, limit)
        await self.setup()
        return await self._text_entries(
            scope, query=query, session_id=session_id, after=after, limit=limit
        )

    async def conversations(
        self,
        scope: ArchiveScope,
        *,
        after: int = 0,
        limit: int = 5,
    ) -> list[ConversationSummary]:
        """List bounded, nonempty sessions within scope.

        Args:
            scope: Trusted channel and chat identity.
            after: Previous session cursor.
            limit: Maximum result count, from 1 to 20.
        """
        _bounds(after, limit)
        await self.setup()
        async with self.records.access():
            scoped = await self.records.get(scope_key(scope)) or {}
            results: list[ConversationSummary] = []
            async for cursor, record in self.records.chain(
                number(scoped, "sessions"), "previous_scope"
            ):
                if (
                    record.get("kind") == "session"
                    and record.get("scope") == scope
                    and record.get("head")
                    and not record.get("deleting")
                    and (not after or cursor < after)
                ):
                    results.append(
                        cast(
                            "ConversationSummary",
                            {
                                key: record[key]
                                for key in (
                                    "cursor",
                                    "session_id",
                                    "started_at",
                                    "updated_at",
                                    "message_count",
                                    "preview",
                                )
                            },
                        )
                    )
                    if len(results) == limit:
                        break
            return results

    async def sessions(self, scope: ArchiveScope) -> list[str]:
        """List owned registrations, including pending resets and empty sessions.

        Args:
            scope: Trusted channel and chat identity.
        """
        await self.setup()
        async with self.records.access():
            scoped = await self.records.get(scope_key(scope)) or {}
            return [
                str(record["session_id"])
                async for _, record in self.records.chain(
                    number(scoped, "sessions"), "previous_scope"
                )
                if record.get("kind") == "session" and record.get("scope") == scope
            ]

    async def delete_session(self, session_id: str) -> None:
        """Delete transcripts, retaining registration on failure for retry.

        Args:
            session_id: Trusted session identifier returned by the archive.
        """
        await self.setup()
        async with self.records.access():
            await self.mark_deleted(session_id)
            await self.delete_text(session_id)

    async def mark_deleted(self, session_id: str) -> None:
        """Persist deletion state while holding the records lock."""
        session = await self.session(session_id)
        if session is None or session.get("deleting"):
            return
        root = await self.records.root()
        await self.records.commit(
            [
                (
                    str(session["cursor"]),
                    {**session, "deleting": True},
                ),
                ("root", {**root, "deletions": number(root, "deletions") + 1}),
            ]
        )

    async def delete_text(self, session_id: str) -> None:
        """Erase text and dedup records under the records lock."""
        session = await self.session(session_id)
        if session is None:
            return
        while number(session, "head"):
            cursor = number(session, "head")
            record = cast("Record", await self.records.get(str(cursor)))
            session = {**session, "head": number(record, "previous_session")}
            links = {key: record[key] for key in ("previous_scope", "previous_session")}
            writes: list[Write] = [(str(cursor), links), (str(session["cursor"]), session)]
            writes.extend((str(record[key]), None) for key in ("dedup", "message"))
            await self.records.commit(writes)
        root = await self.records.root()
        links = {key: session[key] for key in ("previous_scope", "previous_global")}
        scoped = await self.records.get(scope_key(cast("ArchiveScope", session["scope"]))) or {}
        others = await self._other_sessions(scoped, number(session, "cursor"))
        await self.records.commit(
            [
                (scope_key(cast("ArchiveScope", session["scope"])), scoped if others else None),
                (str(session["cursor"]), links),
                ("session:" + digest(session_id), None),
                ("root", {**root, "deletions": number(root, "deletions") - 1}),
            ]
        )

    async def _other_sessions(self, scoped: Record, cursor: int) -> bool:
        async for identifier, record in self.records.chain(
            number(scoped, "sessions"), "previous_scope"
        ):
            if record.get("kind") == "session" and identifier != cursor:
                return True
        return False


def _matches(text: str, query: str) -> bool:
    def normalize(value: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFD", value.casefold())
            if unicodedata.category(char) != "Mn"
        )

    if not query.strip():
        return True
    normalized = normalize(text)
    return all(word in normalized for word in normalize(query).split())
