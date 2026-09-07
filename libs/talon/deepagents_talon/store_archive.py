"""Conversation archives on any durable LangGraph key/value Store.

Warning:
    Experimental API; subject to change with the Talon runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import unicodedata
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_talon.archive import CHUNK_SIZE, _search_page
from deepagents_talon.store_records import Record, StoreRecords, Write, digest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from langchain_core.messages import BaseMessage
    from langgraph.store.base import BaseStore

    from deepagents_talon.archive import (
        ArchiveEntry,
        ArchiveScope,
        ConversationSummary,
        SearchPage,
        SearchVisibility,
    )
    from deepagents_talon.history_profiles import EmbeddingProfile

_MAX_PAGE_SIZE = 20
_MAX_SCAN = 500
_MAX_SEARCH_PAGES = 32


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
    for index, message in enumerate(messages):
        if not isinstance(message, (HumanMessage, AIMessage, ToolMessage)):
            continue
        if isinstance(message, ToolMessage) and message.name in {
            "search_conversations",
            "read_conversation",
            "list_conversations",
        }:
            continue
        text = message.text
        if isinstance(message, AIMessage) and message.tool_calls:
            text += "\nTool calls: " + json.dumps(message.tool_calls, ensure_ascii=False)
        revision = hashlib.sha256(text.encode()).hexdigest()
        for part, start in enumerate(range(0, len(text), CHUNK_SIZE)):
            yield {
                "message_id": message.id or f"talon-history:{timestamp}:{index}",
                "revision": revision,
                "part": part,
                "role": message.type,
                "text": text[start : start + CHUNK_SIZE],
                "search_text": text if part == 0 else "",
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
        vector_store: Separate optional Store configured to embed transcript text.
        vector_search: Enable indexing and search, or only vector deletion.
        embedding_profile: Optional embedding identity and resource limits.
        search_visibility: Whether acknowledged vector writes are immediately searchable.
    """

    def __init__(  # noqa: PLR0913  # Preserve existing arguments when adding optional profile settings.
        self,
        store: BaseStore,
        *,
        namespace: tuple[str, ...],
        vector_store: BaseStore | None = None,
        vector_search: bool = True,
        embedding_profile: EmbeddingProfile | None = None,
        search_visibility: SearchVisibility = "unknown",
    ) -> None:
        """Keep ownership of both Store connections with the caller."""
        if store is vector_store:
            msg = "Archive metadata and vector indexing require separate Store instances"
            raise ValueError(msg)
        self.records = StoreRecords(store, namespace)
        self.vector_search = vector_search
        self._pages: OrderedDict[str, tuple[str, str, str, int]] = OrderedDict()
        self._ready = False
        self._closed = False
        self._setup_lock = asyncio.Lock()
        self.vectors = None
        if vector_store is not None:
            from deepagents_talon.history_vectors import HistoryVectorIndex  # noqa: PLC0415
            from deepagents_talon.store_archive_index import StoreVectorArchive  # noqa: PLC0415

            self.vectors = HistoryVectorIndex(
                StoreVectorArchive(self),
                vector_store,
                indexing=vector_search,
                profile=embedding_profile,
                search_visibility=search_visibility,
            )

    @asynccontextmanager
    async def open(self) -> AsyncIterator[StoreConversationArchive]:
        """Initialize and finish workers while leaving caller-owned Stores open."""
        try:
            await self.setup()
            yield self
        finally:
            await self.aclose()

    async def setup(self) -> None:
        """Recover interrupted writes before enabling retrieval or background indexing."""
        async with self._setup_lock:
            if self._closed:
                msg = "Conversation archive is closed; create a new instance to reopen it"
                raise RuntimeError(msg)
            if self._ready:
                return
            async with self.records.access():
                root = await self.records.root()
                await self.records.commit([("root", root)])
            if self.vectors is not None:
                await self.vectors.start()
            self._ready = True

    async def aclose(self) -> None:
        """Wait for the active indexing batch before Store connections can close."""
        if self.vectors is not None:
            await self.vectors.close()
        self._pages.clear()
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
                ("root", {**root, "last": cursor}),
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
        if self.vectors is not None:
            self.vectors.wake.set()

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
            "next_session": 0,
            "dedup": key,
            "message": message_key,
            "search_text": chunk["search_text"],
        }
        updated = self._summary_update(session, cursor, timestamp, str(chunk["text"]), fresh=fresh)
        writes: list[Write] = [
            (str(cursor), record),
            (str(session["cursor"]), updated),
            (key, {"cursor": cursor}),
            (message_key, {"owner": session["cursor"]}),
            (scope_key(scope), {**scoped, "head": cursor}),
            ("root", {**root, "last": cursor}),
        ]
        if previous := number(session, "head"):
            prior = cast("Record", await self.records.get(str(previous)))
            writes.append((str(previous), {**prior, "next_session": cursor}))
        await self.records.commit(writes)

    @staticmethod
    def _summary_update(
        session: Record, cursor: int, timestamp: str, text: str, *, fresh: bool
    ) -> Record:
        return {
            **session,
            "head": cursor,
            "tail": number(session, "tail") or cursor,
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

    async def _text_entries(  # noqa: PLR0913  # Preserve existing arguments when adding partial scans.
        self,
        scope: ArchiveScope,
        *,
        query: str,
        session_id: str,
        after: int,
        limit: int,
        partial: bool = False,
    ) -> list[ArchiveEntry]:
        async with self.records.access():
            if session_id:
                session = await self.session(session_id)
                if session is None or session["scope"] != scope or session.get("deleting"):
                    return []
                cursor, link = number(session, "tail"), "next_session"
            else:
                scoped = await self.records.get(scope_key(scope)) or {}
                cursor, link = number(scoped, "head"), "previous_scope"
            if after:
                previous = await self.records.get(str(after))
                entry = await self.visible(previous, scope) if previous else None
                if entry is None or (session_id and entry["session_id"] != session_id):
                    return []
                cursor = number(cast("Record", previous), link)
                if cursor and (cursor == after or (cursor > after) != bool(session_id)):
                    msg = "Conversation archive contains an invalid ordering link"
                    raise RuntimeError(msg)
            hits: list[ArchiveEntry] = []
            async for _, record in self._retrieval_chain(cursor, link, partial=partial):
                entry = await self.visible(record, scope)
                if (
                    entry is not None
                    and (not session_id or entry["session_id"] == session_id)
                    and _matches(str(record["search_text"]), query)
                ):
                    hits.append(entry)
                    if len(hits) == limit:
                        break
            return hits

    async def _retrieval_chain(
        self, cursor: int, link: str, *, partial: bool = False
    ) -> AsyncIterator[tuple[int, Record]]:
        scanned = 0
        async for identifier, record in self.records.chain(cursor, link):
            yield identifier, record
            scanned += 1
            if scanned == _MAX_SCAN and number(record, link):
                # Ranked retrieval already scores a bounded candidate pool, so a
                # truncated scan only narrows it. A page from entries() or
                # conversations() claims completeness, so truncating one there
                # would report "no more results" while records remain.
                if partial:
                    return
                msg = (
                    "Conversation history scan limit exceeded (500 records); "
                    "no partial page returned"
                )
                raise RuntimeError(msg)

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

        Scans at most 500 ordering records, including skipped and deleted records.

        Args:
            scope: Trusted channel and chat identity.
            query: Literal words to match against complete message revisions.
            session_id: Read this session chronologically when supplied.
            after: Previous result cursor.
            limit: Maximum result count, from 1 to 20.

        Raises:
            RuntimeError: The scan budget is exhausted before a complete page is known.
        """
        _bounds(after, limit)
        await self.setup()
        return await self._text_entries(
            scope, query=query, session_id=session_id, after=after, limit=limit
        )

    async def search_page(
        self,
        scope: ArchiveScope,
        *,
        query: str = "",
        after: str = "",
        limit: int = 5,
    ) -> SearchPage:
        """Search scoped history with explicit retrieval coverage.

        Args:
            scope: Trusted channel and chat identity.
            query: Search text; empty lists recent history.
            after: Opaque continuation token from the previous page.
            limit: Maximum result count, from 1 to 20.
        """
        _bounds(0, limit)
        await self.setup()
        if self.vectors is not None and self.vector_search and query.strip():
            return await self.vectors.search_page(scope, query, after, limit)
        context = (scope["talon_history_channel"], scope["talon_history_chat"], query)
        continuation = self._pages.get(after) if after else None
        status = "disabled" if query.strip() else "not_requested"
        if after and (continuation is None or continuation[:3] != context):
            return _search_page([], limit, status, expired=True)
        cursor = continuation[3] if continuation else 0
        hits = await self._text_entries(
            scope, query=query, session_id="", after=cursor, limit=limit + 1
        )
        page = _search_page(hits, limit, status, expired=bool(after and not hits))
        if page["has_more"]:
            token = uuid4().hex
            self._pages[token] = (*context, page["results"][-1]["cursor"])
            page["next_after"] = token
            if len(self._pages) > _MAX_SEARCH_PAGES:
                self._pages.popitem(last=False)
        return page

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

        Raises:
            RuntimeError: More than 500 ordering records are needed to complete the page.
        """
        _bounds(after, limit)
        await self.setup()
        async with self.records.access():
            scoped = await self.records.get(scope_key(scope)) or {}
            cursor = number(scoped, "sessions")
            if after:
                previous = await self.records.get(str(after))
                if previous is None or previous.get("scope") != scope:
                    return []
                cursor = number(previous, "previous_scope")
                if cursor >= after:
                    msg = "Conversation archive contains an invalid ordering link"
                    raise RuntimeError(msg)
            results: list[ConversationSummary] = []
            async for _, record in self._retrieval_chain(cursor, "previous_scope"):
                if (
                    record.get("kind") == "session"
                    and record.get("scope") == scope
                    and record.get("head")
                    and not record.get("deleting")
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
        """Delete vectors before transcripts, retaining registration on failure.

        Args:
            session_id: Trusted session identifier returned by the archive.
        """
        await self.setup()
        if self.vectors is not None:
            await self.vectors.delete_session(session_id)
        else:
            async with self.records.access():
                if (await self.records.root())["vectors"]:
                    msg = "Reopen the archive with its vector Store to complete history deletion"
                    raise RuntimeError(msg)
                await self.mark_deleted(session_id)
                await self.delete_text(session_id)

    async def mark_deleted(self, session_id: str) -> None:
        """Persist deletion state while holding the records lock."""
        session = await self.session(session_id)
        if session is None or session.get("deleting"):
            return
        root = await self.records.root()
        deletion = number(root, "last") + 1
        previous = number(root, "deleting")
        writes: list[Write] = [
            (
                str(session["cursor"]),
                {
                    **session,
                    "deleting": True,
                    "delete_cursor": session["head"],
                    "delete_marker": deletion,
                },
            ),
            (
                str(deletion),
                # Linked in both directions so a completed marker leaves the chain
                # without a walk; see `_unlink_deletion`.
                {
                    "owner": session["cursor"],
                    "previous_deleting": previous,
                    "next_deleting": 0,
                },
            ),
            (
                "root",
                {
                    **root,
                    "last": deletion,
                    "deletions": number(root, "deletions") + 1,
                    "deleting": deletion,
                },
            ),
        ]
        if previous and (marker := await self.records.get(str(previous))) is not None:
            writes.append((str(previous), {**marker, "next_deleting": deletion}))
        await self.records.commit(writes)

    async def delete_text(self, session_id: str) -> None:
        """Erase text and dedup records after vector deletion, under the records lock."""
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
        deletions = number(root, "deletions") - 1
        markers, deleting = await self._unlink_deletion(session, root, drained=deletions <= 0)
        links = {"previous_scope": session["previous_scope"]}
        scoped = await self.records.get(scope_key(cast("ArchiveScope", session["scope"]))) or {}
        others = await self._other_sessions(scoped, number(session, "cursor"))
        await self.records.commit(
            [
                *markers,
                (scope_key(cast("ArchiveScope", session["scope"])), scoped if others else None),
                (str(session["cursor"]), links),
                ("session:" + digest(session_id), None),
                ("root", {**root, "deletions": deletions, "deleting": deleting}),
            ]
        )

    async def _unlink_deletion(
        self, session: Record, root: Record, *, drained: bool
    ) -> tuple[list[Write], int]:
        """Remove a completed deletion marker, returning its writes and the new chain head.

        The chain exists so an interrupted deletion can be found again after a restart.
        Retaining completed markers would instead record every deletion the assistant
        has ever performed, and the indexing worker re-walks that chain on every poll
        for as long as any one deletion is still outstanding. Archives written before
        markers were linked in both directions have no `delete_marker`; their chain is
        abandoned wholesale once the last outstanding deletion drains.
        """
        deleting = 0 if drained else number(root, "deleting")
        cursor = number(session, "delete_marker")
        marker = await self.records.get(str(cursor)) if cursor else None
        if marker is None:
            return [], deleting
        previous, following = number(marker, "previous_deleting"), number(marker, "next_deleting")
        writes: list[Write] = [(str(cursor), None)]
        if not following and deleting == cursor:
            deleting = previous
        neighbors = (
            (following, "previous_deleting", previous),
            (previous, "next_deleting", following),
        )
        for neighbor, field, value in neighbors:
            if neighbor and (record := await self.records.get(str(neighbor))) is not None:
                writes.append((str(neighbor), {**record, field: value}))
        return writes, deleting

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
