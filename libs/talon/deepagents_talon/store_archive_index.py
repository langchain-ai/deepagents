"""Portable Store archive operations for the shared vector indexing worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents_talon.history_index import VectorArchive
from deepagents_talon.store_records import digest, number, scope_key

if TYPE_CHECKING:
    from deepagents_talon.archive import ArchiveEntry, ArchiveScope
    from deepagents_talon.history_index import Row
    from deepagents_talon.store_archive import StoreConversationArchive
    from deepagents_talon.store_records import Record


class StoreVectorArchive(VectorArchive):
    """Reconcile immutable archive entries instead of maintaining a SQL queue."""

    def __init__(self, archive: StoreConversationArchive) -> None:
        """Track a bounded scan of unacknowledged archive entries."""
        self.archive = archive
        self.records = archive.records
        self.indexed = 0
        self.scanned = 0
        self.cataloged = 0
        self.cleanup_end = 0

    async def prepare(self) -> str:
        """Resume acknowledged progress against the archive's durable vector Store."""
        async with self.records.access():
            root = await self.records.root()
            if self.archive.vector_search and root.get("semantic_policy") != 1:
                root = {
                    **root,
                    "semantic_policy": 1,
                    "semantic_cleanup_end": number(root, "indexed"),
                    "indexed": 0,
                    "vector_content_cursor": 0,
                }
            self.indexed = number(root, "indexed")
            self.cleanup_end = number(root, "semantic_cleanup_end")
            self.scanned = self.indexed
            self.cataloged = number(root, "vector_content_cursor")
            await self.records.commit([("root", {**root, "vectors": True})])
        return str(root["identity"])

    async def _deletion(self, session_id: str) -> Record | None:
        if session_id:
            return await self.archive.session(session_id)
        root = await self.records.root()
        if not root["deletions"]:
            return None
        async for _, deletion in self.records.chain(number(root, "deleting"), "previous_deleting"):
            session = await self.records.get(str(number(deletion, "owner")))
            if session is not None and session.get("kind") == "session" and session.get("deleting"):
                if number(session, "delete_cursor"):
                    return session
                await self.archive.delete_text(str(session["session_id"]))
        return None

    async def rows(self, session: str, *, indexing: bool, limit: int) -> list[Row]:
        """Read deletion work first, then a bounded range of durable sequence slots."""
        async with self.records.access():
            deleting = await self._deletion(session)
            if deleting is not None:
                return await self._deleted_rows(deleting, limit)
            if session or not indexing:
                return []
            if self.cataloged < self.indexed:
                await self._backfill(limit)
                return []
            last = number(await self.records.root(), "last")
            self.scanned = min(last, self.indexed + limit)
            rows: list[Row] = []
            for cursor in range(self.indexed + 1, self.scanned + 1):
                record = await self.records.get(str(cursor))
                if record is not None and (row := await self._index_row(cursor, record)):
                    rows.append(row)
            if not rows and self.scanned < last:
                self._wake()
            return rows

    async def _index_row(self, cursor: int, record: Record) -> Row | None:
        scanned = cursor
        if record.get("kind") == "index-request":
            cursor = number(record, "target")
            record = await self.records.get(str(cursor)) or {}
        if record.get("kind") != "chunk":
            return None
        owner = await self.records.get(str(number(record, "owner")))
        if owner is None or owner.get("deleting"):
            return None
        if not record.get("indexable"):
            if scanned <= self.cleanup_end:
                return self._row(cursor, record, owner, deleted=2)
            return None
        canonical = await self._canonical(cursor, record)
        if canonical == cursor and scanned > self.cleanup_end:
            return self._row(cursor, record, owner, deleted=0)
        return None

    def _wake(self) -> None:
        if self.archive.vectors is not None:
            self.archive.vectors.wake.set()

    async def _canonical(self, cursor: int, record: Record) -> int:
        """Journal a session-scoped content identity and its deletion reference."""
        entry = cast("ArchiveEntry", record["entry"])
        key = "vector-content:" + digest(entry["session_id"], entry["text"])
        existing = await self.records.get(key)
        if existing is not None:
            target = await self.records.get(str(number(existing, "cursor")))
            if target is not None and target.get("indexable"):
                return number(existing, "cursor")
        await self.records.commit(
            [
                (key, {"cursor": cursor}),
                (str(cursor), {**record, "vector_content": key}),
            ]
        )
        return cursor

    async def _backfill(self, limit: int) -> None:
        """Catalog legacy vectors incrementally without embedding their text again."""
        stop = min(self.indexed, self.cataloged + limit)
        for cursor in range(self.cataloged + 1, stop + 1):
            record = await self.records.get(str(cursor))
            if record is not None and record.get("kind") == "chunk" and record.get("indexable"):
                owner = await self.records.get(str(number(record, "owner")))
                if owner is not None and not owner.get("deleting"):
                    await self._canonical(cursor, record)
        root = await self.records.root()
        await self.records.commit([("root", {**root, "vector_content_cursor": stop})])
        self.cataloged = stop
        self._wake()

    async def _deleted_rows(self, session: Record, limit: int) -> list[Row]:
        rows: list[Row] = []
        async for cursor, record in self.records.chain(
            number(session, "delete_cursor"), "previous_session"
        ):
            rows.append(self._row(cursor, record, session, deleted=True))
            if len(rows) == limit:
                break
        return rows

    @staticmethod
    def _row(cursor: int, record: Record, owner: Record, *, deleted: int) -> Row:
        entry = cast("ArchiveEntry", record["entry"])
        scope = cast("ArchiveScope", owner["scope"])
        return (
            cursor,
            scope["talon_history_channel"],
            scope["talon_history_chat"],
            entry["session_id"],
            int(deleted),
            entry["text"],
        )

    async def acknowledge(self, rows: list[Row]) -> None:
        """Advance only acknowledged work, with deletion progress durable across restarts."""
        async with self.records.access():
            if (not rows or rows[0][4] != 1) and self.scanned > self.indexed:
                root = await self.records.root()
                await self.records.commit(
                    [
                        (
                            "root",
                            {
                                **root,
                                "indexed": self.scanned,
                                "vector_content_cursor": max(self.cataloged, self.scanned),
                                "semantic_cleanup_end": self.cleanup_end
                                if self.scanned < self.cleanup_end
                                else 0,
                            },
                        )
                    ]
                )
                self.indexed = self.scanned
                self.cataloged = max(self.cataloged, self.scanned)
                if self.scanned >= self.cleanup_end:
                    self.cleanup_end = 0
            for cursor, _, _, session_id, deleted, _ in rows:
                if deleted == 1:
                    session = cast("Record", await self.archive.session(session_id))
                    record = cast("Record", await self.records.get(str(cursor)))
                    await self.records.commit(
                        [
                            (
                                str(session["cursor"]),
                                {
                                    **session,
                                    "delete_cursor": number(record, "previous_session"),
                                },
                            )
                        ]
                    )

    async def mark_deleted(self, session: str) -> None:
        """Persist deletion intent before vector removal."""
        async with self.records.access():
            await self.archive.mark_deleted(session)

    async def delete_text(self, session: str) -> None:
        """Remove text and registrations after all vectors have been acknowledged."""
        async with self.records.access():
            await self.archive.delete_text(session)

    async def pending(self, scope: ArchiveScope) -> bool:
        """Report source records that have not yet been reconciled with the Store."""
        async with self.records.access():
            scoped = await self.records.get(scope_key(scope)) or {}
            if max(number(scoped, "head"), number(scoped, "vector_head")) > self.indexed:
                return True
            async for _, session in self.records.chain(
                number(scoped, "sessions"), "previous_scope"
            ):
                if session.get("deleting"):
                    return True
            return False

    async def lexical(self, scope: ArchiveScope, query: str, limit: int) -> list[str]:
        """Find literal keyword candidates without backend-specific full-text operators.

        Stops at the archive's scan budget rather than raising: these candidates are
        fused with the semantic ranking, so a short list degrades recall while a
        raised error would fail a search the semantic leg had already answered.
        """
        entries = await self.archive.text_entries(
            scope,
            query=query,
            session_id="",
            after=0,
            limit=limit,
            partial=True,
        )
        return [str(entry["cursor"]) for entry in entries]

    async def semantic(self, scope: ArchiveScope, keys: list[str]) -> list[str]:
        """Revalidate eligibility while a previous index is being cleaned up.

        Args:
            scope: Trusted chat scope.
            keys: Bounded vector-search candidate identifiers.
        """
        async with self.records.access():
            eligible: list[str] = []
            for key in keys:
                record = await self.records.get(key)
                if record and record.get("indexable") and await self.archive.visible(record, scope):
                    eligible.append(key)
            return eligible

    async def ranked(
        self,
        scope: ArchiveScope,
        keys: list[str],
        after: int,
        limit: int,
    ) -> list[ArchiveEntry]:
        """Read and authorize only bounded vector candidates, preserving their rank."""
        if after:
            if str(after) not in keys:
                return []
            keys = keys[keys.index(str(after)) + 1 :]
        async with self.records.access():
            entries: list[ArchiveEntry] = []
            for key in keys:
                record = await self.records.get(key)
                entry = await self.archive.visible(record, scope) if record else None
                if entry is not None:
                    entries.append(entry)
                    if len(entries) == limit:
                        break
            return entries
