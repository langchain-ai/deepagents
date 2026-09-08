"""Portable Store archive operations for the shared vector indexing worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents_talon.store_archive import number, scope_key

if TYPE_CHECKING:
    from deepagents_talon.archive import ArchiveEntry, ArchiveScope
    from deepagents_talon.history_index import Row
    from deepagents_talon.store_archive import StoreConversationArchive
    from deepagents_talon.store_records import Record


class StoreVectorArchive:
    """Reconcile immutable archive entries instead of maintaining a SQL queue."""

    def __init__(self, archive: StoreConversationArchive) -> None:
        """Track a bounded scan of unacknowledged archive entries."""
        self.archive = archive
        self.records = archive.records
        self.indexed = 0
        self.scanned = 0

    async def prepare(self) -> str:
        """Resume acknowledged progress against the archive's durable vector Store."""
        async with self.records.access():
            root = await self.records.root()
            self.indexed = number(root, "indexed")
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
            last = number(await self.records.root(), "last")
            self.scanned = min(last, self.indexed + limit)
            rows: list[Row] = []
            for cursor in range(self.indexed + 1, self.scanned + 1):
                record = await self.records.get(str(cursor))
                if record is not None and record.get("kind") == "chunk":
                    owner = await self.records.get(str(number(record, "owner")))
                    if owner is not None and not owner.get("deleting"):
                        rows.append(self._row(cursor, record, owner, deleted=False))
            # Empty ranges contain only registrations or content-free deleted slots.
            if not rows:
                self.indexed = self.scanned
                if self.scanned < last and self.archive.vectors is not None:
                    self.archive.vectors.wake.set()
            return rows

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
    def _row(cursor: int, record: Record, owner: Record, *, deleted: bool) -> Row:
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
            if rows and not rows[0][4]:
                root = await self.records.root()
                await self.records.commit([("root", {**root, "indexed": self.scanned})])
                self.indexed = self.scanned
            for cursor, _, _, session_id, deleted, _ in rows:
                if deleted:
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
            if number(scoped, "head") > self.indexed:
                return True
            async for _, session in self.records.chain(
                number(scoped, "sessions"), "previous_scope"
            ):
                if session.get("deleting"):
                    return True
            return False

    async def lexical(self, scope: ArchiveScope, query: str, limit: int) -> list[str]:
        """Find literal keyword candidates without backend-specific full-text operators."""
        entries = await self.archive._text_entries(  # noqa: SLF001  # Storage adapter shares archive retrieval.
            scope,
            query=query,
            session_id="",
            after=0,
            limit=limit,
        )
        return [str(entry["cursor"]) for entry in entries]

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
