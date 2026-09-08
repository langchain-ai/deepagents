"""Storage operations required by the shared vector indexing worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from deepagents_talon.archive import ArchiveEntry, ArchiveScope

Row = tuple[int, str, str, str, int, str | None]


class VectorArchive(Protocol):
    """Provide durable indexing work and authoritative, scope-checked retrieval."""

    async def prepare(self) -> str:
        """Reconcile pending work and return a stable archive identity."""
        ...

    async def rows(self, session: str, *, indexing: bool, limit: int) -> list[Row]:
        """Read pending rows; a session restricts work to its deletions."""
        ...

    async def acknowledge(self, rows: list[Row]) -> None:
        """Acknowledge only successful vector writes."""
        ...

    async def mark_deleted(self, session: str) -> None:
        """Durably schedule vector removal, retaining transcript ownership."""
        ...

    async def delete_text(self, session: str) -> None:
        """Release transcripts and ownership after vector removal succeeds."""
        ...

    async def pending(self, scope: ArchiveScope) -> bool:
        """Report outstanding indexing work in the trusted scope."""
        ...

    async def lexical(self, scope: ArchiveScope, query: str, limit: int) -> list[str]:
        """Find bounded keyword candidates within scope."""
        ...

    async def ranked(
        self, scope: ArchiveScope, keys: list[str], after: int, limit: int
    ) -> list[ArchiveEntry]:
        """Revalidate and read ranked candidates within the trusted scope."""
        ...
