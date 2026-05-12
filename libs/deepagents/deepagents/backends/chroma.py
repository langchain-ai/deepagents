"""ChromaDB vector backend for deepagents.

This module provides :class:`ChromaVectorBackend`, an implementation of
:class:`deepagents.backends.protocol.BackendProtocol` that stores documents
in a ChromaDB collection and exposes them through a virtual filesystem
rooted at ``/logs/``.

Path convention
---------------
* ``/logs/query: <text>``   -- semantic similarity search via ``collection.query``
* ``/logs/id: <doc_id>``    -- exact document fetch via ``collection.get``
* ``/logs/``                -- list every document in the collection

Typical usage::

    import chromadb
    from deepagents.backends import CompositeBackend, ChromaVectorBackend

    client = chromadb.PersistentClient(path="./chroma_db")
    logs_backend = ChromaVectorBackend(client=client, collection="agent_logs")
    backend = CompositeBackend(mounts={"/logs": logs_backend})

The backend is designed to be plugged into a :class:`CompositeBackend` so
that deep agents can read, write, edit, grep, glob and bulk-transfer
log-like documents using semantic retrieval rather than literal file paths.

All eight :class:`BackendProtocol` method pairs (sync + async) are
implemented. The async variants are thin wrappers that off-load the
blocking ChromaDB call to a worker thread via :func:`asyncio.to_thread`,
so they never stall the event loop.
"""

from __future__ import annotations

import asyncio
import fnmatch
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)

_CHROMADB_IMPORT_ERROR = (
    "ChromaVectorBackend requires the 'chromadb' package. "
    "Install it with:  pip install 'deepagents[chroma]'  "
    "or directly:  pip install 'chromadb>=0.4.0'."
)

try:
    import chromadb  # noqa: F401  (imported for availability check)
except ImportError as exc:  # pragma: no cover - exercised only when extra missing
    raise ImportError(_CHROMADB_IMPORT_ERROR) from exc

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.types import EmbeddingFunction


# ---------------------------------------------------------------------------
# Path-parsing helpers
# ---------------------------------------------------------------------------

_QUERY_PREFIX = "query:"
_ID_PREFIX = "id:"


def _strip_root(path: str) -> str:
    """Return *path* with any leading ``/logs`` prefix and slashes removed."""
    return (path or "").strip().removeprefix("/logs").lstrip("/").strip()


def _parse_path(path: str) -> tuple[str, str]:
    """Parse a virtual path into ``(kind, payload)``.

    Returns:
        ``kind`` is one of ``"query"``, ``"id"``, ``"ls"`` or ``"unknown"``.
        ``payload`` is the trimmed text following the prefix (or ``""``).
    """
    body = _strip_root(path)
    if not body:
        return "ls", ""
    lower = body.lower()
    if lower.startswith(_QUERY_PREFIX):
        return "query", body[len(_QUERY_PREFIX) :].strip()
    if lower.startswith(_ID_PREFIX):
        return "id", body[len(_ID_PREFIX) :].strip()
    return "unknown", body


def _distance_to_score(distance: float | None) -> float:
    """Convert a Chroma distance (lower=closer) into a 0..1 similarity score."""
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return round(1.0 / (1.0 + max(d, 0.0)), 4)


def _source_of(metadata: dict[str, Any] | None) -> str:
    """Return the ``source`` field from a Chroma metadata dict, or ``"unknown"``."""
    if not metadata:
        return "unknown"
    src = metadata.get("source")
    return str(src) if src else "unknown"


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class ChromaVectorBackend(BackendProtocol):
    """A :class:`BackendProtocol` implementation backed by a ChromaDB collection.

    Documents in the underlying Chroma collection are exposed as if they were
    files under ``/logs/``. The backend supports two read modes:

    * **Semantic search** -- ``read("/logs/query: <text>", ...)`` runs a
      similarity query against the collection and returns the top
      ``n_results`` hits formatted as numbered lines.
    * **Direct lookup** -- ``read("/logs/id: <doc_id>", ...)`` fetches a
      single document by its Chroma ID.

    Parameters
    ----------
    client:
        A ChromaDB ``ClientAPI`` instance (e.g. from ``chromadb.EphemeralClient``,
        ``chromadb.PersistentClient`` or ``chromadb.HttpClient``).
    collection:
        Name of the Chroma collection to use. It will be created via
        ``get_or_create_collection`` if it does not already exist.
    n_results:
        Default number of hits returned by semantic queries. Defaults to 10.
    embedding_function:
        Optional Chroma embedding function. If ``None``, Chroma's default
        embedding function is used.
    """

    def __init__(
        self,
        client: ClientAPI,
        collection: str,
        n_results: int = 10,
        embedding_function: EmbeddingFunction[Any] | None = None,
    ) -> None:
        """Create a backend bound to *collection* inside *client*.

        See class docstring for parameter semantics.
        """
        self._client = client
        self._collection_name = collection
        self._n_results = max(1, int(n_results))
        if embedding_function is not None:
            self._col = client.get_or_create_collection(
                name=collection, embedding_function=embedding_function
            )
        else:
            self._col = client.get_or_create_collection(name=collection)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count(self) -> int:
        try:
            return int(self._col.count())
        except Exception:  # noqa: BLE001 -- Chroma backends raise heterogeneously
            return 0

    def _id_path(self, doc_id: str) -> str:
        return f"/logs/id: {doc_id}"

    def _normalize_upload_path(self, raw_path: str) -> str:
        """Return a canonical ``/logs/id: <name>`` path for an upload key."""
        kind, payload = _parse_path(raw_path)
        if kind == "id" and payload:
            return self._id_path(payload)
        # Fall back to using the raw basename as the id.
        name = (raw_path or "").strip().lstrip("/")
        name = name.rsplit("/", 1)[-1] or "untitled"
        return self._id_path(name)

    # ------------------------------------------------------------------
    # BackendProtocol -- ls_info / als_info
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:  # noqa: ARG002 -- protocol signature
        """List every document in the collection as a :class:`FileInfo`.

        The *path* argument is accepted for protocol compatibility but is
        ignored beyond an optional ``/logs`` prefix -- a Chroma collection is
        flat, so there are no sub-directories to descend into.
        """
        if self._count() == 0:
            return []
        result = self._col.get()
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        infos: list[FileInfo] = []
        for idx, doc_id in enumerate(ids):
            content = documents[idx] if idx < len(documents) else ""
            metadata = metadatas[idx] if idx < len(metadatas) else None
            size = len(content) if isinstance(content, str) else 0
            infos.append(
                FileInfo(
                    path=self._id_path(str(doc_id)),
                    size=size,
                    metadata={"source": _source_of(metadata)},
                )
            )
        return infos

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async wrapper around :meth:`ls_info`."""
        return await asyncio.to_thread(self.ls_info, path)

    # ------------------------------------------------------------------
    # BackendProtocol -- read / aread
    # ------------------------------------------------------------------

    def read(self, file_path: str, offset: int, limit: int) -> str:
        """Read documents from the collection.

        ``file_path`` must use one of the supported prefixes:

        * ``/logs/query: <text>`` -- top-N semantic hits, each rendered as
          ``[N] [score=0.XX] (source): content``.
        * ``/logs/id: <doc_id>`` -- the single matching document, rendered
          the same way (with ``score=1.00``).

        ``offset`` and ``limit`` slice the rendered line list. Negative or
        oversized values are clamped rather than raised.
        """
        kind, payload = _parse_path(file_path)
        message = self._read_dispatch(kind, payload, file_path)
        if message is not None:
            return message

        lines = (
            self._format_query(payload)
            if kind == "query"
            else self._format_by_id(payload)
        )
        if not lines:
            return "(no matches)"

        start = max(0, int(offset)) if offset else 0
        sliced = (
            lines[start:]
            if limit is None or int(limit) <= 0
            else lines[start : start + int(limit)]
        )
        return "\n".join(sliced)

    def _read_dispatch(
        self, kind: str, payload: str, file_path: str
    ) -> str | None:
        """Return an early-exit string for ``read``, or ``None`` to continue."""
        if kind == "ls":
            return (
                "Use '/logs/query: <text>' for semantic search or "
                "'/logs/id: <doc_id>' for direct lookup."
            )
        if kind == "unknown":
            return (
                f"Unsupported path: {file_path!r}. "
                "Expected '/logs/query: <text>' or '/logs/id: <doc_id>'."
            )
        if self._count() == 0:
            return "(no documents in collection)"
        if kind == "query" and not payload:
            return (
                "Empty query. Use '/logs/query: <text>' with a non-empty "
                "search string."
            )
        if kind == "id" and not payload:
            return (
                "Empty id. Use '/logs/id: <doc_id>' with a non-empty "
                "document id."
            )
        return None

    async def aread(self, file_path: str, offset: int, limit: int) -> str:
        """Async wrapper around :meth:`read`."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def _format_query(self, text: str) -> list[str]:
        k = min(self._n_results, max(1, self._count()))
        result = self._col.query(query_texts=[text], n_results=k)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        lines: list[str] = []
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else None
            distance = distances[i] if i < len(distances) else None
            score = _distance_to_score(distance)
            source = _source_of(metadata)
            lines.append(
                f"[{i + 1}] [score={score:.2f}] (id={doc_id}, source={source}): "
                f"{content}"
            )
        return lines

    def _format_by_id(self, doc_id: str) -> list[str]:
        result = self._col.get(ids=[doc_id])
        ids = result.get("ids") or []
        if not ids:
            return [f"document not found: {doc_id}"]
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        content = documents[0] if documents else ""
        metadata = metadatas[0] if metadatas else None
        source = _source_of(metadata)
        return [f"[1] [score=1.00] (id={ids[0]}, source={source}): {content}"]

    # ------------------------------------------------------------------
    # BackendProtocol -- write / awrite
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Upsert a document into the collection.

        The Chroma ID is derived from the ``id:`` portion of *file_path*.
        Writing via a ``query:`` path is rejected with an error result, since
        a semantic query is not a valid storage key.

        Returns a :class:`WriteResult` whose ``files_update`` is ``None``
        because this backend does not surface filesystem-style updates.
        """
        kind, payload = _parse_path(file_path)
        if kind != "id" or not payload:
            return WriteResult(path=file_path, files_update=None)

        metadata = {"source": payload}
        self._col.upsert(
            ids=[payload],
            documents=[content],
            metadatas=[metadata],
        )
        return WriteResult(path=self._id_path(payload), files_update=None)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async wrapper around :meth:`write`."""
        return await asyncio.to_thread(self.write, file_path, content)

    # ------------------------------------------------------------------
    # BackendProtocol -- edit / aedit
    # ------------------------------------------------------------------

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text inside a stored document and upsert the result.

        With ``replace_all=False`` only the first occurrence of *old_string*
        is replaced; with ``replace_all=True`` every occurrence is replaced.
        If the document is missing or the substring is not present, an
        :class:`EditResult` with ``occurrences=0`` is returned -- no
        exception is raised.
        """
        kind, payload = _parse_path(file_path)
        if kind != "id" or not payload:
            return EditResult(path=file_path, occurrences=0, files_update=None)

        fetched = self._col.get(ids=[payload])
        ids = fetched.get("ids") or []
        if not ids:
            return EditResult(
                path=self._id_path(payload),
                occurrences=0,
                files_update=None,
            )

        documents = fetched.get("documents") or [""]
        metadatas = fetched.get("metadatas") or [None]
        original = documents[0] if documents else ""
        metadata = metadatas[0] if metadatas else None

        if not old_string or old_string not in original:
            return EditResult(
                path=self._id_path(payload),
                occurrences=0,
                files_update=None,
            )

        if replace_all:
            occurrences = original.count(old_string)
            updated = original.replace(old_string, new_string)
        else:
            occurrences = 1
            updated = original.replace(old_string, new_string, 1)

        self._col.upsert(
            ids=[payload],
            documents=[updated],
            metadatas=[metadata] if metadata is not None else None,
        )
        return EditResult(
            path=self._id_path(payload),
            occurrences=occurrences,
            files_update=None,
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Async wrapper around :meth:`edit`."""
        return await asyncio.to_thread(
            self.edit,
            file_path,
            old_string,
            new_string,
            replace_all=replace_all,
        )

    # ------------------------------------------------------------------
    # BackendProtocol -- grep_raw / agrep_raw
    # ------------------------------------------------------------------

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,  # noqa: ARG002 -- protocol signature
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Run a semantic query and return matches as a list of :class:`GrepMatch`.

        Unlike a traditional regex grep, this backend interprets *pattern*
        as natural-language text and uses ``collection.query`` to find
        semantically similar documents. The optional *glob* argument filters
        results by the virtual ``/logs/id: <doc_id>`` path using
        :func:`fnmatch.fnmatch`.

        Returns:
            A list of :class:`GrepMatch` on success (empty list if there is
            nothing to search). A ``str`` is returned only for unrecoverable
            input errors (e.g. an empty pattern).
        """
        if not pattern or not pattern.strip():
            return "grep_raw requires a non-empty pattern"

        if self._count() == 0:
            return []

        k = min(self._n_results, max(1, self._count()))
        result = self._col.query(query_texts=[pattern], n_results=k)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        matches: list[GrepMatch] = []
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else None
            doc_path = self._id_path(str(doc_id))
            if glob and not fnmatch.fnmatch(doc_path, glob):
                continue
            matches.append(
                GrepMatch(
                    path=doc_path,
                    line_number=1,
                    line=content if isinstance(content, str) else str(content),
                    metadata={"source": _source_of(metadata)},
                )
            )
        return matches

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async wrapper around :meth:`grep_raw`."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    # ------------------------------------------------------------------
    # BackendProtocol -- glob_info / aglob_info
    # ------------------------------------------------------------------

    def glob_info(self, pattern: str, path: str | None = None) -> list[FileInfo]:
        """Filter :meth:`ls_info` results by a :mod:`fnmatch` pattern.

        The pattern is matched against the virtual ``/logs/id: <doc_id>``
        path of each document. Use ``"*"`` to match every document.
        """
        infos = self.ls_info(path or "/logs/")
        if not pattern:
            return infos
        return [info for info in infos if fnmatch.fnmatch(info["path"], pattern)]

    async def aglob_info(
        self, pattern: str, path: str | None = None
    ) -> list[FileInfo]:
        """Async wrapper around :meth:`glob_info`."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    # ------------------------------------------------------------------
    # BackendProtocol -- upload_files / aupload_files
    # ------------------------------------------------------------------

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Bulk-upload binary payloads as documents.

        Each ``(path, blob)`` tuple is decoded as UTF-8 (with ``errors="replace"``
        so non-UTF-8 input never raises) and stored via :meth:`write`. The
        returned list parallels the input order. On a write failure the
        corresponding :class:`FileUploadResponse` carries an ``error`` string
        and no exception escapes.
        """
        responses: list[FileUploadResponse] = []
        for raw_path, blob in files:
            target_path = self._normalize_upload_path(raw_path)
            try:
                content = (
                    bytes(blob).decode("utf-8", errors="replace")
                    if isinstance(blob, (bytes, bytearray))
                    else str(blob)
                )
                result = self.write(target_path, content)
                responses.append(
                    FileUploadResponse(path=result.path, error=None)
                )
            except Exception as exc:  # noqa: BLE001 -- surface as response error
                responses.append(
                    FileUploadResponse(path=target_path, error=str(exc))
                )
        return responses

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Async wrapper around :meth:`upload_files`."""
        return await asyncio.to_thread(self.upload_files, files)

    # ------------------------------------------------------------------
    # BackendProtocol -- download_files / adownload_files
    # ------------------------------------------------------------------

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Bulk-download documents by path.

        Each path is read via :meth:`read` with ``offset=0, limit=0`` (i.e.
        the full rendered content). Missing documents are reported as
        ``FileDownloadResponse(path=p, content=None, error="file_not_found")``
        rather than raising. ``query:`` paths return the rendered query
        results just as :meth:`read` would.
        """
        responses: list[FileDownloadResponse] = []
        for raw_path in paths:
            kind, payload = _parse_path(raw_path)
            if kind == "id" and payload:
                fetched = self._col.get(ids=[payload])
                if not (fetched.get("ids") or []):
                    responses.append(
                        FileDownloadResponse(
                            path=raw_path,
                            content=None,
                            error="file_not_found",
                        )
                    )
                    continue
            try:
                content = self.read(raw_path, 0, 0)
                responses.append(
                    FileDownloadResponse(
                        path=raw_path, content=content, error=None
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- surface as response error
                responses.append(
                    FileDownloadResponse(
                        path=raw_path, content=None, error=str(exc)
                    )
                )
        return responses

    async def adownload_files(
        self, paths: list[str]
    ) -> list[FileDownloadResponse]:
        """Async wrapper around :meth:`download_files`."""
        return await asyncio.to_thread(self.download_files, paths)


__all__ = ["ChromaVectorBackend"]
