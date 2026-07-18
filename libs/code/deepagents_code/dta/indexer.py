"""Hybrid BM25 + TF-IDF tool indexer for Dynamic Tool Allocation (DTA)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import operator
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Hybrid scoring weights: BM25 (sparse) and TF-IDF cosine (dense).
_BM25_WEIGHT = 0.7
_TFIDF_WEIGHT = 0.3


@dataclass
class ToolCandidate:
    """A single tool entry held in the :class:`HybridToolIndexer`.

    Attributes:
        name: Unique tool name.
        description: Human-readable description used for retrieval.
        schema: JSON-serialisable schema dict (name, description, parameters).
        namespace: Logical grouping, e.g. ``"mcp:github"`` or ``"builtin"``.
        original_tool: The raw tool object from the agent runtime, retained
            for pass-through reconstruction.
    """

    name: str
    description: str
    schema: dict[str, Any]
    namespace: str
    original_tool: Any = field(default=None, repr=False)


def compute_functional_hash(description: str, schema: dict[str, Any]) -> str:
    """Return a content-based hash used for deduplication.

    The hash covers the description and the parameter property *keys* only, so
    two tools with the same description and the same argument surface are
    considered identical regardless of minor schema metadata differences.

    Args:
        description: Tool description string.
        schema: Full tool schema dict; only ``schema["properties"]`` is hashed.

    Returns:
        A hex-encoded MD5 string identifying the tool's functional signature.
    """
    signature = (
        f"{description}|{json.dumps(schema.get('properties', {}), sort_keys=True)}"
    )
    return hashlib.md5(signature.encode()).hexdigest()  # noqa: S324 — non-security hash


class HybridToolIndexer:
    """In-process BM25 + TF-IDF hybrid retrieval index over agent tools.

    Tools are ingested via :meth:`sync_tools` on every middleware turn.
    Duplicate tools (same functional hash) are silently dropped; built-in
    duplicates of MCP tools take precedence over the MCP copy.

    Stage 1 retrieval (:meth:`search`) combines sparse (BM25) and dense
    (TF-IDF cosine) scores using configurable weights and supports namespace
    filtering for pre-gated retrieval.
    """

    def __init__(self, registry: object) -> None:
        """Initialise the indexer with a namespace registry.

        Args:
            registry: A :class:`~deepagents_code.dta.gating.ToolNamespaceRegistry`
                instance (or any object implementing ``classify_tool``).
        """
        self.registry = registry
        self.tools: dict[str, ToolCandidate] = {}

        # Deduplication: functional-hash → tool name
        self._dedup_map: dict[str, str] = {}

        # BM25 parameters (Okapi BM25)
        self.k1: float = 1.5
        self.b: float = 0.75

        # Per-document term frequency tables
        self.term_freqs: dict[str, dict[str, int]] = {}
        self.doc_freqs: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.total_docs: int = 0

        # Normalised TF-IDF vectors (recomputed on each add)
        self.tfidf_vectors: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def sync_tools(self, request_tools: list[Any]) -> None:
        """Synchronise tools from a ``ModelRequest.tools`` list.

        For each tool the method extracts its name, description, and schema,
        then delegates namespace classification to ``self.registry``.  Tools
        that are already present with the same functional signature are skipped.

        Args:
            request_tools: The raw tool list from the model request (may be
                ``BaseTool`` instances or plain dicts).
        """
        for tool in request_tools:
            name: str = (
                getattr(
                    tool, "name", tool.get("name") if isinstance(tool, dict) else ""
                )
                or ""
            )
            description: str = (
                getattr(
                    tool,
                    "description",
                    tool.get("description") if isinstance(tool, dict) else "",
                )
                or ""
            )

            # Best-effort schema extraction — prefer Pydantic v2, then v1, then dict
            schema: dict[str, Any] = {}
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                if hasattr(args_schema, "model_json_schema"):
                    with contextlib.suppress(Exception):
                        schema = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    with contextlib.suppress(Exception):
                        schema = args_schema.schema()
            if not schema and isinstance(tool, dict):
                schema = tool.get("parameters", {})

            namespace = "builtin"
            if hasattr(self.registry, "classify_tool"):
                namespace = self.registry.classify_tool(tool)

            candidate = ToolCandidate(
                name=name,
                description=description,
                schema={"name": name, "description": description, "parameters": schema},
                namespace=namespace,
                original_tool=tool,
            )
            self.add_tool(candidate)

    def add_tool(self, tool: ToolCandidate) -> None:
        """Add *tool* to the index, respecting deduplication rules.

        If a functionally identical tool already exists, the new entry is only
        accepted when it is a ``"builtin"`` tool displacing an MCP copy.  In
        all other duplicate cases the existing entry is kept unchanged.

        Args:
            tool: The :class:`ToolCandidate` to add.
        """
        func_hash = compute_functional_hash(tool.description, tool.schema)

        if func_hash in self._dedup_map:
            existing_name = self._dedup_map[func_hash]
            existing = self.tools.get(existing_name)
            # Built-in tools take precedence over MCP duplicates
            if (
                tool.namespace == "builtin"
                and existing is not None
                and existing.namespace != "builtin"
            ):
                del self.tools[existing_name]
                self._index_tool(tool, func_hash)
            # Otherwise silently drop the duplicate
        else:
            self._index_tool(tool, func_hash)

    # ------------------------------------------------------------------
    # Internal indexing
    # ------------------------------------------------------------------

    def _index_tool(self, tool: ToolCandidate, func_hash: str) -> None:
        self.tools[tool.name] = tool
        self._dedup_map[func_hash] = tool.name
        self._index_doc(tool.name, f"{tool.name} {tool.description}")
        self._recompute_tfidf()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split *text* into lowercase alphanumeric tokens.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        return [w.lower() for w in text.split() if w.isalnum()]

    def _index_doc(self, doc_id: str, text: str) -> None:
        """Update in-memory frequency tables for a newly indexed document.

        Args:
            doc_id: Unique identifier for the document (tool name).
            text: Full text to tokenize and index.
        """
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        self.avgdl = sum(self.doc_lengths.values()) / self.total_docs

        freqs = Counter(tokens)
        self.term_freqs[doc_id] = dict(freqs)
        for token in freqs:
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _recompute_tfidf(self) -> None:
        """Rebuild normalised TF-IDF vectors for all indexed documents."""
        self.tfidf_vectors.clear()
        for doc_id, freqs in self.term_freqs.items():
            vector: dict[str, float] = {}
            for token, tf in freqs.items():
                df = self.doc_freqs.get(token, 0)
                idf = math.log((self.total_docs + 1) / (df + 1)) + 1
                vector[token] = tf * idf
            norm = math.sqrt(sum(v * v for v in vector.values()))
            if norm > 0:
                self.tfidf_vectors[doc_id] = {k: v / norm for k, v in vector.items()}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        active_task: str = "",
        namespaces: set[str] | None = None,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """Search the index and return tool schemas ranked by hybrid score.

        The query is fused with the current active task string to improve
        recall.  Results are pre-filtered by *namespaces* when provided.

        Args:
            query: The user's current query string.
            active_task: Optional active plan/todo item for multi-query fusion.
            namespaces: When set, only tools whose namespace appears in this
                set are considered.  ``None`` means all tools are eligible.
            top_k: Maximum number of results to return.

        Returns:
            List of tool schema dicts (``{"name", "description", "parameters"}``),
            ordered by descending hybrid relevance score.
        """
        fused_query = f"{query} {active_task}".strip()
        tokens = self._tokenize(fused_query)

        # Build normalised query TF-IDF vector
        query_freqs = Counter(tokens)
        query_vector: dict[str, float] = {}
        for token, tf in query_freqs.items():
            df = self.doc_freqs.get(token, 0)
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1
            query_vector[token] = tf * idf

        norm = math.sqrt(sum(v * v for v in query_vector.values()))
        if norm > 0:
            query_vector = {k: v / norm for k, v in query_vector.items()}

        scores: dict[str, float] = {}

        for doc_id, candidate in self.tools.items():
            if namespaces is not None and candidate.namespace not in namespaces:
                continue

            # Sparse BM25 score
            bm25_score = 0.0
            dl = self.doc_lengths.get(doc_id, 0)
            if dl > 0:
                for token in tokens:
                    if token in self.term_freqs.get(doc_id, {}):
                        tf = self.term_freqs[doc_id][token]
                        df = self.doc_freqs.get(token, 0)
                        idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (
                            1 - self.b + self.b * dl / self.avgdl
                        )
                        bm25_score += idf * (numerator / denominator)

            # Dense TF-IDF cosine score
            tfidf_score = 0.0
            doc_vector = self.tfidf_vectors.get(doc_id, {})
            for token, qv in query_vector.items():
                if token in doc_vector:
                    tfidf_score += qv * doc_vector[token]

            scores[doc_id] = _BM25_WEIGHT * bm25_score + _TFIDF_WEIGHT * tfidf_score

        sorted_docs = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return [self.tools[doc_id].schema for doc_id, _ in sorted_docs[:top_k]]
