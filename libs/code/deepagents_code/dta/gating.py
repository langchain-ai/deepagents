"""Namespace gating and LLM router for Dynamic Tool Allocation (DTA)."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deepagents_code.dta.utils import get_dta_fast_model

try:
    from deepagents_code.config import create_model
except ImportError:
    create_model = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Minimum number of namespaces available before the LLM router is invoked.
# When the total is at or below this threshold every namespace is trivially
# relevant, so we skip the LLM call to save latency.
_FAST_PATH_NAMESPACE_THRESHOLD = 3

# Number of underscore-separated parts needed to extract an MCP server name.
_MCP_PREFIX_MIN_PARTS = 2


class ActiveNamespacesResult(BaseModel):
    """Structured output returned by the namespace router LLM."""

    active_namespaces: list[str] = Field(
        description="Namespaces relevant to the user query"
    )


class ToolNamespaceRegistry:
    """Registry that classifies tools into logical namespaces.

    Namespace assignment drives Stage 1 retrieval filtering so that only tools
    from relevant servers are considered as candidates for a given query.
    """

    @staticmethod
    def classify_tool(tool: object) -> str:
        """Classify *tool* into a namespace string.

        Classification priority:
        1. ``metadata["server_name"]`` -- set by the MCP tool loader.
        2. ``mcp_`` name prefix -- heuristic for tools wrapped without metadata.
        3. Known domain prefixes (``git_``, ``db_``, ``postgres_``, ``sql_``).
        4. Fallback: ``"builtin"``.

        Args:
            tool: A ``BaseTool`` instance, a plain dict with a ``"name"`` key,
                or any object that exposes a ``.name`` attribute.

        Returns:
            A namespace string, e.g. ``"mcp:github"``, ``"git"``,
            ``"database"``, or ``"builtin"``.
        """
        name: str = (
            getattr(tool, "name", tool.get("name") if isinstance(tool, dict) else "")
            or ""
        ).lower()

        # 1. MCP server detection via metadata
        metadata: dict[str, object] = getattr(tool, "metadata", {}) or {}
        if isinstance(metadata, dict) and "server_name" in metadata:
            return f"mcp:{metadata['server_name']}"

        # 2. Heuristic: tools whose name starts with "mcp_<server>_"
        if name.startswith("mcp_"):
            parts = name.split("_")
            if len(parts) >= _MCP_PREFIX_MIN_PARTS:
                return f"mcp:{parts[1]}"

        # 3. Known domain prefixes
        if name.startswith("git_"):
            return "git"
        if any(name.startswith(p) for p in ("db_", "postgres_", "sql_")):
            return "database"

        return "builtin"


class ToolNamespaceRouterNode:
    """LLM-based router that selects the active namespaces for a user query.

    For a single turn the router narrows the full set of registered namespaces
    down to only those relevant to the current query.  This drives Stage 1
    retrieval so that irrelevant MCP servers do not pollute candidate results.

    When the total number of available namespaces is at or below
    :data:`_FAST_PATH_NAMESPACE_THRESHOLD` the LLM call is skipped entirely to
    eliminate unnecessary latency — all namespaces are trivially relevant at
    that scale.

    All LLM errors are caught and handled by failing open: the full set of
    available namespaces is returned so the agent is never blocked.
    """

    def __init__(self, model_spec: str | None = None) -> None:
        """Initialise the router node.

        Args:
            model_spec: Optional explicit model spec (e.g. ``"openai:gpt-4o-mini"``).
                When ``None``, the spec is resolved at call time from the active
                provider via :func:`~deepagents_code.dta.utils.get_dta_fast_model`.
        """
        self.model_spec = model_spec

    def _get_model_spec(self) -> str | None:
        if self.model_spec:
            return self.model_spec
        return get_dta_fast_model()

    @staticmethod
    def _build_messages(
        query: str, available_namespaces: set[str]
    ) -> list[HumanMessage | SystemMessage]:
        sys_prompt = (
            "You are a Tool Namespace Router. "
            "Based on the user's query, select the necessary namespaces.\n"
            f"Available namespaces: {sorted(available_namespaces)}\n"
            "Return only the namespaces that might be needed. "
            "If unsure, include the namespace."
        )
        return [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}"),
        ]

    @staticmethod
    def _validate_result(
        result: ActiveNamespacesResult, available_namespaces: set[str]
    ) -> set[str]:
        """Filter LLM output to only valid, available namespaces.

        Always ensures ``"builtin"`` is included when it exists in the
        available set so core agent tools are never accidentally gated out.

        Args:
            result: Structured output from the router LLM.
            available_namespaces: The full set passed to the router.

        Returns:
            Validated subset of *available_namespaces* to activate.
        """
        valid = {ns for ns in result.active_namespaces if ns in available_namespaces}
        if "builtin" in available_namespaces:
            valid.add("builtin")
        return valid

    def route(self, query: str, available_namespaces: set[str]) -> set[str]:
        """Synchronously select active namespaces for *query*.

        Args:
            query: The current user query string.
            available_namespaces: All namespaces present in the tool index.

        Returns:
            Subset of *available_namespaces* that should be active for this turn.
        """
        if not available_namespaces:
            return set()

        if len(available_namespaces) <= _FAST_PATH_NAMESPACE_THRESHOLD:
            logger.debug(
                "DTA Router fast-path: all %d namespace(s) enabled",
                len(available_namespaces),
            )
            return available_namespaces

        try:
            if create_model is None:
                return available_namespaces

            spec = self._get_model_spec()
            model_res = create_model(spec)
            llm = model_res.model.with_structured_output(ActiveNamespacesResult)
            messages = self._build_messages(query, available_namespaces)
            result: ActiveNamespacesResult = llm.invoke(
                messages, config={"tags": ["dta_router"]}
            )
        except Exception:
            logger.warning(
                "DTA Router LLM failed, failing open to all namespaces",
                exc_info=True,
            )
            return available_namespaces
        else:
            active = self._validate_result(result, available_namespaces)
            logger.info("DTA Router active namespaces: %s", active)
            return active

    async def aroute(self, query: str, available_namespaces: set[str]) -> set[str]:
        """Asynchronously select active namespaces for *query*.

        Mirrors :meth:`route` but uses ``ainvoke`` for async event loops.

        Args:
            query: The current user query string.
            available_namespaces: All namespaces present in the tool index.

        Returns:
            Subset of *available_namespaces* that should be active for this turn.
        """
        if not available_namespaces:
            return set()

        if len(available_namespaces) <= _FAST_PATH_NAMESPACE_THRESHOLD:
            logger.debug(
                "DTA Router fast-path: all %d namespace(s) enabled",
                len(available_namespaces),
            )
            return available_namespaces

        try:
            if create_model is None:
                return available_namespaces

            spec = self._get_model_spec()
            model_res = create_model(spec)
            llm = model_res.model.with_structured_output(ActiveNamespacesResult)
            messages = self._build_messages(query, available_namespaces)
            result: ActiveNamespacesResult = await llm.ainvoke(
                messages, config={"tags": ["dta_router"]}
            )
        except Exception:
            logger.warning(
                "DTA Router LLM failed, failing open to all namespaces",
                exc_info=True,
            )
            return available_namespaces
        else:
            active = self._validate_result(result, available_namespaces)
            logger.info("DTA Router active namespaces: %s", active)
            return active
