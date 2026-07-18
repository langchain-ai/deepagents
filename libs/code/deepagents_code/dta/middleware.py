"""Dynamic Tool Allocation middleware for deepagents-code.

This module implements :class:`DynamicToolAllocationMiddleware`, an
:class:`~langchain.agents.middleware.types.AgentMiddleware` that intercepts
every model call and trims the full tool roster down to a context-relevant
budget before forwarding the request to the LLM.

Pipeline (per turn):
    1. **Dynamic ingestion** — sync new tools from the live request into the
       in-process :class:`~deepagents_code.dta.indexer.HybridToolIndexer`.
    2. **Smart cache** — skip retrieval when the latest message is a tool
       output (the toolset shouldn't change mid-execution).
    3. **Cold reset** — evict temporal history when semantic drift between
       turns is high (Jaccard similarity < 0.2).
    4. **Namespace gating** — invoke the LLM router to limit Stage 1 to
       only relevant MCP servers / namespaces.
    5. **Stage 1 retrieval** — BM25 + TF-IDF hybrid search over gated tools.
    6. **Stage 2 ranking** — LLM re-ranks candidates to a strict budget.
    7. **Reconstruction** — filter ``request.tools`` to the final set and
       forward the modified request to the model.

All errors in steps 4-7 are caught and fail open: the original unmodified
``request`` is returned so the agent always proceeds.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents_code.dta.indexer import HybridToolIndexer
    from deepagents_code.dta.selector import ToolSelectorNode

logger = logging.getLogger(__name__)

# Jaccard similarity threshold below which a Cold Reset is triggered.
_COLD_RESET_THRESHOLD: float = 0.2

# Number of recent messages scanned for temporal tool extraction.
_TEMPORAL_WINDOW: int = 6

# Number of recent messages scanned to detect a refresh_tools call.
_REFRESH_SCAN_WINDOW: int = 2

# Maximum Stage 1 candidates passed to Stage 2.
_STAGE1_TOP_K: int = 30


class DynamicToolAllocationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Middleware that dynamically allocates a context-relevant subset of tools.

    The middleware sits between the agent graph and the LLM, intercepting
    :meth:`wrap_model_call` / :meth:`awrap_model_call` to trim
    ``request.tools`` to a configurable budget before the model sees them.

    Attributes:
        indexer: The hybrid retrieval index used for Stage 1 search.
        selector_node: The LLM-based Stage 2 ranker.
        router_node: The LLM-based namespace router (Stage 0 gating).
        max_tools_budget: Hard cap on the total number of tools passed to the
            model per turn (sticky tools count toward this budget).
        sticky_tools: Tool names that are always included regardless of
            relevance scoring.
    """

    def __init__(
        self,
        indexer: HybridToolIndexer,
        selector_node: ToolSelectorNode,
        router_node: object = None,
        max_tools_budget: int = 12,
    ) -> None:
        """Initialise the middleware.

        Args:
            indexer: Pre-constructed
                :class:`~deepagents_code.dta.indexer.HybridToolIndexer`.
            selector_node: Pre-constructed
                :class:`~deepagents_code.dta.selector.ToolSelectorNode`.
            router_node: Optional pre-constructed
                :class:`~deepagents_code.dta.gating.ToolNamespaceRouterNode`.
                When ``None`` a default instance is created lazily.
            max_tools_budget: Total tool budget per turn including sticky tools.
        """
        self.indexer = indexer
        self.selector_node = selector_node
        if router_node is None:
            from deepagents_code.dta.gating import ToolNamespaceRouterNode

            self.router_node: object = ToolNamespaceRouterNode()
        else:
            self.router_node = router_node
        self.max_tools_budget = max_tools_budget
        self.sticky_tools: set[str] = {
            "read_file",
            "edit_file",
            "execute",
            "ask_user",
            "refresh_tools",
        }
        self._last_user_query: str = ""
        self._cached_toolset: set[str] = set()

    # ------------------------------------------------------------------
    # Context extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context_query(messages: list[Any]) -> str:
        """Return the most recent human message content from *messages*.

        Args:
            messages: Conversation history from the model request.

        Returns:
            The content string of the latest human message, or a safe default.
        """
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human" and isinstance(
                getattr(msg, "content", None), str
            ):
                return msg.content
        return "default context"

    @staticmethod
    def _extract_recent_tools(messages: list[Any]) -> set[str]:
        """Collect tool names used in the last :data:`_TEMPORAL_WINDOW` messages.

        Args:
            messages: Conversation history from the model request.

        Returns:
            Set of tool names invoked recently (used for temporal continuation).
        """
        recent_tools: set[str] = set()
        for msg in messages[-_TEMPORAL_WINDOW:]:
            if getattr(msg, "type", "") == "ai" and hasattr(msg, "tool_calls"):
                recent_tools.update(tc.get("name", "") for tc in msg.tool_calls)
            elif getattr(msg, "type", "") == "tool":
                recent_tools.add(getattr(msg, "name", ""))
        return recent_tools

    @staticmethod
    def _is_tool_output_turn(messages: list[Any]) -> bool:
        """Return ``True`` when the last message is a tool output.

        Args:
            messages: Conversation history from the model request.

        Returns:
            ``True`` if the latest message is a tool-output message.
        """
        if not messages:
            return False
        return getattr(messages[-1], "type", "") == "tool"

    @staticmethod
    def _was_refresh_requested(messages: list[Any]) -> bool:
        """Detect whether ``refresh_tools`` was called in recent messages.

        Args:
            messages: Conversation history from the model request.

        Returns:
            ``True`` if a refresh_tools invocation is detected.
        """
        if not messages:
            return False
        for msg in messages[-_REFRESH_SCAN_WINDOW:]:
            if getattr(msg, "type", "") == "ai" and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if tc.get("name") == "refresh_tools":
                        return True
            elif (
                getattr(msg, "type", "") == "tool"
                and getattr(msg, "name", "") == "refresh_tools"
            ):
                return True
        return False

    def _check_cold_reset(self, current_query: str) -> bool:
        """Return ``True`` when semantic drift warrants a Cold Reset.

        Jaccard similarity between the previous and current query word sets is
        computed; values below :data:`_COLD_RESET_THRESHOLD` indicate a topic
        switch sufficient to evict temporal history.

        Args:
            current_query: The query extracted from the current turn.

        Returns:
            ``True`` if a Cold Reset should be triggered.
        """
        if not self._last_user_query or not current_query:
            return False

        def get_words(text: str) -> set[str]:
            return {w.lower() for w in text.split() if w.isalnum()}

        w1 = get_words(self._last_user_query)
        w2 = get_words(current_query)
        if not w1 or not w2:
            return False

        jaccard = len(w1 & w2) / len(w1 | w2)
        return jaccard < _COLD_RESET_THRESHOLD

    # ------------------------------------------------------------------
    # Request reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_request(
        request: ModelRequest[Any], final_toolset_names: set[str]
    ) -> ModelRequest[Any]:
        """Return a copy of *request* with tools filtered to *final_toolset_names*.

        Tries, in order:
        1. ``request.override(tools=...)`` — preferred immutable pattern.
        2. ``request.copy(update={"tools": ...})`` — Pydantic v1 copy.
        3. ``dataclasses.replace(request, tools=...)`` — dataclass pattern.
        4. Direct attribute mutation as a last resort.

        Args:
            request: The original model request.
            final_toolset_names: Set of tool names that should be forwarded.

        Returns:
            A (possibly new) model request with tools trimmed to the final set.
        """
        filtered_tools = [
            t
            for t in request.tools
            if (getattr(t, "name", t.get("name") if isinstance(t, dict) else ""))
            in final_toolset_names
        ]

        try:
            if hasattr(request, "override"):
                return request.override(tools=filtered_tools)
            if hasattr(request, "copy"):
                return request.copy(update={"tools": filtered_tools})
            if dataclasses.is_dataclass(request):
                return dataclasses.replace(request, tools=filtered_tools)
        except Exception:
            logger.debug(
                "Request reconstruction via API failed; falling back to mutation",
                exc_info=True,
            )
        # Fallback: mutate in place when no immutable copy mechanism exists
        request.tools = filtered_tools
        return request

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _process_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Run the full DTA pipeline synchronously.

        Args:
            request: Incoming model request from the agent graph.

        Returns:
            A model request with tools trimmed to the DTA budget.  On any
            unrecoverable error the original *request* is returned unchanged.
        """
        try:
            # 1. Dynamic ingestion
            self.indexer.sync_tools(request.tools)

            # 2. Extract context & state
            state = getattr(request, "state", {}) or {}
            plan_item = (
                state.get("active_todo")
                if isinstance(state, dict)
                else getattr(state, "active_todo", None)
            )
            context_query = self._extract_context_query(request.messages)

            # 3. Smart cache & refresh detection
            is_refresh = self._was_refresh_requested(request.messages)
            if is_refresh:
                logger.info("DTA: refresh_tools detected — clearing cache")
                self._cached_toolset.clear()

            if (
                not is_refresh
                and self._is_tool_output_turn(request.messages)
                and self._cached_toolset
            ):
                logger.debug("DTA: smart cache hit on tool-output turn")
                return self._reconstruct_request(request, self._cached_toolset)

            # 4. Cold Reset
            if is_refresh or self._check_cold_reset(context_query):
                logger.info("DTA: Cold Reset triggered")
                temporal_tools: set[str] = set()
            else:
                temporal_tools = self._extract_recent_tools(request.messages)

            self._last_user_query = context_query

            # 5. Namespace gating
            available_namespaces = {t.namespace for t in self.indexer.tools.values()}
            active_namespaces = self.router_node.route(
                query=context_query, available_namespaces=available_namespaces
            )

            # 6. Stage 1: BM25/TF-IDF retrieval
            candidates = self.indexer.search(
                query=context_query,
                active_task=plan_item or "",
                namespaces=active_namespaces,
                top_k=_STAGE1_TOP_K,
            )

            # 7. Stage 2: LLM selector
            budget = max(0, self.max_tools_budget - len(self.sticky_tools))
            allocated_names = self.selector_node.select(
                messages=request.messages, candidates=candidates, budget=budget
            )

            final_toolset_names = self.sticky_tools | allocated_names | temporal_tools
            self._cached_toolset = final_toolset_names
            logger.info("DTA final toolset: %s", final_toolset_names)

            return self._reconstruct_request(request, final_toolset_names)
        except Exception:
            logger.exception(
                "DTA _process_request failed catastrophically — failing open"
            )
            return request

    async def _aprocess_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Run the full DTA pipeline asynchronously.

        Mirrors :meth:`_process_request` but uses ``await`` on async LLM calls.

        Args:
            request: Incoming model request from the agent graph.

        Returns:
            A model request with tools trimmed to the DTA budget.  On any
            unrecoverable error the original *request* is returned unchanged.
        """
        try:
            # 1. Dynamic ingestion
            self.indexer.sync_tools(request.tools)

            # 2. Extract context & state
            state = getattr(request, "state", {}) or {}
            plan_item = (
                state.get("active_todo")
                if isinstance(state, dict)
                else getattr(state, "active_todo", None)
            )
            context_query = self._extract_context_query(request.messages)

            # 3. Smart cache & refresh detection
            is_refresh = self._was_refresh_requested(request.messages)
            if is_refresh:
                logger.info("DTA: refresh_tools detected — clearing cache")
                self._cached_toolset.clear()

            if (
                not is_refresh
                and self._is_tool_output_turn(request.messages)
                and self._cached_toolset
            ):
                logger.debug("DTA: smart cache hit on tool-output turn")
                return self._reconstruct_request(request, self._cached_toolset)

            # 4. Cold Reset
            if is_refresh or self._check_cold_reset(context_query):
                logger.info("DTA: Cold Reset triggered")
                temporal_tools_: set[str] = set()
            else:
                temporal_tools_ = self._extract_recent_tools(request.messages)

            self._last_user_query = context_query

            # 5. Namespace gating
            available_namespaces = {t.namespace for t in self.indexer.tools.values()}
            active_namespaces = await self.router_node.aroute(
                query=context_query, available_namespaces=available_namespaces
            )

            # 6. Stage 1: BM25/TF-IDF retrieval
            candidates = self.indexer.search(
                query=context_query,
                active_task=plan_item or "",
                namespaces=active_namespaces,
                top_k=_STAGE1_TOP_K,
            )

            # 7. Stage 2: LLM selector (async)
            budget = max(0, self.max_tools_budget - len(self.sticky_tools))
            allocated_names = await self.selector_node.aselect(
                messages=request.messages, candidates=candidates, budget=budget
            )

            final_toolset_names = self.sticky_tools | allocated_names | temporal_tools_
            self._cached_toolset = final_toolset_names
            logger.info("DTA final toolset: %s", final_toolset_names)

            return self._reconstruct_request(request, final_toolset_names)
        except Exception:
            logger.exception(
                "DTA _aprocess_request failed catastrophically — failing open"
            )
            return request

    # ------------------------------------------------------------------
    # AgentMiddleware interface
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Synchronous middleware entry point.

        Args:
            request: The model request to intercept.
            handler: The downstream model call to invoke with the trimmed request.

        Returns:
            The model response from *handler*.
        """
        return handler(self._process_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Asynchronous middleware entry point.

        Args:
            request: The model request to intercept.
            handler: The downstream model call to invoke with the trimmed request.

        Returns:
            The model response from *handler*.
        """
        return await handler(await self._aprocess_request(request))
