"""Middleware for on-demand tool discovery via search.

This module implements a tool search middleware that enables on-demand tool discovery
when agents have access to many tools (50+). This solves token bloat and accuracy
degradation issues by deferring tool definitions until they're needed.

Problem Statement
-----------------
- Tool definitions consume 10-20K+ tokens upfront, wasting context window
- Tool selection accuracy degrades beyond 30-50 tools
- Most tools unused per session, leading to inefficient token spend

Solution
--------
A hybrid middleware + meta-tool approach:

1. Index all tools at agent initialization
2. If tools exceed token threshold (default 10% of context), activate "deferred mode"
3. In deferred mode: only ``search_tools`` meta-tool + ``always_include`` tools available
4. Agent calls ``search_tools(query)`` to discover relevant tools
5. Discovered tools are "expanded" and become available for subsequent calls

Example:
--------
::

    from deepagents import create_deep_agent
    from deepagents.middleware.tool_search import ToolSearchMiddleware

    agent = create_deep_agent(
        tools=many_tools,  # 100+ tools from MCP servers
        middleware=[
            ToolSearchMiddleware(
                always_include=["read_file", "write_file"],
                search_mode="hybrid",
            ),
        ],
    )

The middleware automatically determines whether to activate deferred mode based on
the token threshold. If tools are below the threshold, all tools are available
immediately without requiring search.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# Limits per specification
MAX_TOOL_CATALOG_SIZE = 10_000
MAX_REGEX_PATTERN_LENGTH = 200
MAX_SEARCH_RESULTS = 5

# Default context window fallback when model profile is unavailable
DEFAULT_CONTEXT_WINDOW = 128_000


class BM25Index:
    """Simple BM25 implementation for relevance-ranked search.

    This is a lightweight BM25 implementation without external dependencies.
    It provides term-frequency based relevance scoring for tool search.

    The BM25 algorithm uses the following formula:
    score(D, Q) = sum(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

    Where:
    - Q is the query containing terms q1, ..., qn
    - D is a document
    - f(qi, D) is qi's term frequency in D
    - |D| is the length of D in words
    - avgdl is the average document length
    - k1 and b are free parameters (typically k1 = 1.5, b = 0.75)
    - IDF(qi) is the inverse document frequency of qi
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.documents: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.doc_freqs: Counter[str] = Counter()  # Number of documents containing each term
        self.term_freqs: list[Counter[str]] = []  # Term frequencies per document
        self.idf_cache: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"\w+", text.lower())

    def add_documents(self, documents: list[str]) -> None:
        """Add documents to the index.

        Args:
            documents: List of document strings to index.
        """
        for doc in documents:
            tokens = self._tokenize(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies for this document
            tf = Counter(tokens)
            self.term_freqs.append(tf)

            # Update document frequency counts
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.doc_count = len(self.documents)
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # Clear IDF cache since corpus changed
        self.idf_cache.clear()

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            idf = 0.0
        else:
            # IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

        self.idf_cache[term] = idf
        return idf

    def search(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        """Search for documents matching query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of (document_index, score) tuples, sorted by score descending.
        """
        # Early return for empty corpus
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[int, float]] = []

        for doc_idx in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            tf_dict = self.term_freqs[doc_idx]

            for term in query_tokens:
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue

                idf = self._idf(term)

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                # Handle edge case where avgdl is 0 (all empty documents)
                if self.avgdl > 0:
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                else:
                    # Degenerate case: no length normalization
                    denominator = tf + self.k1
                score += idf * numerator / denominator

            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score descending and limit results
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]


class ToolIndex:
    """Index for searching tools by name, description, and parameters.

    Supports multiple search modes:
    - bm25: Relevance-ranked natural language search
    - regex: Pattern matching for tool names/descriptions
    - hybrid: Combines both, boosts regex matches
    """

    def __init__(self) -> None:
        """Initialize tool index."""
        self.tools: list[BaseTool | dict[str, Any]] = []
        self.tool_names: list[str] = []
        self.tool_texts: list[str] = []  # Combined searchable text
        self.bm25_index = BM25Index()
        self._indexed = False

    def add_tools(self, tools: Sequence[BaseTool | dict[str, Any]]) -> None:
        """Add tools to the index.

        Args:
            tools: List of tools to index.
        """
        for tool in tools:
            self.tools.append(tool)

            # Extract tool info
            if isinstance(tool, BaseTool):
                name = tool.name
                description = tool.description or ""
                # Try to get parameter info from schema
                params_text = ""
                if hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        schema = tool.args_schema.model_json_schema()
                        if "properties" in schema:
                            params_text = " ".join(
                                f"{k} {v.get('description', '')}"
                                for k, v in schema["properties"].items()
                            )
                    except Exception:
                        pass
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                params_text = ""
                if "parameters" in tool and "properties" in tool["parameters"]:
                    params_text = " ".join(
                        f"{k} {v.get('description', '')}"
                        for k, v in tool["parameters"]["properties"].items()
                    )

            self.tool_names.append(name)
            # Combine name, description, and params for searchability
            # Weight name more heavily by repeating it
            self.tool_texts.append(f"{name} {name} {name} {description} {params_text}")

        # Index all tool texts
        self.bm25_index.add_documents(self.tool_texts)
        self._indexed = True

    def get_tool_name(self, index: int) -> str:
        """Get tool name by index."""
        return self.tool_names[index]

    def get_tool(self, index: int) -> BaseTool | dict[str, Any]:
        """Get tool by index."""
        return self.tools[index]

    def search_bm25(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Search tools using BM25 ranking.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of (tool_name, score) tuples.
        """
        if not self._indexed:
            return []

        results = self.bm25_index.search(query, limit=limit)
        return [(self.tool_names[idx], score) for idx, score in results]

    def search_regex(self, pattern: str, limit: int = 5) -> list[tuple[str, float]]:
        """Search tools using regex pattern matching.

        Args:
            pattern: Regex pattern to match against tool names and descriptions.
                Must be 200 characters or fewer.
            limit: Maximum results to return.

        Returns:
            List of (tool_name, score) tuples. Score is 2.0 for name match, 1.0 for description match.

        Raises:
            ValueError: If pattern exceeds 200 characters.
        """
        if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
            raise ValueError(f"Regex pattern exceeds {MAX_REGEX_PATTERN_LENGTH} character limit")

        results: list[tuple[str, float]] = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Invalid regex, fall back to literal search
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)

        for i, (name, text) in enumerate(zip(self.tool_names, self.tool_texts)):
            # Higher score for name match
            if compiled.search(name):
                results.append((name, 2.0))
            elif compiled.search(text):
                results.append((name, 1.0))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search(
        self,
        query: str,
        mode: Literal["bm25", "regex", "hybrid"] = "bm25",
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for tools matching query.

        Args:
            query: Search query.
            mode: Search mode - "bm25", "regex", or "hybrid".
            limit: Maximum results to return.

        Returns:
            List of (tool_name, score) tuples, sorted by relevance.
        """
        if mode == "bm25":
            return self.search_bm25(query, limit=limit)
        elif mode == "regex":
            return self.search_regex(query, limit=limit)
        else:  # hybrid
            bm25_results = dict(self.search_bm25(query, limit=limit * 2))
            regex_results = dict(self.search_regex(query, limit=limit * 2))

            # Combine scores, boosting regex matches
            combined: dict[str, float] = {}
            for name, score in bm25_results.items():
                combined[name] = score
            for name, score in regex_results.items():
                # Boost regex matches by adding to BM25 score
                combined[name] = combined.get(name, 0) + score * 2

            # Sort and limit
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:limit]


def _estimate_tool_tokens(tool: BaseTool | dict[str, Any]) -> int:
    """Estimate token count for a tool definition.

    Uses a simple heuristic of ~4 characters per token.

    Args:
        tool: Tool to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if isinstance(tool, BaseTool):
        name = tool.name
        description = tool.description or ""
        schema_text = ""
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema_text = json.dumps(tool.args_schema.model_json_schema())
            except Exception:
                pass
        total_chars = len(name) + len(description) + len(schema_text)
    else:
        total_chars = len(json.dumps(tool))

    # Rough estimate: 4 chars per token
    return max(1, total_chars // 4)


def _get_context_window(model: Any) -> int:
    """Get context window size for a model.

    Uses model.profile["max_input_tokens"] if available, otherwise falls back
    to DEFAULT_CONTEXT_WINDOW.

    Args:
        model: The model to get context window for.

    Returns:
        Context window size in tokens.
    """
    if model is None:
        return DEFAULT_CONTEXT_WINDOW

    # Try to get context window from model profile (like graph.py does)
    if (
        hasattr(model, "profile")
        and model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        return model.profile["max_input_tokens"]

    return DEFAULT_CONTEXT_WINDOW


class ToolSearchState(AgentState):
    """State for the tool search middleware."""

    expanded_tools: NotRequired[Annotated[set[str], PrivateStateAttr]]
    """Names of tools that have been discovered and are available."""


class ToolSearchStateUpdate(TypedDict):
    """State update for tool search middleware."""

    expanded_tools: set[str]
    """Updated set of expanded tool names."""


SEARCH_TOOLS_DESCRIPTION = """Search for available tools by query.

Use this tool when you need to find tools for a specific task. The search will return
relevant tools based on your query, and those tools will then become available for use.

Args:
    query: Natural language description of what you want to do. Be specific about the
        task or action you're trying to accomplish. Examples:
        - "read and write files"
        - "search the web"
        - "execute shell commands"
        - "interact with databases"
    mode: Search mode (default: "bm25")
        - "bm25": Natural language search (recommended)
        - "regex": Pattern matching on tool names
        - "hybrid": Combines both methods
    limit: Maximum number of tools to return (default: 5)

Returns:
    List of discovered tools with their names and descriptions. These tools will
    automatically become available for use in subsequent messages.

Example usage:
    search_tools(query="file operations") -> Returns file-related tools
    search_tools(query="http.*", mode="regex") -> Returns HTTP-related tools
"""

TOOL_SEARCH_SYSTEM_PROMPT = """## Tool Search

Some tools are available on-demand through search. Use the `search_tools` tool to discover
and enable tools for specific tasks.

**When to search for tools:**
- When you need a capability that isn't immediately available
- When the user asks you to do something that requires a specific tool
- Before attempting a task that might have dedicated tools

**How it works:**
1. Call `search_tools(query="description of what you need")`
2. Review the returned tool descriptions
3. The discovered tools become available immediately
4. Use the discovered tools to complete your task

**Tips for effective searches:**
- Use descriptive queries: "read files" instead of "file"
- Search early: find tools before you need them
- Be specific: "execute SQL queries" instead of "database"
"""


class ToolSearchMiddleware(AgentMiddleware):
    """Middleware for on-demand tool discovery via search.

    This middleware enables on-demand tool discovery when agents have access to many
    tools. When the total tool definitions exceed a token threshold (default 10% of
    context), the middleware activates "deferred mode":

    - Only the `search_tools` meta-tool and `always_include` tools are available initially
    - The agent calls `search_tools(query)` to discover relevant tools
    - Discovered tools are "expanded" and become available for subsequent calls

    This approach reduces token usage and improves tool selection accuracy by avoiding
    overwhelming the model with too many tool definitions upfront.

    Args:
        token_threshold: Token threshold for activating deferred mode.
            Can be a float (fraction of context window, e.g., 0.10 for 10%) or
            an int (absolute token count). Default is 0.10.
        context_window: Context window size in tokens. If not provided, it's
            auto-detected from the model. Default is 128,000.
        always_include: List of tool names that should always be available,
            even in deferred mode. These tools bypass search.
        search_mode: Default search mode for the search_tools tool.
            Options: "bm25", "regex", "hybrid". Default is "bm25".

    Example:
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware.tool_search import ToolSearchMiddleware

        agent = create_deep_agent(
            tools=many_tools,  # 100+ tools
            middleware=[
                ToolSearchMiddleware(
                    always_include=["read_file", "write_file"],
                    search_mode="hybrid",
                ),
            ],
        )
        ```
    """

    state_schema = ToolSearchState

    def __init__(
        self,
        *,
        token_threshold: int | float = 0.10,
        context_window: int | None = None,
        always_include: list[str] | None = None,
        search_mode: Literal["bm25", "regex", "hybrid"] = "bm25",
    ) -> None:
        """Initialize the tool search middleware."""
        super().__init__()
        self.token_threshold = token_threshold
        self.context_window = context_window or DEFAULT_CONTEXT_WINDOW
        self.always_include = set(always_include or [])
        self.search_mode = search_mode

        # Index and deferred mode state - populated on first use
        self._tool_index: ToolIndex | None = None
        self._all_tools: list[BaseTool | dict[str, Any]] = []
        self._tool_name_to_tool: dict[str, BaseTool | dict[str, Any]] = {}
        self._deferred_mode: bool | None = None
        self._total_tool_tokens: int = 0

        # The search_tools tool is created lazily to capture the search_mode
        self._search_tool: BaseTool | None = None

        # Thread safety for lazy initialization
        self._init_lock = threading.Lock()

    @property
    def tools(self) -> Sequence[BaseTool]:
        """Return the search_tools meta-tool."""
        if self._search_tool is None:
            self._search_tool = self._create_search_tool()
        return [self._search_tool]

    def _create_search_tool(self) -> BaseTool:
        """Create the search_tools meta-tool."""
        default_mode = self.search_mode

        def search_tools(
            query: str,
            runtime: ToolRuntime,
            mode: Literal["bm25", "regex", "hybrid"] | None = None,
            limit: int = 5,
        ) -> str | Command:
            """Search for available tools and expand them."""
            if self._tool_index is None:
                return "Tool index not initialized. No tools available for search."

            # Enforce maximum search results limit
            actual_limit = min(limit, MAX_SEARCH_RESULTS)
            actual_mode = mode or default_mode
            results = self._tool_index.search(query, mode=actual_mode, limit=actual_limit)

            if not results:
                return f"No tools found matching '{query}'. Try a different query or search mode."

            # Get current expanded tools from state
            current_expanded: set[str] = set(runtime.state.get("expanded_tools", set()))

            # Build response with tool descriptions
            lines = [f"Found {len(results)} tool(s) matching '{query}':\n"]
            newly_expanded = []

            for tool_name, score in results:
                tool = self._tool_name_to_tool.get(tool_name)
                if tool:
                    if isinstance(tool, BaseTool):
                        desc = tool.description or "No description"
                    else:
                        desc = tool.get("description", "No description")
                    lines.append(f"- **{tool_name}** (score: {score:.2f})")
                    lines.append(f"  {desc[:200]}{'...' if len(desc) > 200 else ''}")

                    if tool_name not in current_expanded:
                        newly_expanded.append(tool_name)
                        current_expanded.add(tool_name)

            lines.append("\nThese tools are now available for use.")

            if newly_expanded:
                lines.append(f"Newly enabled: {', '.join(newly_expanded)}")

            # Return Command to update state
            if not runtime.tool_call_id:
                return "\n".join(lines)

            return Command(
                update={
                    "expanded_tools": current_expanded,
                    "messages": [
                        ToolMessage(
                            content="\n".join(lines),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        async def asearch_tools(
            query: str,
            runtime: ToolRuntime,
            mode: Literal["bm25", "regex", "hybrid"] | None = None,
            limit: int = 5,
        ) -> str | Command:
            """Async version of search_tools."""
            # The actual search is synchronous, so just delegate
            return search_tools(query, runtime, mode, limit)

        return StructuredTool.from_function(
            name="search_tools",
            func=search_tools,
            coroutine=asearch_tools,
            description=SEARCH_TOOLS_DESCRIPTION,
        )

    def _initialize_index(self, tools: Sequence[BaseTool | dict[str, Any]], model: Any) -> None:
        """Initialize the tool index and determine deferred mode.

        This method is thread-safe via double-checked locking.

        Args:
            tools: The tools to index.
            model: The model being used (for context window detection).

        Raises:
            ValueError: If the tool catalog exceeds the maximum size limit.
        """
        if self._tool_index is not None:
            return  # Already initialized

        with self._init_lock:
            # Double-check after acquiring lock
            if self._tool_index is not None:
                return

            # Validate tool catalog size
            if len(tools) > MAX_TOOL_CATALOG_SIZE:
                raise ValueError(
                    f"Tool catalog size ({len(tools)}) exceeds maximum of {MAX_TOOL_CATALOG_SIZE} tools"
                )

            self._tool_index = ToolIndex()
            self._all_tools = list(tools)

            # Build name to tool mapping
            for tool in tools:
                if isinstance(tool, BaseTool):
                    self._tool_name_to_tool[tool.name] = tool
                else:
                    name = tool.get("name", "")
                    if name:
                        self._tool_name_to_tool[name] = tool

            # Add all tools to index
            self._tool_index.add_tools(tools)

            # Calculate total token usage for tools
            self._total_tool_tokens = sum(_estimate_tool_tokens(t) for t in tools)

            # Determine threshold
            if isinstance(self.token_threshold, float):
                context_window = self.context_window
                if model is not None:
                    context_window = _get_context_window(model)
                threshold_tokens = int(context_window * self.token_threshold)
            else:
                threshold_tokens = self.token_threshold

            # Activate deferred mode if tools exceed threshold
            self._deferred_mode = self._total_tool_tokens > threshold_tokens

            logger.info(
                "ToolSearchMiddleware: %d tools, ~%d tokens. Threshold: %d. Deferred mode: %s",
                len(tools),
                self._total_tool_tokens,
                threshold_tokens,
                self._deferred_mode,
            )

    def _filter_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any]],
        expanded_tools: set[str],
    ) -> list[BaseTool | dict[str, Any]]:
        """Filter tools based on deferred mode and expansion state.

        In deferred mode, only include:
        1. Always-include tools (specified via constructor)
        2. Expanded tools (discovered via search)
        3. The search_tools tool (added by middleware.tools)

        When not in deferred mode, all tools are available.

        Args:
            tools: Full list of tools to filter.
            expanded_tools: Set of tool names that have been expanded.

        Returns:
            Filtered list of tools.
        """
        if not self._deferred_mode:
            # All tools available when not in deferred mode
            return list(tools)

        # In deferred mode, filter to allowed tools
        filtered: list[BaseTool | dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                name = tool.name
            else:
                name = tool.get("name", "")

            if name in self.always_include or name in expanded_tools:
                filtered.append(tool)

        return filtered

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        """Prepare the model request with filtered tools and system prompt.

        Common logic shared between wrap_model_call and awrap_model_call.

        Args:
            request: The original model request.

        Returns:
            Modified model request with filtered tools and updated system prompt.
        """
        # Initialize index on first model call
        self._initialize_index(request.tools, request.model)

        # Get expanded tools from state
        expanded_tools: set[str] = set(request.state.get("expanded_tools", self.always_include))

        # Filter tools
        filtered_tools = self._filter_tools(request.tools, expanded_tools)

        # Add system prompt if in deferred mode
        if self._deferred_mode:
            new_system_message = append_to_system_message(
                request.system_message, TOOL_SEARCH_SYSTEM_PROMPT
            )
            return request.override(
                tools=filtered_tools,
                system_message=new_system_message,
            )
        else:
            # Still update tools list to ensure search_tools is included
            return request.override(tools=filtered_tools)

    def before_agent(
        self,
        state: ToolSearchState,
        runtime: Runtime,
        config: RunnableConfig | None = None,
    ) -> ToolSearchStateUpdate | None:
        """Initialize expanded_tools in state if not present."""
        if "expanded_tools" in state:
            return None

        # Initialize with always_include tools
        return ToolSearchStateUpdate(expanded_tools=set(self.always_include))

    async def abefore_agent(
        self,
        state: ToolSearchState,
        runtime: Runtime,
        config: RunnableConfig | None = None,
    ) -> ToolSearchStateUpdate | None:
        """Async version of before_agent."""
        return self.before_agent(state, runtime, config)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Filter tools based on expansion state and add system prompt."""
        prepared_request = self._prepare_request(request)
        return handler(prepared_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        prepared_request = self._prepare_request(request)
        return await handler(prepared_request)


__all__ = [
    "BM25Index",
    "ToolIndex",
    "ToolSearchMiddleware",
    "ToolSearchState",
]
