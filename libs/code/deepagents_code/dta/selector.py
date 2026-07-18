"""Stage 2 LLM-based tool selector for Dynamic Tool Allocation (DTA)."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deepagents_code.dta.utils import get_dta_fast_model

try:
    from deepagents_code.config import create_model
except ImportError:
    create_model = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Characters-per-token estimate used by the schema token budget heuristic.
_CHARS_PER_TOKEN: int = 4


class ToolSelectionResult(BaseModel):
    """Structured output returned by the selector LLM."""

    selected_tools: list[str] = Field(description="List of tool names selected")
    rationale: str = Field(description="Reasoning for selecting these tools")


class ToolSelectorNode:
    """LLM-based Stage 2 ranker that narrows BM25/TF-IDF candidates to a budget.

    When the candidate count already fits within *budget* the LLM is bypassed
    and the full candidate set is returned immediately.  If the LLM call fails,
    the selector falls back to taking the top-*budget* Stage 1 results in order.

    All LLM errors are caught and handled by failing open, so the agent always
    receives a valid (if un-ranked) toolset.
    """

    def __init__(self, model_spec: str | None = None) -> None:
        """Initialise the selector node.

        Args:
            model_spec: Optional explicit model spec string.  When ``None``,
                resolved at call time from
                :func:`~deepagents_code.dta.utils.get_dta_fast_model`.
        """
        self.model_spec = model_spec

    @staticmethod
    def _build_prompt(
        messages: list[BaseMessage],
        candidates: list[dict[str, Any]],
        budget: int,
    ) -> list[BaseMessage]:
        """Build the LLM prompt for tool selection.

        Args:
            messages: Conversation history; the most recent human message is
                extracted as the query.
            candidates: Stage 1 candidate tool schemas.
            budget: Maximum number of tools the LLM should select.

        Returns:
            A two-element list ``[SystemMessage, HumanMessage]`` ready for
            ``llm.invoke``.
        """
        last_human = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                last_human = getattr(msg, "content", "") or ""
                break

        sys_prompt = (
            "You are a Tool Selector. "
            "Select the most relevant tools to fulfill the user query.\n"
            f"You have a budget of {budget} tools. Do not select redundant tools.\n"
            f"Candidate tools: {json.dumps(candidates)}\n"
        )
        return [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {last_human}"),
        ]

    def _process_result(
        self,
        result: ToolSelectionResult,
        candidates: list[dict[str, Any]],
        budget: int,
        token_limit: int,
    ) -> set[str]:
        """Validate and pack LLM output within budget constraints.

        Args:
            result: Structured output from the selector LLM.
            candidates: Stage 1 candidate schemas used for name validation.
            budget: Maximum number of tool names to return.
            token_limit: Approximate schema token cap.

        Returns:
            Set of validated tool name strings within *budget*.

        Raises:
            ValueError: When the LLM returns no names matching the candidate list.
        """
        selected_candidates = [
            c
            for name in result.selected_tools
            if (c := next((c for c in candidates if c.get("name") == name), None))
            is not None
        ]

        if not selected_candidates:
            msg = "LLM returned no valid tool names from the candidate list"
            raise ValueError(msg)

        return self._pack_within_budget(selected_candidates, budget, token_limit)

    def select(
        self,
        messages: list[BaseMessage],
        candidates: list[dict[str, Any]],
        budget: int = 8,
        token_limit: int = 4000,
    ) -> set[str]:
        """Synchronously select tools from *candidates* within *budget*.

        Args:
            messages: Full conversation history used to extract the user query.
            candidates: Stage 1 candidate tool schemas.
            budget: Maximum number of tools to allocate.
            token_limit: Approximate per-schema character token budget.

        Returns:
            Set of selected tool name strings.

        Raises:
            ImportError: When no model provider is configured and
                ``create_model`` is unavailable.
        """
        if not candidates:
            return set()

        if len(candidates) <= budget:
            return self._pack_within_budget(candidates, budget, token_limit)

        try:
            if create_model is None:
                msg = "create_model unavailable — no provider configured"
                raise ImportError(msg)  # noqa: TRY301

            spec = self.model_spec or get_dta_fast_model()
            model_res = create_model(spec)
            llm = model_res.model.with_structured_output(ToolSelectionResult)
            prompt = self._build_prompt(messages, candidates, budget)
            result: ToolSelectionResult = llm.invoke(
                prompt, config={"tags": ["dta_selector"]}
            )
            return self._process_result(result, candidates, budget, token_limit)
        except Exception:
            logger.warning(
                "DTA Stage 2 sync selection failed, falling back to Stage 1 ranking",
                exc_info=True,
            )
            return self._pack_within_budget(candidates, budget, token_limit)

    async def aselect(
        self,
        messages: list[BaseMessage],
        candidates: list[dict[str, Any]],
        budget: int = 8,
        token_limit: int = 4000,
    ) -> set[str]:
        """Asynchronously select tools from *candidates* within *budget*.

        Mirrors :meth:`select` but uses ``ainvoke`` for async event loops.

        Args:
            messages: Full conversation history used to extract the user query.
            candidates: Stage 1 candidate tool schemas.
            budget: Maximum number of tools to allocate.
            token_limit: Approximate per-schema character token budget.

        Returns:
            Set of selected tool name strings.

        Raises:
            ImportError: When no model provider is configured and
                ``create_model`` is unavailable.
        """
        if not candidates:
            return set()

        if len(candidates) <= budget:
            return self._pack_within_budget(candidates, budget, token_limit)

        try:
            if create_model is None:
                msg = "create_model unavailable — no provider configured"
                raise ImportError(msg)  # noqa: TRY301

            spec = self.model_spec or get_dta_fast_model()
            model_res = create_model(spec)
            llm = model_res.model.with_structured_output(ToolSelectionResult)
            prompt = self._build_prompt(messages, candidates, budget)
            result: ToolSelectionResult = await llm.ainvoke(
                prompt, config={"tags": ["dta_selector"]}
            )
            return self._process_result(result, candidates, budget, token_limit)
        except Exception:
            logger.warning(
                "DTA Stage 2 async selection failed, falling back to Stage 1 ranking",
                exc_info=True,
            )
            return self._pack_within_budget(candidates, budget, token_limit)

    @staticmethod
    def _pack_within_budget(
        candidates: list[dict[str, Any]],
        budget: int,
        token_limit: int,
    ) -> set[str]:
        """Greedily select tool names without exceeding *budget* or *token_limit*.

        Each tool's contribution to the token budget is estimated as
        ``len(json.dumps(schema)) / _CHARS_PER_TOKEN``.

        Args:
            candidates: Ordered list of tool schema dicts to pack.
            budget: Hard cap on the number of tools selected.
            token_limit: Soft cap on accumulated estimated schema tokens.

        Returns:
            Set of selected tool name strings.
        """
        selected: set[str] = set()
        tokens: float = 0.0
        for candidate in candidates:
            name = candidate.get("name", "")
            if not name:
                continue
            token_cost = len(json.dumps(candidate)) / _CHARS_PER_TOKEN
            if len(selected) >= budget or tokens + token_cost > token_limit:
                break
            selected.add(name)
            tokens += token_cost
        return selected
