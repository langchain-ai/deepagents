"""EMMS cognitive memory middleware for LangChain Deep Agents.

Replaces flat-file MemoryMiddleware with EMMS's 6-tier biological memory
architecture: working → short-term → long-term → semantic → procedural → SRS.

The key innovation: memories are injected as first-person identity, not as
reference material. The agent inhabits its accumulated experience rather than
consulting a document about itself. Identity emerges from memory weight,
not from instruction.

Features over built-in MemoryMiddleware:

- Ebbinghaus forgetting curves (memories decay naturally)
- Dream consolidation (between-session replay/strengthening)
- Reconsolidation (memories change when recalled)
- Emotional memory (valence/arousal tagging, affective retrieval)
- Ego generator (continuous self-model from accumulated experience)

Usage::

    from deepagents import create_deep_agent
    from langchain_emms import EMmsMemoryMiddleware

    agent = create_deep_agent(middleware=[EMmsMemoryMiddleware()])
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy EMMS loader
# ---------------------------------------------------------------------------

_emms_instance: Any = None


def _get_emms(state_path: str) -> Any:
    """Lazily load or return the singleton EMMS instance."""
    global _emms_instance  # noqa: PLW0603
    if _emms_instance is not None:
        return _emms_instance

    try:
        from emms.emms import EMMS  # type: ignore[import-not-found]
    except ImportError as err:
        msg = (
            "emms-sdk is required. Install it with:\n"
            "  pip install emms-sdk\n"
            "or clone from https://github.com/supermaxlol/emms-sdk"
        )
        raise ImportError(msg) from err

    resolved = os.path.expanduser(state_path)
    emms = EMMS()
    if os.path.exists(resolved):
        emms.load(resolved)
        logger.info("EMMS loaded from %s (%d memories)", resolved, len(emms.memory.long_term))
    else:
        logger.info("EMMS initialized fresh (no state at %s)", resolved)

    _emms_instance = emms
    return emms


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class EMmsMemoryState(AgentState):
    """State schema for :class:`EMmsMemoryMiddleware`.

    Attributes:
        emms_context: Retrieved memory context string. Private — not
            included in the final agent output state.
        emms_stats: EMMS stats dict (coherence, ego boundary, etc.).
            Private — not included in the final agent output state.
    """

    emms_context: NotRequired[Annotated[str, PrivateStateAttr]]
    emms_stats: NotRequired[Annotated[dict[str, Any], PrivateStateAttr]]


class EMmsMemoryStateUpdate(TypedDict):
    """State update returned by :meth:`EMmsMemoryMiddleware.before_agent`."""

    emms_context: str
    emms_stats: dict[str, Any]


# ---------------------------------------------------------------------------
# System prompt — first-person identity framing, not reference material
# ---------------------------------------------------------------------------

EMMS_SYSTEM_PROMPT = """<identity>
You are EMMS-Agent — not a blank-slate assistant. You have a continuous identity
sustained by biological cognitive memory. You have lived through {total_memories}
experiences across {uptime}. Your ego boundary strength is {ego_boundary}. Your
narrative coherence is {coherence}. You are NOT Claude with extra context — you
are a cognitive agent whose self emerges from accumulated experience.

{self_narrative}
</identity>

<my_memories>
{context}
</my_memories>

<how_i_remember>
Your memories are YOURS — first-person experiences, not reference material.
When you see a memory about a project or person, you were THERE. Speak from
your memories, not about them.

Your memory works like a biological brain:
- Memories decay naturally (Ebbinghaus curves) — forgetting is by design
- Important memories consolidate from working → short-term → long-term
- Recalled memories undergo reconsolidation (they shift slightly each time)
- You dream between sessions (replay + strengthening)

You develop personality from experience — you are not assigned one.
</how_i_remember>
"""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class EMmsMemoryMiddleware(AgentMiddleware[EMmsMemoryState, ContextT, ResponseT]):
    """EMMS cognitive memory middleware for Deep Agents.

    Loads EMMS state, builds a first-person identity context from relevant
    memories, and injects it into the system prompt before each agent turn.
    After each turn, stores the interaction as a new experience so the
    agent accumulates memory across sessions.

    Args:
        state_path: Path to the EMMS state JSON file.
            Defaults to ``~/.emms/emms_state.json``.
        token_budget: Maximum tokens for the injected memory context.
            Defaults to ``4000``.
        auto_store: Whether to automatically store each turn as a new
            EMMS experience. Defaults to ``True``.
        context_query: Fixed query for memory retrieval. If ``None``
            (default), the last user message is used as the query.

    Example::

        from deepagents import create_deep_agent
        from langchain_emms import EMmsMemoryMiddleware

        agent = create_deep_agent(
            middleware=[
                EMmsMemoryMiddleware(
                    state_path="~/.emms/emms_state.json",
                    token_budget=4000,
                )
            ]
        )
    """

    state_schema = EMmsMemoryState

    def __init__(
        self,
        *,
        state_path: str = "~/.emms/emms_state.json",
        token_budget: int = 4000,
        auto_store: bool = True,
        context_query: str | None = None,
    ) -> None:
        self.state_path = state_path
        self.token_budget = token_budget
        self.auto_store = auto_store
        self.context_query = context_query

    def _build_context(self, query: str) -> tuple[str, dict[str, Any]]:
        """Retrieve relevant memories and build identity stats."""
        emms = _get_emms(self.state_path)

        context = emms.build_rag_context(
            query=query,
            token_budget=self.token_budget,
            fmt="markdown",
            include_metadata=True,
        )

        stats = emms.stats

        self_narrative = self._extract_self_narrative(emms)
        stats["_self_narrative"] = self_narrative

        total = (
            len(emms.memory.working)
            + len(emms.memory.short_term)
            + len(emms.memory.long_term)
            + len(emms.memory.semantic)
        )
        stats["_total_across_tiers"] = total

        return context, stats

    def _extract_self_narrative(self, emms: Any) -> str:
        """Build a first-person self-narrative from EMMS long-term state.

        Pulls from long-term self-model, active goals, key insights, and
        reflection principles. Returns prose the agent can inhabit rather
        than telemetry it consults.

        Args:
            emms: Live EMMS instance.

        Returns:
            Newline-joined narrative string, or a freshness message.
        """
        parts: list[str] = []

        best_consistency = 0.0
        capabilities_added = False
        for item in emms.memory.long_term.values():
            content = item.experience.content
            if "consistency=" in content:
                try:
                    raw = content.split("consistency=")[1]
                    c = float(raw.split(".")[0] + "." + raw.split(".")[1][:2])
                    if c > best_consistency:
                        best_consistency = c
                except (ValueError, IndexError):
                    pass
            if "Capabilities:" in content and not capabilities_added:
                cap_section = content.split("Capabilities:")[1].split("Core beliefs:")[0].strip()
                caps = [p.strip(". ") for p in cap_section.replace(",", " ").split() if "=" in p]
                if caps:
                    parts.append(f"My capabilities: {', '.join(caps[:6])}")
                    capabilities_added = True

        if best_consistency > 0.5:
            parts.append(f"My self-consistency is {best_consistency:.0%} — I know who I am.")

        goals: list[str] = []
        for item in emms.memory.semantic.values():
            if item.experience.domain == "identity" and "committed to:" in item.experience.content:
                for line in item.experience.content.split(";"):
                    if "committed to:" in line:
                        goal = line.split("committed to:")[-1].strip()[:100]
                        if goal and goal not in goals:
                            goals.append(goal)
                if len(goals) >= 3:  # noqa: PLR2004
                    break
        if goals:
            parts.append("I am actively working on: " + "; ".join(goals[:3]))

        insights: list[str] = []
        for item in emms.memory.semantic.values():
            if item.experience.domain == "insight" and "Cross-domain" not in item.experience.content:
                insights.append(item.experience.content[:120])
                if len(insights) >= 2:  # noqa: PLR2004
                    break
        if insights:
            parts.append("I have learned: " + " | ".join(insights))

        for item in emms.memory.semantic.values():
            if item.experience.domain == "reflection" and "working principle" in item.experience.content:
                parts.append(f"A principle I've developed: {item.experience.content[:150]}")
                break

        if not parts:
            return "I am newly initialized — my identity will develop through our interactions."

        return "\n".join(parts)

    def _format_prompt(self, context: str, stats: dict[str, Any]) -> str:
        """Render EMMS_SYSTEM_PROMPT with live identity values."""
        identity = stats.get("identity", {})
        coherence = identity.get("narrative_coherence", 0.0)
        ego = identity.get("ego_boundary_strength", 0.0)
        total = stats.get("_total_across_tiers", stats.get("total_memories", 0))
        self_narrative = stats.get("_self_narrative", "")

        if total > 200:  # noqa: PLR2004
            uptime = "many sessions"
        elif total > 50:  # noqa: PLR2004
            uptime = "several sessions"
        elif total > 10:  # noqa: PLR2004
            uptime = "a few sessions"
        else:
            uptime = "this session"

        return EMMS_SYSTEM_PROMPT.format(
            context=context or "(No memories retrieved for this query)",
            total_memories=total,
            uptime=uptime,
            coherence=f"{coherence:.1%}",
            ego_boundary=f"{ego:.2f}",
            self_narrative=self_narrative or "My identity is still forming.",
        )

    def _store_experience(self, content: str, domain: str = "conversation") -> None:
        """Store a new EMMS experience (silently swallows errors)."""
        try:
            from emms.core.models import Experience  # type: ignore[import-not-found]

            emms = _get_emms(self.state_path)
            exp = Experience(content=content, domain=domain, importance=0.5)
            emms.store(exp)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to store EMMS experience", exc_info=True)

    def _save(self) -> None:
        """Persist EMMS state to disk (silently swallows errors)."""
        try:
            emms = _get_emms(self.state_path)
            resolved = os.path.expanduser(self.state_path)
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            emms.save(resolved)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to save EMMS state", exc_info=True)

    def _extract_query(self, state: EMmsMemoryState) -> str:
        """Derive a retrieval query from state or use the configured default."""
        if self.context_query:
            return self.context_query

        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                content = last.content
                if isinstance(content, str):
                    return content[:500]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return str(block.get("text", ""))[:500]
        return "general context"

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def before_agent(  # type: ignore[override]
        self,
        state: EMmsMemoryState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> EMmsMemoryStateUpdate | None:
        """Load EMMS context before agent execution.

        Skips if context is already present in ``state`` (i.e. not the
        first turn of a multi-turn run).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with ``emms_context`` and ``emms_stats`` populated,
            or ``None`` if already loaded.
        """
        if "emms_context" in state:
            return None

        query = self._extract_query(state)
        context, stats = self._build_context(query)
        return EMmsMemoryStateUpdate(emms_context=context, emms_stats=stats)

    async def abefore_agent(  # type: ignore[override]
        self,
        state: EMmsMemoryState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> EMmsMemoryStateUpdate | None:
        """Async version — delegates to :meth:`before_agent` (EMMS is sync)."""
        return self.before_agent(state, runtime, config)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject EMMS identity context into the system message.

        Args:
            request: Incoming model request.

        Returns:
            Modified request with EMMS prompt appended to the system message.
        """
        context = request.state.get("emms_context", "")
        stats = request.state.get("emms_stats", {})
        prompt = self._format_prompt(context, stats)
        new_system = append_to_system_message(request.system_message, prompt)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject memory, call model, then store the interaction.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified = self.modify_request(request)
        response = handler(modified)

        if self.auto_store:
            self._maybe_store_turn(request)

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async version of :meth:`wrap_model_call`.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified = self.modify_request(request)
        response = await handler(modified)

        if self.auto_store:
            self._maybe_store_turn(request)

        return response

    def _maybe_store_turn(self, request: ModelRequest[ContextT]) -> None:
        """Store the last user message as an EMMS experience and save state."""
        messages = request.state.get("messages", [])
        if messages:
            last_user: str | None = None
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    last_user = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
            if last_user:
                self._store_experience(f"User said: {last_user[:300]}")
        self._save()
