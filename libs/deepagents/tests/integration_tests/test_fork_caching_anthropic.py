"""End-to-end verification that fork mode preserves Anthropic prompt caching.

Runs against a live Claude Haiku 4.5 endpoint. The test sends two back-to-back
invocations that route through a forked subagent and asserts the fork's second
invocation hits the prompt cache. A negative control asserts that a non-fork
subagent (same structure, `fork=False`) does not hit the cache on the second
run — the delta confirms fork is what unlocks the reuse.

Haiku's minimum cacheable block is 2048 tokens (vs. 1024 for Sonnet/Opus). The
shared system prompt below is sized well above that floor so the cache
breakpoint actually fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from deepagents.graph import create_deep_agent

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.outputs import LLMResult

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Haiku's prompt-cache minimum is 2048 tokens. Build a deterministic prefix
# well above that floor so we're not racing the boundary when tokenizers
# drift slightly between model versions.
_FILLER_SENTENCE = (
    "You are an exhaustive research assistant. You help the user synthesize "
    "long-form technical documentation into concise, actionable summaries. "
    "You always cite your sources, flag uncertainty, and prefer precise "
    "terminology over colloquialisms. "
)
LARGE_SHARED_PREFIX = _FILLER_SENTENCE * 120  # ~3500+ tokens, safely above 2048


_FORK_SUFFIX_MARKER = "Reply with exactly one short sentence."
"""Text unique to the worker subagent's own system_prompt. Used below to
classify which agent made each LLM call by inspecting the system message
content — ``ls_agent_type`` is propagated via the LangSmith tracer, not the
callback's ``metadata``, so we cannot key on it here."""

_PARENT_PREFIX_MARKER = "exhaustive research assistant"
"""Text unique to the main agent's large shared system prompt. Combined
with ``_FORK_SUFFIX_MARKER``, its presence distinguishes a fork call (both
markers) from a non-fork subagent call (suffix only)."""


class _UsageCapture(BaseCallbackHandler):
    """Records per-LLM-call cache metrics, classified by agent.

    Classification works by pairing ``on_chat_model_start`` (where the full
    message list is visible) with ``on_llm_end`` (where usage metadata is
    available). Each run_id's classification is set when the call starts
    and read when it ends.

    Classes:
    - ``fork-subagent``: system message contains both the parent prefix
      and the worker's fork suffix — the composed fork prompt.
    - ``subagent``: system message contains only the worker's suffix —
      non-fork subagent.
    - ``main``: everything else — the top-level agent's own calls.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._run_class: dict[Any, str] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        system_text = ""
        for msg_list in messages:
            for msg in msg_list:
                if getattr(msg, "type", None) == "system":
                    system_text += str(getattr(msg, "content", "") or "")
        has_parent = _PARENT_PREFIX_MARKER in system_text
        has_fork_suffix = _FORK_SUFFIX_MARKER in system_text
        if has_parent and has_fork_suffix:
            self._run_class[run_id] = "fork-subagent"
        elif has_fork_suffix:
            self._run_class[run_id] = "subagent"
        else:
            self._run_class[run_id] = "main"

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        agent_type = self._run_class.pop(run_id, "main")
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                # Read raw Anthropic fields directly rather than the normalized
                # ``usage_metadata.input_token_details`` path: LangChain's
                # normalizer zeroes ``cache_creation`` whenever the ephemeral
                # TTL breakdown is present, making that surface unusable for
                # verifying cache writes. See
                # https://github.com/langchain-ai/langchain/issues/36991.
                usage = (getattr(msg, "response_metadata", None) or {}).get("usage", {})
                self.events.append(
                    {
                        "agent_type": agent_type,
                        "cache_read": int(usage.get("cache_read_input_tokens", 0)),
                        "cache_creation": int(usage.get("cache_creation_input_tokens", 0)),
                    }
                )


def _build_agent(*, fork: bool) -> object:
    model = ChatAnthropic(model=HAIKU_MODEL, temperature=0)
    return create_deep_agent(
        model=model,
        system_prompt=LARGE_SHARED_PREFIX,
        tools=[],
        subagents=[
            {
                "name": "worker",
                "description": "Worker subagent that answers briefly.",
                "system_prompt": "Reply with exactly one short sentence.",
                "tools": [],
                "fork": fork,
            },
        ],
    )


_DELEGATE_PROMPT = "Call the `task` tool with subagent_type='worker' and description='say hi'. Do this immediately."


class TestForkPromptCachingAnthropic:
    """Live verification that fork mode reuses the Anthropic prompt cache.

    The whole point of ``fork=True`` is that the subagent's prefix matches
    the parent's byte-for-byte, so the provider's prompt cache serves the
    second invocation. These tests verify that directly by capturing every
    LLM call via a callback and filtering for events tagged
    ``ls_agent_type="fork-subagent"``.
    """

    def test_fork_subagent_reuses_prompt_cache_across_invocations(self) -> None:
        """Forked subagent's second call must read a meaningful chunk of tokens from cache.

        Two back-to-back invocations of the agent, each asking it to delegate
        to a forked subagent. Across the two fork calls, at least one must
        report ``cache_read_input_tokens > 0``. The exact boundary between
        creation and read depends on prior cache state, so we only assert
        the reuse — it's the signal that proves the fork's prefix aligns
        with a previously-seen prefix.
        """
        capture = _UsageCapture()
        agent = _build_agent(fork=True)

        agent.invoke(
            {"messages": [HumanMessage(content=_DELEGATE_PROMPT)]},
            config={"callbacks": [capture]},
        )
        agent.invoke(
            {"messages": [HumanMessage(content=_DELEGATE_PROMPT)]},
            config={"callbacks": [capture]},
        )

        fork_events = [e for e in capture.events if e["agent_type"] == "fork-subagent"]
        assert fork_events, (
            f"No fork-subagent LLM calls were observed. All events: {capture.events}. "
            "The model may have ignored the `task` tool; update the delegate prompt."
        )
        assert any(e["cache_read"] > 0 for e in fork_events), (
            f"Fork subagent never reused the prompt cache. Fork events: {fork_events}. "
            "The fork's prefix drifted between invocations — check that "
            "_compose_fork_system_prompt is byte-stable and that no middleware "
            "between fork composition and the model mutates the system message."
        )

    def test_nonfork_subagent_does_not_reuse_prompt_cache(self) -> None:
        """Negative control: ``fork=False`` subagent shows zero cache reuse.

        A non-fork subagent is seeded with only ``[HumanMessage(description)]``
        and its own tiny system prompt (far under Haiku's 2048-token floor),
        so its prefix does not align with any previous call and no cache
        breakpoint can fire. If this test ever shows a cache read, either
        the fork and non-fork paths have converged (a real regression) or
        ambient cross-call caching is masking the fork-specific behavior.
        """
        capture = _UsageCapture()
        agent = _build_agent(fork=False)

        agent.invoke(
            {"messages": [HumanMessage(content=_DELEGATE_PROMPT)]},
            config={"callbacks": [capture]},
        )
        agent.invoke(
            {"messages": [HumanMessage(content=_DELEGATE_PROMPT)]},
            config={"callbacks": [capture]},
        )

        subagent_events = [e for e in capture.events if e["agent_type"] == "subagent"]
        assert subagent_events, (
            f"No subagent LLM calls were observed. All events: {capture.events}. "
            "The model may have ignored the `task` tool; update the delegate prompt."
        )
        for event in subagent_events:
            assert event["cache_read"] == 0, (
                f"Non-fork subagent unexpectedly reused the prompt cache: {event}. "
                "This defeats the fork-vs-non-fork distinction — either the fork "
                "path is silently active when it should not be, or ambient caching "
                "is sharing prefix state in a way the middleware doesn't control."
            )
