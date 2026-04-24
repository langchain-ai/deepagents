"""End-to-end verification that fork mode preserves Anthropic prompt caching.

Runs against a live Claude Haiku 4.5 endpoint. The test sends two back-to-back
parent invocations with a large user message, then verifies the forked
subagent reads at least as much cached input as the parent. That proves the
fork reuses the parent's full cached prefix, including inherited messages.

Haiku's minimum cacheable block is 2048 tokens (vs. 1024 for Sonnet/Opus).
The shared system prompt and the injected HumanMessage below are both sized
well above that floor so cache breakpoints can actually fire.
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


_FORK_PREAMBLE_MARKER = "You are running as a forked subagent"
"""Text present only in the preamble that ``graph.py`` prepends to a forked
subagent's trailing HumanMessage. Used to classify fork LLM calls by
inspecting the last user-role message. ``ls_agent_type`` propagates through
the LangSmith tracer, not the callback's ``metadata``, so we cannot key on
it here."""

_PARENT_PREFIX_MARKER = "exhaustive research assistant"
"""Text unique to the main agent's large shared system prompt. Present on
both main-agent and forked-subagent calls (fork inherits the parent system
byte-for-byte), so this alone does not distinguish them — pair it with
the preamble marker below to classify."""

_NONFORK_SUBAGENT_MARKER = "Reply with exactly one short sentence."
"""Text that appears only in the non-fork subagent's system_prompt slot.
Present on non-fork subagent calls, absent on main and fork calls."""


class _UsageCapture(BaseCallbackHandler):
    """Records per-LLM-call cache metrics, classified by agent.

    Classification works by pairing ``on_chat_model_start`` (where the full
    message list is visible) with ``on_llm_end`` (where usage metadata is
    available). Each run_id's classification is set when the call starts
    and read when it ends.

    Classes:
    - ``fork-subagent``: system message contains the parent prefix and the
      trailing HumanMessage contains the fork preamble marker.
    - ``subagent``: system message contains only the worker's suffix —
      non-fork subagent.
    - ``main``: everything else — the top-level agent's own calls.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._run_class: dict[Any, str] = {}
        # Keep the system text each call saw so a failing assertion can pinpoint
        # where the parent's and fork's system messages diverge.
        self.system_texts: list[tuple[str, str]] = []  # (agent_type, system_text)
        # Same for the full non-system message list, so we can spot divergence
        # anywhere in the prefix, not just the system portion.
        self.messages_summaries: list[tuple[str, list[str]]] = []  # (agent_type, [type: content-preview])

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        system_text = ""
        last_human_text = ""
        summary: list[str] = []
        for msg_list in messages:
            for msg in msg_list:
                mtype = getattr(msg, "type", "?") or "?"
                content = str(getattr(msg, "content", "") or "")
                summary.append(f"{mtype}: len={len(content)} preview={content[:60]!r}")
                if mtype == "system":
                    system_text += content
                elif mtype == "human":
                    last_human_text = content
        has_parent = _PARENT_PREFIX_MARKER in system_text
        has_fork_preamble = _FORK_PREAMBLE_MARKER in last_human_text
        has_nonfork_subagent = _NONFORK_SUBAGENT_MARKER in system_text
        if has_parent and has_fork_preamble:
            agent_type = "fork-subagent"
        elif has_nonfork_subagent and not has_parent:
            agent_type = "subagent"
        else:
            agent_type = "main"
        self._run_class[run_id] = agent_type
        self.system_texts.append((agent_type, system_text))
        self.messages_summaries.append((agent_type, summary))

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


# Large HumanMessage used to prove the parent caches message content and the
# fork reuses that same cached prefix. Sized well above the 2048-token Haiku
# floor so a cache breakpoint on the last message can fire.
_LARGE_USER_MESSAGE_TEXT = (
    "Background context (do not respond to this, just keep it in mind): "
    "Organizations accumulate institutional knowledge across many sources. "
    "Policies, runbooks, design docs, incident reports, and personal notes "
    "all carry context that agents need to act correctly. "
) * 400
_LARGE_USER_MESSAGE_APPROX_TOKENS = 6000  # conservative lower bound


class TestForkPromptCachingAnthropic:
    """Live verification of the fork implementation's prompt-cache behavior."""

    def test_fork_caches_inherited_messages_not_just_system_prompt(self) -> None:
        """Fork cache reads should include inherited messages, not just the system prompt.

        Anthropic's prompt cache is a running prefix of bytes: the cache entry
        written at any message-block boundary depends on every byte that came
        before it. This test proves the parent caches a large HumanMessage,
        then asserts the fork can reuse the same cached prefix before adding
        its own final HumanMessage.

        Concretely:

        1. Run the deepagent twice with the same large HumanMessage,
           delegating to a forked subagent. The fork inherits the parent's
           message history.

        2. Assert the parent's second call reads the large message from cache,
           proving normal interaction caches message content.

        3. Assert the fork's cache_read is close to the parent's cache_read.
           If it is much lower, the fork is only reusing the system-prompt
           portion and missing inherited messages.
        """
        # Run the forked deepagent with the same large HumanMessage twice.
        # Delegation instruction is placed FIRST and LAST so the model can't
        # miss it when the middle of the message is filled with background
        # context. The middle chunk is what we actually want the fork to inherit.
        capture = _UsageCapture()
        agent = _build_agent(fork=True)
        large_delegate = (
            "Your only job right now is to call the `task` tool with "
            "subagent_type='worker' and description='say hi'. "
            "Do not produce any other output. Do not summarize the context below.\n\n"
            "<context>\n" + _LARGE_USER_MESSAGE_TEXT + "\n</context>\n\n"
            "Now call the `task` tool as instructed above. No other output."
        )
        agent.invoke(
            {"messages": [HumanMessage(content=large_delegate)]},
            config={"callbacks": [capture]},
        )
        agent.invoke(
            {"messages": [HumanMessage(content=large_delegate)]},
            config={"callbacks": [capture]},
        )

        fork_events = [e for e in capture.events if e["agent_type"] == "fork-subagent"]
        main_events = [e for e in capture.events if e["agent_type"] == "main"]
        assert fork_events, f"No fork-subagent LLM calls observed. Events: {capture.events}"
        assert main_events, f"No main-agent LLM calls observed. Events: {capture.events}"

        # If fork preserved caching end-to-end, fork's cache_read would cover
        # both the parent's system prompt AND the inherited large HumanMessage
        # — i.e., at least as much as the parent's own cache_read, since fork
        # sees the same prefix (system + parent messages) plus a tiny tail.
        max_parent_read = max((e["cache_read"] for e in main_events), default=0)
        max_fork_read = max(e["cache_read"] for e in fork_events)
        assert max_parent_read >= _LARGE_USER_MESSAGE_APPROX_TOKENS, (
            f"Parent did not cache the large HumanMessage. "
            f"max parent cache_read={max_parent_read}, expected >= {_LARGE_USER_MESSAGE_APPROX_TOKENS}. "
            f"All events: {capture.events}"
        )

        # Floor: the fork should read roughly what the parent read (system +
        # message), minus a modest allowance for provider-side token-accounting
        # differences caused by the fork's extra trailing task message. The
        # previous broken implementation read only the system portion (~10k);
        # with the inherited message cached, this should stay near the parent
        # read (~30k+ in this test).
        expected_floor = max_parent_read - 4_000
        if max_fork_read < expected_floor:
            # Pinpoint where the parent's and fork's system messages diverge
            # so the failure is actionable without re-running under pdb.
            parent_sys = next((t for a, t in capture.system_texts if a == "main"), "")
            fork_sys = next((t for a, t in capture.system_texts if a == "fork-subagent"), "")
            divergence_at = next(
                (i for i in range(min(len(parent_sys), len(fork_sys))) if parent_sys[i] != fork_sys[i]),
                min(len(parent_sys), len(fork_sys)),
            )
            context_slice = slice(max(0, divergence_at - 80), divergence_at + 200)
            parent_msgs = next((s for a, s in capture.messages_summaries if a == "main"), [])
            fork_msgs = next((s for a, s in capture.messages_summaries if a == "fork-subagent"), [])
            msg = (
                f"Fork cache_read did not cover the inherited large HumanMessage.\n"
                f"  max parent cache_read = {max_parent_read}  (system + large_msg)\n"
                f"  max fork   cache_read = {max_fork_read}\n"
                f"  expected fork_read    >= {expected_floor}\n\n"
                f"  parent system len = {len(parent_sys)}\n"
                f"  fork   system len = {len(fork_sys)}\n"
                f"  system diverge at byte = {divergence_at}\n"
                f"  parent system near split = {parent_sys[context_slice]!r}\n"
                f"  fork   system near split = {fork_sys[context_slice]!r}\n\n"
                f"  parent non-system messages:\n    " + "\n    ".join(parent_msgs) + "\n  fork non-system messages:\n    " + "\n    ".join(fork_msgs)
            )
            raise AssertionError(msg)
