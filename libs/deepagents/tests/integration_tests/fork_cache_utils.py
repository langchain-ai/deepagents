"""Shared integration-test helpers for fork prompt-cache verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from deepagents.graph import create_deep_agent

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import LLMResult


class AgentLike(Protocol):
    """Minimal protocol for the compiled agent returned by `create_deep_agent`."""

    def invoke(self, input_: object, config: object | None = None) -> object:
        """Invoke the compiled agent."""


_FILLER_SENTENCE = (
    "You are an exhaustive research assistant. You help the user synthesize "
    "long-form technical documentation into concise, actionable summaries. "
    "You always cite your sources, flag uncertainty, and prefer precise "
    "terminology over colloquialisms. "
)
LARGE_SHARED_PREFIX = _FILLER_SENTENCE * 120

_FORK_PREAMBLE_MARKER = "You are running as a forked subagent"
_PARENT_PREFIX_MARKER = "exhaustive research assistant"
_NONFORK_SUBAGENT_MARKER = "Reply with exactly one short sentence."

LARGE_USER_MESSAGE_TEXT = (
    "Background context (do not respond to this, just keep it in mind): "
    "Organizations accumulate institutional knowledge across many sources. "
    "Policies, runbooks, design docs, incident reports, and personal notes "
    "all carry context that agents need to act correctly. "
) * 400
LARGE_USER_MESSAGE_APPROX_TOKENS = 6000


class UsageCapture(BaseCallbackHandler):
    """Records per-LLM-call cache metrics, classified by agent."""

    def __init__(self, cache_read_tokens: Callable[[Any], int]) -> None:
        self._cache_read_tokens = cache_read_tokens
        self.events: list[dict[str, Any]] = []
        self._run_class: dict[Any, str] = {}
        self.system_texts: list[tuple[str, str]] = []
        self.messages_summaries: list[tuple[str, list[str]]] = []

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
                self.events.append(
                    {
                        "agent_type": agent_type,
                        "cache_read": self._cache_read_tokens(msg),
                    }
                )


def build_fork_agent(model: BaseChatModel, *, fork: bool = True) -> AgentLike:
    """Build the common deepagent used by live prompt-cache tests."""
    return cast(
        "AgentLike",
        create_deep_agent(
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
        ),
    )


def invoke_twice_with_large_message(agent: AgentLike, capture: UsageCapture) -> None:
    """Warm the provider cache, then invoke the same parent/fork prefix again."""
    large_delegate = (
        "Your only job right now is to call the `task` tool with "
        "subagent_type='worker' and description='say hi'. "
        "Do not produce any other output. Do not summarize the context below.\n\n"
        "<context>\n" + LARGE_USER_MESSAGE_TEXT + "\n</context>\n\n"
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


def assert_fork_reuses_inherited_message_cache(capture: UsageCapture, *, provider: str) -> None:
    """Assert fork cache reads cover the inherited large message, not just system text."""
    fork_events = [e for e in capture.events if e["agent_type"] == "fork-subagent"]
    main_events = [e for e in capture.events if e["agent_type"] == "main"]
    assert fork_events, f"No fork-subagent LLM calls observed for {provider}. Events: {capture.events}"
    assert main_events, f"No main-agent LLM calls observed for {provider}. Events: {capture.events}"

    max_parent_read = max((e["cache_read"] for e in main_events), default=0)
    max_fork_read = max(e["cache_read"] for e in fork_events)
    assert max_parent_read >= LARGE_USER_MESSAGE_APPROX_TOKENS, (
        f"{provider} parent did not cache the large HumanMessage. "
        f"max parent cache_read={max_parent_read}, expected >= {LARGE_USER_MESSAGE_APPROX_TOKENS}. "
        f"All events: {capture.events}"
    )

    expected_floor = max_parent_read - 4_000
    if max_fork_read >= expected_floor:
        return

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
        f"{provider} fork cache_read did not cover the inherited large HumanMessage.\n"
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
