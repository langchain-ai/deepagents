"""Integration tests for PTC against real deepagents middlewares.

Uses a real model and a real ``SubAgentMiddleware``-provided
``task`` tool. The assertion is coarse — "the subagent actually ran" —
because the model's phrasing is not deterministic, but the wiring
between PTC, ``task``, and a spawned subagent graph is covered
end-to-end.

Both invocation paths are exercised:

- ``agent.invoke`` (sync path)
- ``agent.ainvoke`` (async path)

``REPLMiddleware`` routes both paths through async QuickJS eval under
the hood so PTC host-function bridges work consistently in either mode.

Requires ``ANTHROPIC_API_KEY`` in the environment. Run with
``make integration_tests``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import pytest
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_quickjs import REPLMiddleware

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real-model integration tests",
)


_MODEL = "claude-sonnet-4-6"
InvokeMode = Literal["invoke", "ainvoke"]


def _researcher_subagent() -> dict:
    """A trivial subagent the outer agent can dispatch `task` calls to.

    Uses the real model with a tight system prompt that keeps responses
    short and deterministic in the ways the test assertion cares about
    (one-word topical answer).
    """
    return {
        "name": "researcher",
        "description": (
            "Returns a one-sentence fact about a topic. "
            "Use this subagent for any research-style request."
        ),
        "system_prompt": (
            "You are a research assistant. Given a topic, reply with exactly "
            "one short sentence stating a well-known fact about it. "
            "Do not use any tools. Do not ask questions."
        ),
        "model": _MODEL,
        "tools": [],
    }


async def _invoke_agent(
    agent: Any,
    payload: dict[str, Any],
    invoke_mode: InvokeMode,
) -> dict[str, Any]:
    if invoke_mode == "ainvoke":
        return await agent.ainvoke(payload)
    return agent.invoke(payload)


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_ptc_spawns_subagent_through_eval(invoke_mode: InvokeMode) -> None:
    """A real model, given access to `eval` + PTC(`task`), actually runs a subagent.

    We assert on graph-observable effects, not on the model's phrasing:

    - A ``ToolMessage`` from the outer ``eval`` call exists.
    - Its content mentions the topic we asked about, which can only
      happen if PTC ran ``tools.task`` and the subagent's response
      round-tripped back through the REPL.
    """
    agent = create_agent(
        model=ChatAnthropic(model=_MODEL),
        middleware=[
            SubAgentMiddleware(
                backend=None,  # not used by this trivial subagent
                subagents=[_researcher_subagent()],
            ),
            REPLMiddleware(ptc=["task"]),
        ],
    )

    # Prompt that nudges toward PTC: "use your REPL to run two research
    # tasks in parallel". The model isn't obligated to take the bait,
    # but `claude-sonnet-4-6` with these tools routinely does.
    prompt = (
        "Use your `eval` tool to write one piece of JavaScript that calls "
        "`tools.task({description, subagent_type: 'researcher'})` for the "
        "topics 'the moon' and 'the ocean' in parallel via Promise.all, "
        "and returns the joined result. Then summarise what you got."
    )
    response = await _invoke_agent(
        agent,
        {"messages": [HumanMessage(content=prompt)]},
        invoke_mode,
    )

    tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
    eval_messages = [m for m in tool_messages if m.name == "eval"]
    assert eval_messages, "expected the model to call the eval tool"

    # The eval ToolMessage body contains whatever the REPL returned —
    # for PTC-routed subagent calls, that's the subagent's final text.
    # We accept either topic as evidence the subagent actually ran.
    combined = "\n".join(m.content for m in eval_messages).lower()
    assert "moon" in combined or "ocean" in combined, (
        f"eval output did not reference the requested topics: {combined[:500]}"
    )


@pytest.mark.parametrize(
    "invoke_mode",
    ["invoke", "ainvoke"],
    ids=["sync_invoke", "async_ainvoke"],
)
async def test_ptc_respects_allowlist_config(invoke_mode: InvokeMode) -> None:
    """When ptc allowlist omits `task`, the model cannot call it from the REPL.

    We give the model both `task` as a regular tool and `eval` with
    PTC configured with an empty allowlist. The REPL's `tools` namespace
    should therefore be empty (or at least not include `task`).
    """
    agent = create_agent(
        model=ChatAnthropic(model=_MODEL),
        middleware=[
            SubAgentMiddleware(
                backend=None,
                subagents=[_researcher_subagent()],
            ),
            REPLMiddleware(ptc=[]),
        ],
    )

    response = await _invoke_agent(
        agent,
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Inside the `eval` tool, run the JavaScript expression "
                        "`typeof tools.task` and return what it says."
                    )
                )
            ],
        },
        invoke_mode,
    )

    tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
    eval_messages = [m for m in tool_messages if m.name == "eval"]
    assert eval_messages, "expected the model to call the eval tool"
    # The model may or may not have called eval usefully, but if it did,
    # `typeof tools.task` should be "undefined".
    combined = "\n".join(m.content for m in eval_messages).lower()
    assert "undefined" in combined, (
        f"expected 'undefined' from typeof tools.task; got: {combined[:500]}"
    )
