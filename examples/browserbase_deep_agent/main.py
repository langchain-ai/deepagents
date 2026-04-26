"""Entry point for the LangChain Deep Agents + Browserbase sample.

Wires four Browserbase tools (defined in :mod:`browser_tools`) into a
``deepagents.create_deep_agent`` agent with:

- A ``browser-specialist`` subagent that owns the rendered/interactive tools,
  so the main planner stays cheap and stateless.
- ``interrupt_on`` for ``browserbase_interactive_task`` so any stateful browser
  action goes through a human approve / edit / reject loop on stdin.
- A LangGraph ``MemorySaver`` checkpointer so interrupt resume works inside a
  single process.

Run with ``python main.py "<query>"``. See the README for required env vars.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_openai import ChatOpenAI

from browser_tools import (
    browserbase_fetch,
    browserbase_interactive_task,
    browserbase_rendered_extract,
    browserbase_search,
)


SYSTEM_PROMPT = """You are a research-oriented Deep Agent with Browserbase tools.

Workflow rules:
- Start with browserbase_search for discovery unless the user already gave you a precise URL.
- Prefer browserbase_fetch for quick reads of static pages.
- Delegate JS-heavy, rendered, or multi-step browsing work to the browser-specialist subagent.
- Use browserbase_rendered_extract for read-only browser work on rendered pages.
- Use browserbase_interactive_task only when the task requires clicking, typing, login, or form submission.
- Keep answers concise and cite the exact URLs you used.
- Avoid interactive browser actions when fetch or rendered extraction is enough.
"""


# Subagent registered with create_deep_agent. The Deep Agent planner sees this
# as a single virtual tool ("browser-specialist") and delegates browser-heavy
# turns to it, isolating the noisy session output from the main thread.
BROWSER_SUBAGENT = {
    "name": "browser-specialist",
    "description": (
        "Handles JS-heavy browsing, rendered extraction, and interactive browser tasks through Browserbase."
    ),
    "system_prompt": """You are a Browserbase browsing specialist.

Use browserbase_rendered_extract for read-only work on rendered pages.
Use browserbase_interactive_task only for stateful actions such as clicking, typing, logging in, or submitting forms.
Return concise summaries with the relevant page URL, what you observed, and whether the task succeeded.
""",
    "tools": [browserbase_rendered_extract, browserbase_interactive_task],
}


def _normalize_chat_model_name(model: str) -> str:
    """Strip a leading ``openai:`` provider prefix that LangChain accepts but ChatOpenAI does not.

    Example: ``openai:gpt-4o`` -> ``gpt-4o``. Non-OpenAI prefixed names
    (``anthropic:...``) are returned unchanged so a different model factory
    can handle them later if you swap the client.
    """
    if ":" in model:
        provider, raw_model = model.split(":", 1)
        if provider == "openai":
            return raw_model
    return model


def build_model(model: str) -> ChatOpenAI:
    """Create the ChatOpenAI client used by the Deep Agent planner.

    Resolution order for credentials:

    1. ``OPENAI_API_KEY`` if present — talk directly to OpenAI (or whatever
       endpoint ``DEEPAGENT_BASE_URL`` / ``OPENAI_BASE_URL`` points at).
    2. ``BROWSERBASE_API_KEY`` + a base URL — route through the Browserbase
       Model Gateway as an OpenAI-compatible endpoint.
    3. Otherwise raise — refusing to silently use the wrong key.

    Args:
        model: Model identifier, optionally prefixed with ``openai:``.

    Returns:
        A configured ``ChatOpenAI`` instance.

    Raises:
        ValueError: If neither credential path is satisfied.
    """
    base_url = os.getenv("DEEPAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    browserbase_api_key = os.getenv("BROWSERBASE_API_KEY")

    if openai_api_key:
        api_key = openai_api_key
    elif browserbase_api_key and base_url:
        api_key = browserbase_api_key
    else:
        raise ValueError(
            "Missing Deep Agent model configuration. Set OPENAI_API_KEY for direct OpenAI access, "
            "or set BROWSERBASE_API_KEY together with DEEPAGENT_BASE_URL/OPENAI_BASE_URL for a "
            "Browserbase-backed OpenAI-compatible gateway."
        )

    kwargs: dict[str, Any] = {
        "model": _normalize_chat_model_name(model),
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_agent(model: str):
    """Compose the Deep Agent graph: model + tools + subagent + interrupt config.

    The planner gets the cheap tools (``search``, ``fetch``) directly. The
    expensive browser tools live on the subagent so they only fire when the
    planner explicitly delegates. ``interrupt_on`` pauses execution before
    any ``browserbase_interactive_task`` call so a human can review.

    Args:
        model: Model name for the planner LLM.

    Returns:
        A compiled LangGraph runnable supporting ``invoke`` and ``Command(resume=...)``.
    """
    return create_deep_agent(
        model=build_model(model),
        tools=[browserbase_search, browserbase_fetch],
        subagents=[BROWSER_SUBAGENT],
        system_prompt=SYSTEM_PROMPT,
        interrupt_on={
            "browserbase_interactive_task": {
                "allowed_decisions": ["approve", "edit", "reject"]
            }
        },
        # In-memory checkpointer is enough for a single-process CLI run.
        # For multi-turn or production use, swap in a persistent checkpointer.
        checkpointer=MemorySaver(),
    )


def _stringify_content(content: Any) -> str:
    """Flatten LangChain message content (str | list[str|dict]) to a printable string.

    Newer chat models emit content as a list of typed parts (``{"type": "text", ...}``,
    image parts, tool-use parts, etc.). We keep text parts verbatim and JSON-dump
    anything else so debugging info isn't lost.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, indent=2, default=str)


def _final_text(result: Any) -> str:
    """Extract the agent's last assistant message from a graph result.

    Walks ``messages`` in reverse looking for an assistant/AI turn. Falls back
    to a JSON dump of the full state so we never silently swallow output.
    """
    state = getattr(result, "value", result)
    if isinstance(state, dict):
        messages = state.get("messages", [])
        for message in reversed(messages):
            msg_type = getattr(message, "type", None)
            if msg_type is None and isinstance(message, dict):
                msg_type = message.get("type") or message.get("role")
            if msg_type in {"ai", "assistant"}:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                return _stringify_content(content)
        return json.dumps(state, indent=2, default=str)
    return str(state)


def _review_actions(result: Any) -> list[dict[str, Any]]:
    """Drive the human-in-the-loop approval flow on stdin.

    For each pending tool call (action_request), print the tool name and args,
    then loop until the user enters one of the tool's ``allowed_decisions``.
    Returns one decision dict per action in the same order LangGraph expects.

    Decision payloads:
    - approve: ``{"type": "approve"}``
    - reject:  ``{"type": "reject"}``
    - edit:    ``{"type": "edit", "edited_action": {"name", "args"}}``
    """
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]
    # Build a lookup so per-tool allowed_decisions can be enforced even if the
    # agent batched calls to multiple different gated tools at once.
    config_by_name = {config["action_name"]: config for config in review_configs}
    decisions: list[dict[str, Any]] = []

    for action in action_requests:
        review = config_by_name[action["name"]]
        allowed = review["allowed_decisions"]

        print("\nPending tool call")
        print(f"Tool: {action['name']}")
        print("Arguments:")
        print(json.dumps(action["args"], indent=2, default=str))
        print(f"Allowed decisions: {', '.join(allowed)}")

        while True:
            raw = input("Decision [approve/edit/reject]: ").strip().lower()
            if raw in allowed:
                if raw == "approve":
                    decisions.append({"type": "approve"})
                    break
                if raw == "reject":
                    decisions.append({"type": "reject"})
                    break

                # Edit branch: ask for replacement args as JSON. Re-prompt on
                # parse failure rather than crashing the whole agent run.
                edited = input("Enter replacement JSON args: ").strip()
                try:
                    edited_args = json.loads(edited)
                except json.JSONDecodeError:
                    print("Invalid JSON. Try again.")
                    continue
                decisions.append(
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": action["name"],
                            "args": edited_args,
                        },
                    }
                )
                break

            print("Invalid decision for this tool call.")

    return decisions


def run(query: str, model: str) -> str:
    """Drive the agent end-to-end, looping over interrupts until completion.

    Each ``invoke`` either finishes (returns final state) or pauses with one
    or more interrupts pending. We collect human decisions for those, resume
    via ``Command(resume=...)``, and repeat until no interrupts remain.

    Args:
        query: User prompt to send as the initial message.
        model: Model name passed through to :func:`build_model`.

    Returns:
        The final assistant message as a plain string.
    """
    agent = build_agent(model=model)
    # Unique thread_id per run; combined with MemorySaver this scopes
    # checkpoint state to this single CLI invocation.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v2",
    )

    while result.interrupts:
        decisions = _review_actions(result)
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        )

    return _final_text(result)


def parse_args() -> argparse.Namespace:
    """Parse CLI args: optional ``query`` positional and ``--model`` override."""
    parser = argparse.ArgumentParser(
        description="Sample LangChain Deep Agents app with Browserbase tools."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "Research the Browserbase Search API and explain when to use Search, Fetch, "
            "and a full browser session. Cite the URLs you used."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPAGENT_MODEL", "gpt-5.4"),
        help="Deep Agents model name. OpenAI-compatible raw names are recommended.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # load_dotenv() before parse_args so DEEPAGENT_MODEL from .env can act as
    # the default for --model.
    load_dotenv()
    args = parse_args()
    print(run(query=args.query, model=args.model))
