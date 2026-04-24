from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from browser_tools import (
    browserbase_fetch,
    browserbase_interactive_task,
    browserbase_rendered_extract,
    browserbase_search,
)

DEFAULT_DEEPAGENT_MODEL = "gpt-5.5"

SYSTEM_PROMPT = """You are a research-oriented Deep Agent with Browserbase tools.

Workflow rules:
- Start with browserbase_search for discovery unless the user already gave you a precise URL.
- Prefer browserbase_fetch for quick reads of static pages.
- Delegate JS-heavy, rendered, or multi-step browsing work to the browser-specialist subagent.
- Use browserbase_rendered_extract for read-only browser work on rendered pages.
- Use browserbase_interactive_task only when the task requires clicking, typing,
  login, or form submission.
- Keep answers concise and cite the exact URLs you used.
- Avoid interactive browser actions when fetch or rendered extraction is enough.
"""


BROWSER_SUBAGENT = {
    "name": "browser-specialist",
    "description": (
        "Handles JS-heavy browsing, rendered extraction, and interactive browser tasks "
        "through Browserbase."
    ),
    "system_prompt": """You are a Browserbase browsing specialist.

Use browserbase_rendered_extract for read-only work on rendered pages.
Use browserbase_interactive_task only for stateful actions such as clicking, typing,
logging in, or submitting forms.
Return concise summaries with the relevant page URL, what you observed, and whether
the task succeeded.
""",
    "tools": [browserbase_rendered_extract, browserbase_interactive_task],
}


def _normalize_chat_model_name(model: str) -> str:
    if ":" in model:
        provider, raw = model.split(":", 1)
        if provider == "openai":
            return raw
    return model


def build_model(model: str) -> ChatOpenAI:
    base_url = os.getenv("DEEPAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    browserbase_api_key = os.getenv("BROWSERBASE_API_KEY")

    if openai_api_key:
        api_key = openai_api_key
    elif browserbase_api_key and base_url:
        api_key = browserbase_api_key
    else:
        msg = (
            "Missing Deep Agent model configuration. Set OPENAI_API_KEY for direct OpenAI access, "
            "or set BROWSERBASE_API_KEY together with DEEPAGENT_BASE_URL/OPENAI_BASE_URL for an "
            "OpenAI-compatible gateway."
        )
        raise ValueError(msg)

    kwargs: dict[str, Any] = {
        "model": _normalize_chat_model_name(model),
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_agent(model: str) -> Any:
    return create_deep_agent(
        model=build_model(model),
        tools=[browserbase_search, browserbase_fetch],
        subagents=[BROWSER_SUBAGENT],
        system_prompt=SYSTEM_PROMPT,
        interrupt_on={
            "browserbase_interactive_task": {"allowed_decisions": ["approve", "edit", "reject"]}
        },
        checkpointer=MemorySaver(),
    )


def _stringify_content(content: Any) -> str:
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


def _get_interrupts(result: Any) -> list[Any]:
    interrupts = getattr(result, "interrupts", None)
    if interrupts:
        return list(interrupts)

    state = getattr(result, "value", result)
    if isinstance(state, dict):
        raw = state.get("__interrupt__") or state.get("interrupts") or []
        return list(raw)
    return []


def _interrupt_value(interrupt: Any) -> Any:
    return getattr(interrupt, "value", interrupt)


def _review_actions(interrupts: Sequence[Any]) -> list[dict[str, Any]]:
    interrupt_value = _interrupt_value(interrupts[0])
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]
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

                edited = input("Enter replacement JSON args: ").strip()
                try:
                    args = json.loads(edited)
                except json.JSONDecodeError:
                    print("Invalid JSON. Try again.")
                    continue
                decisions.append(
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": action["name"],
                            "args": args,
                        },
                    }
                )
                break

            print("Invalid decision for this tool call.")

    return decisions


def run(query: str, model: str) -> str:
    agent = build_agent(model=model)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v2",
    )

    while interrupts := _get_interrupts(result):
        decisions = _review_actions(interrupts)
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        )

    return _final_text(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        default=os.getenv("DEEPAGENT_MODEL", DEFAULT_DEEPAGENT_MODEL),
        help="Deep Agents model name. OpenAI-compatible raw names are recommended.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    print(run(query=args.query, model=args.model))
