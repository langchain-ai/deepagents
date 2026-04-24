"""
Deep Agents harness — optimizable by better-harness.

Edit everything **above** the FIXED ADAPTER BOUNDARY line.
The outer agent will only modify the editable section.

To use: set agent_import_path = "deepagents_harness:HarborAgent" in your cases.toml.
"""

# ============================================================
# EDITABLE SECTION — outer agent modifies everything here
# ============================================================

SYSTEM_PROMPT = """You are a helpful AI agent.

When a request is underspecified, ask the minimum number of followup
questions needed before acting. Use reasonable defaults when the intent
is clearly implied.
"""

MODEL = "claude-sonnet-4-6"


def create_tools():
    """Return the list of tools available to the agent.

    Add, remove, or modify tools here. Examples:
        from langchain_community.tools import ShellTool
        return [ShellTool()]
    """
    return []


def create_agent():
    """Build and return the configured Deep Agent."""
    from deepagents import create_deep_agent  # noqa: PLC0415
    return create_deep_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=create_tools(),
    )


def run_task(task_input: str) -> list:
    """Run one task and return the deepagents message history.

    Returns the full list of LangChain messages produced by the agent,
    which the fixed adapter boundary serializes into a trace.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    agent = create_agent()
    result = agent.invoke({"messages": [HumanMessage(content=task_input)]})
    return result.get("messages", [])


# ============================================================
# FIXED ADAPTER BOUNDARY — do not edit below this line
# ============================================================

import json  # noqa: E402
import os    # noqa: E402
from pathlib import Path  # noqa: E402

_TRACE_ENV = "BETTER_HARNESS_TRACE_FILE"
_MAX_CONTENT = 800


def _serialize_messages(messages: list, task_input: str, error: str | None) -> dict:
    """Convert deepagents message history to clean trace JSON.

    Format matches better_harness.core.Trace so the optimizer can load it
    directly via better_harness.traces.load_trace().
    """
    turns = []
    pending: dict | None = None

    for msg in messages:
        mtype = getattr(msg, "type", "") or ""

        if mtype == "human":
            continue  # skip the initial task message

        if mtype == "ai":
            # Extract text content (skip tool_use blocks if content is a list).
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            else:
                text = str(content).strip()

            # Extract tool calls.
            raw_calls = getattr(msg, "tool_calls", []) or []
            calls = [
                {
                    "_id": tc.get("id", ""),   # internal, stripped before output
                    "tool": tc.get("name", ""),
                    "input": tc.get("args", {}),
                    "output": None,
                    "error": None,
                }
                for tc in raw_calls
            ]

            if text or calls:
                pending = {"agent": text, "calls": calls}
                turns.append(pending)

        elif mtype == "tool" and pending is not None:
            call_id = getattr(msg, "tool_call_id", "")
            raw_content = str(getattr(msg, "content", "") or "")
            content_trimmed = raw_content[:_MAX_CONTENT]

            for call in pending["calls"]:
                if call["_id"] == call_id:
                    # Heuristic: flag obvious errors so the outer agent can spot them.
                    if (
                        raw_content.startswith(("Error", "Traceback"))
                        or "Exception" in raw_content[:80]
                    ):
                        call["error"] = content_trimmed
                    else:
                        call["output"] = content_trimmed
                    break

    # Final output: the last non-empty AI message text.
    final_output = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            c = getattr(msg, "content", "") or ""
            if isinstance(c, str) and c.strip():
                final_output = c.strip()[:500]
                break

    # Strip internal _id fields before writing.
    clean_turns = [
        {
            "agent": t["agent"],
            "calls": [{k: v for k, v in c.items() if k != "_id"} for c in t["calls"]],
        }
        for t in turns
    ]

    return {
        "task": task_input,
        "total_turns": len(clean_turns),
        "turns": clean_turns,
        "final_output": final_output,
        "failure": error,
    }


class HarborAgent:
    """Harbor adapter. DO NOT MODIFY THIS CLASS.

    Harbor imports this class and calls run() to execute one task.
    The result is written to result.json by Harbor's judge.
    We write trace.json here so the optimizer can show the outer agent
    exactly what the inner agent did on each failing case.
    """

    def run(self, task_input, **_kwargs):
        task_str = (
            task_input
            if isinstance(task_input, str)
            else str(task_input.get("task", task_input))
        )

        messages: list = []
        error: str | None = None

        try:
            messages = run_task(task_str)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        trace = _serialize_messages(messages, task_str, error)

        # Write trace.json to the path the optimizer told us to use.
        trace_path = os.environ.get(_TRACE_ENV)
        if trace_path:
            p = Path(trace_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(trace, indent=2) + "\n")

        # Return the agent's final output for Harbor's judge.
        if error:
            return f"Error: {error}"
        return trace["final_output"]
