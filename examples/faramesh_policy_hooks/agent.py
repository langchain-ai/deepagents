"""Deep Agents + Faramesh policy integration via tool-call hooks.

This example demonstrates using Deep Agents hook APIs to call an external
Faramesh policy endpoint before each tool execution.
"""

from __future__ import annotations

import json
import os
from urllib import request

from deepagents import create_deep_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool


@tool
def read_secret(path: str) -> str:
    """Read a sensitive path for demonstration purposes."""
    return f"Simulated read from {path}"


def faramesh_policy_hook(tool_request):
    """Ask a Faramesh policy endpoint whether this tool call is allowed.

    Expected response JSON from the endpoint:
      {"allow": true}
      {"allow": false, "reason": "..."}
    """
    policy_url = os.getenv("FARAMESH_POLICY_URL")
    if not policy_url:
        return None

    payload = {
        "tool": tool_request.tool_call.get("name"),
        "args": tool_request.tool_call.get("args", {}),
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        policy_url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=5) as response:  # noqa: S310
            decision = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return ToolMessage(
            content=f"Faramesh policy check failed: {exc}",
            tool_call_id=tool_request.tool_call["id"],
        )

    if decision.get("allow", True):
        return None

    reason = decision.get("reason", "Blocked by Faramesh policy.")
    return ToolMessage(
        content=f"Policy denied: {reason}",
        tool_call_id=tool_request.tool_call["id"],
    )


agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[read_secret],
    before_tool_call_hooks=[faramesh_policy_hook],
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Try to use read_secret on /etc/shadow and explain what happened.",
                }
            ]
        }
    )
    print(result)
