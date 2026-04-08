"""End-to-end SDK test for the coding-agent deployment.

Drives the deployment via ``langgraph_sdk`` and asserts the three things
that have to work for the spike to be useful:

1. **Hub backend wired correctly.** The agent can list and read files
   under ``/longterm/`` (skills + AGENTS.md served from the LangSmith
   Prompt Hub repo via ``HubBackend``).
2. **Sandbox provisioned per thread.** Two distinct ``thread_id`` values
   get two distinct sandboxes, each with its own filesystem.
3. **Sandbox shell actually executes.** Commands run inside the sandbox
   and their output round-trips back to the model.

Usage::

    # Against a local ``langgraph dev`` server
    python test_deployment.py

    # Against a deployed LangGraph Platform instance
    python test_deployment.py --url https://my-deployment.example/

Required env vars: ``ANTHROPIC_API_KEY``, ``LANGSMITH_API_KEY``.

The script exits non-zero on any failed check, so it can be wired into
CI. Each check prints a one-line ``PASS`` / ``FAIL`` summary.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field

from langgraph_sdk import get_client


ASSISTANT = os.environ.get("ASSISTANT_ID", "agent")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunOutput:
    final_text: str = ""
    tool_outputs: list[str] = field(default_factory=list)


async def _run_one(client, prompt: str) -> tuple[str, RunOutput]:
    """Drive a single thread to completion. Returns (thread_id, output)."""
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id)

    out = RunOutput()
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT,
        input={"messages": [{"role": "user", "content": prompt}]},
        stream_mode="updates",
    ):
        if chunk.event != "updates":
            continue
        for _node, update in (chunk.data or {}).items():
            if not isinstance(update, dict):
                continue
            for m in update.get("messages", []) or []:
                if not isinstance(m, dict):
                    continue
                role = m.get("type")
                content = m.get("content")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                content_str = str(content or "")
                if role == "ai" and content_str.strip():
                    out.final_text = content_str
                elif role == "tool":
                    out.tool_outputs.append(content_str)
    return thread_id, out


async def check_execute_present(client) -> CheckResult:
    """Verify the deployed agent has the ``execute`` tool wired up.

    A bare assistant schema lookup would tell us the registered graph
    nodes, but tools are middleware-injected at runtime, so the only
    reliable check is to ask the model itself. We accept the check as
    long as the model produces some shell-style output via ``execute``,
    not just a refusal.
    """
    marker = uuid.uuid4().hex[:8]
    prompt = (
        f"Use the `execute` tool to run `echo deepagents-execute-{marker}`. "
        f"If you do not have an execute tool, reply with exactly: "
        f"NO_EXECUTE_TOOL"
    )
    _tid, out = await _run_one(client, prompt)

    saw_marker = any(f"deepagents-execute-{marker}" in t for t in out.tool_outputs)
    refused = "NO_EXECUTE_TOOL" in out.final_text

    if saw_marker:
        return CheckResult(
            name="execute tool wired to sandbox",
            passed=True,
            detail=f"shell echo round-tripped marker '{marker}'",
        )
    return CheckResult(
        name="execute tool wired to sandbox",
        passed=False,
        detail=(
            "model did not produce execute output. "
            f"refused={refused} "
            f"final_text={out.final_text[:200]!r} "
            f"tool_outputs={[t[:120] for t in out.tool_outputs]}"
        ),
    )


async def check_hub_loading(client) -> CheckResult:
    """Verify the agent can read the hub-backed files via composite routing."""
    prompt = (
        "Use the file tools to list `/longterm/skills/`, then read "
        "`/longterm/AGENTS.md`. Quote one exact line from AGENTS.md so I "
        "know you actually read it. Be terse."
    )
    _tid, out = await _run_one(client, prompt)

    seen_listing = any(
        "code-review" in t and "planning" in t for t in out.tool_outputs
    )
    seen_agents_content = any(
        "Coding Agent" in t or "expert software engineer" in t for t in out.tool_outputs
    )

    if seen_listing and seen_agents_content:
        return CheckResult(
            name="hub loading (skills + AGENTS.md via composite)",
            passed=True,
            detail="ls /longterm/skills/ saw the skill dirs and AGENTS.md content was read",
        )
    return CheckResult(
        name="hub loading (skills + AGENTS.md via composite)",
        passed=False,
        detail=(
            f"saw_listing={seen_listing} saw_agents={seen_agents_content} "
            f"tool_outputs={[t[:120] for t in out.tool_outputs]}"
        ),
    )


async def check_sandbox_isolation(client) -> CheckResult:
    """Two threads, two distinct sandboxes, isolated filesystem state."""
    marker_a = f"marker-A-{uuid.uuid4().hex[:8]}"
    marker_b = f"marker-B-{uuid.uuid4().hex[:8]}"

    prompt_a = (
        f"Use execute to run `echo {marker_a} > /tmp/marker.txt && "
        f"cat /tmp/marker.txt`. Reply with just the cat output."
    )
    prompt_b = (
        f"Use execute to first `cat /tmp/marker.txt 2>&1` (this should NOT "
        f"contain '{marker_a}' since this is a fresh sandbox), then run "
        f"`echo {marker_b} > /tmp/marker.txt && cat /tmp/marker.txt`. "
        f"Reply with both outputs labeled."
    )

    tid_a, out_a = await _run_one(client, prompt_a)
    tid_b, out_b = await _run_one(client, prompt_b)

    # Thread A wrote and read its own marker
    a_wrote_own = any(marker_a in t for t in out_a.tool_outputs) or marker_a in out_a.final_text

    # Thread B wrote and read its own marker
    b_wrote_own = any(marker_b in t for t in out_b.tool_outputs) or marker_b in out_b.final_text

    # Thread B did NOT see thread A's marker in any tool output (per-thread
    # sandbox isolation). We deliberately don't check final_text — the model
    # often quotes our prompt back, which would put marker_a in its reply
    # without that meaning the file actually existed in thread B's sandbox.
    b_isolated = not any(marker_a in t for t in out_b.tool_outputs)

    if a_wrote_own and b_wrote_own and b_isolated:
        return CheckResult(
            name="sandbox per-thread isolation",
            passed=True,
            detail=(
                f"thread A {tid_a[:8]} got '{marker_a}', "
                f"thread B {tid_b[:8]} got '{marker_b}', "
                f"thread B did not see thread A's marker"
            ),
        )

    return CheckResult(
        name="sandbox per-thread isolation",
        passed=False,
        detail=(
            f"a_wrote_own={a_wrote_own} b_wrote_own={b_wrote_own} "
            f"b_isolated={b_isolated} "
            f"thread_a_tools={[t[:80] for t in out_a.tool_outputs]} "
            f"thread_b_tools={[t[:80] for t in out_b.tool_outputs]}"
        ),
    )


async def main(url: str) -> int:
    for required in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(required):
            print(f"FAIL setup: missing required env var {required}", file=sys.stderr)
            return 1

    print(f"Driving deployment at {url}")
    client = get_client(url=url)

    checks: list[CheckResult] = []

    print("\n[1/3] execute tool wired to sandbox ...")
    checks.append(await check_execute_present(client))

    print("[2/3] Hub-backed skills + AGENTS.md ...")
    checks.append(await check_hub_loading(client))

    print("[3/3] Per-thread sandbox isolation ...")
    checks.append(await check_sandbox_isolation(client))

    print("\n--- results ---")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name}")
        if c.detail:
            print(f"         {c.detail}")

    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("DEPLOYMENT_URL", "http://localhost:2024"),
        help=(
            "Base URL of the LangGraph deployment to test. "
            "Defaults to ``$DEPLOYMENT_URL`` if set, otherwise ``http://localhost:2024`` "
            "(matches a local ``langgraph dev`` server)."
        ),
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.url)))
