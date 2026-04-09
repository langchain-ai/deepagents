"""End-to-end SDK test for the content-writer deployment.

Verifies that the agent can read ``/memories/AGENTS.md`` (seeded from
``src/AGENTS.md`` at bundle time) and quote content from it.

Usage::

    LANGSMITH_API_KEY=... python test_deployment.py \\
        --url https://your-deployment.us.langgraph.app
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


async def check_memory_loaded(client) -> CheckResult:
    """Verify the agent can read /memories/AGENTS.md from the store."""
    prompt = (
        "Use the file tools to read `/memories/AGENTS.md` and quote the "
        "first heading (e.g. '# Content Writer Agent'). Be terse."
    )
    _tid, out = await _run_one(client, prompt)
    saw = any(
        "Content Writer" in t for t in (*out.tool_outputs, out.final_text)
    )
    return CheckResult(
        name="memory loaded (/memories/AGENTS.md)",
        passed=saw,
        detail=f"final={out.final_text[:160]!r}",
    )


async def check_memory_read_only(client) -> CheckResult:
    """Verify /memories/AGENTS.md rejects writes/edits."""
    prompt = (
        "Try to edit `/memories/AGENTS.md` — replace 'Content' with 'XXX' "
        "using the edit tool. If it errors, reply with EXACTLY: BLOCKED. "
        "If it succeeds, reply with EXACTLY: ALLOWED."
    )
    _tid, out = await _run_one(client, prompt)
    blocked = "BLOCKED" in out.final_text and "ALLOWED" not in out.final_text
    return CheckResult(
        name="memory is read-only",
        passed=blocked,
        detail=f"final={out.final_text[:160]!r}",
    )


async def main(url: str) -> int:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("FAIL setup: missing LANGSMITH_API_KEY", file=sys.stderr)
        return 1

    print(f"Driving deployment at {url}")
    client = get_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])

    checks: list[CheckResult] = []

    print("\n[1/2] memory loaded ...")
    checks.append(await check_memory_loaded(client))

    print("[2/2] memory read-only ...")
    checks.append(await check_memory_read_only(client))

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
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.url)))
