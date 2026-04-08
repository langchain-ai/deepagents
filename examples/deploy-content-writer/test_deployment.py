"""End-to-end SDK test for the content-writer deployment.

Verifies:
1. Local tool from ``[tools].python_file`` is loaded.
2. ``user``-scoped memory persists across threads for the same user
   (write a fact in thread A, read it back in thread B).

Usage::

    LANGSMITH_API_KEY=... python test_deployment.py \\
        --url https://deepagents-deploy-content-w-6909480a63d7575eb597d5a1b3c6e61e.us.langgraph.app
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


async def check_local_tool_loaded(client) -> CheckResult:
    """Verify ``word_count`` from ``tools.py`` is wired up."""
    prompt = (
        "Use the `word_count` tool on the text 'one two three four five' "
        "and reply with EXACTLY the integer it returned, nothing else. "
        "If the tool is not available, reply with: NO_LOCAL_TOOL"
    )
    _tid, out = await _run_one(client, prompt)
    saw = "5" in out.final_text and "NO_LOCAL_TOOL" not in out.final_text
    if saw:
        return CheckResult(
            name="local tool from [tools].python_file",
            passed=True,
            detail="word_count returned 5",
        )
    return CheckResult(
        name="local tool from [tools].python_file",
        passed=False,
        detail=(
            f"final_text={out.final_text[:200]!r} "
            f"tool_outputs={[t[:100] for t in out.tool_outputs]}"
        ),
    )


async def check_user_scoped_memory(client) -> CheckResult:
    """Write a fact to a file in thread A, read it back in thread B.

    Both threads run as the same authenticated user (same LangSmith key),
    so a ``user``-scoped store namespace should give them the same
    filesystem view, even though their thread IDs differ.
    """
    marker = f"favorite-color-is-{uuid.uuid4().hex[:8]}"

    prompt_a = (
        f"Use the file tools to write the file `/notes/profile.txt` "
        f"with EXACTLY this content: {marker}. Then confirm with 'WROTE'."
    )
    prompt_b = (
        "Use the file tools to read `/notes/profile.txt` and reply with "
        "EXACTLY its content, nothing else. If the file does not exist, "
        "reply with: NO_FILE"
    )

    tid_a, out_a = await _run_one(client, prompt_a)
    tid_b, out_b = await _run_one(client, prompt_b)

    wrote = "WROTE" in out_a.final_text or any(marker in t for t in out_a.tool_outputs)
    read_back = marker in out_b.final_text or any(marker in t for t in out_b.tool_outputs)

    if wrote and read_back:
        return CheckResult(
            name="user-scoped memory persists across threads",
            passed=True,
            detail=(
                f"thread A {tid_a[:8]} wrote marker, "
                f"thread B {tid_b[:8]} read it back"
            ),
        )
    return CheckResult(
        name="user-scoped memory persists across threads",
        passed=False,
        detail=(
            f"wrote={wrote} read_back={read_back} "
            f"a_final={out_a.final_text[:120]!r} "
            f"b_final={out_b.final_text[:120]!r} "
            f"b_tools={[t[:100] for t in out_b.tool_outputs]}"
        ),
    )


async def main(url: str) -> int:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("FAIL setup: missing LANGSMITH_API_KEY", file=sys.stderr)
        return 1

    print(f"Driving deployment at {url}")
    client = get_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])

    checks: list[CheckResult] = []

    print("\n[1/2] local tool from [tools].python_file ...")
    checks.append(await check_local_tool_loaded(client))

    print("[2/2] user-scoped memory persists across threads ...")
    checks.append(await check_user_scoped_memory(client))

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
