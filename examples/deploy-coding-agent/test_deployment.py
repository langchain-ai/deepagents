"""End-to-end SDK test for the coding-agent deployment.

Drives the deployment via ``langgraph_sdk`` and asserts the three things
that have to work for the spike to be useful:

1. **Store backend wired correctly.** The agent can list skills under
   ``/skills/`` and read ``/memories/AGENTS.md`` from the LangGraph store.
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


async def _run_turn(client, thread_id: str, prompt: str) -> RunOutput:
    """Drive one turn of an existing thread to completion."""
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
    return out


async def _run_one(client, prompt: str) -> tuple[str, RunOutput]:
    """Drive a fresh thread for a single turn. Returns ``(thread_id, output)``."""
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id)
    out = await _run_turn(client, thread_id, prompt)
    return thread_id, out


async def check_mcp_tool_loaded(client) -> CheckResult:
    """Verify an MCP tool from ``[mcp].config`` is loaded.

    The example bundles a docs-langchain MCP server. If the loader is
    wired correctly, the model should report having a docs-search tool.
    """
    prompt = (
        "List every tool you have available. Just the names, one per "
        "line, no commentary. If no tools, reply NONE."
    )
    _tid, out = await _run_one(client, prompt)
    text = out.final_text.lower()
    saw_mcp = "search_docs_by_lang_chain" in text or "get_page_docs_by_lang_chain" in text
    if saw_mcp:
        return CheckResult(
            name="MCP tool from [mcp].config",
            passed=True,
            detail="docs-langchain MCP tools were exposed to the model",
        )
    return CheckResult(
        name="MCP tool from [mcp].config",
        passed=False,
        detail=f"docs MCP tools missing from tool list. final_text={out.final_text[:300]!r}",
    )


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


async def check_store_loading(client) -> CheckResult:
    """Verify the agent can read the store-backed files via composite routing."""
    prompt = (
        "Use the file tools to list `/skills/`, then read "
        "`/memories/AGENTS.md`. Quote one exact line from AGENTS.md so I "
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
            name="store loading (skills + AGENTS.md)",
            passed=True,
            detail="ls /skills/ saw the skill dirs and AGENTS.md content was read",
        )
    return CheckResult(
        name="store loading (skills + AGENTS.md)",
        passed=False,
        detail=(
            f"saw_listing={seen_listing} saw_agents={seen_agents_content} "
            f"tool_outputs={[t[:120] for t in out.tool_outputs]}"
        ),
    )


async def check_fresh_thread_starts_empty(client) -> CheckResult:
    """A brand-new thread should get a fresh sandbox with no leftover state.

    1. Thread A: write a unique marker into ``/tmp/leftover.txt``.
    2. Thread B (new ``thread_id``): try to ``cat /tmp/leftover.txt``.

    If thread B sees the marker, sandboxes are leaking across threads
    (cache too aggressive, or all threads share one sandbox). If thread
    B reports "no such file", thread B got its own fresh sandbox — the
    expected behavior.
    """
    marker = f"leftover-{uuid.uuid4().hex[:8]}"

    write_prompt = (
        f"Use `execute` to run `echo {marker} > /tmp/leftover.txt`. "
        f"Just confirm the write."
    )
    _tid_a, out_a = await _run_one(client, write_prompt)

    read_prompt = (
        "Use `execute` to run `cat /tmp/leftover.txt 2>&1`. "
        "Reply with EXACTLY the file contents (or the error)."
    )
    _tid_b, out_b = await _run_one(client, read_prompt)

    saw_marker_in_b = any(marker in t for t in out_b.tool_outputs)

    if not saw_marker_in_b:
        return CheckResult(
            name="fresh thread starts with empty sandbox",
            passed=True,
            detail=(
                f"thread B did not see marker '{marker}' from thread A — "
                f"each new thread provisions its own sandbox"
            ),
        )
    return CheckResult(
        name="fresh thread starts with empty sandbox",
        passed=False,
        detail=(
            f"thread B saw thread A's marker '{marker}' — "
            f"sandboxes are leaking across threads. "
            f"thread_a_tools={[t[:80] for t in out_a.tool_outputs]} "
            f"thread_b_tools={[t[:80] for t in out_b.tool_outputs]}"
        ),
    )


async def check_sandbox_scoping_across_assistants(client) -> CheckResult:
    """Document how the per-thread sandbox cache interacts with assistant_id.

    Creates a fresh thread, runs one turn against the default ``agent``
    assistant that writes a marker to ``/tmp/marker.txt``, then creates a
    SECOND assistant on the same graph and runs another turn on the
    SAME thread asking it to ``cat /tmp/marker.txt``.

    Outcomes:

    - **Marker visible** → sandbox is shared across assistants for the
      same ``thread_id`` (current implementation: cache key is just
      ``thread_id``).
    - **Marker missing** → sandbox is isolated per ``(assistant_id,
      thread_id)`` (a future improvement we may want).

    Today this PASSES with "shared" semantics. The check name documents
    the observed behavior so a regression in either direction is loud.
    """
    marker = f"cross-assist-{uuid.uuid4().hex[:8]}"
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id)

    write_prompt = (
        f"Use `execute` to run `echo {marker} > /tmp/marker.txt`. "
        f"Just confirm the write succeeded."
    )
    out_write = await _run_turn(client, thread_id, write_prompt)

    second_assistant = await client.assistants.create(
        graph_id="agent",
        name=f"probe-{uuid.uuid4().hex[:8]}",
    )
    second_aid = second_assistant["assistant_id"]

    read_prompt = (
        "Use `execute` to run `cat /tmp/marker.txt 2>&1`. "
        "Reply with EXACTLY the file contents (or the error)."
    )
    out_read = RunOutput()
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=second_aid,
        input={"messages": [{"role": "user", "content": read_prompt}]},
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
                    out_read.final_text = content_str
                elif role == "tool":
                    out_read.tool_outputs.append(content_str)

    saw_marker = any(marker in t for t in out_read.tool_outputs) or marker in out_read.final_text

    if saw_marker:
        return CheckResult(
            name="sandbox shared across assistants on same thread",
            passed=True,
            detail=(
                f"second assistant on the same thread saw marker '{marker}' — "
                f"sandbox cache key is thread_id only (current behavior)"
            ),
        )
    return CheckResult(
        name="sandbox shared across assistants on same thread",
        passed=False,
        detail=(
            f"second assistant did NOT see marker '{marker}' — "
            f"sandbox is isolated per (assistant_id, thread_id) "
            f"or some other behavior. "
            f"write_tools={[t[:80] for t in out_write.tool_outputs]} "
            f"read_tools={[t[:80] for t in out_read.tool_outputs]} "
            f"read_final={out_read.final_text[:200]!r}"
        ),
    )


async def check_sandbox_persists_within_thread(client) -> CheckResult:
    """Verify the sandbox persists across turns within the same thread.

    Two turns on one thread:

    1. Write a unique marker to ``/tmp/marker.txt``.
    2. ``cat /tmp/marker.txt`` and report the output.

    If the same sandbox is reused between turns, turn 2 sees the file
    turn 1 wrote. If a fresh sandbox is created on every turn (cache
    miss / wrong cache key), turn 2 will report "no such file".
    """
    marker = f"persist-{uuid.uuid4().hex[:8]}"
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id)

    write_prompt = (
        f"Use the `execute` tool to run "
        f"`echo {marker} > /tmp/marker.txt`. Just confirm the command ran."
    )
    out_write = await _run_turn(client, thread_id, write_prompt)

    read_prompt = (
        "Use the `execute` tool to run `cat /tmp/marker.txt 2>&1`. "
        "Reply with EXACTLY the file contents (or the error)."
    )
    out_read = await _run_turn(client, thread_id, read_prompt)

    saw_marker = any(marker in t for t in out_read.tool_outputs) or marker in out_read.final_text
    if saw_marker:
        return CheckResult(
            name="sandbox persists within thread",
            passed=True,
            detail=(
                f"thread {thread_id[:8]} reused its sandbox across turns; "
                f"marker '{marker}' round-tripped in turn 2"
            ),
        )
    return CheckResult(
        name="sandbox persists within thread",
        passed=False,
        detail=(
            f"thread {thread_id[:8]} did NOT see marker '{marker}' in turn 2 — "
            f"sandbox was likely re-created. "
            f"turn1_tools={[t[:80] for t in out_write.tool_outputs]} "
            f"turn2_tools={[t[:80] for t in out_read.tool_outputs]} "
            f"turn2_final={out_read.final_text[:200]!r}"
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

    print("\n[1/7] execute tool wired to sandbox ...")
    checks.append(await check_execute_present(client))

    print("[2/7] Store-backed skills + AGENTS.md ...")
    checks.append(await check_store_loading(client))

    print("[3/7] Per-thread sandbox isolation ...")
    checks.append(await check_sandbox_isolation(client))

    print("[4/7] Sandbox persists within thread ...")
    checks.append(await check_sandbox_persists_within_thread(client))

    print("[5/7] Fresh thread starts with empty sandbox ...")
    checks.append(await check_fresh_thread_starts_empty(client))

    print("[6/7] Sandbox scoping across assistants ...")
    checks.append(await check_sandbox_scoping_across_assistants(client))

    print("[7/7] MCP tool from mcp.json ...")
    checks.append(await check_mcp_tool_loaded(client))

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
