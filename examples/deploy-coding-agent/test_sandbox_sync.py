"""Minimal test: verify skill scripts are synced into the sandbox and executable."""

import asyncio
import os
import sys
import uuid

from langgraph_sdk import get_client


async def main(url: str) -> int:
    client = get_client(url=url)
    thread_id = str(uuid.uuid4())
    await client.threads.create(thread_id=thread_id)

    prompt = (
        "Use the `execute` tool to run: "
        "python3 /skills/code-review/lint_check.py --help 2>&1 || "
        "python /skills/code-review/lint_check.py --help 2>&1\n"
        "Reply with the raw output only."
    )

    tool_outputs = []
    final_text = ""
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id="agent",
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
                    final_text = content_str
                elif role == "tool":
                    tool_outputs.append(content_str)

    print(f"tool_outputs ({len(tool_outputs)}):")
    for t in tool_outputs:
        print(f"  {t[:200]}")
    print(f"\nfinal_text: {final_text[:300]!r}")

    saw_script = any(
        "lint_check" in t.lower() or "scan" in t.lower() or "usage" in t.lower()
        or "docstring" in t.lower() or "no warnings found" in t.lower()
        or "warning(s) found" in t.lower()
        for t in tool_outputs
    )
    saw_error = any("no such file" in t.lower() for t in tool_outputs)

    if saw_script and not saw_error:
        print("\nPASS: skill script executed in sandbox")
        return 0
    else:
        print(f"\nFAIL: saw_script={saw_script} saw_error={saw_error}")
        return 1


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:2024"
    sys.exit(asyncio.run(main(url)))
