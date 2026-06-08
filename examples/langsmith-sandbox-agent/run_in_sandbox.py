"""Run a DeepAgent inside a LangSmith sandbox and verify it.

This driver runs on your machine. It boots a LangSmith sandbox, uploads the
agent and a small coding task, runs the agent *inside the box*, and checks the
result with pytest, also inside the box.

Auth:
  - LangSmith: resolves `LANGSMITH_PROFILE` (or `LANGSMITH_API_KEY`) via
    `langsmith.Client()`, then passes the key + endpoint to `SandboxClient`.
  - Model: the real `ANTHROPIC_API_KEY` is handed to the sandbox auth proxy
    (an `opaque` rule on api.anthropic.com). The box runs with a dummy
    `ANTHROPIC_API_KEY=foo`; the proxy swaps in the real key on egress, so the
    real key never enters the box.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.sandbox import SandboxClient

load_dotenv()

HERE = Path(__file__).parent
FS_CAPACITY = 32 * 1024**3  # 32 GiB (the Dockerfile builder base snapshot is 16 GiB)
SNAPSHOT_NAME = "deepagents-sandbox-example"
BOX_NAME = f"deepagents-sandbox-example-{int(time.time())}"


def _print_result(label, res):
    out = getattr(res, "stdout", "") or ""
    err = getattr(res, "stderr", "") or ""
    code = getattr(res, "exit_code", "?")
    print(f"--- {label}: exit={code} ---")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print("[stderr]", err.rstrip())
    return code


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY (injected via the proxy, never into the box).")
        return 2

    # Resolve LangSmith auth from the profile/env, then build a sandbox client.
    base = Client()  # reads LANGSMITH_PROFILE / LANGSMITH_API_KEY
    sandbox_endpoint = base.api_url.rstrip("/") + "/v2/sandboxes"
    client = SandboxClient(api_key=base.api_key, api_endpoint=sandbox_endpoint, timeout=600.0)

    # Build the snapshot (agent runtime baked in) once; reuse it on later runs.
    ready = [
        s for s in client.list_snapshots()
        if (getattr(s, "name", "") or "") == SNAPSHOT_NAME
        and getattr(s, "status", "") == "ready"
    ]
    if ready:
        print(f"[driver] reusing ready snapshot '{SNAPSHOT_NAME}'")
    else:
        print("[driver] building snapshot from Dockerfile (deepagents preinstalled)...")
        client.create_snapshot_from_dockerfile(
            SNAPSHOT_NAME,
            dockerfile="Dockerfile",
            fs_capacity_bytes=FS_CAPACITY,
            context=str(HERE),
            on_build_log=lambda line: print("  build:", line.rstrip()),
            timeout=900,
        )
        print(f"[driver] snapshot '{SNAPSHOT_NAME}' built")

    # Boot a box. The auth-proxy rule injects the real Anthropic key on egress.
    proxy_config = {
        "rules": [
            {
                "name": "anthropic-api",
                "match_hosts": ["api.anthropic.com"],
                "headers": [
                    {"name": "x-api-key", "type": "opaque", "value": os.environ["ANTHROPIC_API_KEY"]},
                    {"name": "anthropic-version", "type": "plaintext", "value": "2023-06-01"},
                ],
            }
        ]
    }
    print(f"[driver] booting sandbox '{BOX_NAME}'...")
    sb = client.create_sandbox(
        snapshot_name=SNAPSHOT_NAME,
        name=BOX_NAME,
        proxy_config=proxy_config,
        idle_ttl_seconds=0,
        delete_after_stop_seconds=300,
        wait_for_ready=True,
        timeout=180,
    )

    try:
        print("[driver] uploading agent and task into the box...")
        sb.write("/app/solution.py", (HERE / "task" / "solution.py").read_text())
        sb.write("/app/test_solution.py", (HERE / "task" / "test_solution.py").read_text())
        sb.write("/app/agent.py", (HERE / "agent.py").read_text())

        # Run the agent in the box. ANTHROPIC_API_KEY=foo is a dummy; the proxy
        # swaps in the real key on the outbound call to Anthropic.
        print("[driver] running the DeepAgent inside the box...")
        agent_res = sb.run(
            "cd /app && ANTHROPIC_API_KEY=foo python agent.py",
            timeout=900,
            idle_timeout=900,
            on_stdout=lambda c: print(c, end=""),
            on_stderr=lambda c: print(c, end=""),
        )
        _print_result("agent run", agent_res)

        print("[driver] verifying with pytest inside the box...")
        test_code = _print_result(
            "pytest", sb.run("cd /app && python -m pytest test_solution.py -q", timeout=180)
        )

        print("\n==== RESULT:", "PASS ✅" if test_code == 0 else f"FAIL ❌ (exit={test_code})", "====")
        return 0 if test_code == 0 else 1
    finally:
        print(f"[driver] deleting sandbox '{BOX_NAME}'...")
        try:
            client.delete_sandbox(BOX_NAME)
        except Exception as e:  # noqa: BLE001
            print("[driver] delete failed:", e)


if __name__ == "__main__":
    sys.exit(main())
