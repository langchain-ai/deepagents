"""A Deep Agent backed by Vercel Sandbox, wired the way you'd run it in production.

This example layers three patterns on top of the ``langchain-vercel-sandbox``
backend that the thin wrapper deliberately leaves to the caller:

1. **Pre-baked snapshots** for fast cold starts (see ``bake_snapshot.py``).
2. **A deny-by-default network policy with credential brokering**
   (see ``network_policy.py``).
3. **Per-conversation sandbox lifecycle** via a ``BackendFactory`` so each
   thread reuses one warm sandbox instead of booting a fresh one per turn.

Run it:

    python agent.py "Generate a bar chart of [3, 1, 4, 1, 5, 9, 2, 6] and save it to /tmp/out.png"

Required environment (see ``.env.example``):
    ANTHROPIC_API_KEY, VERCEL_TOKEN, VERCEL_TEAM_ID, VERCEL_PROJECT_ID
Optional:
    VERCEL_SANDBOX_SNAPSHOT_ID  — skip per-run pip installs
    APP_API_HOST / APP_API_TOKEN — demonstrate credential brokering
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_vercel_sandbox import VercelSandbox
from vercel.sandbox import Sandbox, SnapshotSource

from network_policy import build_network_policy

load_dotenv()

# Vercel caps a sandbox's wall-clock lifetime (max 45 minutes). Recreate before
# the platform kills it so an in-flight turn never hits a dead sandbox. Note the
# unit: `Sandbox.create(timeout=...)` is MILLISECONDS (the sandbox lifetime),
# while `VercelSandbox(timeout=...)` is SECONDS (a per-command ceiling).
SANDBOX_LIFETIME_MS = 30 * 60 * 1000
PREEMPTIVE_RECREATE_S = 25 * 60  # recreate at 25 min, well inside the ceiling
COMMAND_TIMEOUT_S = 120


@dataclass
class _WarmSandbox:
    backend: VercelSandbox
    created_at: float


# Process-level cache of warm sandboxes keyed by conversation (thread) id. A
# real multi-worker server would back this with a shared store; for a single
# process this dict + lock is enough to stop every turn from cold-booting.
_WARM: dict[str, _WarmSandbox] = {}
_WARM_LOCK = threading.Lock()


def _brokered_hosts() -> dict[str, str]:
    """Wire up credential brokering for your own API host, if configured."""
    host = os.environ.get("APP_API_HOST")
    token = os.environ.get("APP_API_TOKEN")
    if host and token:
        # The sandbox can call `host` authenticated, yet never holds `token`:
        # the firewall injects the Authorization header on egress.
        return {host: token}
    return {}


def _create_sandbox() -> VercelSandbox:
    """Boot a fresh Vercel sandbox with snapshot + network policy applied."""
    create_kwargs: dict[str, object] = {
        "runtime": "python3.13",
        "timeout": SANDBOX_LIFETIME_MS,
        "network_policy": build_network_policy(brokered_hosts=_brokered_hosts()),
    }

    snapshot_id = os.environ.get("VERCEL_SANDBOX_SNAPSHOT_ID")
    if snapshot_id:
        # Boot from the pre-baked image: pandas/matplotlib/etc. are already
        # installed, so cold start is a few seconds instead of a minute.
        create_kwargs["source"] = SnapshotSource(
            type="snapshot", snapshot_id=snapshot_id
        )

    sandbox = Sandbox.create(**create_kwargs)

    if not snapshot_id:
        # No snapshot: install the data/plotting stack once, on boot.
        sandbox.run_command("pip", ["install", "-q", "pandas", "matplotlib"])

    return VercelSandbox(sandbox=sandbox, timeout=COMMAND_TIMEOUT_S)


def get_backend(runtime: object) -> VercelSandbox:
    """`BackendFactory`: return a warm sandbox for this conversation thread.

    `create_deep_agent(backend=...)` calls this with a `ToolRuntime`. We key the
    warm-sandbox cache by the LangGraph thread id so each conversation reuses one
    sandbox across turns, and recreate it before Vercel's lifetime ceiling.
    """
    config = getattr(runtime, "config", None) or {}
    thread_id = (config.get("configurable") or {}).get("thread_id") or "default"

    with _WARM_LOCK:
        warm = _WARM.get(thread_id)
        if warm is not None and (time.monotonic() - warm.created_at) < PREEMPTIVE_RECREATE_S:
            return warm.backend
        # Missing or too old: boot a fresh one. (A dropped reference to the old
        # sandbox is reaped by Vercel at its lifetime ceiling.)
        backend = _create_sandbox()
        _WARM[thread_id] = _WarmSandbox(backend=backend, created_at=time.monotonic())
        return backend


SYSTEM_PROMPT = """You are a data analysis agent.

You have an `execute` tool that runs shell commands in an isolated Linux
sandbox, plus filesystem tools. Write and run Python to answer the user's
question. Save any generated artifacts under /tmp and report their paths.
"""


def build_agent() -> object:
    """Assemble the Deep Agent with the Vercel sandbox backend factory."""
    model = init_chat_model("anthropic:claude-sonnet-4-6")
    # Pass the factory (not an instance) so each thread gets its own sandbox.
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=get_backend,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question"')
        raise SystemExit(2)

    question = sys.argv[1]
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        # The thread id keys the warm-sandbox cache in get_backend().
        config={"configurable": {"thread_id": "example-thread"}},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
