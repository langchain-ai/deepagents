"""Nest LangChain calls made in a sandbox under the Deep Agents execute span."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.backends.protocol import ExecuteResponse
from langsmith.run_helpers import get_current_run_tree
from langsmith.sandbox import SandboxClient

if TYPE_CHECKING:
    from langsmith.sandbox import AsyncSandbox, ExecutionResult, Sandbox

SNAPSHOT_NAME = "deepagents-distributed-tracing"
SNAPSHOT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
FS_CAPACITY = 5 * 1024**3

TRACE_INSTRUCTIONS = """
When you write Python that calls an LLM in the sandbox, continue the active trace with this pattern:

    import json
    import os
    import langsmith as ls

    headers = json.loads(os.environ["LANGSMITH_TRACE_HEADERS"])
    with ls.tracing_context(parent=headers):
        result = traced_entrypoint()

Decorate `traced_entrypoint` with `@ls.traceable(name="sandbox_rlm_task")`. Use
`os.environ["SANDBOX_MODEL"]` as the model passed to `init_chat_model`. The sandbox
already receives the model provider key, LangSmith key, and tracing configuration.
"""


def _execute_response(result: ExecutionResult) -> ExecuteResponse:
    """Convert a LangSmith sandbox result to a Deep Agents response."""
    output = result.stdout or ""
    if result.stderr:
        output += f"\n{result.stderr}" if output else result.stderr
    return ExecuteResponse(output=output, exit_code=result.exit_code)


class TracedLangSmithSandbox(LangSmithSandbox):
    """Inject trusted distributed-tracing context into each sandbox command."""

    def __init__(self, sandbox: Sandbox, env: dict[str, str]) -> None:
        """Create a sandbox backend with an explicit environment allowlist."""
        super().__init__(sandbox)
        self._env = env

    def _command_env(self) -> dict[str, str]:
        """Build per-command environment variables from the active run tree."""
        env = dict(self._env)
        if run_tree := get_current_run_tree():
            env["LANGSMITH_TRACE_HEADERS"] = json.dumps(run_tree.to_headers())
        return env

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute with the current run tree propagated out of process."""
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.run(
            command, timeout=effective_timeout, env=self._command_env()
        )
        return _execute_response(result)

    async def aexecute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        """Execute asynchronously with the current run tree propagated out of process."""
        effective_timeout = timeout if timeout is not None else self._default_timeout
        sandbox: AsyncSandbox = self._aget_sandbox()
        result = await sandbox.run(
            command, timeout=effective_timeout, env=self._command_env()
        )
        return _execute_response(result)


def _sandbox_env() -> dict[str, str]:
    """Return only the credentials and configuration needed by sandbox LLM code."""
    env = {
        "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY"],
        "LANGSMITH_TRACING": "true",
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "SANDBOX_MODEL": os.environ["SANDBOX_MODEL"],
    }
    if endpoint := os.environ.get("LANGSMITH_ENDPOINT"):
        env["LANGSMITH_ENDPOINT"] = endpoint
    return env


def _create_sandbox(client: SandboxClient) -> Sandbox:
    """Create the reusable snapshot when needed, then start a sandbox."""
    snapshots = client.list_snapshots(name_contains=SNAPSHOT_NAME)
    if not any(
        snapshot.name == SNAPSHOT_NAME and snapshot.status == "ready"
        for snapshot in snapshots
    ):
        client.create_snapshot(
            name=SNAPSHOT_NAME,
            docker_image=SNAPSHOT_IMAGE,
            fs_capacity_bytes=FS_CAPACITY,
        )
    return client.create_sandbox(snapshot_name=SNAPSHOT_NAME)


def main() -> None:
    """Run a Deep Agent whose generated sandbox LLM code joins the same trace."""
    client = SandboxClient(api_key=os.environ["LANGSMITH_API_KEY"])
    sandbox = _create_sandbox(client)
    backend = TracedLangSmithSandbox(sandbox, _sandbox_env())
    agent = create_deep_agent(
        model=os.environ["HOST_MODEL"],
        backend=backend,
        system_prompt=TRACE_INSTRUCTIONS,
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write and run /workspace/rlm_task.py. Have its LangChain model "
                            "produce three candidate explanations of distributed tracing, then "
                            "make a final model call that selects the clearest one. Print the result."
                        ),
                    }
                ]
            }
        )
        print(result["messages"][-1].content)
    finally:
        client.delete_sandbox(sandbox.name)


if __name__ == "__main__":
    main()
