"""Shell runner — agent backed by a per-task :class:`LangSmithSandbox`.

Each task gets its own LangSmith sandbox spun up from the
``deepagents-cli`` template. ``/context.txt`` is uploaded into the
sandbox (via ``aupload_files``) *before* the agent runs so its shell
commands (``cat /context.txt``, ``grep``, etc.) find the file at the
path the system prompt advertises.

The sandbox's ``execute`` tool is picked up automatically by
``FilesystemMiddleware`` since ``LangSmithSandbox`` implements
``SandboxBackendProtocol``. Returning ``initial_files=None`` tells
:func:`run_agent_async` to skip state-level file seeding — the shell
runner's filesystem lives entirely in the sandbox, not in agent state.

Teardown exits the underlying ``SandboxClient.sandbox`` context so the
sandbox is torn down even on test failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from langsmith.sandbox import SandboxClient

from tests.evals.oolong.runners._common import SYSTEM_PROMPT, RunnerContext

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.oolong.data_utils import OolongTask


logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "deepagents-cli"
"""LangSmith sandbox template the shell runner boots from.

Matches the template used by the swarm path in
``cc/quick-repl-swarm-table`` — keeps the runtime environment
consistent across swarm-style evals. Change this only if the shell
eval needs a different image."""


async def build_runner(
    *,
    model: BaseChatModel,
    task: OolongTask,
) -> RunnerContext:
    """Build the shell runner.

    Enters the sandbox context, seeds ``/context.txt`` via
    ``aupload_files``, and hands back a teardown that exits the
    context. If seeding fails mid-setup we still exit the context so
    the sandbox doesn't leak.
    """
    client = SandboxClient()
    sandbox_cm = client.sandbox(template_name=_TEMPLATE_NAME)
    ls_sandbox = sandbox_cm.__enter__()
    try:
        backend = LangSmithSandbox(sandbox=ls_sandbox)
        responses = await backend.aupload_files(
            [("/context.txt", task.context_window_text.encode("utf-8"))]
        )
        failures = [r for r in responses if r.error]
        if failures:
            details = ", ".join(f"{r.path}: {r.error}" for r in failures)
            msg = f"Failed to seed files into LangSmithSandbox: {details}"
            raise RuntimeError(msg)

        agent = create_deep_agent(
            model=model,
            backend=backend,
            system_prompt=SYSTEM_PROMPT,
        )
    except BaseException:
        # Roll back the context enter so we don't leak the sandbox on
        # a setup error.
        sandbox_cm.__exit__(None, None, None)
        raise

    def teardown() -> None:
        try:
            sandbox_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to close LangSmithSandbox")

    return RunnerContext(
        agent=agent,
        query_addendum="",
        # Shell runner owns its filesystem inside the sandbox; the
        # state-seeded path would stage ``/context.txt`` into agent
        # state and leave it invisible to shell commands.
        initial_files=None,
        teardown=teardown,
    )


__all__ = ["build_runner"]
