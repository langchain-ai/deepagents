"""Demo coding agent using ACP."""

import asyncio
import os
from pathlib import Path

from acp import (
    run_agent as run_acp_agent,
)
from acp.schema import (
    SessionMode,
    SessionModeState,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import Checkpointer, CompiledStateGraph
from langgraph.prebuilt import ToolRuntime

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from examples.local_context import LocalContextMiddleware


def _get_interrupt_config(mode_id: str) -> dict:
    """Get interrupt configuration for a given mode."""
    mode_to_interrupt = {
        "ask_before_edits": {
            "edit_file": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
            "write_todos": {"allowed_decisions": ["approve", "reject"]},
            "execute": {"allowed_decisions": ["approve", "reject"]},
        },
        "accept_edits": {
            "write_todos": {"allowed_decisions": ["approve", "reject"]},
            "execute": {"allowed_decisions": ["approve", "reject"]},
        },
        "accept_everything": {},
    }
    return mode_to_interrupt.get(mode_id, {})


async def _serve_example_agent() -> None:
    """Run example agent from the root of the repository with ACP integration."""
    load_dotenv()

    # Create a data directory for storing session databases
    # This keeps checkpoint databases out of project directories and gitignored
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Use a single database for all sessions
    # Sessions are distinguished by thread_id (session_id)
    db_path = data_dir / "sessions.db"

    # Use AsyncSqliteSaver for persistent session storage with async support
    # This allows sessions to be loaded across restarts with better performance
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        await _serve_with_checkpointer(checkpointer)


async def _serve_with_checkpointer(checkpointer: Checkpointer) -> None:
    """Serve the agent with the provided checkpointer."""

    def build_agent(context: AgentSessionContext) -> CompiledStateGraph:
        """Agent factory based in the given root directory."""
        _root_dir = context.cwd
        interrupt_config = _get_interrupt_config(context.mode)

        def create_backend(tr: ToolRuntime | None = None) -> CompositeBackend:
            ephemeral_backend = StateBackend(tr) if tr is not None else None
            shell_env = os.environ.copy()

            # Use CLIShellBackend for filesystem + shell execution.
            # Provides `execute` tool via FilesystemMiddleware with per-command
            # timeout support.
            shell_backend = LocalShellBackend(
                root_dir=_root_dir,
                inherit_env=True,
                env=shell_env,
            )
            return CompositeBackend(
                default=shell_backend,
                routes={
                    "/memories/": ephemeral_backend,
                    "/conversation_history/": ephemeral_backend,
                }
                if ephemeral_backend is not None
                else {},
            )

        return create_deep_agent(
            checkpointer=checkpointer,
            backend=create_backend,
            interrupt_on=interrupt_config,
            middleware=[LocalContextMiddleware(backend=create_backend())],
        )

    modes = SessionModeState(
        current_mode_id="accept_edits",
        available_modes=[
            SessionMode(
                id="ask_before_edits",
                name="Ask before edits",
                description="Ask permission before edits, writes, shell commands, and plans",
            ),
            SessionMode(
                id="accept_edits",
                name="Accept edits",
                description="Auto-accept edit operations, but ask before shell commands and plans",
            ),
            SessionMode(
                id="accept_everything",
                name="Accept everything",
                description="Auto-accept all operations without asking permission",
            ),
        ],
    )

    acp_agent = AgentServerACP(agent=build_agent, modes=modes, checkpointer=checkpointer)
    await run_acp_agent(acp_agent)


def main() -> None:
    """Run the demo agent."""
    asyncio.run(_serve_example_agent())


if __name__ == "__main__":
    main()
