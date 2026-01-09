"""Server script to run DeepAgents in browser via textual-serve."""
import asyncio
from pathlib import Path

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.config import create_model, settings
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search


async def main():
    """Run the DeepAgents app with a real agent."""
    model = create_model()

    # Create tools list
    tools = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    async with get_checkpointer() as checkpointer:
        # Create the real agent
        agent, composite_backend = create_cli_agent(
            model=model,
            assistant_id="browser-agent",
            tools=tools,
            sandbox=None,
            sandbox_type=None,
            auto_approve=False,
            checkpointer=checkpointer,
        )

        # Create and run the app
        app = DeepAgentsApp(
            agent=agent,
            assistant_id="browser-agent",
            backend=composite_backend,
            auto_approve=False,
            cwd=Path.cwd(),
            thread_id=generate_thread_id(),
        )
        await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
