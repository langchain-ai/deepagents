"""Main entry point and CLI loop for deepagents."""
# ruff: noqa: E402, BLE001, PLR0912, PLR0915

# Suppress deprecation warnings from langchain_core (e.g., Pydantic V1 on Python 3.14+)
# ruff: noqa: E402
import warnings

warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")

import asyncio

# Suppress Pydantic v1 compatibility warnings from langchain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import create_server_agent

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import settings
from deepagents_cli.integrations.sandbox_factory import create_sandbox_async
from deepagents_cli.tools import fetch_url, http_request, web_search

tools = [http_request, fetch_url]
if settings.has_tavily:
    tools.append(web_search)

from langgraph_sdk import get_client

client = get_client()

# Sentinel value to indicate sandbox creation is in progress
SANDBOX_CREATING = "__creating__"
# How long to wait for sandbox creation (seconds)
SANDBOX_CREATION_TIMEOUT = 180
# How often to poll for sandbox_id (seconds)
SANDBOX_POLL_INTERVAL = 1.0


async def _get_sandbox_id_from_metadata(thread_id: str) -> str | None:
    """Get sandbox_id from thread metadata."""
    thread = await client.threads.get(thread_id=thread_id)
    return thread.get("metadata", {}).get("sandbox_id")


async def _wait_for_sandbox_id(thread_id: str) -> str:
    """Wait for sandbox_id to be set in thread metadata.
    
    Polls thread metadata until sandbox_id is set to a real value
    (not the creating sentinel).
    
    Raises:
        TimeoutError: If sandbox creation takes too long
    """
    elapsed = 0.0
    while elapsed < SANDBOX_CREATION_TIMEOUT:
        sandbox_id = await _get_sandbox_id_from_metadata(thread_id)
        if sandbox_id is not None and sandbox_id != SANDBOX_CREATING:
            return sandbox_id
        await asyncio.sleep(SANDBOX_POLL_INTERVAL)
        elapsed += SANDBOX_POLL_INTERVAL
    
    msg = f"Timeout waiting for sandbox creation for thread {thread_id}"
    raise TimeoutError(msg)


async def get_agent(config):
    """Get or create an agent with a sandbox for the given thread."""
    thread_id = config["configurable"].get("thread_id", None)
    
    if thread_id is None:
        # No thread_id means no sandbox
        return create_server_agent(
            model=None,
            assistant_id="agent",
            tools=tools,
            sandbox=None,
            sandbox_type=None,
            auto_approve=True,
        )
    
    # Check if sandbox already exists or is being created
    sandbox_id = await _get_sandbox_id_from_metadata(thread_id)
    
    if sandbox_id == SANDBOX_CREATING:
        # Another call is creating the sandbox, wait for it
        sandbox_id = await _wait_for_sandbox_id(thread_id)
    
    if sandbox_id is None:
        # No sandbox yet - we need to create one
        # First, set sentinel to prevent other callers from also creating
        await client.threads.update(
            thread_id=thread_id,
            metadata={"sandbox_id": SANDBOX_CREATING}
        )
        
        try:
            # Create the sandbox
            sandbox_cm = create_sandbox_async("langsmith", cleanup=False)
            sandbox_backend = await sandbox_cm.__aenter__()
            
            # Update metadata with real sandbox_id
            await client.threads.update(
                thread_id=thread_id,
                metadata={"sandbox_id": sandbox_backend._sandbox.name}
            )
        except Exception:
            # Clear sentinel on failure so others can retry
            await client.threads.update(
                thread_id=thread_id,
                metadata={"sandbox_id": None}
            )
            raise
    else:
        # Connect to existing sandbox
        sandbox_cm = create_sandbox_async("langsmith", sandbox_id=sandbox_id, cleanup=False)
        sandbox_backend = await sandbox_cm.__aenter__()
    
    return create_server_agent(
        model=None,
        assistant_id="agent",
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type="langsmith",
        auto_approve=True,
    )
