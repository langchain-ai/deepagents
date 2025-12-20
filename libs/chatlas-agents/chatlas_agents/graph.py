"""DeepAgents-based agent implementation for ChATLAS.

This module provides integration between ChATLAS and the deepagents framework,
with support for MCP tools and container-based sandbox execution.
"""

import logging
from typing import Any, Dict, List, Optional
from deepagents import create_deep_agent
import inspect
from langchain_core.tools import BaseTool

from chatlas_agents.config import AgentConfig
from chatlas_agents.llm import create_llm_from_config
from chatlas_agents.tools import load_mcp_tools
from chatlas_agents.sandbox import (
    DockerSandboxBackend,
    ApptainerSandboxBackend,
    SandboxBackendType,
)

logger = logging.getLogger(__name__)


async def create_chatlas_deep_agent(
    config: AgentConfig,
    custom_tools: Optional[List[BaseTool]] = None,
    use_docker_sandbox: bool = False,
    docker_image: str = "python:3.13-slim",
    sandbox_backend: SandboxBackendType = SandboxBackendType.APPTAINER,
):
    """Create a DeepAgent for ChATLAS with MCP server integration.
    
    Args:
        config: Agent configuration
        custom_tools: Optional additional custom tools
        use_docker_sandbox: Whether to use container for sandbox execution
        docker_image: Container image to use for sandbox
        sandbox_backend: Type of sandbox backend to use (docker or apptainer), defaults to apptainer
        
    Returns:
        Compiled DeepAgent (LangGraph StateGraph)
        
    Note:
        DeepAgents come with built-in tools:
        - Planning (TODO list management)
        - File system operations
        - Sub-agent spawning
        - Computer access capabilities (with container sandbox)
    """
    # Create LLM
    llm = create_llm_from_config(config.llm)
    
    # Load tools from MCP server
    mcp_tools = await load_mcp_tools(config.mcp)
    
    logger.info(f"Loaded {len(mcp_tools)} tools from MCP server")
    
    # Combine MCP tools with any custom tools
    all_tools = mcp_tools if mcp_tools else []
    if custom_tools:
        all_tools.extend(custom_tools)
        logger.info(f"Added {len(custom_tools)} custom tools")
    
    # Build system prompt
    sandbox_info = ""
    if use_docker_sandbox:
        if sandbox_backend == SandboxBackendType.APPTAINER:
            sandbox_info = "\nYou are running in an Apptainer sandbox environment with full shell access for executing commands securely."
        else:
            sandbox_info = "\nYou are running in a Docker sandbox environment with full shell access for executing commands securely."
    
    system_prompt = f"""You are {config.name}, an AI assistant for the ChATLAS system.

{config.description}

You have access to tools from the ChATLAS MCP server for interacting with the system.
Use the built-in planning, file system, and sub-agent capabilities to tackle complex tasks.{sandbox_info}

Be helpful, accurate, and methodical in your approach. Break down complex tasks into manageable steps.
"""
    
    # Determine whether the installed deepagents supports injecting a backend
    backend = None
    supports_backend = False
    try:
        sig = inspect.signature(create_deep_agent)
        supports_backend = "backend" in sig.parameters
    except Exception:
        supports_backend = False

    # Create sandbox backend only if the factory supports it
    if use_docker_sandbox and supports_backend:
        try:
            if sandbox_backend == SandboxBackendType.APPTAINER:
                logger.info(f"Creating Apptainer sandbox with image: {docker_image}")
                # Ensure docker_image has proper format for Apptainer
                apptainer_image = docker_image
                if not any(docker_image.startswith(prefix) for prefix in ["docker://", "oras://", "library://", "/"]):
                    apptainer_image = f"docker://{docker_image}"
                backend = ApptainerSandboxBackend(image=apptainer_image)
                logger.info(f"Apptainer sandbox created: {backend.id}")
            else:
                logger.info(f"Creating Docker sandbox with image: {docker_image}")
                backend = DockerSandboxBackend(image=docker_image)
                logger.info(f"Docker sandbox created: {backend.id[:12]}")
        except Exception as e:
            logger.error(f"Failed to create {sandbox_backend.value} sandbox: {e}")
            logger.warning("Falling back to in-memory backend")
            backend = None
    elif use_docker_sandbox and not supports_backend:
        logger.warning(f"Installed deepagents.create_deep_agent does not accept a 'backend' parameter; cannot enable {sandbox_backend.value} sandbox. Falling back to default filesystem backend.")
    
    # Don't manually add FilesystemMiddleware here — pass backend to create_deep_agent
    # so the deepagents factory can create and de-duplicate middleware correctly.
    middleware = []
    
    # Create the DeepAgent
    # DeepAgents includes built-in middleware for:
    # - FilesystemMiddleware: File system operations (with Docker sandbox if provided)
    # - SubAgentMiddleware: Spawn and manage sub-agents
    # - Planning capabilities with TODO lists
    # Build kwargs and include `backend` only if supported by the installed deepagents
    create_kwargs = {
        "model": llm,
        "tools": all_tools if all_tools else None,
        "system_prompt": system_prompt,
        "middleware": middleware if middleware else (),
    }
    if supports_backend and backend is not None:
        create_kwargs["backend"] = backend

    agent = create_deep_agent(**create_kwargs)
    
    logger.info(f"DeepAgent '{config.name}' created successfully with {len(all_tools)} tools")
    if use_docker_sandbox:
        logger.info(f"{sandbox_backend.value.capitalize()} sandbox enabled for command execution")
    
    return agent


# Create default agent instance for CLI usage
graph = None


async def get_graph():
    """Get or initialize the default DeepAgent for CLI usage.
    
    Returns:
        Compiled DeepAgent application
    """
    global graph
    if graph is None:
        from chatlas_agents.config import load_config_from_env
        
        config = load_config_from_env()
        graph = await create_chatlas_deep_agent(config)
    return graph


def get_graph_sync():
    """Synchronous wrapper for get_graph for CLI compatibility.
    
    Returns:
        Compiled DeepAgent application
    """
    import asyncio
    return asyncio.run(get_graph())

