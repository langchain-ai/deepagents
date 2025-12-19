"""ChATLAS agent implementation using DeepAgents framework."""

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from chatlas_agents.config import AgentConfig
from chatlas_agents.mcp import create_mcp_client, cleanup_mcp_session
from chatlas_agents.sandbox import SandboxBackendType

logger = logging.getLogger(__name__)


class DeepAgentWrapper:
    """ChATLAS agent using DeepAgents framework for advanced capabilities."""

    def __init__(
        self, 
        config: AgentConfig, 
        use_docker_sandbox: bool = False, 
        docker_image: str = "python:3.13-slim",
        sandbox_backend: SandboxBackendType = SandboxBackendType.APPTAINER,
    ):
        """Initialize DeepAgent wrapper.

        Args:
            config: Agent configuration
            use_docker_sandbox: Whether to use container for sandbox execution
            docker_image: Container image to use for sandbox
            sandbox_backend: Type of sandbox backend (docker or apptainer), defaults to apptainer
        """
        self.config = config
        self.app = None
        self.mcp_client = create_mcp_client(config.mcp)
        self.use_docker_sandbox = use_docker_sandbox
        self.docker_image = docker_image
        self.sandbox_backend = sandbox_backend

    async def initialize(self):
        """Initialize the DeepAgent application."""
        from chatlas_agents.graph import create_chatlas_deep_agent
        
        self.app = await create_chatlas_deep_agent(
            self.config,
            use_docker_sandbox=self.use_docker_sandbox,
            docker_image=self.docker_image,
            sandbox_backend=self.sandbox_backend,
        )
        logger.info(f"DeepAgent '{self.config.name}' initialized")
        if self.use_docker_sandbox:
            logger.info(f"{self.sandbox_backend.value.capitalize()} sandbox enabled for command execution")

    async def run(
        self,
        input_text: str,
        thread_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Run the agent with the given input.

        Args:
            input_text: User input
            thread_id: Thread ID for conversation persistence
            **kwargs: Additional arguments

        Returns:
            Agent response
        """
        if self.app is None:
            await self.initialize()

        try:
            # DeepAgents expect messages in LangChain format
            config = {"configurable": {"thread_id": thread_id}}
            input_state = {
                "messages": [HumanMessage(content=input_text)]
            }

            # Invoke the DeepAgent
            result = await self.app.ainvoke(input_state, config)

            # Extract output from the final state
            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None

            if last_message:
                output = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                output = "No response generated"

            return {
                "input": input_text,
                "output": output,
                "thread_id": thread_id,
            }
        except Exception as e:
            logger.error(f"Error running DeepAgent: {e}", exc_info=True)
            return {"error": str(e), "input": input_text}

    async def stream(
        self,
        input_text: str,
        thread_id: str = "default",
        **kwargs
    ):
        """Stream agent responses.

        Args:
            input_text: User input
            thread_id: Thread ID for conversation persistence
            **kwargs: Additional arguments

        Yields:
            Agent response chunks
        """
        if self.app is None:
            await self.initialize()

        try:
            config = {"configurable": {"thread_id": thread_id}}
            input_state = {
                "messages": [HumanMessage(content=input_text)]
            }

            async for event in self.app.astream(input_state, config):
                yield event
        except Exception as e:
            logger.error(f"Error streaming from DeepAgent: {e}")
            yield {"error": str(e)}

    async def close(self):
        """Clean up resources."""
        # Clean up MCP session resources
        await cleanup_mcp_session()



async def create_deep_agent(
    config: AgentConfig,
    use_docker_sandbox: bool = False,
    docker_image: str = "python:3.13-slim",
    sandbox_backend: SandboxBackendType = SandboxBackendType.APPTAINER,
) -> DeepAgentWrapper:
    """Create and initialize a DeepAgent-based ChATLAS agent.

    Args:
        config: Agent configuration
        use_docker_sandbox: Whether to use container for sandbox execution
        docker_image: Container image to use for sandbox
        sandbox_backend: Type of sandbox backend (docker or apptainer), defaults to apptainer

    Returns:
        Initialized DeepAgent wrapper
    """
    agent = DeepAgentWrapper(
        config, 
        use_docker_sandbox=use_docker_sandbox, 
        docker_image=docker_image,
        sandbox_backend=sandbox_backend,
    )
    await agent.initialize()
    return agent
