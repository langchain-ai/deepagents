"""DeepAgents ACP server implementation."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Literal

from acp import (
    Agent,
    AgentSideConnection,
    PROTOCOL_VERSION,
    stdio_streams,
)
from acp.schema import (
    AgentMessageChunk,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    SessionNotification,
    TextContentBlock,
    Implementation,
    AgentThoughtChunk,
    ToolCallProgress,
    ContentToolCallContent,
    LoadSessionResponse,
    SetSessionModeResponse,
    SetSessionModelResponse,
    CancelNotification,
    LoadSessionRequest,
    SetSessionModeRequest,
    SetSessionModelRequest,
    AgentPlanUpdate,
    PlanEntry,
    PermissionOption,
    RequestPermissionRequest,
    AllowedOutcome,
    DeniedOutcome,
    ToolCall as ACPToolCall,
)
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages.content import ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt


class DeepagentsACP(Agent):
    """ACP Agent implementation wrapping deepagents."""

    def __init__(
        self,
        connection: AgentSideConnection,
        agent_graph: CompiledStateGraph,
    ) -> None:
        """Initialize the DeepAgents agent.

        Args:
            connection: The ACP connection for communicating with the client
            agent_graph: A compiled LangGraph StateGraph (output of create_deep_agent)
        """
        self._connection = connection
        self._agent_graph = agent_graph
        self._sessions: dict[str, dict[str, Any]] = {}
        # Track tool calls by ID for matching with ToolMessages
        # Maps tool_call_id -> ToolCall TypedDict
        self._tool_calls: dict[str, ToolCall] = {}

    async def initialize(
        self,
        params: InitializeRequest,
    ) -> InitializeResponse:
        """Initialize the agent and return capabilities."""
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentInfo=Implementation(
                name="DeepAgents ACP Server",
                version="0.1.0",
                title="DeepAgents ACP Server",
            ),
        )

    async def newSession(
        self,
        params: NewSessionRequest,
    ) -> NewSessionResponse:
        """Create a new session with a deepagents instance."""
        session_id = str(uuid.uuid4())
        # Store session state with the shared agent graph
        self._sessions[session_id] = {
            "agent": self._agent_graph,
            "thread_id": str(uuid.uuid4()),
        }

        return NewSessionResponse(sessionId=session_id)

    async def _handle_ai_message_chunk(
        self,
        params: PromptRequest,
        message: AIMessageChunk,
    ) -> None:
        """Handle an AIMessageChunk and send appropriate notifications.

        Args:
            params: The prompt request parameters
            message: An AIMessageChunk from the streaming response

        Note:
            According to LangChain's content block types, message.content_blocks
            returns a list of ContentBlock unions. Each block is a TypedDict with
            a "type" field that discriminates the block type:
            - TextContentBlock: type="text", has "text" field
            - ReasoningContentBlock: type="reasoning", has "reasoning" field
            - ToolCallChunk: type="tool_call_chunk"
            - And many others (image, audio, video, etc.)
        """
        for block in message.content_blocks:
            # All content blocks have a "type" field for discrimination
            block_type = block.get("type")

            if block_type == "text":
                # TextContentBlock has a required "text" field
                text = block.get("text", "")
                if not text:  # Only yield non-empty text
                    continue
                await self._connection.sessionUpdate(
                    SessionNotification(
                        update=AgentMessageChunk(
                            content=TextContentBlock(text=text, type="text"),
                            sessionUpdate="agent_message_chunk",
                        ),
                        sessionId=params.sessionId,
                    )
                )
            elif block_type == "reasoning":
                # ReasoningContentBlock has a "reasoning" field (NotRequired)
                reasoning = block.get("reasoning", "")
                if not reasoning:
                    continue

                await self._connection.sessionUpdate(
                    SessionNotification(
                        update=AgentThoughtChunk(
                            content=TextContentBlock(text=reasoning, type="text"),
                            sessionUpdate="agent_thought_chunk",
                        ),
                        sessionId=params.sessionId,
                    )
                )

    async def _handle_completed_tool_calls(
        self,
        params: PromptRequest,
        message: AIMessage,
    ) -> None:
        """Handle completed tool calls from an AIMessage and send notifications.

        Args:
            params: The prompt request parameters
            message: An AIMessage containing tool_calls

        Note:
            According to LangChain's AIMessage type:
            - message.tool_calls: list[ToolCall] where ToolCall is a TypedDict with:
              - name: str (required)
              - args: dict[str, Any] (required)
              - id: str | None (required field, but can be None)
              - type: Literal["tool_call"] (optional, NotRequired)
        """
        # Use direct attribute access - tool_calls is a defined field on AIMessage
        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            # Access TypedDict fields directly (they're required fields)
            tool_call_id = tool_call["id"]  # str | None
            tool_name = tool_call["name"]  # str
            tool_args = tool_call["args"]  # dict[str, Any]

            # Skip tool calls without an ID (shouldn't happen in practice)
            if tool_call_id is None:
                continue

            # Skip todo tool calls as they're handled separately
            if tool_name == "todo":
                raise NotImplementedError("TODO tool call handling not implemented yet")

            # Send tool call progress update showing the tool is running
            await self._connection.sessionUpdate(
                SessionNotification(
                    update=ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId=tool_call_id,
                        title=tool_name,
                        rawInput=tool_args,
                        status="pending",
                    ),
                    sessionId=params.sessionId,
                )
            )

            # Store the tool call for later matching with ToolMessage
            self._tool_calls[tool_call_id] = tool_call

    async def _handle_tool_message(
        self,
        params: PromptRequest,
        tool_call: ToolCall,
        message: ToolMessage,
    ) -> None:
        """Handle a ToolMessage and send appropriate notifications.

        Args:
            params: The prompt request parameters
            tool_call: The original ToolCall that this message is responding to
            message: A ToolMessage containing the tool execution result

        Note:
            According to LangChain's ToolMessage type (inherits from BaseMessage):
            - message.content: str | list[str | dict] (from BaseMessage)
            - message.tool_call_id: str (specific to ToolMessage)
            - message.status: str | None (e.g., "error" for failed tool calls)
        """
        # Determine status based on message status or content
        status: Literal["completed", "failed"] = "completed"
        if hasattr(message, "status") and message.status == "error":
            status = "failed"

        # Build content blocks if message has content
        content_blocks = []
        for content_block in message.content_blocks:
            if content_block.get("type") == "text":
                text = content_block.get("text", "")
                if text:
                    content_blocks.append(
                        ContentToolCallContent(
                            type="content",
                            content=TextContentBlock(text=text, type="text"),
                        )
                    )
        # Send tool call progress update with the result
        await self._connection.sessionUpdate(
            SessionNotification(
                update=ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId=message.tool_call_id,
                    title=tool_call["name"],
                    content=content_blocks,
                    rawOutput=message.content,
                    status=status,
                ),
                sessionId=params.sessionId,
            )
        )

    async def _handle_todo_update(
        self,
        params: PromptRequest,
        todos: list[dict[str, Any]],
    ) -> None:
        """Handle todo list updates from the tools node.

        Args:
            params: The prompt request parameters
            todos: List of todo dictionaries with 'content' and 'status' fields

        Note:
            Todos come from the deepagents graph's write_todos tool and have the structure:
            [{'content': 'Task description', 'status': 'pending'|'in_progress'|'completed'}, ...]
        """
        # Convert todos to PlanEntry objects
        entries = []
        for todo in todos:
            # Extract fields from todo dict
            content = todo.get("content", "")
            status = todo.get("status", "pending")

            # Validate and cast status to PlanEntryStatus
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"

            # Create PlanEntry with default priority of "medium"
            entry = PlanEntry(
                content=content,
                status=status,  # type: ignore
                priority="medium",
            )
            entries.append(entry)

        # Send plan update notification
        await self._connection.sessionUpdate(
            SessionNotification(
                update=AgentPlanUpdate(
                    sessionUpdate="plan",
                    entries=entries,
                ),
                sessionId=params.sessionId,
            )
        )

    async def _handle_interrupt(
        self,
        params: PromptRequest,
        interrupt: Interrupt,
    ) -> list[dict[str, Any]]:
        """Handle a LangGraph interrupt and request permission from the client.

        Args:
            params: The prompt request parameters
            interrupt: The interrupt from LangGraph containing action_requests and review_configs

        Returns:
            List of decisions to pass to Command(resume={...})

        Note:
            The interrupt.value contains:
            - action_requests: [{'name': str, 'args': dict, 'description': str}, ...]
            - review_configs: [{'action_name': str, 'allowed_decisions': list[str]}, ...]
        """
        interrupt_data = interrupt.value
        action_requests = interrupt_data.get("action_requests", [])
        review_configs = interrupt_data.get("review_configs", [])

        # Create a mapping of action names to their allowed decisions
        allowed_decisions_map = {}
        for review_config in review_configs:
            action_name = review_config.get("action_name")
            allowed_decisions = review_config.get("allowed_decisions", [])
            allowed_decisions_map[action_name] = allowed_decisions

        # Collect decisions for all action requests
        decisions = []

        for action_request in action_requests:
            tool_name = action_request.get("name")
            tool_args = action_request.get("args", {})

            # Get allowed decisions for this action
            allowed_decisions = allowed_decisions_map.get(
                tool_name, ["approve", "reject"]
            )

            # Build permission options based on allowed decisions
            options = []
            if "approve" in allowed_decisions:
                options.append(
                    PermissionOption(
                        optionId="allow-once",
                        name="Allow once",
                        kind="allow_once",
                    )
                )
            if "reject" in allowed_decisions:
                options.append(
                    PermissionOption(
                        optionId="reject-once",
                        name="Reject",
                        kind="reject_once",
                    )
                )
            # Generate a tool call ID for this permission request
            # We need to find the corresponding tool call from the stored calls
            # For now, use a generated ID
            tool_call_id = f"perm_{uuid.uuid4().hex[:8]}"

            # Create ACP ToolCall object for the permission request
            acp_tool_call = ACPToolCall(
                toolCallId=tool_call_id,
                title=tool_name,
                rawInput=tool_args,
                status="pending",
            )

            # Send permission request to client
            response = await self._connection.requestPermission(
                RequestPermissionRequest(
                    sessionId=params.sessionId,
                    toolCall=acp_tool_call,
                    options=options,
                )
            )

            # Convert ACP response to LangGraph decision
            outcome = response.outcome

            if isinstance(outcome, AllowedOutcome):
                option_id = outcome.optionId
                if option_id == "allow-once":
                    # Check if this was actually an edit option
                    selected_option = next(
                        (opt for opt in options if opt.optionId == option_id), None
                    )
                    if selected_option and selected_option.field_meta:
                        # This is an edit - for now, just approve
                        # TODO: Implement actual edit functionality
                        decisions.append({"type": "approve"})
                    else:
                        decisions.append({"type": "approve"})
                elif option_id == "edit":
                    # Edit option - for now, just approve
                    # TODO: Implement actual edit functionality to collect edited args
                    decisions.append({"type": "approve"})
            elif isinstance(outcome, DeniedOutcome):
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Action rejected by user",
                    }
                )

        return decisions

    async def _stream_and_handle_updates(
        self,
        params: PromptRequest,
        agent: Any,
        stream_input: dict[str, Any] | Command,
        config: dict[str, Any],
    ) -> list[Interrupt]:
        """Stream agent execution and handle updates, returning any interrupts.

        Args:
            params: The prompt request parameters
            agent: The agent to stream from
            stream_input: Input to pass to agent.astream (initial message or Command)
            config: Configuration with thread_id

        Returns:
            List of interrupts that occurred during streaming
        """
        interrupts = []

        async for stream_mode, data in agent.astream(
            stream_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                # Handle streaming message chunks (AIMessageChunk)
                message, metadata = data
                if isinstance(message, AIMessageChunk):
                    await self._handle_ai_message_chunk(params, message)
            elif stream_mode == "updates":
                # Handle completed node updates
                for node_name, update in data.items():
                    # Check for interrupts
                    if node_name == "__interrupt__":
                        # Extract interrupts from the update
                        interrupts.extend(update)
                        continue

                    # Only process model and tools nodes
                    if node_name not in ("model", "tools"):
                        continue

                    # Handle todos from tools node
                    if node_name == "tools" and "todos" in update:
                        todos = update.get("todos", [])
                        if todos:
                            await self._handle_todo_update(params, todos)

                    # Get messages from the update
                    messages = update.get("messages", [])
                    if not messages:
                        continue

                    # Process the last message from this node
                    last_message = messages[-1]

                    # Handle completed AI messages from model node
                    if node_name == "model" and isinstance(last_message, AIMessage):
                        # Check if this AIMessage has tool calls
                        if last_message.tool_calls:
                            await self._handle_completed_tool_calls(
                                params, last_message
                            )

                    # Handle tool execution results from tools node
                    elif node_name == "tools" and isinstance(last_message, ToolMessage):
                        # Look up the original tool call by ID
                        tool_call = self._tool_calls.get(last_message.tool_call_id)
                        if tool_call:
                            await self._handle_tool_message(
                                params, tool_call, last_message
                            )

        return interrupts

    async def prompt(
        self,
        params: PromptRequest,
    ) -> PromptResponse:
        """Handle a user prompt and stream responses."""
        session_id = params.sessionId
        session = self._sessions.get(session_id)

        # Extract text from prompt content blocks
        prompt_text = ""
        for block in params.prompt:
            if hasattr(block, "text"):
                prompt_text += block.text
            elif isinstance(block, dict) and "text" in block:
                prompt_text += block["text"]

        # Stream the agent's response
        agent = session["agent"]
        thread_id = session["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}

        # Start with the initial user message
        stream_input: dict[str, Any] | Command = {
            "messages": [{"role": "user", "content": prompt_text}]
        }

        # Loop until there are no more interrupts
        while True:
            # Stream and collect any interrupts
            interrupts = await self._stream_and_handle_updates(
                params, agent, stream_input, config
            )

            # If no interrupts, we're done
            if not interrupts:
                break

            # Process each interrupt and collect decisions
            all_decisions = []
            for interrupt in interrupts:
                decisions = await self._handle_interrupt(params, interrupt)
                all_decisions.extend(decisions)

            # Prepare to resume with the collected decisions
            stream_input = Command(resume={"decisions": all_decisions})

        return PromptResponse(stopReason="end_turn")

    async def authenticate(self, params: Any) -> Any | None:
        """Authenticate (optional)."""
        # Authentication not required for now
        return None

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods (optional)."""
        raise NotImplementedError(f"Extension method {method} not supported")

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications (optional)."""
        pass

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel a running session."""
        # TODO: Implement cancellation logic
        pass

    async def loadSession(
        self,
        params: LoadSessionRequest,
    ) -> LoadSessionResponse | None:
        """Load an existing session (optional)."""
        # Not implemented yet - would need to serialize/deserialize session state
        return None

    async def setSessionMode(
        self,
        params: SetSessionModeRequest,
    ) -> SetSessionModeResponse | None:
        """Set session mode (optional)."""
        # Could be used to switch between different agent modes
        return None

    async def setSessionModel(
        self,
        params: SetSessionModelRequest,
    ) -> SetSessionModelResponse | None:
        """Set session model (optional)."""
        # Not supported - model is configured at agent graph creation time
        return None


async def main() -> None:
    """Main entry point for running the ACP server."""
    # from deepagents_cli.agent import create_agent_with_config
    # from deepagents_cli.config import create_model
    # from deepagents_cli.tools import fetch_url, http_request, web_search
    #
    # # Create model using CLI configuration
    # model = create_model()
    #
    # # Setup tools - conditionally include web_search if Tavily is available
    # tools = [http_request, fetch_url]
    # if os.environ.get("TAVILY_API_KEY"):
    #     tools.append(web_search)
    #
    # # Create CLI agent with shell access and other CLI features
    # # Using default assistant_id "agent" for ACP server
    # agent_graph, composite_backend = create_agent_with_config(
    #     model=model,
    #     assistant_id="agent",
    #     tools=tools,
    #     sandbox=None,  # Local mode
    #     sandbox_type=None,
    #     system_prompt=None,  # Use default CLI system prompt
    #     auto_approve=False,  # Require user approval for destructive operations
    #     enable_memory=True,  # Enable persistent memory
    #     enable_skills=True,  # Enable custom skills
    #     enable_shell=True,  # Enable shell access
    # )
    #
    # Define default tools

    from langchain.agents.middleware import HumanInTheLoopMiddleware

    @tool()
    def get_weather(location: str) -> str:
        """Get the weather for a given location."""
        return f"The weather in {location} is sunny with a high of 75°F."

    # Create the agent graph with default configuration
    model = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )

    agent_graph = create_deep_agent(
        model=model,
        tools=[get_weather],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "get_weather": True,
                }
            )
        ],
    )

    # Start the ACP server
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: DeepagentsACP(conn, agent_graph), writer, reader)
    await asyncio.Event().wait()


def cli_main() -> None:
    """Synchronous CLI entry point for the ACP server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
