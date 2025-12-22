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
    PlanEntryStatus,
)
from deepagents.graph import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages.content import ToolCall
from langchain_core.tools import tool


@tool()
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # Dummy implementation for example purposes
    return f"The weather in {location} is sunny with a high of 75°F."


class DeepagentsACP(Agent):
    """ACP Agent implementation wrapping deepagents."""

    def __init__(
        self,
        connection: AgentSideConnection,
        model: str | BaseChatModel | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        """Initialize the DeepAgents agent.

        Args:
            connection: The ACP connection for communicating with the client
            model: The model to use (string or BaseChatModel instance)
            tools: List of tools to provide to the agent
        """
        self._connection = connection
        self._tools = tools or [get_weather]
        self._sessions: dict[str, dict[str, Any]] = {}
        # Track tool calls by ID for matching with ToolMessages
        # Maps tool_call_id -> ToolCall TypedDict
        self._tool_calls: dict[str, ToolCall] = {}

        # Handle model parameter
        if model is None:
            # Use default Claude Sonnet
            from langchain_anthropic import ChatAnthropic

            self._model = ChatAnthropic(
                model_name="claude-sonnet-4-5-20250929",
                max_tokens=20000,
            )
        elif isinstance(model, str):
            # Try to create ChatAnthropic from string
            # Support common model name formats
            from langchain_anthropic import ChatAnthropic

            model_name = model
            if "/" in model_name:
                # Strip provider prefix if present (e.g., "anthropic/claude-...")
                model_name = model_name.split("/", 1)[1]
            self._model = ChatAnthropic(
                model_name=model_name,
                max_tokens=20000,
            )
        else:
            # Use provided BaseChatModel instance
            self._model = model

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
        # Create deepagents instance
        agent = create_deep_agent(
            model=self._model,
            tools=self._tools,
            checkpointer=InMemorySaver(),
        )

        session_id = str(uuid.uuid4())
        # Store session state
        self._sessions[session_id] = {
            "agent": agent,
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
        # # Stream the agent's response
        agent = session["agent"]
        thread_id = session["thread_id"]

        async for stream_mode, data in agent.astream(
            {"messages": [{"role": "user", "content": prompt_text}]},
            config={"configurable": {"thread_id": thread_id}},
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
                            await self._handle_completed_tool_calls(params, last_message)

                    # Handle tool execution results from tools node
                    elif node_name == "tools" and isinstance(last_message, ToolMessage):
                        # Look up the original tool call by ID
                        tool_call = self._tool_calls.get(last_message.tool_call_id)
                        if tool_call:
                            await self._handle_tool_message(params, tool_call, last_message)

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
        session_id = params.sessionId
        self._cancelled_sessions.add(session_id)

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
        session_id = params.sessionId
        session = self._sessions.get(session_id)

        if session and params.model:
            # Would need to recreate the agent with new model
            # For now, just store it but don't recreate
            session["model"] = params.model
            return SetSessionModelResponse()

        return None


async def main() -> None:
    """Main entry point for running the ACP server."""
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: DeepagentsACP(conn), writer, reader)
    await asyncio.Event().wait()


def cli_main() -> None:
    """Synchronous CLI entry point for the ACP server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
