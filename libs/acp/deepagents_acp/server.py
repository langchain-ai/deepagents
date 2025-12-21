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
)
from deepagents.graph import create_deep_agent
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
            elif block_type == "tool_call_chunk":
                # ToolCallChunk: type="tool_call_chunk"
                # Has fields: id (str | None), name (str | None), args (str | None)
                # TODO: Add tool call chunk handling when ACP schema supports it
                pass

        # Check for complete tool calls (not chunks)
        # Note: During streaming, tool_calls on AIMessageChunk accumulates as chunks
        # are merged. Complete tool calls are only reliably available in the
        # "updates" stream mode when the model node completes.
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # If this is a todo tool call, send a plan update
                if tool_call["name"] == "todo":
                    await self._handle_todo_tool_call(params, tool_call)

    async def _handle_todo_tool_call(
        self,
        params: PromptRequest,
        tool_call: ToolCall,
    ) -> None:
        """Handle a todo tool call and send plan update."""
        try:
            # Get the tool call arguments
            args = tool_call.get("args", {})

            # Extract todos list
            todos = args.get("todos", [])

            # Convert todos to PlanEntry objects
            plan_entries = []
            for todo in todos:
                if isinstance(todo, dict):
                    content = todo.get("content", "")
                    status_str = todo.get("status", "pending")

                    # Map status to PlanEntry status
                    entry_status: Literal["pending", "in_progress", "completed"] = "pending"
                    if status_str in ("pending", "in_progress", "completed"):
                        entry_status = status_str

                    plan_entries.append(
                        PlanEntry(
                            content=content,
                            status=entry_status,
                        )
                    )

            # Send plan update
            if plan_entries:
                await self._connection.sessionUpdate(
                    SessionNotification(
                        update=AgentPlanUpdate(
                            sessionUpdate="plan",
                            entries=plan_entries,
                        ),
                        sessionId=params.sessionId,
                    )
                )
        except Exception:
            # If parsing fails, just skip the plan update
            pass

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
                continue

            # Send tool call progress update showing the tool is running
            await self._connection.sessionUpdate(
                SessionNotification(
                    update=ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId=tool_call_id,
                        title=tool_name,
                        rawInput=tool_args,
                        status="running",
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
        content_blocks = None
        if message.content:
            # Convert tool message content to text
            # message.content can be str or list[str | dict]
            if isinstance(message.content, str):
                content_text = message.content
            elif isinstance(message.content, list):
                # Join list items into a single string
                content_text = "\n".join(
                    item if isinstance(item, str) else str(item) for item in message.content
                )
            else:
                content_text = str(message.content)

            content_blocks = [
                ContentToolCallContent(
                    type="content",
                    content=TextContentBlock(text=content_text, type="text"),
                )
            ]

        # Extract tool_call_id - it's a required attribute on ToolMessage
        tool_call_id = message.tool_call_id

        # Send tool call progress update with the result
        await self._connection.sessionUpdate(
            SessionNotification(
                update=ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId=tool_call_id,
                    title=tool_call["name"],
                    content=content_blocks,
                    rawOutput=message.content,
                    status=status,
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

        ai_message = None

        async for stream_mode, data in agent.astream(
            {"messages": [{"role": "user", "content": prompt_text}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if stream_mode == "messages":
                message, metadata = data
                if isinstance(message, AIMessageChunk):
                    ai_message = message if ai_message is None else ai_message + message
                    await self._handle_ai_message_chunk(params, message)
            elif stream_mode == "updates":
                # Handle updates from the agent (completed messages)
                for source, update in data.items():
                    if source in ("model", "tools"):
                        # Get the last message from the update
                        messages = update.get("messages", [])
                        if not messages:
                            continue

                        last_message = messages[-1]

                        # Handle completed tool calls (AIMessage with tool_calls)
                        if isinstance(last_message, AIMessage) and source == "model":
                            await self._handle_completed_tool_calls(params, last_message)

                        # Handle tool results (ToolMessage)
                        elif isinstance(last_message, ToolMessage) and source == "tools":
                            # Look up the tool call by ID
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
