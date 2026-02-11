import json
from typing import Any
from uuid import uuid4

from acp import (
    Agent as ACPAgent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionModeResponse,
    run_agent as run_acp_agent,
    start_edit_tool_call,
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AgentPlanUpdate,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    PermissionOption,
    PlanEntry,
    PromptCapabilities,
    ResourceContentBlock,
    SessionMode,
    SessionModeState,
    SseMcpServer,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.graph import Checkpointer, CompiledStateGraph
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, StateSnapshot

from deepagents_acp.utils import (
    convert_audio_block_to_content_blocks,
    convert_embedded_resource_block_to_content_blocks,
    convert_image_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
)

load_dotenv()


class ACPAgentServer(ACPAgent):
    _conn: Client

    _deepagent: CompiledStateGraph
    _root_dir: str
    _checkpointer: Checkpointer
    _mode: str

    @staticmethod
    def _get_interrupt_config(mode_id: str) -> dict:
        """Get interrupt configuration for a given mode"""
        mode_to_interrupt = {
            "ask_before_edits": {
                "edit_file": {"allowed_decisions": ["approve", "reject"]},
                "write_file": {"allowed_decisions": ["approve", "reject"]},
                "write_todos": {"allowed_decisions": ["approve", "reject"]},
            },
            "auto": {
                "write_todos": {"allowed_decisions": ["approve", "reject"]},
            },
        }
        return mode_to_interrupt.get(mode_id, {})

    def _create_deepagent(self, mode: str):
        """Create a DeepAgent with the appropriate configuration for the given mode"""
        interrupt_config = self._get_interrupt_config(mode)

        def create_backend(tr):
            ephemeral_backend = StateBackend(tr)
            return CompositeBackend(
                default=FilesystemBackend(root_dir=self._root_dir, virtual_mode=True),
                routes={
                    "/memories/": ephemeral_backend,
                    "/conversation_history/": ephemeral_backend,
                },
            )

        return create_deep_agent(
            checkpointer=self._checkpointer,
            backend=create_backend,
            interrupt_on=interrupt_config,
        )

    def __init__(
        self,
        root_dir: str,
        checkpointer: Checkpointer,
        mode: str,
    ):
        self._root_dir = root_dir
        self._checkpointer = checkpointer
        self._mode = mode
        self._deepagent = self._create_deepagent(mode)
        self._cancelled = False
        self._session_plans: dict[str, list[dict[str, Any]]] = {}  # Track current plan per session
        super().__init__()

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    # embedded_context=True,
                )
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        # Define available modes
        available_modes = [
            SessionMode(
                id="ask_before_edits",
                name="Ask before edits",
                description="Ask permission before edits and writes",
            ),
            SessionMode(
                id="auto",
                name="Accept edits",
                description="Auto-accept edit operations",
            ),
        ]

        return NewSessionResponse(
            session_id=uuid4().hex,
            modes=SessionModeState(
                available_modes=available_modes,
                current_mode_id=self._mode,
            ),
        )

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse:
        # Recreate the deep agent with new mode configuration
        self._deepagent = self._create_deepagent(mode_id)
        self._mode = mode_id

        return SetSessionModeResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel the current execution."""
        self._cancelled = True

    async def _log_text(self, session_id: str, text: str):
        update = update_agent_message(text_block(text))
        await self._conn.session_update(session_id=session_id, update=update, source="DeepAgent")

    def _all_tasks_completed(self, plan: list[dict[str, Any]]) -> bool:
        """Check if all tasks in a plan are completed.

        Args:
            plan: List of todo dictionaries

        Returns:
            True if all tasks have status 'completed', False otherwise
        """
        if not plan:
            return True

        return all(todo.get("status") == "completed" for todo in plan)

    async def _clear_plan(self, session_id: str) -> None:
        """Clear the plan by sending an empty plan update.

        Args:
            session_id: The session ID
        """
        update = AgentPlanUpdate(
            session_update="plan",
            entries=[],
        )
        await self._conn.session_update(
            session_id=session_id,
            update=update,
            source="DeepAgent",
        )
        # Clear the stored plan for this session
        self._session_plans[session_id] = []

    async def _handle_todo_update(
        self,
        session_id: str,
        todos: list[dict[str, Any]],
        log_plan: bool = True,
    ) -> None:
        """Handle todo list updates from write_todos tool.

        Args:
            session_id: The session ID
            todos: List of todo dictionaries with 'content' and 'status' fields
            log_plan: Whether to log the plan as a visible text message
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
        update = AgentPlanUpdate(
            session_update="plan",
            entries=entries,
        )
        await self._conn.session_update(
            session_id=session_id,
            update=update,
            source="DeepAgent",
        )

        # Optionally send a visible text message showing the plan
        if log_plan:
            plan_text = "## Plan\n\n"
            for i, todo in enumerate(todos, 1):
                content = todo.get("content", "")
                plan_text += f"{i}. {content}\n"

            await self._log_text(session_id=session_id, text=plan_text)

    async def _process_tool_call_chunks(
        self,
        session_id: str,
        message_chunk: Any,
        active_tool_calls: dict,
        tool_call_accumulator: dict,
    ) -> None:
        """Process tool call chunks and start tool calls when complete."""
        if (
            not isinstance(message_chunk, str)
            and hasattr(message_chunk, "tool_call_chunks")
            and message_chunk.tool_call_chunks
        ):
            for chunk in message_chunk.tool_call_chunks:
                chunk_id = chunk.get("id")
                chunk_name = chunk.get("name")
                chunk_args = chunk.get("args", "")
                chunk_index = chunk.get("index", 0)

                # Initialize accumulator for this index if we have id and name
                if chunk_id and chunk_name:
                    if (
                        chunk_index not in tool_call_accumulator
                        or chunk_id != tool_call_accumulator[chunk_index]
                    ):
                        tool_call_accumulator[chunk_index] = {
                            "id": chunk_id,
                            "name": chunk_name,
                            "args_str": "",
                        }

                # Accumulate args string chunks using index
                if chunk_args and chunk_index in tool_call_accumulator:
                    tool_call_accumulator[chunk_index]["args_str"] += chunk_args

            # After processing chunks, try to start any tool calls with complete args
            for index, acc in tool_call_accumulator.items():
                tool_id = acc.get("id")
                tool_name = acc.get("name")
                args_str = acc.get("args_str", "")

                # Only start if we haven't started yet and have parseable args
                if tool_id and tool_id not in active_tool_calls and args_str:
                    try:
                        tool_args = json.loads(args_str)

                        # Mark as started and store args for later reference
                        active_tool_calls[tool_id] = {
                            "name": tool_name,
                            "args": tool_args,
                        }

                        # Create the appropriate tool call update
                        update = self._create_tool_call_update(tool_id, tool_name, tool_args)

                        await self._conn.session_update(
                            session_id=session_id,
                            update=update,
                            source="DeepAgent",
                        )

                        # If this is write_todos, send the plan update immediately
                        if tool_name == "write_todos" and isinstance(tool_args, dict):
                            todos = tool_args.get("todos", [])
                            await self._handle_todo_update(session_id, todos, log_plan=False)
                    except json.JSONDecodeError:
                        pass

    def _create_tool_call_update(
        self, tool_id: str, tool_name: str, tool_args: dict[str, Any]
    ) -> ToolCallStart:
        """Create a tool call update based on tool type and arguments."""
        kind_map: dict = {
            "read_file": "read",
            "edit_file": "edit",
            "write_file": "edit",
            "ls": "search",
            "glob": "search",
            "grep": "search",
        }
        tool_kind = kind_map.get(tool_name, "other")

        # Determine title and create appropriate update based on tool type
        if tool_name == "read_file" and isinstance(tool_args, dict):
            path = tool_args.get("file_path")
            title = f"Read `{path}`" if path else tool_name
            return start_tool_call(
                tool_call_id=tool_id,
                title=title,
                kind=tool_kind,
                status="pending",
            )
        elif tool_name == "edit_file" and isinstance(tool_args, dict):
            path = tool_args.get("file_path", "")
            old_string = tool_args.get("old_string", "")
            new_string = tool_args.get("new_string", "")
            title = f"Edit `{path}`" if path else tool_name

            # Only create diff if we have both old and new strings
            if path and old_string and new_string:
                diff_content = tool_diff_content(
                    path=path,
                    new_text=new_string,
                    old_text=old_string,
                )
                return start_edit_tool_call(
                    tool_call_id=tool_id,
                    title=title,
                    path=path,
                    content=diff_content,
                    # This is silly but for some reason content isn't passed through
                    extra_options=[diff_content],
                )
            else:
                # Fallback to generic tool call if data incomplete
                return start_tool_call(
                    tool_call_id=tool_id,
                    title=title,
                    kind=tool_kind,
                    status="pending",
                )
        elif tool_name == "write_file" and isinstance(tool_args, dict):
            path = tool_args.get("file_path")
            title = f"Write `{path}`" if path else tool_name
            return start_tool_call(
                tool_call_id=tool_id,
                title=title,
                kind=tool_kind,
                status="pending",
            )
        else:
            title = tool_name
            return start_tool_call(
                tool_call_id=tool_id,
                title=title,
                kind=tool_kind,
                status="pending",
            )

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        # Reset cancellation flag for new prompt
        self._cancelled = False

        # Convert ACP content blocks to LangChain multimodal content format
        content_blocks = []

        for block in prompt:
            if isinstance(block, TextContentBlock):
                content_blocks.extend(convert_text_block_to_content_blocks(block))
            elif isinstance(block, ImageContentBlock):
                content_blocks.extend(convert_image_block_to_content_blocks(block))
            elif isinstance(block, AudioContentBlock):
                content_blocks.extend(convert_audio_block_to_content_blocks(block))
            elif isinstance(block, ResourceContentBlock):
                content_blocks.extend(
                    convert_resource_block_to_content_blocks(block, root_dir=self._root_dir)
                )
            elif isinstance(block, EmbeddedResourceContentBlock):
                content_blocks.extend(convert_embedded_resource_block_to_content_blocks(block))
        # Stream the deep agent response with multimodal content
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        # Track active tool calls and accumulate chunks by index
        active_tool_calls = {}
        tool_call_accumulator = {}  # index -> {id, name, args_str}

        current_state = None
        user_decisions = []

        while current_state is None or current_state.interrupts:
            # Check for cancellation
            if self._cancelled:
                self._cancelled = False  # Reset for next prompt
                return PromptResponse(stop_reason="cancelled")

            async for message_chunk, metadata in self._deepagent.astream(
                Command(resume={"decisions": user_decisions})
                if user_decisions
                else {"messages": [{"role": "user", "content": content_blocks}]},
                config=config,
                stream_mode="messages",
            ):
                # Check for cancellation during streaming
                if self._cancelled:
                    self._cancelled = False  # Reset for next prompt
                    return PromptResponse(stop_reason="cancelled")

                # Process tool call chunks
                await self._process_tool_call_chunks(
                    session_id,
                    message_chunk,
                    active_tool_calls,
                    tool_call_accumulator,
                )

                if isinstance(message_chunk, str):
                    await self._log_text(text=message_chunk, session_id=session_id)
                # Check for tool results (ToolMessage responses)
                elif hasattr(message_chunk, "type") and message_chunk.type == "tool":
                    # This is a tool result message
                    tool_call_id = getattr(message_chunk, "tool_call_id", None)
                    if tool_call_id and tool_call_id in active_tool_calls:
                        if active_tool_calls[tool_call_id].get("name") != "edit_file":
                            # Update the tool call with completion status and result
                            content = getattr(message_chunk, "content", "")
                            update = update_tool_call(
                                tool_call_id=tool_call_id,
                                status="completed",
                                content=[tool_content(text_block(str(content)))],
                            )
                            await self._conn.session_update(
                                session_id=session_id, update=update, source="DeepAgent"
                            )

                elif message_chunk.content:
                    # content can be a string or a list of content blocks
                    if isinstance(message_chunk.content, str):
                        text = message_chunk.content
                    elif isinstance(message_chunk.content, list):
                        # Extract text from content blocks
                        text = ""
                        for block in message_chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text += block.get("text", "")
                            elif isinstance(block, str):
                                text += block
                    else:
                        text = str(message_chunk.content)

                    if text:
                        await self._log_text(text=text, session_id=session_id)

            # Check if the agent is interrupted (waiting for HITL approval)
            current_state = await self._deepagent.aget_state(config)
            user_decisions = await self._handle_interrupts(
                current_state=current_state,
                session_id=session_id,
                active_tool_calls=active_tool_calls,
            )

        return PromptResponse(stop_reason="end_turn")

    async def _handle_interrupts(
        self, *, current_state: StateSnapshot, session_id: str, active_tool_calls: dict
    ):
        user_decisions = []
        if current_state.next and current_state.interrupts:
            # Agent is interrupted, request permission from user
            for interrupt in current_state.interrupts:
                # Get the tool call info from the interrupt
                tool_call_id = interrupt.id
                interrupt_value = interrupt.value

                # Extract action requests from interrupt_value
                action_requests = []
                if isinstance(interrupt_value, dict):
                    # Deep Agents wraps tool calls in action_requests
                    action_requests = interrupt_value.get("action_requests", [])

                # Process each action request
                for action in action_requests:
                    tool_name = action.get("name", "tool")
                    tool_args = action.get("args", {})

                    # Check if this is write_todos - auto-approve updates to existing plan
                    if tool_name == "write_todos" and isinstance(tool_args, dict):
                        new_todos = tool_args.get("todos", [])

                        # Auto-approve if there's an existing plan that's not fully completed
                        if session_id in self._session_plans:
                            existing_plan = self._session_plans[session_id]
                            all_completed = self._all_tasks_completed(existing_plan)

                            if not all_completed:
                                # Plan is in progress, auto-approve updates
                                # Store the updated plan (status and content may have changed)
                                self._session_plans[session_id] = new_todos
                                user_decisions.append({"type": "approve"})
                                continue

                    # Create a title for the permission request
                    if tool_name == "write_todos":
                        title = "Review Plan"
                        # Log the plan text when requesting approval
                        todos = tool_args.get("todos", [])
                        plan_text = "## Plan\n\n"
                        for i, todo in enumerate(todos, 1):
                            content = todo.get("content", "")
                            plan_text += f"{i}. {content}\n"
                        await self._log_text(session_id=session_id, text=plan_text)
                    elif tool_name == "edit_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Edit `{file_path}`"
                    elif tool_name == "write_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Write `{file_path}`"
                    else:
                        title = tool_name

                    # Create permission options
                    options = [
                        PermissionOption(
                            option_id="approve",
                            name="Approve",
                            kind="allow_once",
                        ),
                        PermissionOption(
                            option_id="reject",
                            name="Reject",
                            kind="reject_once",
                        ),
                    ]

                    # Request permission from the client
                    tool_call_update = ToolCallUpdate(
                        tool_call_id=tool_call_id,
                        title=title,
                    )
                    response = await self._conn.request_permission(
                        session_id=session_id,
                        tool_call=tool_call_update,
                        options=options,
                    )
                    # Handle the user's decision
                    if response.outcome.outcome == "selected":
                        decision_type = response.outcome.option_id

                        # If rejecting a plan, clear it and provide feedback
                        if tool_name == "write_todos" and decision_type == "reject":
                            await self._clear_plan(session_id)
                            user_decisions.append(
                                {
                                    "type": decision_type,
                                    "feedback": (
                                        "The user rejected the plan. Please ask them for feedback "
                                        "on how the plan can be improved, then create a new "
                                        "and improved plan using this same write_todos tool."
                                    ),
                                }
                            )
                        elif tool_name == "write_todos" and decision_type == "approve":
                            # Store the approved plan for future comparisons
                            self._session_plans[session_id] = tool_args.get("todos", [])
                            user_decisions.append({"type": decision_type})
                        else:
                            user_decisions.append({"type": decision_type})
                    else:
                        # User cancelled, treat as rejection
                        user_decisions.append({"type": "reject"})

                        # If cancelling a plan, clear it
                        if tool_name == "write_todos":
                            await self._clear_plan(session_id)
            return user_decisions


async def serve_acp_stdio(root_dir: str) -> None:
    checkpointer = MemorySaver()

    # Start with ask_before_edits mode (ask before edits)
    mode_id = "ask_before_edits"

    acp_agent = ACPAgentServer(
        root_dir=root_dir,
        mode=mode_id,
        checkpointer=checkpointer,
    )
    await run_acp_agent(acp_agent)
