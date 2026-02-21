"""ACP server implementation for Deep Agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
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
from acp.exceptions import RequestError
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
    SessionModeState,
    SseMcpServer,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
    ToolKind,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from acp.interfaces import Client
    from deepagents.graph import Checkpointer
    from langchain.tools import ToolRuntime
    from langchain_core.runnables import RunnableConfig

from deepagents_acp.utils import (
    convert_audio_block_to_content_blocks,
    convert_embedded_resource_block_to_content_blocks,
    convert_image_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
    extract_command_types,
    format_execute_result,
    truncate_execute_command_for_display,
)


@dataclass(frozen=True, slots=True)
class AgentSessionContext:
    """Context for an agent session, including working directory and mode."""

    cwd: str
    mode: str


class AgentServerACP(ACPAgent):
    """ACP agent server that bridges Deep Agents with the Agent Client Protocol."""

    _conn: Client

    def __init__(
        self,
        agent: CompiledStateGraph | Callable[[AgentSessionContext], CompiledStateGraph],
        *,
        modes: SessionModeState | None = None,
    ) -> None:
        """Initialize the ACP agent server with the given agent factory or compiled graph."""
        super().__init__()
        self._cwd = ""
        self._agent_factory = agent
        self._agent: CompiledStateGraph | None = None

        if isinstance(agent, CompiledStateGraph):
            if modes is not None:
                msg = "modes can only be provided when agent is a factory"
                raise ValueError(msg)
            self._modes: SessionModeState | None = None
        else:
            self._modes = modes

        self._session_modes: dict[str, str] = {}
        self._session_mode_states: dict[str, SessionModeState] = {}
        self._cancelled = False
        self._session_plans: dict[str, list[dict[str, Any]]] = {}
        self._session_cwds: dict[str, str] = {}
        self._allowed_command_types: dict[
            str, set[tuple[str, str | None]]
        ] = {}  # Track allowed command types per session

    def on_connect(self, conn: Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,  # noqa: ARG002  # ACP protocol interface parameter
        client_info: Implementation | None = None,  # noqa: ARG002  # ACP protocol interface parameter
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> InitializeResponse:
        """Return server capabilities to the ACP client."""
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(
                    image=True,
                )
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> NewSessionResponse:
        """Create a new agent session with the given working directory."""
        if mcp_servers is None:
            mcp_servers = []
        session_id = uuid4().hex
        self._session_cwds[session_id] = cwd

        if self._modes is not None:
            self._session_modes[session_id] = self._modes.current_mode_id
            self._session_mode_states[session_id] = self._modes
            return NewSessionResponse(session_id=session_id, modes=self._modes)

        if not isinstance(self._agent_factory, CompiledStateGraph):
            return NewSessionResponse(session_id=session_id)

        return NewSessionResponse(session_id=session_id)

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> SetSessionModeResponse:
        """Switch the session to a different mode, resetting the agent."""
        if self._modes is not None and session_id in self._session_mode_states:
            state = self._session_mode_states[session_id]
            self._session_modes[session_id] = mode_id
            self._session_mode_states[session_id] = SessionModeState(
                available_modes=state.available_modes,
                current_mode_id=mode_id,
            )
            self._reset_agent(session_id)
        return SetSessionModeResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # ACP protocol interface parameters
        """Cancel the current execution."""
        self._cancelled = True

    async def _log_text(self, session_id: str, text: str) -> None:
        """Send a text message update to the client."""
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
        *,
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
                status=status,
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
                is_new_tool_call = (
                    chunk_index not in tool_call_accumulator
                    or chunk_id != tool_call_accumulator[chunk_index].get("id")
                )
                if chunk_id and chunk_name and is_new_tool_call:
                    tool_call_accumulator[chunk_index] = {
                        "id": chunk_id,
                        "name": chunk_name,
                        "args_str": "",
                    }

                # Accumulate args string chunks using index
                if chunk_args and chunk_index in tool_call_accumulator:
                    tool_call_accumulator[chunk_index]["args_str"] += chunk_args

            # After processing chunks, try to start any tool calls with complete args
            for _index, acc in list(tool_call_accumulator.items()):
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

                        # Create the appropriate tool call start
                        update = self._create_tool_call_start(tool_id, tool_name, tool_args)

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

    def _create_tool_call_start(
        self, tool_id: str, tool_name: str, tool_args: dict[str, Any]
    ) -> ToolCallStart:
        """Create a tool call update based on tool type and arguments."""
        kind_map: dict[str, ToolKind] = {
            "read_file": "read",
            "edit_file": "edit",
            "write_file": "edit",
            "ls": "search",
            "glob": "search",
            "grep": "search",
            "execute": "execute",
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
                raw_input=tool_args,
            )
        if tool_name == "edit_file" and isinstance(tool_args, dict):
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
            # Fallback to generic tool call if data incomplete
            return start_tool_call(
                tool_call_id=tool_id,
                title=title,
                kind=tool_kind,
                status="pending",
                raw_input=tool_args,
            )
        if tool_name == "write_file" and isinstance(tool_args, dict):
            path = tool_args.get("file_path")
            title = f"Write `{path}`" if path else tool_name
            return start_tool_call(
                tool_call_id=tool_id,
                title=title,
                kind=tool_kind,
                status="pending",
                raw_input=tool_args,
            )
        if tool_name == "execute" and isinstance(tool_args, dict):
            command = tool_args.get("command", "")
            return start_tool_call(
                tool_call_id=tool_id,
                title=command or "Execute command",
                kind=tool_kind,
                status="pending",
                raw_input=tool_args,
            )
        title = tool_name
        return start_tool_call(
            tool_call_id=tool_id,
            title=title,
            kind=tool_kind,
            status="pending",
            raw_input=tool_args,
        )

    def _reset_agent(self, session_id: str) -> None:
        """Reset the agent instance, re-creating it from the factory if applicable."""
        if isinstance(self._agent_factory, CompiledStateGraph):
            self._agent = self._agent_factory
        else:
            mode = self._session_modes.get(
                session_id,
                self._modes.current_mode_id if self._modes is not None else "auto",
            )
            context = AgentSessionContext(cwd=self._cwd, mode=mode)
            self._agent = self._agent_factory(context)

    async def prompt(  # noqa: C901, PLR0912, PLR0915  # Complex streaming protocol handler with many branches
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> PromptResponse:
        """Process a user prompt and stream the agent response."""
        if self._agent is None:
            cwd = self._session_cwds.get(session_id)
            if cwd is not None:
                self._cwd = cwd
            self._reset_agent(session_id)

            if getattr(self._agent, "checkpointer", None) is None:
                self._agent.checkpointer = MemorySaver()  # ty: ignore[unresolved-attribute]  # Guarded by getattr check above

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
                    convert_resource_block_to_content_blocks(block, root_dir=self._cwd)
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

            async for stream_chunk in self._agent.astream(  # ty: ignore[possibly-missing-attribute]  # _agent is set in new_session before prompt is called
                Command(resume={"decisions": user_decisions})
                if user_decisions
                else {"messages": [{"role": "user", "content": content_blocks}]},
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                _expected_len = 3  # (namespace, stream_mode, data)
                if not isinstance(stream_chunk, tuple) or len(stream_chunk) != _expected_len:
                    continue

                _namespace, stream_mode, data = stream_chunk
                # Check for cancellation during streaming
                if self._cancelled:
                    self._cancelled = False  # Reset for next prompt
                    return PromptResponse(stop_reason="cancelled")

                if stream_mode == "updates":
                    updates = data
                    if isinstance(updates, dict) and "__interrupt__" in updates:
                        interrupt_objs = updates.get("__interrupt__")
                        if interrupt_objs:
                            for interrupt_obj in interrupt_objs:
                                interrupt_value = interrupt_obj.value
                                if not isinstance(interrupt_value, dict):
                                    raise RequestError(
                                        -32600,
                                        (
                                            "ACP limitation: this agent raised a free-form "
                                            "LangGraph interrupt(), which ACP cannot display.\n\n"
                                            "ACP only supports human-in-the-loop permission "
                                            "prompts with a fixed set of decisions "
                                            "(approve/reject/edit).\n"
                                            "Spec: https://agentclientprotocol.com/protocol/overview\n\n"
                                            "Fix: use LangChain HumanInTheLoopMiddleware-style "
                                            "interrupts (action_requests/review_configs).\n"
                                            "Docs: https://docs.langchain.com/oss/python/langchain/"
                                            "human-in-the-loop\n\n"
                                            "This is a protocol limitation, not a bug in the agent."
                                        ),
                                        {"interrupt_value": interrupt_value},
                                    )

                            current_state = await self._agent.aget_state(config)  # ty: ignore[possibly-missing-attribute]  # _agent is set in new_session before prompt is called
                            user_decisions = await self._handle_interrupts(
                                current_state=current_state,
                                session_id=session_id,
                            )
                            break

                    for node_name, update in updates.items():
                        if node_name == "tools" and isinstance(update, dict) and "todos" in update:
                            todos = update.get("todos", [])
                            if todos:
                                await self._handle_todo_update(session_id, todos, log_plan=False)

                    continue

                message_chunk, _metadata = data

                # Process tool call chunks
                await self._process_tool_call_chunks(
                    session_id,
                    message_chunk,
                    active_tool_calls,
                    tool_call_accumulator,
                )

                if isinstance(message_chunk, str):
                    if not _namespace:
                        await self._log_text(text=message_chunk, session_id=session_id)
                # Check for tool results (ToolMessage responses)
                elif hasattr(message_chunk, "type") and message_chunk.type == "tool":
                    # This is a tool result message
                    tool_call_id = getattr(message_chunk, "tool_call_id", None)
                    if (
                        tool_call_id
                        and tool_call_id in active_tool_calls
                        and active_tool_calls[tool_call_id].get("name") != "edit_file"
                    ):
                        # Update the tool call with completion status and result
                        content = getattr(message_chunk, "content", "")
                        tool_info = active_tool_calls[tool_call_id]
                        tool_name = tool_info.get("name")

                        # Format execute tool results specially
                        if tool_name == "execute":
                            tool_args = tool_info.get("args", {})
                            command = tool_args.get("command", "")
                            formatted_content = format_execute_result(
                                command=command, result=str(content)
                            )
                        else:
                            formatted_content = str(content)
                        update = update_tool_call(
                            tool_call_id=tool_call_id,
                            status="completed",
                            content=[tool_content(text_block(formatted_content))],
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

                    if text and not _namespace:
                        await self._log_text(text=text, session_id=session_id)

            # After streaming completes, check if we need to exit the loop
            # The loop continues while there are interrupts (line 467)
            # We get the current state to check the loop condition
            current_state = await self._agent.aget_state(config)  # ty: ignore[possibly-missing-attribute]  # _agent is set in new_session before prompt is called
            # Note: Interrupts are handled during streaming via __interrupt__ updates
            # This state check is only for the while loop condition

        return PromptResponse(stop_reason="end_turn")

    async def _handle_interrupts(  # noqa: C901, PLR0912, PLR0915  # Complex HITL permission handling with many branches
        self,
        *,
        current_state: StateSnapshot,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Handle agent interrupts by requesting permission from the client."""
        user_decisions: list[dict[str, Any]] = []
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

                    if session_id in self._allowed_command_types:
                        if tool_name == "execute" and isinstance(tool_args, dict):
                            command = tool_args.get("command", "")
                            command_types = extract_command_types(command)

                            if command_types:
                                # Check if ALL command types are already allowed for this session
                                all_allowed = all(
                                    ("execute", cmd_type) in self._allowed_command_types[session_id]
                                    for cmd_type in command_types
                                )
                                if all_allowed:
                                    # Auto-approve this command
                                    user_decisions.append({"type": "approve"})
                                    continue
                        elif (tool_name, None) in self._allowed_command_types[session_id]:
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
                    elif tool_name == "execute" and isinstance(tool_args, dict):
                        command = tool_args.get("command", "")
                        # Truncate long commands for display
                        display_command = truncate_execute_command_for_display(command=command)
                        title = f"Execute: `{display_command}`" if command else "Execute command"
                    else:
                        title = tool_name

                    desc = tool_name
                    if tool_name == "execute" and isinstance(tool_args, dict):
                        command = tool_args.get("command", "")
                        command_types = extract_command_types(command)
                        if command_types:
                            # Create a descriptive name based on the command types
                            if len(command_types) == 1:
                                desc = f"`{command_types[0]}`"
                            else:
                                # Show all unique command types
                                unique_types = list(
                                    dict.fromkeys(command_types)
                                )  # Preserve order, remove duplicates
                                desc = ", ".join(f"`{ct}`" for ct in unique_types)

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
                        PermissionOption(
                            option_id="approve_always",
                            name=f"Always allow {desc} commands",
                            kind="allow_always",
                        ),
                    ]

                    # Request permission from the client
                    tool_call_update = ToolCallUpdate(
                        tool_call_id=tool_call_id, title=title, raw_input=tool_args
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
                        if decision_type == "approve_always":
                            if session_id not in self._allowed_command_types:
                                self._allowed_command_types[session_id] = set()
                            if tool_name == "execute":
                                command = tool_args.get("command", "")
                                command_types = extract_command_types(command)
                                if command_types:
                                    for cmd_type in command_types:
                                        self._allowed_command_types[session_id].add(
                                            ("execute", cmd_type)
                                        )
                            else:
                                self._allowed_command_types[session_id].add((tool_name, None))
                            # Approve this command
                            user_decisions.append({"type": "approve"})
                        elif tool_name == "write_todos" and decision_type == "reject":
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


async def _serve_test_agent() -> None:
    """Run test agent from the root of the repository with ACP integration."""
    from dotenv import load_dotenv  # noqa: PLC0415  # Lazy import for dev-only entry point

    load_dotenv()

    checkpointer: Checkpointer = MemorySaver()

    def build_agent(context: AgentSessionContext) -> CompiledStateGraph:
        """Agent factory based in the given root directory."""
        agent_root_dir = context.cwd

        def create_backend(run_time: ToolRuntime) -> CompositeBackend:
            ephemeral_backend = StateBackend(run_time)
            return CompositeBackend(
                default=FilesystemBackend(root_dir=agent_root_dir, virtual_mode=True),
                routes={
                    "/memories/": ephemeral_backend,
                    "/conversation_history/": ephemeral_backend,
                },
            )

        return create_deep_agent(
            model="openai:gpt-5.2",
            checkpointer=checkpointer,
            backend=create_backend,
        )

    acp_agent = AgentServerACP(agent=build_agent)
    await run_acp_agent(acp_agent)
