"""Textual UI adapter for agent execution."""
# ruff: noqa: PLR0912, PLR0915, ANN401, PLR2004, BLE001, TRY203
# This module has complex streaming logic ported from execution.py

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    HITLResponse,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError

from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.image_utils import create_multimodal_content
from deepagents_cli.input import ImageTracker, parse_file_mentions
from deepagents_cli.ui import format_tool_display, format_tool_message_content
from deepagents_cli.widgets.messages import (
    AssistantMessage,
    DiffMessage,
    ErrorMessage,
    SystemMessage,
    ToolCallMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)


class TextualUIAdapter:
    """Adapter for rendering agent output to Textual widgets.

    This adapter provides an abstraction layer between the agent execution
    and the Textual UI, allowing streaming output to be rendered as widgets.
    """

    def __init__(
        self,
        mount_message: Callable,
        update_status: Callable[[str], None],
        request_approval: Callable,  # async callable returning Future
        on_auto_approve_enabled: Callable[[], None] | None = None,
        scroll_to_bottom: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            mount_message: Async callable to mount a message widget
            update_status: Callable to update the status bar message
            request_approval: Callable that returns a Future for HITL approval
            on_auto_approve_enabled: Callback when auto-approve is enabled
            scroll_to_bottom: Callback to scroll chat to bottom
        """
        self._mount_message = mount_message
        self._update_status = update_status
        self._request_approval = request_approval
        self._on_auto_approve_enabled = on_auto_approve_enabled
        self._scroll_to_bottom = scroll_to_bottom

        # State tracking
        self._current_assistant_message: AssistantMessage | None = None
        self._current_tool_messages: dict[str, ToolCallMessage] = {}
        self._pending_text = ""
        self._token_tracker: Any = None

    def set_token_tracker(self, tracker: Any) -> None:
        """Set the token tracker for usage tracking."""
        self._token_tracker = tracker


async def execute_task_textual(
    user_input: str,
    agent: Any,
    assistant_id: str | None,
    session_state: Any,
    adapter: TextualUIAdapter,
    backend: Any = None,
    image_tracker: ImageTracker | None = None,
) -> None:
    """Execute a task with output directed to Textual UI.

    This is the Textual-compatible version of execute_task() that uses
    the TextualUIAdapter for all UI operations.

    Args:
        user_input: The user's input message
        agent: The LangGraph agent to execute
        assistant_id: The agent identifier
        session_state: Session state with auto_approve flag
        adapter: The TextualUIAdapter for UI operations
        backend: Optional backend for file operations
        image_tracker: Optional tracker for images
    """
    # Parse file mentions and inject content if any
    prompt_text, mentioned_files = parse_file_mentions(user_input)

    # Max file size to embed inline (256KB, matching mistral-vibe)
    # Larger files get a reference instead - use read_file tool to view them
    max_embed_bytes = 256 * 1024

    if mentioned_files:
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for file_path in mentioned_files:
            try:
                file_size = file_path.stat().st_size
                if file_size > max_embed_bytes:
                    # File too large - include reference instead of content
                    size_kb = file_size // 1024
                    context_parts.append(
                        f"\n### {file_path.name}\n"
                        f"Path: `{file_path}`\n"
                        f"Size: {size_kb}KB (too large to embed, use read_file tool to view)"
                    )
                else:
                    content = file_path.read_text()
                    context_parts.append(
                        f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
                    )
            except Exception as e:
                context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")
        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    # Include images in the message content
    images_to_send = []
    if image_tracker:
        images_to_send = image_tracker.get_images()
    if images_to_send:
        message_content = create_multimodal_content(final_input, images_to_send)
    else:
        message_content = final_input

    thread_id = session_state.thread_id
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "assistant_id": assistant_id,
            "agent_name": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if assistant_id
        else {},
    }

    captured_input_tokens = 0
    captured_output_tokens = 0

    # Update status to show thinking
    adapter._update_status("Agent is thinking...")

    # Hide token display during streaming (will be shown with accurate count at end)
    if adapter._token_tracker:
        adapter._token_tracker.hide()

    file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=backend)
    displayed_tool_ids: set[str] = set()
    tool_call_buffers: dict[str | int, dict] = {}

    # Track pending text and assistant messages PER NAMESPACE to avoid interleaving
    # when multiple subagents stream in parallel
    pending_text_by_namespace: dict[tuple, str] = {}
    assistant_message_by_namespace: dict[tuple, Any] = {}

    # Clear images from tracker after creating the message
    if image_tracker:
        image_tracker.clear()

    stream_input: dict | Command = {"messages": [{"role": "user", "content": message_content}]}

    try:
        while True:
            interrupt_occurred = False
            hitl_response: dict[str, HITLResponse] = {}
            suppress_resumed_output = False
            pending_interrupts: dict[str, HITLRequest] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue

                namespace, current_stream_mode, data = chunk

                # Convert namespace to hashable tuple for dict keys
                ns_key = tuple(namespace) if namespace else ()

                # Filter out subagent outputs - only show main agent (empty namespace)
                # Subagents run via Task tool and should only report back to the main agent
                is_main_agent = ns_key == ()

                # Handle UPDATES stream - for interrupts and todos
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for interrupts
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                try:
                                    validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                                        interrupt_obj.value
                                    )
                                    pending_interrupts[interrupt_obj.id] = validated_request
                                    interrupt_occurred = True
                                except ValidationError:
                                    raise

                    # Check for todo updates (not yet implemented in Textual UI)
                    chunk_data = next(iter(data.values())) if data else None
                    if chunk_data and isinstance(chunk_data, dict) and "todos" in chunk_data:
                        pass  # Future: render todo list widget

                # Handle MESSAGES stream - for content and tool calls
                elif current_stream_mode == "messages":
                    # Skip subagent outputs - only render main agent content in chat
                    if not is_main_agent:
                        continue

                    if not isinstance(data, tuple) or len(data) != 2:
                        continue

                    message, _metadata = data

                    if isinstance(message, HumanMessage):
                        content = message.text
                        # Flush pending text for this namespace
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if content and pending_text:
                            await _flush_assistant_text_ns(
                                adapter, pending_text, ns_key, assistant_message_by_namespace
                            )
                            pending_text_by_namespace[ns_key] = ""
                        continue

                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(message.content)
                        record = file_op_tracker.complete_with_message(message)

                        adapter._update_status("Agent is thinking...")

                        # Update tool call status with output
                        tool_id = getattr(message, "tool_call_id", None)
                        if tool_id and tool_id in adapter._current_tool_messages:
                            tool_msg = adapter._current_tool_messages[tool_id]
                            output_str = str(tool_content) if tool_content else ""
                            if tool_status == "success":
                                tool_msg.set_success(output_str)
                            else:
                                tool_msg.set_error(output_str or "Error")
                            # Clean up - remove from tracking dict after status update
                            del adapter._current_tool_messages[tool_id]

                        # Show shell errors
                        if tool_name == "shell" and tool_status != "success":
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter, pending_text, ns_key, assistant_message_by_namespace
                                )
                                pending_text_by_namespace[ns_key] = ""
                            if tool_content:
                                await adapter._mount_message(ErrorMessage(str(tool_content)))

                        # Show file operation results - always show diffs in chat
                        if record:
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter, pending_text, ns_key, assistant_message_by_namespace
                                )
                                pending_text_by_namespace[ns_key] = ""
                            if record.diff:
                                await adapter._mount_message(
                                    DiffMessage(record.diff, record.display_path)
                                )
                        continue

                    # Extract token usage (before content_blocks check - usage may be on any chunk)
                    if adapter._token_tracker and hasattr(message, "usage_metadata"):
                        usage = message.usage_metadata
                        if usage:
                            # Use total_tokens which includes input + output
                            total_toks = usage.get("total_tokens", 0)
                            if total_toks:
                                captured_input_tokens = max(captured_input_tokens, total_toks)
                            else:
                                # Fallback to input + output if total not provided
                                input_toks = usage.get("input_tokens", 0)
                                output_toks = usage.get("output_tokens", 0)
                                if input_toks or output_toks:
                                    total = input_toks + output_toks
                                    captured_input_tokens = max(captured_input_tokens, total)

                    # Check if this is an AIMessageChunk with content
                    if not hasattr(message, "content_blocks"):
                        continue

                    # Process content blocks
                    for block in message.content_blocks:
                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                # Track accumulated text for reference
                                pending_text = pending_text_by_namespace.get(ns_key, "")
                                pending_text += text
                                pending_text_by_namespace[ns_key] = pending_text

                                # Get or create assistant message for this namespace
                                current_msg = assistant_message_by_namespace.get(ns_key)
                                if current_msg is None:
                                    current_msg = AssistantMessage()
                                    await adapter._mount_message(current_msg)
                                    assistant_message_by_namespace[ns_key] = current_msg
                                    # Anchor scroll once when message is created
                                    # anchor() keeps scroll locked to bottom as content grows
                                    if adapter._scroll_to_bottom:
                                        adapter._scroll_to_bottom()

                                # Append just the new text chunk for smoother streaming
                                # (uses MarkdownStream internally for better performance)
                                await current_msg.append_content(text)

                        elif block_type in ("tool_call_chunk", "tool_call"):
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            buffer_key: str | int
                            if chunk_index is not None:
                                buffer_key = chunk_index
                            elif chunk_id is not None:
                                buffer_key = chunk_id
                            else:
                                buffer_key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                buffer_key,
                                {"name": None, "id": None, "args": None, "args_parts": []},
                            )

                            if chunk_name:
                                buffer["name"] = chunk_name
                            if chunk_id:
                                buffer["id"] = chunk_id

                            if isinstance(chunk_args, dict):
                                buffer["args"] = chunk_args
                                buffer["args_parts"] = []
                            elif isinstance(chunk_args, str):
                                if chunk_args:
                                    parts: list[str] = buffer.setdefault("args_parts", [])
                                    if not parts or chunk_args != parts[-1]:
                                        parts.append(chunk_args)
                                    buffer["args"] = "".join(parts)
                            elif chunk_args is not None:
                                buffer["args"] = chunk_args

                            buffer_name = buffer.get("name")
                            buffer_id = buffer.get("id")
                            if buffer_name is None:
                                continue

                            parsed_args = buffer.get("args")
                            if isinstance(parsed_args, str):
                                if not parsed_args:
                                    continue
                                try:
                                    parsed_args = json.loads(parsed_args)
                                except json.JSONDecodeError:
                                    continue
                            elif parsed_args is None:
                                continue

                            if not isinstance(parsed_args, dict):
                                parsed_args = {"value": parsed_args}

                            # Flush pending text before tool call
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter, pending_text, ns_key, assistant_message_by_namespace
                                )
                                pending_text_by_namespace[ns_key] = ""
                                assistant_message_by_namespace.pop(ns_key, None)

                            if buffer_id is not None and buffer_id not in displayed_tool_ids:
                                displayed_tool_ids.add(buffer_id)
                                file_op_tracker.start_operation(buffer_name, parsed_args, buffer_id)

                                # Mount tool call message
                                tool_msg = ToolCallMessage(buffer_name, parsed_args)
                                await adapter._mount_message(tool_msg)
                                adapter._current_tool_messages[buffer_id] = tool_msg

                            tool_call_buffers.pop(buffer_key, None)
                            display_str = format_tool_display(buffer_name, parsed_args)
                            adapter._update_status(f"Executing {display_str}...")

                    if getattr(message, "chunk_position", None) == "last":
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if pending_text:
                            await _flush_assistant_text_ns(
                                adapter, pending_text, ns_key, assistant_message_by_namespace
                            )
                            pending_text_by_namespace[ns_key] = ""
                            assistant_message_by_namespace.pop(ns_key, None)

            # Flush any remaining text from all namespaces
            for ns_key, pending_text in list(pending_text_by_namespace.items()):
                if pending_text:
                    await _flush_assistant_text_ns(
                        adapter, pending_text, ns_key, assistant_message_by_namespace
                    )
            pending_text_by_namespace.clear()
            assistant_message_by_namespace.clear()

            # Handle HITL after stream completes
            if interrupt_occurred:
                any_rejected = False

                for interrupt_id, hitl_request in pending_interrupts.items():
                    if session_state.auto_approve:
                        # Auto-approve silently (user sees tool calls already)
                        decisions = [{"type": "approve"} for _ in hitl_request["action_requests"]]
                        hitl_response[interrupt_id] = {"decisions": decisions}
                    else:
                        # Request approval via adapter
                        decisions = []

                        def mark_hitl_approved(action_request: ActionRequest) -> None:
                            tool_name = action_request.get("name")
                            if tool_name not in {"write_file", "edit_file"}:
                                return
                            args = action_request.get("args", {})
                            if isinstance(args, dict):
                                file_op_tracker.mark_hitl_approved(tool_name, args)

                        for action_request in hitl_request["action_requests"]:
                            future = await adapter._request_approval(action_request, assistant_id)
                            decision = await future

                            # Check for auto-approve-all
                            if (
                                isinstance(decision, dict)
                                and decision.get("type") == "auto_approve_all"
                            ):
                                session_state.auto_approve = True
                                if adapter._on_auto_approve_enabled:
                                    adapter._on_auto_approve_enabled()
                                decisions.append({"type": "approve"})
                                mark_hitl_approved(action_request)
                                # Approve remaining actions
                                for _ in hitl_request["action_requests"][len(decisions) :]:
                                    decisions.append({"type": "approve"})
                                break

                            decisions.append(decision)
                            # Try multiple keys for tool call id
                            tool_id = (
                                action_request.get("id")
                                or action_request.get("tool_call_id")
                                or action_request.get("call_id")
                            )
                            tool_name = action_request.get("name", "")

                            # Find matching tool message - by id or by name as fallback
                            tool_msg = None
                            tool_msg_key = None  # Track key for cleanup
                            if tool_id and tool_id in adapter._current_tool_messages:
                                tool_msg = adapter._current_tool_messages[tool_id]
                                tool_msg_key = tool_id
                            elif tool_name:
                                # Fallback: find last tool message with matching name
                                for key, msg in reversed(
                                    list(adapter._current_tool_messages.items())
                                ):
                                    if msg._tool_name == tool_name:
                                        tool_msg = msg
                                        tool_msg_key = key
                                        break

                            if isinstance(decision, dict) and decision.get("type") == "approve":
                                mark_hitl_approved(action_request)
                                # Don't call set_success here - wait for actual tool output
                                # The ToolMessage handler will update with real results
                            elif isinstance(decision, dict) and decision.get("type") == "reject":
                                if tool_msg:
                                    tool_msg.set_rejected()
                                # Only remove from tracking on reject (approved tools need output update)
                                if tool_msg_key and tool_msg_key in adapter._current_tool_messages:
                                    del adapter._current_tool_messages[tool_msg_key]

                        if any(d.get("type") == "reject" for d in decisions):
                            any_rejected = True

                        hitl_response[interrupt_id] = {"decisions": decisions}

                suppress_resumed_output = any_rejected

            if interrupt_occurred and hitl_response:
                if suppress_resumed_output:
                    await adapter._mount_message(
                        SystemMessage("Command rejected. Tell the agent what you'd like instead.")
                    )
                    return

                stream_input = Command(resume=hitl_response)
            else:
                break

    except asyncio.CancelledError:
        adapter._update_status("Interrupted")

        # Mark any pending tools as rejected
        for tool_msg in list(adapter._current_tool_messages.values()):
            tool_msg.set_rejected()
        adapter._current_tool_messages.clear()

        await adapter._mount_message(SystemMessage("Interrupted by user"))

        # Append cancellation message to agent state so LLM knows what happened
        # This preserves context rather than rolling back
        try:
            cancellation_msg = HumanMessage(
                content="[SYSTEM] Task interrupted by user. Previous operation was cancelled."
            )
            await agent.aupdate_state(config, {"messages": [cancellation_msg]})
        except Exception:  # noqa: S110
            pass  # State update is best-effort
        # Report tokens even on interrupt (or restore display if none captured)
        if adapter._token_tracker:
            if captured_input_tokens or captured_output_tokens:
                adapter._token_tracker.add(captured_input_tokens, captured_output_tokens)
            else:
                adapter._token_tracker.show()  # Restore previous value
        return

    except KeyboardInterrupt:
        adapter._update_status("Interrupted")

        # Mark any pending tools as rejected
        for tool_msg in list(adapter._current_tool_messages.values()):
            tool_msg.set_rejected()
        adapter._current_tool_messages.clear()

        await adapter._mount_message(SystemMessage("Interrupted by user"))

        # Append cancellation message to agent state
        try:
            cancellation_msg = HumanMessage(
                content="[SYSTEM] Task interrupted by user. Previous operation was cancelled."
            )
            await agent.aupdate_state(config, {"messages": [cancellation_msg]})
        except Exception:  # noqa: S110
            pass  # State update is best-effort
        # Report tokens even on interrupt (or restore display if none captured)
        if adapter._token_tracker:
            if captured_input_tokens or captured_output_tokens:
                adapter._token_tracker.add(captured_input_tokens, captured_output_tokens)
            else:
                adapter._token_tracker.show()  # Restore previous value
        return

    adapter._update_status("Ready")

    # Update token tracker
    if adapter._token_tracker and (captured_input_tokens or captured_output_tokens):
        adapter._token_tracker.add(captured_input_tokens, captured_output_tokens)


async def _flush_assistant_text_ns(
    adapter: TextualUIAdapter,
    text: str,
    ns_key: tuple,
    assistant_message_by_namespace: dict[tuple, Any],
) -> None:
    """Flush accumulated assistant text for a specific namespace.

    Finalizes the streaming by stopping the MarkdownStream.
    If no message exists yet, creates one with the full content.
    """
    if not text.strip():
        return

    current_msg = assistant_message_by_namespace.get(ns_key)
    if current_msg is None:
        # No message was created during streaming - create one with full content
        current_msg = AssistantMessage(text)
        await adapter._mount_message(current_msg)
        await current_msg.write_initial_content()
        assistant_message_by_namespace[ns_key] = current_msg
    else:
        # Stop the stream to finalize the content
        await current_msg.stop_stream()
