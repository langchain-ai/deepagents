"""Middleware for providing Claude's native text_editor_20250728 tool to an agent."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol, EditResult, WriteResult
from deepagents.middleware.filesystem import (
    BACKEND_TYPES,
    FileData,
    _file_data_reducer,
    _validate_path,
)


class ClaudeTextEditorState(AgentState):
    """State for the Claude text editor middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


CLAUDE_TEXT_EDITOR_SYSTEM_PROMPT = """## Claude Native Text Editor Tool

You have access to Claude's native text editor tool `str_replace_based_edit_tool` for file operations.
All file paths must start with a /.

This tool supports four commands:

**view** - View file contents or directory listings
- Parameters: `command: "view"`, `path: str`, `view_range: [start_line, end_line]` (optional)
- Use view_range to read specific line ranges (1-indexed, use -1 for end of file)
- Examples:
  - View entire file: `{"command": "view", "path": "/file.py"}`
  - View lines 1-100: `{"command": "view", "path": "/file.py", "view_range": [1, 100]}`
  - View from line 50 to end: `{"command": "view", "path": "/file.py", "view_range": [50, -1]}`

**str_replace** - Replace text by finding exact string matches
- Parameters: `command: "str_replace"`, `path: str`, `old_str: str`, `new_str: str`
- Replaces ALL occurrences of old_str with new_str by default
- Example: `{"command": "str_replace", "path": "/file.py", "old_str": "old text", "new_str": "new text"}`

**create** - Create new files or overwrite existing ones
- Parameters: `command: "create"`, `path: str`, `file_text: str`
- Will overwrite existing files without warning
- Example: `{"command": "create", "path": "/new_file.py", "file_text": "print('hello')"}`

**insert** - Insert text at a specific line number
- Parameters: `command: "insert"`, `path: str`, `insert_line: int`, `new_str: str`
- Line numbers are 1-indexed
- Text is inserted BEFORE the specified line
- Example: `{"command": "insert", "path": "/file.py", "insert_line": 10, "new_str": "# New comment"}`"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """Get the resolved backend instance from backend or factory.

    Args:
        backend: Backend instance or factory function.
        runtime: The tool runtime context.

    Returns:
        Resolved backend instance.
    """
    if callable(backend):
        return backend(runtime)
    return backend


def _claude_text_editor_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
) -> BaseTool:
    """Generate Claude's native str_replace_based_edit_tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.

    Returns:
        Configured str_replace_based_edit_tool that implements Claude's text_editor_20250728 interface.
    """

    @tool(
        description="Claude's native text editor tool. Supports four commands: 'view' (read files/directories), "
        "'str_replace' (replace text), 'create' (create/overwrite files), and 'insert' (insert at line number)."
    )
    def str_replace_based_edit_tool(
        command: Literal["view", "str_replace", "create", "insert"],
        path: str,
        runtime: ToolRuntime[None, ClaudeTextEditorState],
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        file_text: str | None = None,
        insert_line: int | None = None,
    ) -> str | Command:
        """Claude's native text editor tool with command-based interface.

        Args:
            command: The command to execute (view, str_replace, create, insert).
            path: The file path (must be absolute, starting with /).
            runtime: The tool runtime context.
            view_range: Optional [start_line, end_line] for view command (1-indexed, -1 for end).
            old_str: The string to replace (required for str_replace command).
            new_str: The replacement string (required for str_replace and insert commands).
            file_text: The file content (required for create command).
            insert_line: The line number to insert at (required for insert command, 1-indexed).

        Returns:
            Result message or Command with state updates.
        """
        resolved_backend = _get_backend(backend, runtime)

        # For view command, preserve trailing slash for directory detection
        is_directory_request = command == "view" and path.endswith("/")
        validated_path = _validate_path(path)

        # Restore trailing slash if needed
        if is_directory_request and not validated_path.endswith("/"):
            validated_path = validated_path + "/"

        if command == "view":
            return _handle_view(resolved_backend, validated_path, view_range)
        if command == "str_replace":
            if old_str is None or new_str is None:
                return "Error: str_replace command requires both 'old_str' and 'new_str' parameters"
            return _handle_str_replace(resolved_backend, validated_path, old_str, new_str, runtime)
        if command == "create":
            if file_text is None:
                return "Error: create command requires 'file_text' parameter"
            return _handle_create(resolved_backend, validated_path, file_text, runtime)
        if command == "insert":
            if insert_line is None or new_str is None:
                return "Error: insert command requires both 'insert_line' and 'new_str' parameters"
            return _handle_insert(resolved_backend, validated_path, insert_line, new_str, runtime)
        return f"Error: Unknown command '{command}'. Supported commands: view, str_replace, create, insert"

    return str_replace_based_edit_tool


def _handle_view(backend: BackendProtocol, path: str, view_range: list[int] | None) -> str:
    """Handle the 'view' command.

    Args:
        backend: The backend to use.
        path: The validated file path.
        view_range: Optional [start_line, end_line] range (1-indexed, -1 for end).

    Returns:
        File contents or directory listing.
    """
    # Try to list as directory first (if path ends with /)
    if path.endswith("/"):
        try:
            infos = backend.ls_info(path)
            file_list = [fi.get("path", "") for fi in infos]
            return f"Directory listing for {path}:\n" + "\n".join(file_list)
        except Exception as e:
            return f"Error viewing directory {path}: {e}"

    # Try to read as a file
    try:
        if view_range is None:
            # Read entire file - use very large limit to effectively read all
            return backend.read(path, offset=0, limit=1000000)
        # Read with range (convert from 1-indexed to 0-indexed offset)
        start_line = view_range[0]
        end_line = view_range[1]

        if start_line < 1:
            return f"Error: start_line must be >= 1, got {start_line}"

        # Convert to 0-indexed offset
        offset = start_line - 1

        # Calculate limit (-1 means read to end)
        if end_line == -1:
            limit = 1000000  # Very large limit to read to end
        else:
            if end_line < start_line:
                return f"Error: end_line ({end_line}) must be >= start_line ({start_line})"
            limit = end_line - start_line + 1

        return backend.read(path, offset=offset, limit=limit)
    except Exception as e:
        return f"Error viewing {path}: {e}"


def _handle_str_replace(
    backend: BackendProtocol,
    path: str,
    old_str: str,
    new_str: str,
    runtime: ToolRuntime,
) -> str | Command:
    """Handle the 'str_replace' command.

    Args:
        backend: The backend to use.
        path: The validated file path.
        old_str: The string to replace.
        new_str: The replacement string.
        runtime: The tool runtime context.

    Returns:
        Success message or Command with state updates.
    """
    # Claude's str_replace replaces ALL occurrences by default
    res: EditResult = backend.edit(path, old_str, new_str, replace_all=True)

    if res.error:
        return res.error

    message = f"Successfully replaced {res.occurrences} occurrence(s) in '{res.path}'"

    if res.files_update is not None:
        from langchain_core.messages import ToolMessage

        return Command(
            update={
                "files": res.files_update,
                "messages": [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)],
            }
        )

    return message


def _handle_create(
    backend: BackendProtocol,
    path: str,
    file_text: str,
    runtime: ToolRuntime,
) -> str | Command:
    """Handle the 'create' command.

    Args:
        backend: The backend to use.
        path: The validated file path.
        file_text: The content to write.
        runtime: The tool runtime context.

    Returns:
        Success message or Command with state updates.
    """
    # Claude's create overwrites existing files without warning
    # We need to force the write even if file exists
    res: WriteResult = backend.write(path, file_text)

    if res.error:
        return res.error

    message = f"Created file '{res.path}'"

    if res.files_update is not None:
        from langchain_core.messages import ToolMessage

        return Command(
            update={
                "files": res.files_update,
                "messages": [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)],
            }
        )

    return message


def _handle_insert(
    backend: BackendProtocol,
    path: str,
    insert_line: int,
    new_str: str,
    runtime: ToolRuntime,
) -> str | Command:
    """Handle the 'insert' command.

    Args:
        backend: The backend to use.
        path: The validated file path.
        insert_line: The line number to insert at (1-indexed).
        new_str: The text to insert.
        runtime: The tool runtime context.

    Returns:
        Success message or Command with state updates.
    """
    if insert_line < 1:
        return f"Error: insert_line must be >= 1, got {insert_line}"

    # Access raw file data directly from state
    try:
        files = runtime.state.get("files", {})
        file_data = files.get(path)

        if file_data is None:
            return f"Error: File '{path}' not found"

        # Get raw content lines
        lines = file_data.get("content", [])

        # Insert new text at specified line (convert from 1-indexed to 0-indexed)
        insert_index = insert_line - 1

        if insert_index < 0:
            insert_index = 0
        elif insert_index > len(lines):
            insert_index = len(lines)

        # Make a copy and insert
        new_lines = lines.copy()
        new_lines.insert(insert_index, new_str)

        # Write back the modified content
        new_content = "\n".join(new_lines)
        res: WriteResult = backend.write(path, new_content)

        if res.error:
            # If write fails because file exists, we need to use edit
            # For insert, we'll read->modify->overwrite pattern
            # Let's try a different approach: delete and recreate
            from deepagents.backends.utils import create_file_data

            new_file_data = create_file_data(new_content)
            message = f"Inserted text at line {insert_line} in '{path}'"

            from langchain_core.messages import ToolMessage

            return Command(
                update={
                    "files": {path: new_file_data},
                    "messages": [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)],
                }
            )

        message = f"Inserted text at line {insert_line} in '{res.path}'"

        if res.files_update is not None:
            from langchain_core.messages import ToolMessage

            return Command(
                update={
                    "files": res.files_update,
                    "messages": [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)],
                }
            )

        return message

    except Exception as e:
        return f"Error inserting into {path}: {e}"


class ClaudeTextEditorMiddleware(AgentMiddleware):
    """Middleware for providing Claude's native text_editor_20250728 tool to an agent.

    This middleware adds a single tool `str_replace_based_edit_tool` that implements
    Claude's native text editor interface with four commands: view, str_replace, create,
    and insert. Files are stored using any backend that implements the BackendProtocol.

    This is a pure implementation of Claude's native tool without additional search
    capabilities (no glob/grep). For extended search functionality, use FilesystemMiddleware instead.

    Args:
        backend: Backend for file storage. If not provided, defaults to StateBackend
            (ephemeral storage in agent state). For persistent storage, use StoreBackend
            or CompositeBackend with custom routes.
        system_prompt: Optional custom system prompt override.

    Example:
        ```python
        from deepagents.middleware.claude_text_editor import ClaudeTextEditorMiddleware
        from deepagents import create_deep_agent

        # Create agent with Claude's native text editor
        agent = create_deep_agent(
            use_claude_native_text_editor=True
        )
        ```
    """

    state_schema = ClaudeTextEditorState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the Claude text editor middleware.

        Args:
            backend: Backend for file storage, or a factory callable. Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
        """
        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Set system prompt (allow full override)
        self.system_prompt = system_prompt if system_prompt is not None else CLAUDE_TEXT_EDITOR_SYSTEM_PROMPT

        # Generate the single Claude text editor tool
        self.tools = [_claude_text_editor_tool_generator(self.backend)]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt to include instructions on using Claude's text editor.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt to include instructions on using Claude's text editor.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return await handler(request)
