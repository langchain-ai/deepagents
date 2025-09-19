from langchain.agents.middleware import AgentState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import Annotated, Union
from deepagents.state import Todo
from deepagents.prompts import (
    WRITE_TODOS_TOOL_DESCRIPTION,
    LIST_FILES_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    EDIT_FILE_TOOL_DESCRIPTION,
)


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=LIST_FILES_TOOL_DESCRIPTION)
def ls(state: Annotated[dict, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())


@tool(description=READ_FILE_TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[AgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get file content
    content = mock_filesystem[file_path]

    # Handle empty file
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"

    # Split content into lines
    lines = content.splitlines()

    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]

        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]

        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")

    return "\n".join(result_lines)


@tool(description=WRITE_FILE_TOOL_DESCRIPTION)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_FILE_TOOL_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> Union[Command, str]:
    """Write to a file."""
    mock_filesystem = state.get("files", {})
    # Check if file exists in mock filesystem
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get current file content
    content = mock_filesystem[file_path]

    # Check if old_string exists in the file
    if old_string not in content:
        return f"Error: String not found in file: '{old_string}'"

    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"

    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
    else:
        new_content = content.replace(
            old_string, new_string, 1
        )  # Replace only first occurrence
        result_msg = f"Successfully replaced string in '{file_path}'"

    # Update the mock filesystem
    mock_filesystem[file_path] = new_content
    return Command(
        update={
            "files": mock_filesystem,
            "messages": [ToolMessage(result_msg, tool_call_id=tool_call_id)],
        }
    )
