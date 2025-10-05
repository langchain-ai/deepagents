import os
import logging
from pathlib import Path
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.tools.tool_node import InjectedState
from typing import Annotated, Union
from deepagents.state import Todo, FilesystemState
from deepagents.prompts import (
    WRITE_TODOS_TOOL_DESCRIPTION,
    LIST_FILES_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    EDIT_FILE_TOOL_DESCRIPTION,
)

logger = logging.getLogger(__name__)

try:
    file_system_path = os.environ["FILE_SYSTEM_PATH"]
except Exception as e:
    logger.error(f"FILE_SYSTEM_PATH not set: {e}")
    raise e

def validate_file_path(file_path: str) -> tuple[bool, str, str]:
    """Validate that file path is within the allowed filesystem boundary.
    
    Returns:
        tuple: (is_valid, resolved_path, error_message)
    """
    try:
        # If path is absolute, use it as-is; if relative, resolve from file_system_path
        if os.path.isabs(file_path):
            resolved_path = os.path.realpath(file_path)
        else:
            # For relative paths, resolve from file_system_path (not CWD)
            resolved_path = os.path.realpath(os.path.join(file_system_path, file_path))
        
        # Ensure the resolved path is within file_system_path
        file_system_path_real = os.path.realpath(file_system_path)
        if not resolved_path.startswith(file_system_path_real):
            return False, "", f"Error: Path '{file_path}' (resolved to '{resolved_path}') is outside the allowed filesystem boundary '{file_system_path}'"
        
        return True, resolved_path, ""
    except Exception as e:
        return False, "", f"Error: Invalid path '{file_path}': {str(e)}"


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
def ls(
    state: Annotated[FilesystemState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    path: str = "."
) -> Command:
    """List all files and directories in the specified path"""
    # Validate path
    is_valid, resolved_path, error_msg = validate_file_path(path)
    if not is_valid:
        return error_msg
    
    try:
        if not os.path.exists(resolved_path):
            error_msg = f"Error: Path '{path}' does not exist"
            return Command(
                update={
                    "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
                }
            )
        
        if not os.path.isdir(resolved_path):
            error_msg = f"Error: Path '{path}' is not a directory"
            return Command(
                update={
                    "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
                }
            )
        
        # List directory contents
        items = []
        discovered_files = []
        
        for item in sorted(os.listdir(resolved_path)):
            item_path = os.path.join(resolved_path, item)
            if os.path.isdir(item_path):
                items.append(f"{item}/")
            else:
                items.append(item)
                # Add files (not directories) to discovered list
                if path == ".":
                    discovered_files.append(item)
                else:
                    discovered_files.append(os.path.join(path, item))
        
        # Update state with discovered files
        existing_files = state.get("files", []) if state is not None else []
        updated_files = existing_files.copy()
        for file_path in discovered_files:
            if file_path not in updated_files:
                updated_files.append(file_path)
        
        # Format the result as a string for the message
        result_msg = f"Directory listing for '{path}':\n" + "\n".join(items)
        
        return Command(
            update={
                "files": updated_files,
                "messages": [ToolMessage(result_msg, tool_call_id=tool_call_id)]
            }
        )
        
    except PermissionError:
        error_msg = f"Error: Permission denied accessing '{path}'"
        return Command(
            update={
                "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        error_msg = f"Error: Failed to list directory '{path}': {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
            }
        )


@tool(description=READ_FILE_TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[FilesystemState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    offset: int = 0,
    limit: int = 2000
) -> Union[Command, str]:
    """Read file from the real filesystem with tabular file detection and character limits."""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found"
        
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file"
        
        
        # Read file with encoding detection
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return f"Error: Unable to decode file '{file_path}'. File may be binary or use unsupported encoding."
        
        # Handle empty file
        if not content or content.strip() == "":
            return "System reminder: File exists but has empty contents"
        
        # Apply character limit first (before splitting into lines)
        max_chars = 100000  # 100KB character limit
        if len(content) > max_chars:
            content = content[:max_chars]
            content += f"\n\n[Content truncated at {max_chars} characters. File has {len(content)} total characters.]"
        
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
                line_content = line_content[:2000] + " [line truncated]"
            
            # Line numbers start at 1, so add 1 to the index
            line_number = i + 1
            result_lines.append(f"{line_number:6d}\t{line_content}")
        
        file_content = "\n".join(result_lines)
        
        # Track the successfully read file path in state
        files = state.get("files", [])
        if file_path not in files:
            files = files.copy()
            files.append(file_path)
        
        return Command(
            update={
                "files": files,
                "messages": [ToolMessage(file_content, tool_call_id=tool_call_id)]
            }
        )
        
    except PermissionError:
        return f"Error: Permission denied reading '{file_path}'"
    except Exception as e:
        return f"Error: Failed to read file '{file_path}': {str(e)}"


@tool(description=WRITE_FILE_TOOL_DESCRIPTION)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[FilesystemState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Write content to a file in the real filesystem."""
    # Validate path
    is_valid, resolved_path, error_msg = validate_file_path(file_path)
    if not is_valid:
        return Command(
            update={
                "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
            }
        )
    
    try:
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(resolved_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Write file
        with open(resolved_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update state with the file path (handle None state gracefully)
        files = state.get("files", []) if state is not None else []
        if file_path not in files:
            files = files.copy()
            files.append(file_path)
        
        return Command(
            update={
                "files": files,
                "messages": [
                    ToolMessage(f"Successfully wrote file '{file_path}'", tool_call_id=tool_call_id)
                ],
            }
        )
        
    except PermissionError:
        error_msg = f"Error: Permission denied writing to '{file_path}'"
        return Command(
            update={
                "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        error_msg = f"Error: Failed to write file '{file_path}': {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)]
            }
        )


@tool(description=EDIT_FILE_TOOL_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[FilesystemState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False
) -> Union[Command, str]:
    """Edit a file by replacing old_string with new_string."""
    # Validate path
    is_valid, resolved_path, error_msg = validate_file_path(file_path)
    if not is_valid:
        return error_msg
    
    try:
        # Check if file exists
        if not os.path.exists(resolved_path):
            return f"Error: File '{file_path}' not found"
        
        if not os.path.isfile(resolved_path):
            return f"Error: '{file_path}' is not a file"
        
        # Read current file content
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(resolved_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return f"Error: Unable to decode file '{file_path}'. File may be binary or use unsupported encoding."
        
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
        
        # Write the updated content back to file
        with open(resolved_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Update state with the file path if not already tracked
        files = state.get("files", [])
        if file_path not in files:
            files = files.copy()
            files.append(file_path)
        
        return Command(
            update={
                "files": files,
                "messages": [ToolMessage(result_msg, tool_call_id=tool_call_id)],
            }
        )
        
    except PermissionError:
        return f"Error: Permission denied accessing '{file_path}'"
    except Exception as e:
        return f"Error: Failed to edit file '{file_path}': {str(e)}"
