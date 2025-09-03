from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Optional, Union
from langgraph.prebuilt import InjectedState
import re

from deepagents.prompts import (
    WRITE_TODOS_DESCRIPTION,
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    REGEX_SEARCH_DESCRIPTION,
)
from deepagents.state import Todo, DeepAgentState


@tool(description=WRITE_TODOS_DESCRIPTION)
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


def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())


@tool(description=TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file."""
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


def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write to a file."""
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


@tool(description=EDIT_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
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


@tool(description=REGEX_SEARCH_DESCRIPTION)
def regex_search(
    pattern: str,
    state: Annotated[DeepAgentState, InjectedState],
    file_path: Optional[str] = None,
    max_matches: int = 100,
    context_chars: int = 100,
) -> str:
    """Search for regex patterns across files in the mocked filesystem."""
    mock_filesystem = state.get("files", {})
    
    if not mock_filesystem:
        return "No files found in the filesystem"
    
    # Compile the regex pattern
    try:
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern '{pattern}': {e}, fix your mistakes"
    
    results = []
    total_matches = 0
    
    # Determine which files to search
    files_to_search = {}
    if file_path:
        if file_path not in mock_filesystem:
            return f"Error: File '{file_path}' not found"
        files_to_search[file_path] = mock_filesystem[file_path]
    else:
        files_to_search = mock_filesystem
    
    # Search each file
    for current_file_path, content in files_to_search.items():
        if not content or content.strip() == "":
            continue
            
        file_matches = []
        
        for match in compiled_pattern.finditer(content):
            if len(file_matches) >= max_matches:
                break
            
            match_start = match.start()
            match_end = match.end()
            
            context_start = max(0, match_start - context_chars)
            context_end = min(len(content), match_end + context_chars)
            
            before_context = content[context_start:match_start]
            matched_text = content[match_start:match_end]
            after_context = content[match_end:context_end]
            
            line_number = content[:match_start].count('\n') + 1
            
            # Create context display with match highlighted
            context_display = f"{before_context}>>>{matched_text}<<<{after_context}"
            
            file_matches.append({
                "line_number": line_number,
                "match_text": matched_text,
                "match_start": match_start,
                "match_end": match_end,
                "context": context_display,
            })
        
        if file_matches:
            results.append({
                "file_path": current_file_path,
                "matches": file_matches,
                "match_count": len(file_matches)
            })
            total_matches += len(file_matches)
    
    # Format results
    if not results:
        search_scope = f"file '{file_path}'" if file_path else "all files"
        return f"<search_results>\n<summary>No matches found for pattern '{pattern}' in {search_scope}</summary>\n</search_results>"
    
    output_lines = []
    output_lines.append("<search_results>")
    output_lines.append(f"<summary>Found {total_matches} matches for pattern '{pattern}'</summary>")
    
    for file_result in results:
        output_lines.append(f"<file path='{file_result['file_path']}' match_count='{file_result['match_count']}'>")
        
        for i, match in enumerate(file_result['matches'], 1):
            output_lines.append(f"<match id='{i}' line='{match['line_number']}' start='{match['match_start']}' end='{match['match_end']}'>")
            output_lines.append(f"<matched_text>{match['match_text']}</matched_text>")
            output_lines.append(f"<context>{match['context']}</context>")
            output_lines.append("</match>")
        
        output_lines.append("</file>")
    
    output_lines.append("</search_results>")
    return "\n".join(output_lines)
