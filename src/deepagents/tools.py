from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Optional, Union
from langgraph.prebuilt import InjectedState
import fnmatch
import re

from deepagents.prompts import (
    WRITE_TODOS_DESCRIPTION,
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    GLOB_DESCRIPTION,
    GREP_DESCRIPTION,
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
) -> Command:
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


@tool(description=GLOB_DESCRIPTION)
def glob(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    include_dirs: bool = False,
    recursive: bool = True,
    state: Annotated[DeepAgentState, InjectedState] = None,
) -> str:
    """Find files and directories using glob patterns in the mock filesystem."""
    try:
        mock_filesystem = state.get("files", {}) if state else {}
        if not mock_filesystem:
            return "No files available in the mock filesystem"

        # Normalize the search path
        search_path = path.rstrip("/")
        if search_path == ".":
            search_path = ""

        results = []

        # Get all file paths that match the search path
        candidate_paths = []
        for file_path in mock_filesystem.keys():
            # Check if file is within the search path
            if search_path and not file_path.startswith(search_path):
                continue

            # For non-recursive searches, ensure file is directly in search path
            if not recursive and search_path:
                relative_path = file_path[len(search_path) :].lstrip("/")
                if "/" in relative_path:
                    continue

            candidate_paths.append(file_path)

        # Also generate directory paths if include_dirs is True
        dir_paths = set()
        if include_dirs:
            for file_path in candidate_paths:
                # Generate all parent directory paths
                path_parts = file_path.split("/")
                for i in range(1, len(path_parts)):
                    dir_path = "/".join(path_parts[:i])
                    if not search_path or dir_path.startswith(search_path):
                        dir_paths.add(dir_path + "/")

        # Combine files and directories
        all_paths = candidate_paths + list(dir_paths)

        # Filter paths using glob pattern matching
        for full_path in all_paths:
            if len(results) >= max_results:
                break

            # For pattern matching, we need to consider the relative path from search_path
            if search_path:
                if full_path.startswith(search_path):
                    relative_path = full_path[len(search_path) :].lstrip("/")
                else:
                    continue
            else:
                relative_path = full_path

            # Use fnmatch for glob pattern matching
            if fnmatch.fnmatch(relative_path, pattern):
                results.append(full_path)
            elif recursive and "/" in relative_path:
                # Also try matching just the filename part for recursive searches
                filename = relative_path.split("/")[-1]
                if fnmatch.fnmatch(filename, pattern):
                    results.append(full_path)

        # Sort results for consistent output
        results.sort()

        if not results:
            search_type = "recursive" if recursive else "non-recursive"
            dirs_note = " (including directories)" if include_dirs else ""
            return f"No matches found for pattern '{pattern}' in mock filesystem at '{path}' ({search_type} search{dirs_note})"

        result_count = len(results)
        header = (
            f"Found {result_count} matches for pattern '{pattern}' in mock filesystem"
        )
        if result_count >= max_results:
            header += f" (limited to {max_results} results)"
        header += ":\n\n"

        return header + "\n".join(results)

    except Exception as e:
        return f"Error in mock filesystem glob search: {str(e)}"


@tool(description=GREP_DESCRIPTION)
def grep(
    pattern: str,
    files: Optional[Union[str, list[str]]] = None,
    path: Optional[str] = None,
    file_pattern: str = "*",
    max_results: int = 50,
    case_sensitive: bool = False,
    context_lines: int = 0,
    regex: bool = False,
    recursive: bool = True,
    state: Annotated[DeepAgentState, InjectedState] = None,
) -> str:
    """Search for text patterns within files in the mock filesystem."""
    try:
        mock_filesystem = state.get("files", {}) if state else {}
        if not mock_filesystem:
            return "No files available in the mock filesystem"

        if not files and not path:
            return "Error: Must provide either 'files' parameter or 'path' parameter"

        # Prepare list of files to search
        files_to_search = []

        if files:
            # Convert single file to list
            if isinstance(files, str):
                files = [files]

            # Validate files exist in mock filesystem
            for file_path in files:
                if file_path in mock_filesystem:
                    files_to_search.append(file_path)
                else:
                    return (
                        f"Error: File '{file_path}' does not exist in mock filesystem"
                    )

        elif path:
            # Find files in mock filesystem matching the path and pattern
            search_path = path.rstrip("/")
            if search_path == ".":
                search_path = ""

            for file_path in mock_filesystem.keys():
                # Check if file is within the search path
                if search_path and not file_path.startswith(search_path):
                    continue

                # For non-recursive searches, ensure file is directly in search path
                if not recursive and search_path:
                    relative_path = file_path[len(search_path) :].lstrip("/")
                    if "/" in relative_path:
                        continue

                # Get filename for pattern matching
                filename = file_path.split("/")[-1]

                # Match against file_pattern
                if fnmatch.fnmatch(filename, file_pattern):
                    files_to_search.append(file_path)

        if not files_to_search:
            return f"No files found to search in mock filesystem"

        # Prepare regex pattern
        if regex:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: Invalid regex pattern: {str(e)}"
        else:
            compiled_pattern = None

        results = []
        total_matches = 0

        for file_path in files_to_search:
            if total_matches >= max_results:
                break

            file_content = mock_filesystem[file_path]
            if not file_content:
                continue

            lines = file_content.splitlines()
            file_matches = []

            # Search through lines
            for line_num, line in enumerate(lines, 1):
                match_found = False

                if regex:
                    match_found = compiled_pattern.search(line) is not None
                else:
                    if case_sensitive:
                        match_found = pattern in line
                    else:
                        match_found = pattern.lower() in line.lower()

                if match_found:
                    # Add context lines if requested
                    context_start = max(0, line_num - 1 - context_lines)
                    context_end = min(len(lines), line_num + context_lines)

                    match_info = {
                        "line_num": line_num,
                        "line_content": line,
                        "context_lines": [],
                    }

                    if context_lines > 0:
                        for ctx_line_num in range(context_start, context_end):
                            ctx_line_content = lines[ctx_line_num]
                            is_match_line = (ctx_line_num + 1) == line_num
                            match_info["context_lines"].append(
                                {
                                    "line_num": ctx_line_num + 1,
                                    "content": ctx_line_content,
                                    "is_match": is_match_line,
                                }
                            )

                    file_matches.append(match_info)
                    total_matches += 1

                    if total_matches >= max_results:
                        break

            # Add file results if there were matches
            if file_matches:
                file_result_lines = [f"📄 {file_path}"]

                for match in file_matches:
                    if context_lines > 0:
                        # Show context
                        for ctx in match["context_lines"]:
                            prefix = ">" if ctx["is_match"] else " "
                            file_result_lines.append(
                                f"{prefix} {ctx['line_num']:4d}: {ctx['content']}"
                            )
                        file_result_lines.append("")  # Empty line between matches
                    else:
                        # Show just the matching line
                        file_result_lines.append(
                            f"  {match['line_num']:4d}: {match['line_content']}"
                        )

                results.append("\n".join(file_result_lines))

        if not results:
            pattern_desc = (
                f"regex pattern '{pattern}'" if regex else f"text '{pattern}'"
            )
            case_desc = " (case-sensitive)" if case_sensitive else " (case-insensitive)"
            return f"No matches found for {pattern_desc}{case_desc} in mock filesystem"

        # Format final results
        match_count = len(results)
        header = f"Found matches in {match_count} files in mock filesystem"
        if total_matches >= max_results:
            header += f" (limited to {max_results} total matches)"
        header += ":\n"

        return header + "\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error in mock filesystem grep search: {str(e)}"
