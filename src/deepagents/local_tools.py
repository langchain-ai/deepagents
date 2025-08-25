import os
import pathlib
import re
import subprocess
from langchain_core.tools import tool
from typing import Annotated, Optional, Union, Literal

from deepagents.local_prompts import (
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    GLOB_DESCRIPTION,
    GREP_DESCRIPTION,
    WRITE_DESCRIPTION,
    STR_REPLACE_EDIT_DESCRIPTION,
)

def ls(path: str = ".", state=None) -> list[str]:
    """List all files in the specified directory."""
    try:
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            return [f"Error: Path '{path}' does not exist"]
        if not path_obj.is_dir():
            return [f"Error: Path '{path}' is not a directory"]

        # Get all files and directories in the path
        items = []
        for item in path_obj.iterdir():
            items.append(str(item.name))
        return sorted(items)
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]


@tool(description=TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
    state=None,
) -> str:
    """Read file from disk."""
    try:
        path_obj = pathlib.Path(file_path)

        # Check if file exists
        if not path_obj.exists():
            return f"Error: File '{file_path}' not found"

        if not path_obj.is_file():
            return f"Error: '{file_path}' is not a file"

        # Read file content
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try reading as binary and decode with errors ignored
            with open(path_obj, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")

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
            return (
                f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
            )

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

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool(description=WRITE_DESCRIPTION)
def write_file(
    file_path: str,
    content: str,
    state=None,
) -> str:
    """Write content to a file on disk."""
    try:
        path_obj = pathlib.Path(file_path)

        # Create parent directories if they don't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote to file '{file_path}'"

    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool(description=GLOB_DESCRIPTION)
def glob(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    include_dirs: bool = False,
    recursive: bool = True,
    state=None,
) -> str:
    """Find files and directories using glob patterns."""
    try:
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            return f"Error: Path '{path}' does not exist"

        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"

        results = []

        # Use pathlib.Path.glob or rglob based on recursive setting
        try:
            if recursive:
                # For recursive search, use rglob
                matches = path_obj.rglob(pattern)
            else:
                # For non-recursive search, use glob
                matches = path_obj.glob(pattern)

            # Process matches
            for match in matches:
                if len(results) >= max_results:
                    break

                # Check if we should include this result
                if match.is_file():
                    results.append(str(match))
                elif match.is_dir() and include_dirs:
                    results.append(f"{match}/")

            # Sort results for consistent output
            results.sort()

        except Exception as e:
            return f"Error processing glob pattern: {str(e)}"

        if not results:
            search_type = "recursive" if recursive else "non-recursive"
            dirs_note = " (including directories)" if include_dirs else ""
            return f"No matches found for pattern '{pattern}' in '{path}' ({search_type} search{dirs_note})"

        result_count = len(results)
        header = f"Found {result_count} matches for pattern '{pattern}'"
        if result_count >= max_results:
            header += f" (limited to {max_results} results)"
        header += ":\n\n"

        return header + "\n".join(results)

    except Exception as e:
        return f"Error in glob search: {str(e)}"


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
    state=None,
) -> str:
    """Search for text patterns within files using ripgrep."""
    try:
        if not files and not path:
            return "Error: Must provide either 'files' parameter or 'path' parameter"

        # Build ripgrep command
        cmd = ["rg"]
        
        # Add pattern
        if regex:
            cmd.extend(["-e", pattern])
        else:
            cmd.extend(["-F", pattern])
        
        # Add case sensitivity
        if not case_sensitive:
            cmd.append("-i")
        
        # Add context lines
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        # Add max results
        if max_results > 0:
            cmd.extend(["-m", str(max_results)])
        
        # Add file pattern if specified
        if file_pattern != "*":
            cmd.extend(["-g", file_pattern])
        
        # Add files or path
        if files:
            if isinstance(files, str):
                cmd.append(files)
            else:
                cmd.extend(files)
        elif path:
            cmd.append(path)
        
        # Run ripgrep
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=path if path and os.path.isdir(path) else None
            )
            
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
                # No matches found
                pattern_desc = f"regex pattern '{pattern}'" if regex else f"text '{pattern}'"
                case_desc = " (case-sensitive)" if case_sensitive else " (case-insensitive)"
                return f"No matches found for {pattern_desc}{case_desc}"
            else:
                return f"Error running ripgrep: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: ripgrep search timed out"
        except FileNotFoundError:
            return "Error: ripgrep (rg) not found. Please install ripgrep to use this tool."
        except Exception as e:
            return f"Error running ripgrep: {str(e)}"
            
    except Exception as e:
        return f"Error in grep search: {str(e)}"


@tool(description=STR_REPLACE_EDIT_DESCRIPTION)
def str_replace_based_edit_tool(
    command: Literal["view", "str_replace", "create", "insert"],
    path: str,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None, 
    view_range: Optional[list[int]] = None,
    file_text: Optional[str] = None,
    insert_line: Optional[int] = None,
    state=None,
) -> str:
    """
    A text editor tool that can view, edit, create files, and make targeted edits.
    
    Commands:
    - view: Display file or directory contents
    - str_replace: Replace exact text matches  
    - create: Create a new file
    - insert: Insert text at a specific line
    
    Args:
        command: The operation to perform
        path: File or directory path
        old_str: Exact string to replace (for str_replace)
        new_str: Replacement string (for str_replace/insert)
        view_range: [start_line, end_line] for viewing (1-indexed)
        file_text: Content for new file (for create)
        insert_line: Line number after which to insert (for insert, 0-indexed)
    """
    try:
        path_obj = pathlib.Path(path)
        
        if command == "view":
            if path_obj.is_dir():
                # List directory contents
                try:
                    items = []
                    for item in path_obj.iterdir():
                        if item.is_dir():
                            items.append(f"{item.name}/")
                        else:
                            items.append(item.name)
                    return "\n".join(sorted(items))
                except Exception as e:
                    return f"Error listing directory: {str(e)}"
            
            elif path_obj.is_file():
                # Read file content
                try:
                    with open(path_obj, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    return f"Error: File contains non-UTF-8 content"
                
                lines = content.splitlines()
                
                if view_range:
                    start_line, end_line = view_range
                    # Convert to 0-indexed
                    start_idx = max(0, start_line - 1)
                    end_idx = min(len(lines), end_line)
                    lines = lines[start_idx:end_idx]
                    
                # Format with line numbers
                result_lines = []
                start_num = view_range[0] if view_range else 1
                for i, line in enumerate(lines):
                    line_num = start_num + i
                    result_lines.append(f"{line_num:4d} | {line}")
                
                return "\n".join(result_lines) if result_lines else "File is empty"
            else:
                return f"Error: Path '{path}' does not exist"
        
        elif command == "str_replace":
            if not old_str or new_str is None:
                return "Error: str_replace requires both old_str and new_str parameters"
                
            if not path_obj.exists():
                return f"Error: File '{path}' not found"
                
            if not path_obj.is_file():
                return f"Error: '{path}' is not a file"
                
            # Read file
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                return f"Error: File contains non-UTF-8 content"
                
            # Check if old_str exists
            if old_str not in content:
                return f"Error: String not found in file"
                
            # Count occurrences
            occurrences = content.count(old_str)
            if occurrences > 1:
                return f"Error: String appears {occurrences} times. Please provide more specific context."
                
            # Replace
            new_content = content.replace(old_str, new_str, 1)
            
            # Write back
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_content)
                
            return f"Successfully replaced text in '{path}'"
            
        elif command == "create":
            if file_text is None:
                return "Error: create command requires file_text parameter"
                
            if path_obj.exists():
                return f"Error: File '{path}' already exists"
                
            # Create parent directories if needed
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(file_text)
                
            return f"Successfully created file '{path}'"
            
        elif command == "insert":
            if new_str is None or insert_line is None:
                return "Error: insert command requires both new_str and insert_line parameters"
                
            if not path_obj.exists():
                return f"Error: File '{path}' not found"
                
            if not path_obj.is_file():
                return f"Error: '{path}' is not a file"
                
            # Read file
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                return f"Error: File contains non-UTF-8 content"
                
            # Insert text
            if insert_line < 0 or insert_line > len(lines):
                return f"Error: insert_line {insert_line} out of range (0-{len(lines)})"
                
            # Add newline to new_str if it doesn't end with one
            if new_str and not new_str.endswith('\n'):
                new_str += '\n'
                
            lines.insert(insert_line, new_str)
            
            # Write back
            with open(path_obj, "w", encoding="utf-8") as f:
                f.writelines(lines)
                
            return f"Successfully inserted text at line {insert_line} in '{path}'"
            
        else:
            return f"Error: Unknown command '{command}'"
            
    except Exception as e:
        return f"Error: {str(e)}"