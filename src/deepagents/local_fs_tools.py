import os
import pathlib
import re
import subprocess
from langchain_core.tools import tool
from typing import Annotated, Optional, Union, Literal

from deepagents.prompts import (
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    GLOB_DESCRIPTION,
    GREP_DESCRIPTION,
    WRITE_DESCRIPTION,
)

def ls(path: str = ".") -> list[str]:
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


@tool(description=EDIT_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a file on disk by replacing old_string with new_string."""
    try:
        path_obj = pathlib.Path(file_path)

        # Check if file exists
        if not path_obj.exists():
            return f"Error: File '{file_path}' not found"

        if not path_obj.is_file():
            return f"Error: '{file_path}' is not a file"

        # Read current file content
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: File '{file_path}' contains non-UTF-8 content"

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

        # Write the updated content back to the file
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(new_content)

        return result_msg

    except Exception as e:
        return f"Error editing file: {str(e)}"


@tool(description=GLOB_DESCRIPTION)
def glob(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    include_dirs: bool = False,
    recursive: bool = True,
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



