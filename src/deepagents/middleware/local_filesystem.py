"""Middleware that exposes local filesystem tools to an agent.

This mirrors the structure of `FilesystemMiddleware` but operates on the
host/local filesystem (disk) rather than the in-memory/mock filesystem.
It ports the tool behavior from `src/deepagents/local_fs_tools.py` into
middleware-provided tools so they can be injected via AgentMiddleware.
"""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union
from typing_extensions import NotRequired

import os
import pathlib
import re
import subprocess

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.config import get_config
from langchain_core.tools import tool
from deepagents.middleware.filesystem import (
    _create_file_data,
    _format_content_with_line_numbers,
    FilesystemState,
    FILESYSTEM_SYSTEM_PROMPT,
)
from deepagents.middleware.common import TOO_LARGE_TOOL_MSG, FILESYSTEM_SYSTEM_PROMPT_GLOB_GREP_SUPPLEMENT

from deepagents.prompts import (
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    GLOB_DESCRIPTION,
    GREP_DESCRIPTION,
    WRITE_DESCRIPTION,
    LONGTERM_MEMORY_SYSTEM_PROMPT,
    DEFAULT_MEMORY,
)

LOCAL_LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the specified directory on disk.

Usage:
- The ls tool will return a list of all files in the specified directory.
- The path parameter accepts both absolute paths (starting with /) and relative paths
- Relative paths are resolved relative to the current working directory
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""

LOCAL_READ_FILE_TOOL_DESCRIPTION = TOOL_DESCRIPTION + "\n- You should ALWAYS make sure a file has been read before editing it."
LOCAL_EDIT_FILE_TOOL_DESCRIPTION = EDIT_DESCRIPTION


# -----------------------------
# Path Resolution Helper
# -----------------------------

def _resolve_path(path: str, cwd: str, long_term_memory: bool = False) -> str:
    """Resolve relative paths against CWD, leave absolute paths unchanged.
    
    Special handling: /memories/* paths are redirected to ~/.deepagents/<assistant_id>/
    assistant_id is retrieved from the runtime config metadata. If assistant_id is None, 
    /memories paths will return an error. This only works if long_term_memory=True.
    """
    if path.startswith("/memories"):
        if not long_term_memory:
            raise ValueError(
                "Long-term memory is disabled. "
                "/memories/ access requires long_term_memory=True."
            )
        
        # Get assistant_id from config metadata
        config = get_config()
        assistant_id = config.get("metadata", {}).get("assistant_id") if config else None
        
        if assistant_id is None:
            raise ValueError(
                "Memory access is disabled when no assistant_id is provided. "
                "To use /memories/, pass assistant_id in config metadata."
            )
        
        agent_dir = pathlib.Path.home() / ".deepagents" / assistant_id
        if path == "/memories":
            return str(agent_dir)
        else:
            relative_part = path[len("/memories/"):]
            return str(agent_dir / relative_part)
    
    if os.path.isabs(path):
        return path
    return str(pathlib.Path(cwd) / path)


# -----------------------------
# Tool Implementations (Local)
# -----------------------------

def _ls_impl(path: str = ".", cwd: str | None = None, long_term_memory: bool = False) -> list[str]:
    """List all files in the specified directory on disk."""
    try:
        if cwd:
            path = _resolve_path(path, cwd, long_term_memory)
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            return [f"Error: Path '{path}' does not exist"]
        if not path_obj.is_dir():
            return [f"Error: Path '{path}' is not a directory"]

        items: list[str] = []
        for item in path_obj.iterdir():
            items.append(str(item.name))
        return sorted(items)
    except Exception as e:  # pragma: no cover - defensive
        return [f"Error listing directory: {str(e)}"]


def _read_file_impl(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
    cwd: str | None = None,
    long_term_memory: bool = False,
) -> str:
    """Read a file from the local filesystem and return cat -n formatted content."""
    try:
        if cwd:
            file_path = _resolve_path(file_path, cwd, long_term_memory)
        path_obj = pathlib.Path(file_path)

        if not path_obj.exists():
            return f"Error: File '{file_path}' not found"
        if not path_obj.is_file():
            return f"Error: '{file_path}' is not a file"

        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to binary read and lenient decode
            with open(path_obj, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")

        if not content or content.strip() == "":
            return "System reminder: File exists but has empty contents"

        lines = content.splitlines()
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))

        if start_idx >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

        result_lines = []
        for i in range(start_idx, end_idx):
            line_content = lines[i]
            if len(line_content) > 2000:
                line_content = line_content[:2000]
            result_lines.append(f"{i + 1:6d}\t{line_content}")

        return "\n".join(result_lines)

    except Exception as e:  # pragma: no cover - defensive
        return f"Error reading file: {str(e)}"


def _write_file_impl(
    file_path: str,
    content: str,
    cwd: str | None = None,
    long_term_memory: bool = False,
) -> str:
    """Write content to a file on the local filesystem (creates parents)."""
    try:
        if cwd:
            file_path = _resolve_path(file_path, cwd, long_term_memory)
        path_obj = pathlib.Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to file '{file_path}'"
    except Exception as e:  # pragma: no cover - defensive
        return f"Error writing file: {str(e)}"


def _edit_file_impl(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    cwd: str | None = None,
    long_term_memory: bool = False,
) -> str:
    """Edit a file on disk by replacing old_string with new_string."""
    try:
        if cwd:
            file_path = _resolve_path(file_path, cwd, long_term_memory)
        path_obj = pathlib.Path(file_path)
        if not path_obj.exists():
            return f"Error: File '{file_path}' not found"
        if not path_obj.is_file():
            return f"Error: '{file_path}' is not a file"

        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: File '{file_path}' contains non-UTF-8 content"

        if old_string not in content:
            return f"Error: String not found in file: '{old_string}'"

        if not replace_all:
            occurrences = content.count(old_string)
            if occurrences > 1:
                return (
                    f"Error: String '{old_string}' appears {occurrences} times in file. "
                    "Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
                )
            elif occurrences == 0:
                return f"Error: String not found in file: '{old_string}'"

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(new_content)

        if replace_all:
            replacement_count = content.count(old_string)
            return f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
        return f"Successfully replaced string in '{file_path}'"

    except Exception as e:  # pragma: no cover - defensive
        return f"Error editing file: {str(e)}"


def _glob_impl(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    include_dirs: bool = False,
    recursive: bool = True,
    cwd: str | None = None,
    long_term_memory: bool = False,
) -> str:
    """Find files and directories using glob patterns on local filesystem."""
    try:
        if cwd:
            path = _resolve_path(path, cwd, long_term_memory)
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            return f"Error: Path '{path}' does not exist"
        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"

        results: list[str] = []
        try:
            matches = path_obj.rglob(pattern) if recursive else path_obj.glob(pattern)
            for match in matches:
                if len(results) >= max_results:
                    break
                if match.is_file():
                    results.append(str(match))
                elif match.is_dir() and include_dirs:
                    results.append(f"{match}/")
            results.sort()
        except Exception as e:
            return f"Error processing glob pattern: {str(e)}"

        if not results:
            search_type = "recursive" if recursive else "non-recursive"
            dirs_note = " (including directories)" if include_dirs else ""
            return f"No matches found for pattern '{pattern}' in '{path}' ({search_type} search{dirs_note})"

        header = f"Found {len(results)} matches for pattern '{pattern}'"
        if len(results) >= max_results:
            header += f" (limited to {max_results} results)"
        header += ":\n\n"
        return header + "\n".join(results)

    except Exception as e:  # pragma: no cover - defensive
        return f"Error in glob search: {str(e)}"


def _grep_impl(
    pattern: str,
    files: Optional[Union[str, list[str]]] = None,
    path: Optional[str] = None,
    file_pattern: str = "*",
    max_results: int = 50,
    case_sensitive: bool = False,
    context_lines: int = 0,
    regex: bool = False,
    cwd: str | None = None,
    long_term_memory: bool = False,
) -> str:
    """Search for text patterns within files using ripgrep on local filesystem."""
    try:
        if not files and not path:
            return "Error: Must provide either 'files' parameter or 'path' parameter"
        
        if cwd:
            if files:
                if isinstance(files, str):
                    files = _resolve_path(files, cwd, long_term_memory)
                else:
                    files = [_resolve_path(f, cwd, long_term_memory) for f in files]
            if path:
                path = _resolve_path(path, cwd, long_term_memory)

        cmd: list[str] = ["rg"]
        if regex:
            cmd.extend(["-e", pattern])
        else:
            cmd.extend(["-F", pattern])

        if not case_sensitive:
            cmd.append("-i")

        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        if max_results > 0:
            cmd.extend(["-m", str(max_results)])

        if file_pattern != "*":
            cmd.extend(["-g", file_pattern])

        if files:
            if isinstance(files, str):
                cmd.append(files)
            else:
                cmd.extend(files)
        elif path:
            cmd.append(path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=path if path and os.path.isdir(path) else None,
            )
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
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

    except Exception as e:  # pragma: no cover - defensive
        return f"Error in grep search: {str(e)}"


# --------------------------------
# Middleware: LocalFilesystemMiddleware
# --------------------------------

LOCAL_FILESYSTEM_SYSTEM_PROMPT = FILESYSTEM_SYSTEM_PROMPT + "\n" + FILESYSTEM_SYSTEM_PROMPT_GLOB_GREP_SUPPLEMENT

# Skills discovery paths
STANDARD_SKILL_PATHS = [
    "~/.deepagents/skills",
    "./.deepagents/skills",
]


def _get_local_filesystem_tools(custom_tool_descriptions: dict[str, str] | None = None, cwd: str | None = None, long_term_memory: bool = False):
    """Return tool instances for local filesystem operations.
    
    agent_name is retrieved from runtime config via get_config() when tools are called.
    """
    # We already decorated read/write/edit/glob/grep with @tool including descriptions
    # Only `ls` needs a manual wrapper to attach a description.
    ls_description = (
        custom_tool_descriptions.get("ls") if custom_tool_descriptions else LOCAL_LIST_FILES_TOOL_DESCRIPTION
    )

    @tool(description=ls_description)
    def ls(path: str = ".") -> list[str]:  # noqa: D401 - simple wrapper
        """List all files in the specified directory."""
        return _ls_impl(path, cwd=cwd, long_term_memory=long_term_memory)

    read_desc = (
        (custom_tool_descriptions or {}).get("read_file", LOCAL_READ_FILE_TOOL_DESCRIPTION)
    )

    @tool(description=read_desc)
    def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return _read_file_impl(file_path, offset, limit, cwd=cwd, long_term_memory=long_term_memory)

    write_desc = (
        (custom_tool_descriptions or {}).get("write_file", WRITE_DESCRIPTION)
    )

    @tool(description=write_desc)
    def write_file(file_path: str, content: str) -> str:
        return _write_file_impl(file_path, content, cwd=cwd, long_term_memory=long_term_memory)

    edit_desc = (
        (custom_tool_descriptions or {}).get("edit_file", LOCAL_EDIT_FILE_TOOL_DESCRIPTION)
    )

    @tool(description=edit_desc)
    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        return _edit_file_impl(file_path, old_string, new_string, replace_all, cwd=cwd, long_term_memory=long_term_memory)

    glob_desc = (
        (custom_tool_descriptions or {}).get("glob", GLOB_DESCRIPTION)
    )

    @tool(description=glob_desc)
    def glob(
        pattern: str,
        path: str = ".",
        max_results: int = 100,
        include_dirs: bool = False,
        recursive: bool = True,
    ) -> str:
        return _glob_impl(pattern, path, max_results, include_dirs, recursive, cwd=cwd, long_term_memory=long_term_memory)

    grep_desc = (
        (custom_tool_descriptions or {}).get("grep", GREP_DESCRIPTION)
    )

    @tool(description=grep_desc)
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
        return _grep_impl(
            pattern,
            files=files,
            path=path,
            file_pattern=file_pattern,
            max_results=max_results,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            regex=regex,
            cwd=cwd,
            long_term_memory=long_term_memory,
        )

    return [ls, read_file, write_file, edit_file, glob, grep]


class LocalFilesystemState(FilesystemState):
    """Extended filesystem state with agent memory support."""
    
    agent_memory: NotRequired[str]
    """Content of agent.md file for long-term memory."""


class LocalFilesystemMiddleware(AgentMiddleware):
    """Middleware that injects local filesystem tools into an agent.

    Tools added:
    - ls
    - read_file
    - write_file
    - edit_file
    - glob
    - grep
    
    Skills are automatically discovered from:
    - ~/.deepagents/skills/ (personal skills)
    - ./.deepagents/skills/ (project skills)
    """

    state_schema = LocalFilesystemState

    def _check_ripgrep_installed(self) -> None:
        """Check if ripgrep (rg) is installed on the system.
        
        Raises:
            RuntimeError: If ripgrep is not found on the system.
        """
        try:
            subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ripgrep (rg) is not installed. The grep tool requires ripgrep to function. "
                "Please install it from https://github.com/BurntSushi/ripgrep#installation"
            )
        except Exception as e:
            raise RuntimeError(f"Error checking for ripgrep installation: {str(e)}")

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        cwd: str | None = None,
        long_term_memory: bool = False,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self.long_term_memory = long_term_memory
        
        # Check if ripgrep is installed
        self._check_ripgrep_installed()
        
        # Discover skills from standard locations
        self.skills = self._discover_skills()
        
        # Build system prompt
        cwd_prompt = f"\n\nCurrent working directory: {self.cwd}\n\nWhen using filesystem tools (ls, read_file, write_file, edit_file, glob, grep), relative paths will be resolved relative to this directory."
        
        # Add long-term memory documentation if enabled
        memory_prompt = ""
        if long_term_memory:
            memory_prompt = LONGTERM_MEMORY_SYSTEM_PROMPT.format(memory_path="`/memories/`")
        
        base_prompt = system_prompt or LOCAL_FILESYSTEM_SYSTEM_PROMPT
        skills_prompt = self._build_skills_prompt()
        
        self.system_prompt = base_prompt + cwd_prompt + memory_prompt + skills_prompt
        self.tools = _get_local_filesystem_tools(custom_tool_descriptions, cwd=self.cwd, long_term_memory=long_term_memory)
        self.tool_token_limit_before_evict = tool_token_limit_before_evict
    
    def _discover_skills(self) -> list[dict[str, str]]:
        """Discover skills from standard filesystem locations.
        
        Returns:
            List of skill metadata dictionaries with keys: name, path, description, source.
        """
        from deepagents.skills import parse_skill_frontmatter
        
        discovered = {}
        
        for base_path in STANDARD_SKILL_PATHS:
            # Expand ~ to home directory
            expanded_path = os.path.expanduser(base_path)
            
            # Resolve relative paths against cwd
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.join(self.cwd, expanded_path)
            
            if not os.path.exists(expanded_path):
                continue
            
            # Find all SKILL.md files
            try:
                skill_files = _glob_impl(
                    pattern="**/SKILL.md",
                    path=expanded_path,
                    max_results=1000,
                    recursive=True,
                )
                
                # Parse the glob output (skip header line)
                if "Found" not in skill_files:
                    continue
                    
                lines = skill_files.split('\n')
                skill_paths = [line for line in lines[2:] if line.strip()]  # Skip header and empty
                
                for skill_path in skill_paths:
                    try:
                        # Read SKILL.md file
                        content = _read_file_impl(skill_path, cwd=None)
                        if content.startswith("Error:"):
                            continue
                        
                        # Remove line numbers from cat -n format
                        content_lines = []
                        for line in content.split('\n'):
                            # Format is "     1\tcontent"
                            if '\t' in line:
                                content_lines.append(line.split('\t', 1)[1])
                        actual_content = '\n'.join(content_lines)
                        
                        # Parse YAML frontmatter
                        frontmatter = parse_skill_frontmatter(actual_content)
                        if not frontmatter.get('name'):
                            continue
                        
                        skill_name = frontmatter['name']
                        source = "project" if "./.deepagents" in base_path else "personal"
                        
                        # Project skills override personal skills
                        discovered[skill_name] = {
                            "name": skill_name,
                            "path": skill_path,
                            "description": frontmatter.get('description', ''),
                            "version": frontmatter.get('version', ''),
                            "source": source,
                        }
                    except Exception:
                        # Skip skills that fail to parse
                        continue
                        
            except Exception:
                # Skip paths that fail to glob
                continue
        
        return list(discovered.values())
    
    def _build_skills_prompt(self) -> str:
        """Build the skills section of the system prompt.
        
        Returns:
            System prompt text describing available skills, or empty string if no skills.
        """
        if not self.skills:
            return ""
        
        prompt = "\n\n## Available Skills\n\nYou have access to the following skills:"
        
        for i, skill in enumerate(self.skills, 1):
            prompt += f"\n\n{i}. **{skill['name']}** ({skill['path']})"
            if skill['description']:
                prompt += f"\n   - {skill['description']}"
            prompt += f"\n   - Source: {skill['source']}"
        
        prompt += "\n\nTo use a skill, read its SKILL.md file using `read_file`. Skills may contain additional resources in scripts/, references/, and assets/ subdirectories."
        
        return prompt

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        """Load agent.md from disk at the beginning of agent execution.
        
        This runs once when the agent starts, loading the agent's memory
        from ~/.deepagents/<assistant_id>/agent.md and storing it in state.
        If agent_memory is already in state (from a previous run), skip loading.
        
        Args:
            state: The agent state.
            runtime: The LangGraph runtime.
            
        Returns:
            State update with agent_memory loaded, or None if disabled/no assistant_id/already loaded.
        """
        if not self.long_term_memory:
            return None
        
        # Check if already loaded from a previous run
        if "agent_memory" in state:
            return None
        
        config = get_config()
        assistant_id = config.get("metadata", {}).get("assistant_id") if config else None
        
        if assistant_id is None:
            return None
        
        agent_dir = pathlib.Path.home() / ".deepagents" / assistant_id
        agent_md = agent_dir / "agent.md"
        
        if agent_md.exists():
            agent_memory_content = agent_md.read_text()
        else:
            # Create empty agent.md
            agent_dir.mkdir(parents=True, exist_ok=True)
            agent_memory_content = DEFAULT_MEMORY
            agent_md.write_text(agent_memory_content)
        
        return {"agent_memory": agent_memory_content}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Build final system prompt
        final_prompt = ""
        
        # Add agent memory if available in state
        if self.long_term_memory:
            agent_memory_content = request.state.get("agent_memory", "")
            if agent_memory_content:
                final_prompt = f"<agent_memory>\n{agent_memory_content}\n</agent_memory>\n\n"
        
        # Add middleware system prompt
        if self.system_prompt is not None:
            final_prompt += self.system_prompt
        
        # Prepend to request system prompt
        if final_prompt:
            request.system_prompt = (
                final_prompt + "\n\n" + request.system_prompt
                if request.system_prompt
                else final_prompt
            )
        
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # Build final system prompt
        final_prompt = ""
        
        # Add agent memory if available in state
        if self.long_term_memory:
            agent_memory_content = request.state.get("agent_memory", "")
            if agent_memory_content:
                final_prompt = f"<agent_memory>\n{agent_memory_content}\n</agent_memory>\n\n"
        
        # Add middleware system prompt
        if self.system_prompt is not None:
            final_prompt += self.system_prompt
        
        # Prepend to request system prompt
        if final_prompt:
            request.system_prompt = (
                final_prompt + "\n\n" + request.system_prompt
                if request.system_prompt
                else final_prompt
            )
        
        return await handler(request)

    # --------- Token-eviction to filesystem state (like FilesystemMiddleware) ---------
    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage) and isinstance(tool_result.content, str):
            content = tool_result.content
            if self.tool_token_limit_before_evict and len(content) > 4 * self.tool_token_limit_before_evict:
                file_path = f"/large_tool_results/{tool_result.tool_call_id}"
                file_data = _create_file_data(content)
                state_update = {
                    "messages": [
                        ToolMessage(
                            TOO_LARGE_TOOL_MSG.format(
                                tool_call_id=tool_result.tool_call_id,
                                file_path=file_path,
                                content_sample=_format_content_with_line_numbers(
                                    file_data["content"][:10], format_style="tab", start_line=1
                                ),
                            ),
                            tool_call_id=tool_result.tool_call_id,
                        )
                    ],
                    "files": {file_path: file_data},
                }
                return Command(update=state_update)
        elif isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            message_updates = update.get("messages", [])
            file_updates = update.get("files", {})

            edited_message_updates = []
            for message in message_updates:
                if self.tool_token_limit_before_evict and isinstance(message, ToolMessage) and isinstance(message.content, str):
                    content = message.content
                    if len(content) > 4 * self.tool_token_limit_before_evict:
                        file_path = f"/large_tool_results/{message.tool_call_id}"
                        file_data = _create_file_data(content)
                        edited_message_updates.append(
                            ToolMessage(
                                TOO_LARGE_TOOL_MSG.format(
                                    tool_call_id=message.tool_call_id,
                                    file_path=file_path,
                                    content_sample=_format_content_with_line_numbers(
                                        file_data["content"][:10], format_style="tab", start_line=1
                                    ),
                                ),
                                tool_call_id=message.tool_call_id,
                            )
                        )
                        file_updates[file_path] = file_data
                        continue
                edited_message_updates.append(message)
            return Command(update={**update, "messages": edited_message_updates, "files": file_updates})
        return tool_result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # Skip eviction for local filesystem tools
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}:
            return handler(request)
        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}:
            return await handler(request)
        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result)
