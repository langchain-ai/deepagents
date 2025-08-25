WRITE_TODOS_DESCRIPTION = """
You have access to this tool to help you manage and plan tasks. Use this tool VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
This tool is also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.

Examples:

<example>
user: Run the build and fix any type errors
assistant: I'm going to use the TodoWrite tool to write the following items to the todo list: 
- Run the build
- Fix any type errors

I'm now going to run the build using Bash.

Looks like I found 10 type errors. I'm going to use the TodoWrite tool to write 10 items to the todo list.

marking the first todo as in_progress

Let me start working on the first item...

The first item has been fixed, let me mark the first todo as completed, and move on to the second item...
..
..
</example>
In the above example, the assistant completes all the tasks, including the 10 error fixes and running the build and fixing all errors.

<example>
user: Help me write a new feature that allows users to track their usage metrics and export them to various formats

assistant: I'll help you implement a usage metrics tracking and export feature. Let me first use the TodoWrite tool to plan this task.
Adding the following todos to the todo list:
1. Research existing metrics tracking in the codebase
2. Design the metrics collection system
3. Implement core metrics tracking functionality
4. Create export functionality for different formats

Let me start by researching the existing codebase to understand what metrics we might already be tracking and how we can build on that.

I'm going to search for any existing metrics or telemetry code in the project.

I've found some existing telemetry code. Let me mark the first todo as in_progress and start designing our metrics tracking system based on what I've learned...

[Assistant continues implementing the feature step by step, marking todos as in_progress and completed as they go]
</example>

"""
EDIT_DESCRIPTION = """This is a tool for editing files. For moving or renaming files, you should generally use the Bash tool with the 'mv' command instead. For larger edits, use the Write tool to overwrite files.

Before using this tool:

Use the Read tool to understand the file's contents and context

Verify the directory path is correct (only applicable when creating new files):

Use the LS tool to verify the parent directory exists and is the correct location

To make a file edit, provide the following:

file_path: The absolute path to the file to modify (must be absolute, not relative)
old_string: The text to replace (must be unique within the file, and must match the file contents exactly, including all whitespace and indentation)
new_string: The edited text to replace the old_string

The tool will replace ONE occurrence of old_string with new_string in the specified file.

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change. This means:

Include AT LEAST 3-5 lines of context BEFORE the change point
Include AT LEAST 3-5 lines of context AFTER the change point
Include all whitespace, indentation, and surrounding code exactly as it appears in the file

SINGLE INSTANCE: This tool can only change ONE instance at a time. If you need to change multiple instances:

Make separate calls to this tool for each instance
Each call must uniquely identify its specific instance using extensive context

VERIFICATION: Before using this tool:

Check how many instances of the target text exist in the file
If multiple instances exist, gather enough context to uniquely identify each one
Plan separate tool calls for each instance

WARNING: If you do not follow these requirements:

The tool will fail if old_string matches multiple locations
The tool will fail if old_string doesn't match exactly (including whitespace)
You may change the wrong instance if you don't include enough context

When making edits:

Ensure the edit results in idiomatic, correct code
Do not leave the code in a broken state
Always use absolute file paths (starting with /)

If you want to create a new file, use:

A new file path, including dir name if needed
An empty old_string
The new file's contents as new_string"""

TOOL_DESCRIPTION = """Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful. 
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents

CRITICAL: Always use absolute paths (starting with /)"""

GLOB_DESCRIPTION = """Find files and directories using glob patterns (similar to Unix glob/find commands).

This tool searches for files and directories that match specified patterns. It's ideal for finding files by name, extension, or path patterns.

Usage:
- pattern: Glob pattern to match (e.g., "*.py", "**/*.js", "src/**/test_*.py")
- path: Directory to start search from (defaults to current directory ".")
- max_results: Maximum number of results to return (defaults to 100)
- include_dirs: Include directories in results (defaults to False, files only)
- recursive: Enable recursive search (defaults to True)

Glob Pattern Examples:
- "*.py" - All Python files in current directory
- "**/*.py" - All Python files recursively
- "src/**/*.js" - All JS files under src/ directory recursively  
- "test_*.py" - Files starting with "test_" and ending with ".py"
- "**/node_modules" - All node_modules directories
- "*.{py,js,ts}" - Files with .py, .js, or .ts extensions

Returns: List of matching file/directory paths, one per line

CRITICAL: Always use absolute paths for the path parameter"""

GREP_DESCRIPTION = """A powerful search tool that uses ripgrep (rg) for fast text pattern matching.

This tool searches file contents for specified patterns and returns matching lines with context. It's ideal for finding specific content, functions, variables, or text across your codebase.

Usage:
- pattern: Text pattern to search for (supports regular expressions if regex=True)
- files: List of file paths to search in, or single file path string
- path: Directory to search in (alternative to files parameter)
- file_pattern: Glob pattern for files to search (e.g., "*.py") when using path
- max_results: Maximum number of matching lines to return (defaults to 50)
- case_sensitive: Whether search should be case-sensitive (defaults to False)
- context_lines: Number of lines to show before/after each match (defaults to 0)
- regex: Treat pattern as regular expression (defaults to False)

Examples:
- Search for "TODO" in specific files: pattern="TODO", files=["main.py", "utils.py"]
- Search in all Python files: pattern="def main", path=".", file_pattern="*.py"
- Regex search: pattern=r"function\\s+\\w+", regex=True, file_pattern="*.js"
- Case-sensitive search: pattern="ClassName", case_sensitive=True
- With context: pattern="import", context_lines=2

Returns: File paths with line numbers and matching lines, plus context if requested

CRITICAL: Always use absolute paths for files and path parameters"""

WRITE_DESCRIPTION = """Write a file to the local filesystem. Overwrites the existing file if there is one.

Before using this tool:

- Use the Read tool to understand the file's contents and context.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly instructed to do so.

Directory Verification (only applicable when creating new files):

Use the LS tool to verify the parent directory exists and is the correct location

Usage:
- file_path: Path to the file to write (absolute or relative path)
- content: The content to write to the file

The tool will automatically create parent directories if they don't exist.

CRITICAL: Always use absolute paths (starting with /)"""

STR_REPLACE_EDIT_DESCRIPTION = """A versatile text editor tool for viewing, editing, creating, and inserting content in files.

This tool provides multiple commands for different file operations:

Commands:
- view: Display file contents with line numbers or list directory contents
- str_replace: Replace exact text matches in files (safer than edit_file for single replacements)
- create: Create new files with specified content
- insert: Insert text at specific line numbers

Usage:
- command: The operation to perform (view, str_replace, create, insert)
- path: File or directory path (use absolute paths)
- old_str: Exact string to replace (for str_replace)
- new_str: Replacement string (for str_replace/insert)
- view_range: [start_line, end_line] for viewing specific lines (1-indexed)
- file_text: Content for new file (for create)
- insert_line: Line number after which to insert (for insert, 0-indexed)

Examples:
- View file: command="view", path="/path/to/file.py"
- View specific lines: command="view", path="/path/to/file.py", view_range=[10, 20]
- Replace text: command="str_replace", path="/path/to/file.py", old_str="old text", new_str="new text"
- Create file: command="create", path="/path/to/new.py", file_text="print('hello')"
- Insert line: command="insert", path="/path/to/file.py", insert_line=5, new_str="new line content"

CRITICAL: Always use absolute paths (starting with /)"""

TASK_DESCRIPTION_PREFIX = """Launch a new agent to handle complex, multi-step tasks autonomously. 

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. (Tools: *)
{other_agents}
"""

TASK_DESCRIPTION_SUFFIX = """When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

When to use the Agent tool:
- When you are instructed to execute custom slash commands. Use the Agent tool with the slash command invocation as the entire prompt. The slash command can take arguments. For example: Task(description="Check the file", prompt="/check-file path/to/file.py")

When NOT to use the Agent tool:
- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly
- If you are searching for a specific term or definition within a known location, use the Glob tool instead, to find the match more quickly
- If you are searching for content within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above

Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement."""