"""Common middleware constants shared across filesystem middlewares."""

# Message used when evicting a large tool result to the in-memory filesystem
TOO_LARGE_TOOL_MSG = (
    "Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}\n"
    "You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.\n"
    "You can do this by specifying an offset and limit in the read_file tool call.\n"
    "For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.\n\n"
    "Here are the first 10 lines of the result:\n{content_sample}\n"
)

# Supplement lines for tools that aren't listed in the base filesystem system prompt
FILESYSTEM_SYSTEM_PROMPT_GLOB_GREP_SUPPLEMENT = (
    "\n- glob: find files/directories by pattern\n"
    "- grep: search file contents using ripgrep (rg)"
)
