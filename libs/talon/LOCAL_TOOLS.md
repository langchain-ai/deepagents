# Local Python tools (experimental)

Talon can load Python tools directly, without an MCP server. Set
`DEEPAGENTS_TALON_TOOLS_DIRS` to one or more trusted host directories, separated
by your platform's path separator (`:` on Linux/macOS, `;` on Windows):

```bash
DEEPAGENTS_TALON_TOOLS_DIRS=/opt/my-talon-tools \
AGENT_MODEL=<provider>:<model-id> uv run deepagents-talon
```

Relative paths resolve against Talon's startup working directory; `~` expands to
the operator's home. No directory is discovered or imported by default. The echo
runtime, used when no model is configured, does not import local tools.

Create `/opt/my-talon-tools/greetings.py`:

```python
from langchain_core.tools import tool


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
async def async_greet(name: str) -> str:
    """Greet someone asynchronously."""
    return f"Hello, {name}!"
```

Talon imports top-level `*.py` files in sorted filename order and exposes their
public `BaseTool` instances, including functions decorated with `@tool` and
instances of custom `BaseTool` subclasses. Ordinary functions and classes are
not exposed. Files and variables beginning with `_` are ignored, including
`__init__.py`; subdirectories are not scanned. Multiple aliases of the same tool
instance are exposed once. Install tool dependencies into Talon's Python
environment; the loader does not modify `sys.path` or install packages. Sibling
and relative imports are not provided; put shared helpers in an installed package.

Missing directories, import errors, duplicate tool names, and conflicts with
Talon or MCP tool names fail startup. Symlinked Python files must resolve inside
the configured directory. Send `/tools-reload` after adding, editing, or deleting
Python tool files to activate changes without restarting Talon. For example,
`DEEPAGENTS_TALON_TOOLS_DIRS=./tools` loads a folder in your repository.
On Slack, use `/talon tools-reload` in a DM or mention the bot with
`@Talon /tools-reload` in a channel thread. The command is also listed in `/help`
and registered as a Discord application command.

Reload reimports the configured files and validates the new tool catalog and
subagent attachments before activation. Import or validation failures leave the
previous tools active. MCP and subagent reloads preserve the latest loaded Python
tools. Successful reloads affect subsequent turns; running turns and background
tasks retain their original tools. Retired import generations remain registered
until shutdown, so repeated reloads can increase memory usage. Reload does not
cancel ongoing work or reset conversation history. Installed dependency modules
are not reloaded; restart Talon after changing those.

Embedding hosts can pass `tools_dirs=[Path("/opt/my-talon-tools")]` to
`DeepAgentRuntime`. This is a new optional keyword-only parameter; existing
`tools=` behavior is unchanged. Local tools enter the main agent's tool catalog.
Subagents can select them using their existing `tools: [greet]` frontmatter or
per-task tool selection; they are not attached to every subagent automatically.

## Trust and approvals

**Only load code you trust as much as Talon itself.** Imports execute arbitrary
Python on the host at startup and on every reload, with Talon's privileges and
access to process credentials. Reloading executes import-time side effects again;
failed reloads cannot undo those effects. Local tools also run on the host even
when shell and filesystem tools use a remote sandbox. Filesystem middleware does not constrain arbitrary
Python tool code. If the agent edits tools in your repository, review those edits
before requesting a reload: loading them grants Talon's full host privileges.
Do not point this setting at uploads, downloaded code, or other untrusted content.
The symlink check is not a sandbox or protection against a writer modifying code.

Existing exact-name approval settings in the assistant's `tools.json` apply to
local tool calls; unspecified tools follow the existing no-approval default.
Approvals do not gate import-time code. Custom tools must implement their own
input validation and resource cleanup; filesystem tools should use
`deepagents.backends.utils.validate_path` and enforce their own allowed roots.
