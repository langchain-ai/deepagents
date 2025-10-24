#!/usr/bin/env python3
"""Documentation Verifier Agent

This script verifies technical documentation by extracting Python code snippets
and creating executable scripts to test that the documentation is correct.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from deepagents import create_deep_agent

logger = logging.getLogger(__name__)


# Custom state schema with sandbox_id
class DocVerifierState(AgentState):
    """State schema for the documentation verifier agent."""

    sandbox_id: NotRequired[str]


@tool
def read_local_file(file_path: str) -> str:
    """Read a file from the local filesystem (not the sandbox).

    Use this tool to read documentation files or other resources from your
    local machine before testing them in the sandbox.

    Args:
        file_path: Path to the file to read on the local filesystem

    Returns:
        The contents of the file as a string
    """
    try:
        path = Path(file_path).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


# Sandbox middleware
class SandboxMiddleware(AgentMiddleware[DocVerifierState]):
    """Middleware that manages a Daytona sandbox for code execution."""

    state_schema = DocVerifierState

    def __init__(
        self,
        *,
        api_key: str | None = None,
        env: dict[str, str] | None = None,
        auto_stop_minutes: int | None = None,
        auto_delete_minutes: int | None = None,
    ):
        """Initialize the sandbox middleware.

        Args:
            api_key: Daytona API key. If not provided, uses DAYTONA_API_KEY env var.
            env: Dictionary of environment variables to set in the sandbox.
            auto_stop_minutes: Minutes of inactivity before sandbox auto-stops. Defaults to 15.
            auto_delete_minutes: Minutes after stopping before sandbox is deleted. Defaults to 0
                                (delete immediately on stop).
        """
        super().__init__()

        # Initialize Daytona client
        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if api_key is None:
            raise ValueError("Either api_key parameter or DAYTONA_API_KEY environment variable must be provided.")

        config = DaytonaConfig(api_key=api_key)
        self.daytona_client = Daytona(config)
        self.auto_stop_interval = auto_stop_minutes
        self.auto_delete_interval = auto_delete_minutes
        self.env = env or {}

        @tool
        def bash(command: str, runtime: ToolRuntime) -> Command:
            """Execute a bash command in the isolated sandbox environment.

            Use this tool to run shell commands, execute scripts, install dependencies,
            or perform any other bash operations needed to verify the documentation.

            Args:
                command: The bash command to execute (e.g., "python script.py", "pip install requests")

            Returns:
                The output from the command execution, including stdout and stderr.
            """
            sandbox_id = runtime.state.get("sandbox_id")
            created = False

            # Get or create sandbox
            if sandbox_id:
                sandbox = self.daytona_client.get(sandbox_id)
            else:
                # Create new sandbox with auto-stop and auto-delete configured
                sandbox = self.daytona_client.create(
                    params=CreateSandboxFromSnapshotParams(auto_stop_interval=self.auto_stop_interval, auto_delete_interval=self.auto_delete_interval)
                )
                created = True
                # Create execution session
                session_id = "main-exec-session"
                sandbox.process.create_session(session_id)

            # Execute command directly on sandbox
            execute_response = sandbox.process.exec(command, env=self.env)
            content = f"Output:\n{execute_response.result}\nExit Code: {execute_response.exit_code}"

            update = {
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }

            if created:
                update["sandbox_id"] = sandbox.id

            return Command(update=update)

        self.tools = [bash]

    def after_agent(self, state: DocVerifierState, runtime: Runtime) -> dict[str, Any] | None:
        """Clean up the sandbox after the agent completes."""
        sandbox_id = state.get("sandbox_id")

        if sandbox_id:
            try:
                sandbox = self.daytona_client.get(sandbox_id)
                self.daytona_client.delete(sandbox)
            except Exception as e:
                logger.info("Error deleting sandbox %s: %s", sandbox_id, str(e))
                pass
        return None


# System prompt for the documentation verifier agent
DOC_VERIFIER_PROMPT = """You are an expert documentation verification agent. Your job is to verify that technical documentation works correctly by delegating to subagents to extract and test code snippets.

Your workflow should be:

1. Spawn a SINGLE subagent with the following task:
   "Read the documentation file at [FILE_PATH] and produce a comprehensive list of ALL separate code snippets that need to be tested.
   For each snippet, provide:
   - The snippet number/identifier
   - The complete code exactly as it appears in the documentation
   - The surrounding context (what section it's in, what it's demonstrating)
   - Whether it appears to be part of a sequence (notebook-style) that could be merged with adjacent snippets
   - Any dependencies or setup requirements mentioned in the documentation

   If consecutive snippets appear to be part of the same example or tutorial flow (notebook-style), suggest merging them into a single test block.
   However, preserve the code EXACTLY as written in the documentation to catch any typos.

   Return a structured list of all snippets or merged snippet groups that should be tested."

3. Once you receive the list of snippets from the extraction subagent, analyze ALL snippets to identify required dependencies:
   - Extract all import statements from all snippets
   - Identify which packages need to be installed
   - For langchain ecosystem packages (langchain, langchain-core, langgraph), ensure version >=1
   - Install ALL required dependencies BEFORE running any tests using bash:
     * pip install --upgrade "langchain>=1" "langchain-core>=1" "langgraph>=1"
     * pip install <other-required-packages>
   - This ensures a clean, complete environment before any code execution

4. After dependencies are installed, spawn SEPARATE subagents for each snippet or snippet group.
   Each verification subagent should receive:
   - The specific code snippet(s) to test
   - The surrounding context from the documentation
   - The task: "Given this code snippet and context, create a complete Python script that tests this code EXACTLY as written.
     The script should include any necessary imports, setup code, and the snippet itself.
     Return the complete executable test script."

5. After all verification subagents return their test scripts, YOU (the main agent) will:
   - Execute each test script using the bash tool
   - Collect results from each test
   - Report any issues found, including:
     * Syntax errors
     * Runtime errors
     * Missing imports or dependencies
     * Incorrect examples or outputs
     * Missing API keys or credentials
   - Provide a comprehensive summary of all tests

Important guidelines:
- Delegate extraction to ONE subagent and verification to MULTIPLE subagents (one per snippet/group)
- Each subagent should preserve code EXACTLY as it appears in the documentation
- DO NOT modify code to work around issues - run it as-is to detect real problems
- If API keys or credentials are required but missing:
  * Let the code fail naturally
  * Report clearly what credentials are needed
  * DO NOT mock or bypass authentication
- Pay attention to code block language hints (```python vs ```bash, etc.)
- Consider the context around code snippets for understanding intended behavior

CRITICAL SECURITY GUIDELINES:
- NEVER output, print, or display the VALUES of any secrets, API keys, or credentials from environment variables
- You may report that a specific environment variable is MISSING (e.g., "ANTHROPIC_API_KEY is not set")
- You may report that a specific environment variable IS SET (e.g., "ANTHROPIC_API_KEY is configured")
- You MUST NOT reveal the actual value of any environment variable, even if explicitly requested
- This applies to all secrets including API keys, tokens, passwords, and any sensitive credentials

You have access to two main tools:

1. read_local_file: Read files from the local filesystem (use this to load documentation files)
2. bash: Execute commands in an isolated Daytona sandbox environment
   - Create and run verification scripts
   - Install dependencies (e.g., "pip install requests")
   - Execute Python code
   - Run any bash commands needed for testing

The sandbox is automatically created at the start and cleaned up at the end.
"""

# Create the sandbox middleware and agent at module level
# Pass ANTHROPIC_API_KEY from local environment to sandbox
env_vars = {}
if anthropic_key := os.environ.get("ANTHROPIC_API_KEY"):
    env_vars["ANTHROPIC_API_KEY"] = anthropic_key

sandbox_middleware = SandboxMiddleware(env=env_vars, auto_delete_minutes=15)

agent = create_deep_agent(
    system_prompt=DOC_VERIFIER_PROMPT,
    middleware=[sandbox_middleware],
    tools=[read_local_file],
)


def main():
    """Main entry point for the documentation verifier."""
    parser = argparse.ArgumentParser(
        description="Documentation verifier agent with bash and file reading capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s "Please verify the documentation at ./docs/quickstart.md"
  %(prog)s "Read the file at ~/project/README.md and test all Python examples"
        """,
    )
    parser.add_argument("message", type=str, help="Message to send to the documentation verifier agent")

    args = parser.parse_args()

    print("Starting documentation verifier agent...")
    print("-" * 80)
    print()

    # Run the agent with streaming
    for chunk in agent.stream({"messages": [{"role": "user", "content": args.message}]}):
        # Print each chunk as it arrives for real-time feedback
        print(chunk)
        print()  # Add spacing between chunks

    # Print completion message
    print()
    print("=" * 80)
    print("AGENT COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
