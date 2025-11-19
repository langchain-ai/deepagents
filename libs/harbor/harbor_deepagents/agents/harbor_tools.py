"""Harbor-specific tools for DeepAgents to interact with the environment."""

from typing import Annotated

from langchain_core.tools import tool

from harbor.environments.base import BaseEnvironment


def create_harbor_environment_tools(environment: BaseEnvironment) -> list:
    """Create tools that allow DeepAgent to interact with Harbor environment.

    These tools wrap Harbor's environment API to provide:
    - Bash command execution
    - File reading
    - File writing

    Args:
        environment: Harbor environment instance (Docker, Modal, etc.)

    Returns:
        List of LangChain tool functions
    """

    @tool
    async def bash(command: Annotated[str, "The bash command to execute"]) -> str:
        """Execute a bash command in the task environment.

        Use this for running Python scripts, installing packages, listing files,
        checking output, etc. Returns stdout, stderr, and exit code.

        Examples:
        - bash("python script.py")
        - bash("ls -la /app")
        - bash("cat file.txt")
        """
        result = await environment.exec(command)

        output_parts = []
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout.strip()}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.strip()}")
        output_parts.append(f"[exit_code] {result.return_code}")

        return "\n\n".join(output_parts)

    @tool
    async def read_file(path: Annotated[str, "Path to the file to read"]) -> str:
        """Read the contents of a file from the task environment.

        Provide the full path to the file (e.g., /app/script.py).

        Returns the file contents as a string.
        """
        result = await environment.exec(f"cat {path}")

        if result.return_code != 0:
            return f"Error reading file: {result.stderr}"

        return result.stdout or ""

    @tool
    async def write_file(
        path: Annotated[str, "Path to the file to write"],
        content: Annotated[str, "Content to write to the file"],
    ) -> str:
        """Write content to a file in the task environment.

        This will create or overwrite the file. Provide the full path
        and the content to write.

        Returns success message or error.
        """
        # Escape content for heredoc
        escaped_content = content.replace("'", "'\"'\"'")

        result = await environment.exec(f"cat > {path} << 'EOF'\n{content}\nEOF")

        if result.return_code != 0:
            return f"Error writing file: {result.stderr}"

        return f"Successfully wrote {len(content)} bytes to {path}"

    return [bash, read_file, write_file]
