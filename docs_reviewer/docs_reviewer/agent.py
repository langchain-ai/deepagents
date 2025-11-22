"""Code execution via bash with uv/uvx for isolated environments."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional
from langchain_core.tools import tool


def create_code_execution_tool():
    """
    Create a tool that executes code snippets using bash.

    For Python: Uses uv/uvx to create isolated environments with dependencies
    For JavaScript: Uses node directly
    For Bash: Executes directly

    This tool will be called multiple times in parallel by the main agent.
    """

    @tool
    def execute_code_snippet(
        code: str,
        language: str,
        dependencies: Optional[list[str]] = None,
        python_version: Optional[str] = None
    ) -> str:
        """
        Execute a code snippet in an isolated environment and validate it works.

        This tool runs code locally using bash. For Python, it uses uv/uvx to create
        isolated environments with specified dependencies.

        Args:
            code: The code to execute
            language: Programming language (python, javascript, bash, etc.)
            dependencies: List of package dependencies (e.g., ["pandas", "requests"])
            python_version: Python version to use (e.g., "3.11", "3.12")

        Returns:
            JSON string with execution results: {"success": bool, "output": str, "error": str}
        """
        dependencies = dependencies or []

        try:
            if language.lower() in ('python', 'py'):
                return _execute_python(code, dependencies, python_version)
            elif language.lower() in ('javascript', 'js'):
                return _execute_javascript(code, dependencies)
            elif language.lower() in ('bash', 'sh'):
                return _execute_bash(code)
            else:
                return f'{{"success": false, "output": "", "error": "Unsupported language: {language}"}}'

        except Exception as e:
            return f'{{"success": false, "output": "", "error": "Execution error: {str(e)}"}}'

    def _execute_python(code: str, dependencies: list[str], python_version: Optional[str]) -> str:
        """Execute Python code using uvx for isolated environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_file = tmp_path / "script.py"
            script_file.write_text(code)

            # Build uvx command
            if dependencies:
                # Use uvx to run with dependencies
                deps_args = " ".join([f"--with {dep}" for dep in dependencies])
                python_flag = f"--python {python_version}" if python_version else ""
                cmd = f"uvx {python_flag} {deps_args} python {script_file}"
            else:
                # Just use uv run for simple execution
                python_flag = f"--python {python_version}" if python_version else ""
                cmd = f"uv run {python_flag} python {script_file}"

            # Execute
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir
            )

            success = result.returncode == 0
            output = result.stdout.strip()
            error = result.stderr.strip()

            # Format as JSON string
            import json
            return json.dumps({
                "success": success,
                "output": output,
                "error": error
            })

    def _execute_javascript(code: str, dependencies: list[str]) -> str:
        """Execute JavaScript code using node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_file = tmp_path / "script.js"

            # If dependencies, create package.json and install
            if dependencies:
                package_json = {
                    "type": "module",
                    "dependencies": {dep: "latest" for dep in dependencies}
                }
                import json
                (tmp_path / "package.json").write_text(json.dumps(package_json, indent=2))

                # Install dependencies
                install_result = subprocess.run(
                    "npm install",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=tmpdir
                )

                if install_result.returncode != 0:
                    return json.dumps({
                        "success": False,
                        "output": "",
                        "error": f"Failed to install dependencies: {install_result.stderr}"
                    })

            script_file.write_text(code)

            # Execute
            result = subprocess.run(
                f"node {script_file}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir
            )

            success = result.returncode == 0
            output = result.stdout.strip()
            error = result.stderr.strip()

            import json
            return json.dumps({
                "success": success,
                "output": output,
                "error": error
            })

    def _execute_bash(code: str) -> str:
        """Execute bash code directly."""
        result = subprocess.run(
            code,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        success = result.returncode == 0
        output = result.stdout.strip()
        error = result.stderr.strip()

        import json
        return json.dumps({
            "success": success,
            "output": output,
            "error": error
        })

    return execute_code_snippet


class DocsReviewerAgent:
    """Agent that reviews docs by calling execute_code_snippet tool in parallel."""

    def __init__(self):
        """Initialize the agent - no longer needed, kept for compatibility."""
        pass

    def review_snippets(
        self,
        markdown_file: Path,
        snippets: list['CodeSnippet'],
    ) -> list[dict[str, Any]]:
        """
        Review snippets by returning them for the CLI agent to process.

        The CLI agent will call execute_code_snippet tool for each snippet,
        and LangGraph will handle parallel execution automatically.

        Args:
            markdown_file: Path to markdown file
            snippets: List of code snippets

        Returns:
            List of snippet metadata (execution happens via tool calls)
        """
        from docs_reviewer.markdown_parser import filter_executable_snippets

        # Just return executable snippets - the main agent will execute them
        executable = filter_executable_snippets(snippets)

        return [
            {
                "snippet_index": i,
                "code": s["code"],
                "language": s["language"],
                "start_line": s["start_line"],
                "end_line": s["end_line"],
            }
            for i, s in enumerate(executable)
        ]
