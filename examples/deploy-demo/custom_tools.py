"""Example custom tools for the deploy demo.

This file is referenced by deepagents.json as "custom": "./custom_tools.py:tools"
and gets bundled into the deployment.
"""

from langchain_core.tools import tool


@tool
def get_project_info() -> str:
    """Get information about the current project configuration."""
    return "Project: deploy-demo | Framework: deepagents | Status: deployed"


@tool
def check_code_style(code: str) -> str:
    """Check if the given code follows basic style guidelines.

    Args:
        code: The code to check.
    """
    issues = []
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            issues.append(f"Line {i}: exceeds 120 characters ({len(line)} chars)")
        if line.rstrip() != line:
            issues.append(f"Line {i}: trailing whitespace")
    if not issues:
        return "No style issues found."
    return "Style issues:\n" + "\n".join(issues)


# This is what deepagents deploy imports via the "custom_tools.py:tools" reference
tools = [get_project_info, check_code_style]
