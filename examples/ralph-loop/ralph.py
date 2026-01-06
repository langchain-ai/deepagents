"""
Ralph Loop for DeepAgents

Ralph is a simple pattern: run an agent in a loop until it signals completion.
The agent's work persists in the filesystem, so each iteration builds on the last.

That's it. That's Ralph.
"""

import re


def ralph_loop(agent, task: str, max_iterations: int = 10):
    """
    Run an agent in a Ralph loop.

    Args:
        agent: A DeepAgent with FilesystemBackend
        task: The task to accomplish
        max_iterations: Safety limit

    Returns:
        dict with iterations count and completion status
    """
    iteration = 1

    while iteration <= max_iterations:
        print(f"\n{'='*50}")
        print(f"RALPH ITERATION {iteration}")
        print(f"{'='*50}")

        # Build prompt with iteration context
        prompt = f"""## Iteration {iteration}/{max_iterations}

Your previous work persists in the filesystem.
Check existing files and continue from where you left off.

TASK:
{task}

When complete, output: <promise>DONE</promise>"""

        # Same prompt structure, every iteration
        result = agent.invoke({
            "messages": [{"role": "user", "content": prompt}]
        })

        # Get response text
        response = _get_response(result)

        # Check for completion
        if "<promise>DONE</promise>" in response.upper().replace(" ", ""):
            print(f"\n✓ DONE - Completed in {iteration} iterations")
            return {"iterations": iteration, "completed": True}

        print("Not done yet, continuing...")
        iteration += 1

    print(f"\n✗ Max iterations ({max_iterations}) reached")
    return {"iterations": iteration - 1, "completed": False}


def _get_response(result: dict) -> str:
    """Extract text from agent result."""
    messages = result.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    if hasattr(last, "content"):
        return str(last.content)
    return str(last)
