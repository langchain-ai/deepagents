#!/usr/bin/env python3
"""
Demo: Ralph Loop with DeepAgents
"""

import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from ralph import ralph_loop


def main():
    with tempfile.TemporaryDirectory() as work_dir:
        print(f"Working in: {work_dir}\n")

        # Create agent with persistent filesystem
        agent = create_deep_agent(
            backend=FilesystemBackend(root_dir=work_dir),
        )

        # Run Ralph loop
        result = ralph_loop(
            agent,
            task="""Create a Python calculator:
1. calculator.py with add, subtract, multiply, divide functions
2. test_calculator.py with tests for each function
3. Run the tests to verify they pass

Output <promise>DONE</promise> when complete.""",
            max_iterations=5,
        )

        # Show results
        print(f"\n{'='*50}")
        print(f"Iterations: {result['iterations']}")
        print(f"Completed: {result['completed']}")
        print(f"\nFiles created:")
        for f in Path(work_dir).rglob("*.py"):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
