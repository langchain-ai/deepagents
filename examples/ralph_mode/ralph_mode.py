#!/usr/bin/env python3
"""
Ralph Mode - Single run for Deep Agents

Usage:
    uv pip install deepagents-cli
    python ralph_mode.py
    python ralph_mode.py --model claude-haiku-4-5-20251001
"""

import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import argparse
import asyncio
import tempfile
from pathlib import Path

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.config import create_model


async def ralph(model_name: str = None):
    """Run agent once with beautiful CLI output."""
    work_dir = tempfile.mkdtemp(prefix="ralph-")

    model = create_model(model_name)
    agent, backend = create_cli_agent(
        model=model,
        assistant_id="ralph",
        tools=[],
        auto_approve=True,
    )

    app = DeepAgentsApp(
        agent=agent,
        assistant_id="ralph",
        backend=backend,
        auto_approve=True,
        cwd=work_dir
    )
    await app.run_async()


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Mode - Single run for Deep Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ralph_mode.py
  python ralph_mode.py --model claude-haiku-4-5-20251001
        """
    )
    parser.add_argument("--model", help="Model to use (e.g., claude-haiku-4-5-20251001)")
    args = parser.parse_args()

    try:
        asyncio.run(ralph(args.model))
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C


if __name__ == "__main__":
    main()
