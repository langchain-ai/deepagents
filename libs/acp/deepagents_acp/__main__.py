import argparse
import asyncio
import os

from deepagents_acp.agent import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run ACP DeepAgent with specified root directory")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root directory accessible to the agent (default: current working directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to use for the agent. "
            "Use 'provider:model' format (e.g. 'openai:gpt-5'). "
            "Defaults to Claude Sonnet 4.5 if not specified."
        ),
    )
    args = parser.parse_args()
    root_dir = args.root_dir if args.root_dir else os.getcwd()
    asyncio.run(run_agent(root_dir, model=args.model))


if __name__ == "__main__":
    main()
