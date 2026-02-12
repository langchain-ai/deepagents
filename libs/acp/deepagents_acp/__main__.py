import argparse
import asyncio
import os

from deepagents_acp.server import _serve_test_agent


def main():
    parser = argparse.ArgumentParser(description="Run ACP DeepAgent with specified root directory")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root directory accessible to the agent (default: current working directory)",
    )
    args = parser.parse_args()
    root_dir = args.root_dir if args.root_dir else os.getcwd()
    asyncio.run(_serve_test_agent(root_dir))


if __name__ == "__main__":
    main()
