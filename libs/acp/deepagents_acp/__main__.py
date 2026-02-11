import argparse
import asyncio
import os

from deepagents_acp.server import serve_acp_stdio


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
    asyncio.run(serve_acp_stdio(root_dir))


if __name__ == "__main__":
    main()
