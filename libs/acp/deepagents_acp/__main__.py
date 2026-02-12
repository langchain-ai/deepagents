import asyncio

from deepagents_acp.server import _serve_test_agent


def main():
    asyncio.run(_serve_test_agent())


if __name__ == "__main__":
    main()
