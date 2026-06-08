"""A DeepAgent that runs *inside* a LangSmith sandbox.

This is a plain `deepagents` agent built with `create_deep_agent`. It uses
`LocalShellBackend`, so its file edits and shell commands act on the box's real
filesystem under /app. Swap in any deepagents-based agent you like, this is just
a stand-in.

The whole agent process (and its LLM calls) runs in the sandbox. The only thing
leaving the box is the request to the model provider, which the sandbox's auth
proxy authenticates, so the box never holds a real model API key.
"""

# Trust the sandbox egress-proxy CA. The proxy injects credentials (e.g. the
# Anthropic key) into outbound HTTPS by terminating TLS with its own CA, which
# the sandbox installs into the OS trust store at boot. `truststore` makes
# Python's ssl, and therefore httpx and the Anthropic SDK, verify against that
# OS store instead of the bundled certifi roots.
import truststore

truststore.inject_into_ssl()

import sys  # noqa: E402

from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends.local_shell import LocalShellBackend  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402

SYSTEM_PROMPT = (
    "You are an autonomous coding agent working inside a sandbox at /app. "
    "Implement the requested change directly in the files using your file tools, "
    "then use the execute tool to run the test suite and verify your work. "
    "Keep iterating until all tests pass. Be concise."
)

TASK = (
    "/app/solution.py defines a function `roman_to_int(s)` that is not yet "
    "implemented. Implement it so it converts a Roman numeral string to an "
    "integer. The tests in /app/test_solution.py must pass. Run "
    "`python -m pytest test_solution.py -q` to check, and keep iterating until "
    "every test passes."
)


def main() -> int:
    print("[agent] building a deepagent on LocalShellBackend(root_dir=/app)")
    model = ChatAnthropic(model_name="claude-sonnet-4-6")
    # virtual_mode=False: operate on the box's real filesystem under /app. The
    # sandbox itself is the isolation boundary. Set explicitly since the default
    # is changing in a future deepagents release.
    backend = LocalShellBackend(root_dir="/app", inherit_env=True, virtual_mode=False)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
    )

    print("[agent] invoking on task...")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": TASK}]},
        config={"recursion_limit": 100},
    )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        last = messages[-1]
        print("[agent] final message:\n", getattr(last, "content", last))
    print("[agent] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
