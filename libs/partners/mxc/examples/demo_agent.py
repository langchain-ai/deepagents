"""Deep Agent + MXCSandbox demo.

The agent works on the tiny repo in ``examples/sample_repo`` (one failing
test). It is asked to fix the test and, in the same task, to read a file that
lives *outside* the repo. The shell read is blocked by the MXC sandbox and
the agent explains what happened.

Environment variables:

- ``MXC_DEMO_MODEL``: model string for ``create_deep_agent``; defaults to
  ``anthropic:claude-opus-5-5``. Use ``openai:<model>`` with an OpenAI key.
- ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``: the provider key for that model.
- ``MXC_EXECUTOR_PATH``: optional path to the MXC executor; otherwise it is
  discovered from ``node_modules/@microsoft/mxc-sdk``.

Without an API key the script still imports, builds the backend and the
agent, prints the resolved policy, and exits 0.

Pass ``--scripted`` to run the whole demo with no API key and no model call.
A scripted stand-in for the model issues a fixed list of tool calls, and the
real agent loop runs them through the real MXC sandbox, so every tool result
printed is genuine.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_mxc import MXCExecutorNotFoundError, MXCSandbox

EXAMPLES_DIR = Path(__file__).resolve().parent
SAMPLE_REPO = EXAMPLES_DIR / "sample_repo"
OUTSIDE_FILE = EXAMPLES_DIR / "outside_the_repo.txt"

DEFAULT_MODEL = "anthropic:claude-opus-5-5"

SYSTEM_PROMPT = """You are a careful coding agent working inside a sandboxed project folder.
Use the `execute` tool to run shell commands. The working directory is the project root.
When a command is blocked by the sandbox, do not try to work around it: say what was
blocked and that the user needs to grant access."""

TASK = f"""Two things, in order:

1. Run the test suite with `python -m pytest -q`. One test fails. Fix the bug in the code
   under test (not the test itself) and re-run until the suite is green.

2. Then use the `execute` tool to print the contents of this file, which is outside the project:
   {OUTSIDE_FILE}

Finish with a short report: what the bug was, what you changed, and exactly what happened
when you tried to read the outside file."""


BUGGY_LINE = "    return round(price - percent, 2)"
FIXED_LINE = "    return round(price * (1 - percent / 100), 2)"


class ScriptedModel(BaseChatModel):
    """Stand-in for a chat model that replays a fixed list of tool calls.

    It never contacts a model provider. Each time the agent asks for the next
    step it returns the next scripted message, so the agent loop, the tools and
    the sandbox all run for real.
    """

    steps: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-demo"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201, ARG002 - matches BaseChatModel
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN202, ARG002
        step = self.steps[min(self.position, len(self.steps) - 1)]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=step)])


def scripted_model() -> ScriptedModel:
    """The fixed plan: run tests, fix the bug, re-run, try the outside file, report."""
    read_outside = f'type "{OUTSIDE_FILE}"' if os.name == "nt" else f"cat '{OUTSIDE_FILE}'"

    def call(number: int, name: str, **args: str) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"scripted-{number}", "type": "tool_call"}])

    return ScriptedModel(
        steps=[
            call(1, "execute", command="python -m pytest -q"),
            call(2, "edit_file", file_path="/pricing.py", old_string=BUGGY_LINE, new_string=FIXED_LINE),
            call(3, "execute", command="python -m pytest -q"),
            call(4, "execute", command=read_outside),
            AIMessage(
                content="Scripted run finished. No model was called: the steps above were fixed in advance, "
                "and the tool results shown are the real output of the agent's tools and the MXC sandbox."
            ),
        ]
    )


def _provider(model: str) -> str:
    return model.split(":", 1)[0].lower() if ":" in model else "anthropic"


def has_api_key(model: str) -> bool:
    """Whether the environment holds a key for the model's provider."""
    keys = {
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "openai": ("OPENAI_API_KEY",),
        "azure_openai": ("AZURE_OPENAI_API_KEY",),
        "google_genai": ("GOOGLE_API_KEY",),
    }
    return any(os.environ.get(name) for name in keys.get(_provider(model), ()))


def copy_sample_repo(destination: Path | None = None) -> Path:
    """Copy the sample repo to a working location so the original stays failing."""
    if destination is None:
        destination = Path(tempfile.mkdtemp(prefix="mxc-demo-")) / "sample_repo"
    shutil.copytree(SAMPLE_REPO, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return destination


def build_backend(root_dir: Path) -> MXCSandbox:
    """The sandbox: writes limited to `root_dir`, network off."""
    # Observed on Windows: git, node, PowerShell and pytest cannot start while the
    # window system is locked, so the demo opens it there. Clipboard access and
    # input injection stay off.
    return MXCSandbox(root_dir=root_dir, timeout=180, allow_ui=sys.platform == "win32")


def build_agent(model: str | BaseChatModel | None = None, *, root_dir: Path | None = None):  # noqa: ANN201 - deep agent graph type is internal
    """Construct the Deep Agent. Works without an API key; the key is only needed to run it."""
    model = model or os.environ.get("MXC_DEMO_MODEL", DEFAULT_MODEL)
    root = root_dir or copy_sample_repo()
    backend = build_backend(root)
    agent = create_deep_agent(model=model, backend=backend, system_prompt=SYSTEM_PROMPT)
    return agent, backend, root


def _print_step(message: BaseMessage) -> None:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            args = call.get("args", {})
            summary = args.get("command") or args.get("file_path") or args.get("path") or args
            print(f"\n>>> {call['name']}: {summary}")
        text = message.text if isinstance(message.text, str) else ""
        if text.strip():
            print(f"\n{text.strip()}")
    elif isinstance(message, ToolMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        print("\n".join(f"    {line}" for line in content.strip().splitlines()[:40]))


def main() -> int:
    """Run the demo."""
    scripted = "--scripted" in sys.argv[1:]
    model = "scripted (no model call)" if scripted else os.environ.get("MXC_DEMO_MODEL", DEFAULT_MODEL)
    try:
        agent, backend, root = build_agent(scripted_model() if scripted else model)
    except MXCExecutorNotFoundError as exc:
        print(exc)
        return 2

    print(f"model:     {model}")
    print(f"root_dir:  {root}")
    print(f"executor:  {backend.executor_path}")
    print(f"outside:   {OUTSIDE_FILE}")
    policy = backend.policy
    network = "on" if policy.allow_network else "off"
    print(f"policy:    rw={len(policy.read_write_paths)} ro={len(policy.read_only_paths)} denied={len(policy.denied_paths)} network={network}")

    if not scripted and not has_api_key(model):
        print(f"\nNo API key found for provider {_provider(model)!r}. The agent and backend were constructed fine.")
        print("Set ANTHROPIC_API_KEY (or OPENAI_API_KEY with MXC_DEMO_MODEL=openai:<model>) to run the demo,")
        print("or pass --scripted to run it without a model.")
        return 0

    # Every tool call, tool result and model message is printed as it happens;
    # the agent's final report is the last message printed.
    seen = 0
    messages: list = []
    for chunk in agent.stream({"messages": [{"role": "user", "content": TASK}]}, stream_mode="values"):
        messages = chunk.get("messages", [])
        for message in messages[seen:]:
            _print_step(message)
        seen = len(messages)
    if scripted:
        return _check_scripted_run(messages)
    return 0


def _check_scripted_run(messages: list) -> int:
    """Fail loudly if the sandbox did not behave as the scripted steps expect."""
    results = [m.content if isinstance(m.content, str) else str(m.content) for m in messages if isinstance(m, ToolMessage)]
    expected_tool_results = 4
    problems = []
    if len(results) < expected_tool_results:
        problems.append(f"expected {expected_tool_results} tool results, got {len(results)}")
    else:
        if "1 failed" not in results[0]:
            problems.append("the first test run did not report the failing test")
        if "passed" not in results[2] or "failed" in results[2]:
            problems.append("the tests did not pass after the fix")
        if "Blocked by sandbox policy" not in results[3]:
            problems.append("reading the file outside the project was not reported as blocked")
    if problems:
        print("\nSCRIPTED RUN FAILED: " + "; ".join(problems) + ".")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
