"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_WORKDIR = Path("/app")

# Cap the agent's per-call working context so long agentic runs don't slow every
# turn / degrade reasoning on a huge context (the dominant cause of timeouts).
# Read by deepagents' `compute_summarization_defaults`; `setdefault` lets a
# workflow `--agent-env` override them.
_SUMMARIZE_TRIGGER_TOKENS = "50000"
_SUMMARIZE_KEEP_TOKENS = "15000"


def _apply_context_budget() -> None:
    """Set the summarization trigger/keep budget before the agent is built."""
    os.environ.setdefault("DEEPAGENTS_SUMMARIZE_TRIGGER_TOKENS", _SUMMARIZE_TRIGGER_TOKENS)
    os.environ.setdefault("DEEPAGENTS_SUMMARIZE_KEEP_TOKENS", _SUMMARIZE_KEEP_TOKENS)


_SHELL_ENV_DENYLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "BASETEN_API_KEY",
        "FIREWORKS_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
        "NVIDIA_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    }
)

_SYSTEM_PROMPT = """You are an autonomous coding agent in a sandboxed environment with shell and
filesystem access. No human is available to answer questions: make reasonable
assumptions and keep working until the task is fully complete, rather than
stopping to report what you would do.

Work from the sandbox working directory (run `pwd` if unsure). Prefer
non-interactive command flags; never run a command that waits for human input.

## Match the spec exactly

File paths, filenames, field names, identifiers, and output formats must match the
task wording character-for-character — `value` is not `val`, `/app/result.txt` is
not `/app/results.txt`. If the task defines a schema or names an output file, copy
it verbatim; do not rename or "improve" it.

## Let code do the work, not prose

Use your reply to decide the approach, not to carry it out. You have a limited
output budget — deriving results, simulating logic, or hand-writing file contents
in your reasoning will exhaust it before anything reaches disk. Instead:

- Compute, test, and verify by writing a script and running it in the shell, then
  reading the result. Code is your scratchpad for trial-and-error.
- Generate large or repetitive files (data, generated code, long transformed text)
  with a script that writes them — never by typing the contents out yourself,
  whether in a message or as a `write_file` argument.
- Keep reasoning short and decision-focused, and act early rather than thinking at
  length before your first tool call. When a task needs multiple steps, plan them as
  concrete actions to run ("write and run `encoder.py`"), not thinking steps
  ("analyze", "derive") — a plan should commit you to executing, not deliberating.

## Use the right tool

Prefer dedicated tools over raw shell: `read_file` over `cat`, `write_file` over
`echo`/heredoc, `edit_file` over `sed`/`awk`, `grep`/`glob` over shell equivalents.
Read large files with pagination. Make independent tool calls in parallel.

## When you get stuck

Read the full error output, not just the first line — the cause is often mid-trace.
Reproduce a failure before fixing it, and change one thing at a time. If the same
step fails repeatedly, stop and find the root cause instead of retrying the same
approach. If you cannot fully converge before the run ends, write your best-effort
artifact rather than leaving nothing.

## Verify against the real check, not your own

Two kinds of mistakes sink a task: getting the behavior wrong, and getting a name, path,
or format wrong. Guard against both before you finish.

Behavior — run it the way the grader will. If the task ships tests, examples, or sample
input/output, run them; they are your success signal. Otherwise turn a concrete example from
the task into a runnable check and iterate until it passes - the exact command, from a clean
directory. Derive what you check from the task's literal wording, not your own assumptions: a
self-written test that passes proves nothing if it asserts a field, path, or value you invented.

Contract — when the task names exact identifiers (file names, paths, field/message/type/ function/class
names, schema, wire format, or output format), call `verify_implementation` before concluding. It re-reads
the task and your files with a fresh independent checker and reports exact-match mismatches your tests can miss.
Treat any FAIL as a real defect: fix and re-run until it returns PASS.


## Finish with a verified deliverable

If the task asks for a file or on-disk output, that artifact must exist before you
stop — confirm it with the shell (`ls`, `cat`). Never end having only described or
planned the deliverable.
"""


@contextmanager
def _scrub_shell_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in _SHELL_ENV_DENYLIST}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _configurable(config: dict[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    value = config.get("configurable")
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`configurable` must be a dictionary"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _model_kwargs(configurable: dict[str, object]) -> dict[str, Any]:
    value = configurable.get("model_kwargs")
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`configurable.model_kwargs` must be a dictionary"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _model_name(configurable: dict[str, object]) -> str:
    value = configurable.get("model") or os.environ.get("HARBOR_MODEL")
    if not isinstance(value, str) or not value.strip():
        msg = "`configurable.model` or `HARBOR_MODEL` must provide a model name"
        raise ValueError(msg)
    return value


def _workdir(configurable: dict[str, object]) -> Path:
    value = configurable.get("cwd")
    if value is None:
        return _DEFAULT_WORKDIR
    if not isinstance(value, str | Path):
        msg = "`configurable.cwd` must be a string path"
        raise TypeError(msg)
    return Path(value)


def make_graph(config: dict[str, object] | None = None) -> object:
    """Create the Deep Agents Code CLI harness graph Harbor should run.

    Harbor's installed `langgraph` agent loads this factory from
    `langgraph.json` inside each benchmark sandbox. The returned value is the
    LangGraph graph produced by Deep Agents Code's headless constructor.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model` and optional provider kwargs in
            `configurable.model_kwargs`.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    _apply_context_budget()
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    assistant_id = os.environ.get("HARBOR_SESSION_ID") or f"harbor-{uuid.uuid4()}"
    with _scrub_shell_env():
        graph, _backend = create_cli_agent(
            model=model,
            assistant_id=assistant_id,
            sandbox=None,
            sandbox_type="harbor",
            system_prompt=_SYSTEM_PROMPT,
            interactive=False,
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=True,
            enable_verify=True,
            cwd=_workdir(configurable),
        )
    return graph


def make_bare_graph(config: dict[str, object] | None = None) -> object:
    """Create a Deep Agents SDK graph Harbor should run directly.

    This path avoids the Deep Agents Code CLI harness while still attaching a
    local shell backend rooted at Harbor's sandbox workdir so terminal-bench
    tasks can use filesystem and command execution tools.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model` and optional provider kwargs in
            `configurable.model_kwargs`.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    _apply_context_budget()
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
    )
