"""LangGraph entrypoints for the OOLONG arms under Harbor.

Hosts the two OOLONG-synth arms from PR #4213, both registered in the shared
``langgraph.json``:

- ``oolong_plain`` — a Deep Agent that fans out contiguous line-range
  reads/classification to ``general-purpose`` ``task`` subagents and aggregates.
- ``oolong_code_interpreter`` (RLM) — same, but the agent orchestrates the
  fan-out and aggregation from inside a QuickJS ``eval`` program
  (``CodeInterpreterMiddleware``).

Both read ``/app/context.txt`` and write the final answer to ``/app/answer.txt``
(the file the Harbor verifier grades), via a ``LocalShellBackend`` shared with
their subagents.

Kept separate from ``langgraph_agent.py`` on purpose: that module imports
``deepagents_code`` and ``langchain_mcp_adapters`` at import time, which the
OOLONG arms don't need. The plain arm imports only the ``deepagents`` SDK; the
code-interpreter arm imports ``langchain_quickjs`` lazily, so loading this module
never requires QuickJS unless that arm is actually run.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import os

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_WORKDIR = Path("/app")

# Provider/tracing secrets removed from the shell env the agent's `execute` tool
# sees (the LLM client reads them from os.environ at call time, which is restored
# after graph construction). Defense in depth on top of inherit_env=False.
_SHELL_ENV_DENYLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
        "OPENAI_API_KEY",
    }
)

_OOLONG_PLAIN_SYSTEM_PROMPT = """You are a precise data analyst running in a Harbor \
benchmark sandbox. The document to analyze is at `/app/context.txt`. Answer the \
question in the exact format it requests.

Strategy: split the document into contiguous line-range chunks and fan them out to
`general-purpose` subagents in parallel — emit multiple `task` calls in one turn,
each given a line range to read and classify — then aggregate their per-line
results to compute the answer.

When you have the final answer, use `write_file` to write ONLY the answer to
`/app/answer.txt` in the exact format the question requests (for example
`Label: <answer>`). That file is the sole thing graded; finish only after writing it.
"""

_OOLONG_CODE_INTERPRETER_SYSTEM_PROMPT = """You are a precise data analyst running \
in a Harbor benchmark sandbox. The document to analyze is at `/app/context.txt`. \
Answer the question in the exact format it requests.

Strategy: split the document into contiguous line-range chunks and fan them out to
`general-purpose` subagents in parallel from inside a single `eval` program
(`Promise.all([...])` of `task(...)` calls, each given a line range to read and
classify), then aggregate their per-line results in JavaScript to compute the
answer.

When you have the final answer, use `write_file` to write ONLY the answer to
`/app/answer.txt` in the exact format the question requests (for example
`Label: <answer>`). That file is the sole thing graded; finish only after writing it.
"""

_OOLONG_SUBAGENT_SYSTEM_PROMPT = (
    "You analyze one assigned line range of `/app/context.txt`. Read the entire "
    "range with `read_file` and return the requested per-line result for every line "
    "in the range. Do not aggregate; do not delegate further."
)


@contextmanager
def _scrub_shell_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in _SHELL_ENV_DENYLIST}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
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


def _sub_model_id(configurable: dict[str, object], default: str) -> str:
    """Resolve the subagent model id (configurable.sub_model / OOLONG_SUB_MODEL / root)."""
    value = configurable.get("sub_model") or os.environ.get("OOLONG_SUB_MODEL")
    if isinstance(value, str) and value.strip():
        return value
    return default


def _build_oolong_graph(
    config: dict[str, object] | None,
    *,
    system_prompt: str,
    middleware: list[Any] | None = None,
) -> object:
    """Build an OOLONG Deep Agent for one arm.

    The orchestrator and its ``general-purpose`` subagents share a
    ``LocalShellBackend`` rooted at the sandbox workdir, so the subagents read the
    same on-disk ``/app/context.txt`` the orchestrator sees, and the answer is
    written to ``/app/answer.txt`` on the real container filesystem. The only
    difference between arms is ``system_prompt`` and ``middleware``.

    Args:
        config: LangGraph runtime config. Harbor passes the root model in
            ``configurable.model`` (or ``HARBOR_MODEL``); the subagent model comes
            from ``configurable.sub_model`` / ``OOLONG_SUB_MODEL``, defaulting to
            the root model.
        system_prompt: The arm's orchestrator prompt.
        middleware: Extra middleware for the arm (e.g. the code interpreter).

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    configurable = _configurable(config)
    root_model_id = _model_name(configurable)
    model = init_chat_model(root_model_id, **_model_kwargs(configurable))
    subagents = [
        {
            "name": "general-purpose",
            "description": (
                "Reads a line range of /app/context.txt and extracts per-line facts "
                "as directed."
            ),
            "system_prompt": _OOLONG_SUBAGENT_SYSTEM_PROMPT,
            "model": _sub_model_id(configurable, root_model_id),
        }
    ]
    with _scrub_shell_env():
        return create_deep_agent(
            model=model,
            # virtual_mode=False: absolute paths (/app/context.txt, /app/answer.txt)
            # resolve to the real container files, matching the baked task layout.
            backend=LocalShellBackend(
                root_dir=_workdir(configurable), inherit_env=False, virtual_mode=False
            ),
            system_prompt=system_prompt,
            subagents=subagents,
            middleware=middleware or [],
        )


def make_oolong_plain_graph(config: dict[str, object] | None = None) -> object:
    """OOLONG "plain" arm: the agent fans out to ``general-purpose`` subagents."""
    return _build_oolong_graph(config, system_prompt=_OOLONG_PLAIN_SYSTEM_PROMPT)


def make_oolong_code_interpreter_graph(config: dict[str, object] | None = None) -> object:
    """OOLONG "code interpreter" (RLM) arm: fan-out + aggregation inside a QuickJS ``eval``.

    Adds ``CodeInterpreterMiddleware`` (the ``eval`` tool); the agent dispatches
    ``general-purpose`` subagents via ``task(...)`` from JavaScript and aggregates
    in JS. ``langchain_quickjs`` is imported here, not at module load, so the plain
    arm never requires QuickJS. The caps mirror PR #4213: a per-``eval`` timeout
    long enough for a ``Promise.all`` of subagent dispatches, and a ceiling on
    ``eval`` calls so a retry-spiral can't run away.
    """
    from langchain_quickjs import CodeInterpreterMiddleware

    return _build_oolong_graph(
        config,
        system_prompt=_OOLONG_CODE_INTERPRETER_SYSTEM_PROMPT,
        middleware=[CodeInterpreterMiddleware(timeout=300.0, max_ptc_calls=12)],
    )
