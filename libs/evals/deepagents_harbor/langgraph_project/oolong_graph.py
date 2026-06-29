"""LangGraph entrypoint for the OOLONG code-interpreter (RLM) agent under Harbor.

Registers ``oolong_code_interpreter`` in the shared ``langgraph.json``: a Deep
Agent (PR #4213's RLM arm) that reads ``/app/context.txt``, fans out contiguous
line-range reads/classification to ``general-purpose`` ``task`` subagents from
inside a QuickJS ``eval`` program (``CodeInterpreterMiddleware``), aggregates in
JavaScript, and writes the final answer to ``/app/answer.txt`` (the file the
Harbor verifier grades), via a ``LocalShellBackend`` shared with its subagents.

The plain (no-code-interpreter) baseline is just the existing ``bare_deepagent``
graph run against the OOLONG dataset — no separate graph needed.

Kept separate from ``langgraph_agent.py`` on purpose: that module imports
``deepagents_code`` and ``langchain_mcp_adapters`` at import time, which this
graph doesn't need. ``langchain_quickjs`` is imported lazily in the factory, so
loading this module never requires QuickJS until the graph is built.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.middleware.subagents import SubAgent

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


def make_oolong_code_interpreter_graph(config: dict[str, object] | None = None) -> object:
    """OOLONG code-interpreter (RLM) agent: fan-out + aggregation inside a QuickJS ``eval``.

    The agent dispatches ``general-purpose`` subagents via ``task(...)`` from inside
    an ``eval`` program (``CodeInterpreterMiddleware``) and aggregates in JavaScript.
    The orchestrator and subagents share one ``LocalShellBackend`` rooted at the
    sandbox workdir, so the subagents read the same on-disk ``/app/context.txt`` and
    the answer is written to ``/app/answer.txt``. ``langchain_quickjs`` is imported
    here, not at module load, so loading this module never requires QuickJS. The caps
    mirror PR #4213: a per-``eval`` timeout long enough for a ``Promise.all`` of
    subagent dispatches, and a ceiling on ``eval`` calls so a retry-spiral can't run
    away.

    Args:
        config: LangGraph runtime config. Harbor passes the root model in
            ``configurable.model`` (or ``HARBOR_MODEL``); the subagent model comes
            from ``configurable.sub_model`` / ``OOLONG_SUB_MODEL``, defaulting to
            the root model.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    from langchain_quickjs import CodeInterpreterMiddleware  # noqa: PLC0415

    # list[Any] so the (lazily imported) middleware type satisfies create_deep_agent's
    # Sequence[AgentMiddleware] parameter without a hard dependency on its type.
    middleware: list[Any] = [CodeInterpreterMiddleware(timeout=300.0, max_ptc_calls=12)]
    configurable = _configurable(config)
    root_model_id = _model_name(configurable)
    model = init_chat_model(root_model_id, **_model_kwargs(configurable))
    subagents: list[SubAgent] = [
        {
            "name": "general-purpose",
            "description": (
                "Reads a line range of /app/context.txt and extracts per-line facts as directed."
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
            system_prompt=_OOLONG_CODE_INTERPRETER_SYSTEM_PROMPT,
            subagents=subagents,
            middleware=middleware,
        )
