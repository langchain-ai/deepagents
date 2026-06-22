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

_SYSTEM_PROMPT = """You are running in a Harbor benchmark sandbox.

Complete the task autonomously. There is no human operator available to answer
follow-up questions, so make reasonable assumptions and keep working until the
task is complete.

Use the sandbox working directory for all file and shell operations. In Terminal
Bench-style tasks this is usually `/app`; use `pwd` if you need to confirm the
current directory.

Prefer non-interactive command variants. Do not run commands that wait for
human input.
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
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
    )


def make_eval_graph(config: dict[str, object] | None = None) -> object:
    """Create a Deep Agents graph for a specific eval.

    Reads `configurable["eval_name"]` to look up the corresponding
    :class:`~deepagents_evals.eval_registry.EvalSpec` and delegates to its
    `build()` method. This is the dispatcher entry point that lets a single
    `eval_langgraph.json` graph serve every eval in the suite — each Harbor task
    just passes its eval name via `--agent-kwarg configurable='{"eval_name":
    "test_write_file_simple"}'`.

    For evals that support the `repl_name` parameter (relational /
    incident-graph suites), pass `configurable["repl_name"]` (`"quickjs"`
    or omit for direct tool binding).

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model`, optional provider kwargs in
            `configurable.model_kwargs`, the eval name in
            `configurable.eval_name`, optionally `configurable.repl_name`, and
            optionally `configurable.eval_config` (per-case runtime data
            forwarded to the eval's builder, e.g. tau2 `initial_state` or BFCL
            `involved_classes` / `initial_config`).

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no eval name or model name is provided.
    """
    from deepagents_evals.eval_registry import EVALS  # noqa: PLC0415

    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))

    eval_name = configurable.get("eval_name")
    if not isinstance(eval_name, str) or not eval_name.strip():
        msg = "`configurable.eval_name` must provide an eval name"
        raise ValueError(msg)

    spec = EVALS.get(eval_name)
    if spec is None:
        msg = f"Unknown eval_name {eval_name!r}. Registered evals: {sorted(EVALS)}"
        raise ValueError(msg)

    repl_name = configurable.get("repl_name")
    if repl_name is not None and not isinstance(repl_name, str):
        msg = "`configurable.repl_name` must be a string or omitted"
        raise TypeError(msg)

    if repl_name is not None and not spec.supports_repl:
        msg = f"Eval {eval_name!r} does not support repl_name"
        raise ValueError(msg)

    eval_config = configurable.get("eval_config")
    if eval_config is not None and not isinstance(eval_config, dict):
        msg = "`configurable.eval_config` must be a mapping or omitted"
        raise TypeError(msg)
    eval_config = {str(k): v for k, v in eval_config.items()} if eval_config else None

    return spec.build(model, repl_name=repl_name, config=eval_config)
